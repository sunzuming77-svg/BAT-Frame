import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WavLMModel
from mamba_blocks import MixerModel
from conformer_blocks import ConformerEncoder


class BalanceBCELoss(nn.Module):
    def __init__(self, negative_ratio=5.0, eps=1e-8):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, target, mask=None):
        pred = pred.squeeze(-1)
        target = target.squeeze(-1).float()
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        if mask is not None:
            mask = mask.squeeze(-1).float() if mask.dim() == loss.dim() + 1 else mask.float()
            loss = loss * mask

        positive_index = (target == 1.0)
        negative_index = (target == 0.0)
        if mask is not None:
            valid_mask = mask > 0
            positive_index = positive_index & valid_mask
            negative_index = negative_index & valid_mask

        positive_count = int(positive_index.sum().item())
        negative_count = min(int(negative_index.sum().item()), int(max(1, positive_count) * self.negative_ratio))

        if positive_count == 0 and negative_count == 0:
            return loss.new_zeros(())

        positive_loss = loss[positive_index]
        negative_loss = loss[negative_index]

        if negative_count > 0 and negative_loss.numel() > 0:
            negative_loss, _ = torch.topk(negative_loss.reshape(-1), k=min(negative_count, negative_loss.numel()))
        else:
            negative_loss = loss.new_zeros(0)
            negative_count = 0

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        return balance_loss


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=50.0):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, pred, target, mask=None):
        pred = pred.squeeze(-1)
        target = target.squeeze(-1).float()
        loss = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=self.pos_weight.to(pred.device), reduction='none'
        )

        if mask is not None:
            mask = mask.squeeze(-1).float() if mask.dim() == loss.dim() + 1 else mask.float()
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        return loss.mean()


class TransitionAwareBoundaryLoss(nn.Module):
    def __init__(self, pos_weight=30.0, sigma=1.5, dice_weight=0.5, tversky_weight=0.5, alpha=0.3, beta=0.7, eps=1e-6):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor([pos_weight], dtype=torch.float32))
        self.sigma = sigma
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def _soften_targets(self, target):
        target = target.float()
        if self.sigma <= 0:
            return target
        radius = max(int(round(self.sigma * 3)), 1)
        offsets = torch.arange(-radius, radius + 1, device=target.device, dtype=target.dtype)
        kernel = torch.exp(-0.5 * (offsets / self.sigma).pow(2))
        kernel = kernel.view(1, 1, -1)
        soft = F.conv1d(target.unsqueeze(1), kernel, padding=radius).squeeze(1)
        return soft.clamp(0.0, 1.0)

    def forward(self, pred, target, mask=None):
        pred = pred.squeeze(-1)
        target = target.squeeze(-1).float()
        soft_target = self._soften_targets(target)
        bce = F.binary_cross_entropy_with_logits(
            pred, soft_target, pos_weight=self.pos_weight.to(pred.device), reduction='none'
        )
        prob = torch.sigmoid(pred)

        if mask is not None:
            mask = mask.squeeze(-1).float() if mask.dim() == bce.dim() + 1 else mask.float()
        else:
            mask = torch.ones_like(bce)

        bce_loss = (bce * mask).sum() / (mask.sum() + self.eps)
        prob = prob * mask
        soft_target = soft_target * mask

        intersection = (prob * soft_target).sum()
        prob_sum = prob.sum()
        target_sum = soft_target.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (prob_sum + target_sum + self.eps)

        fp = (prob * (1.0 - soft_target) * mask).sum()
        fn = ((1.0 - prob) * soft_target * mask).sum()
        tversky = (intersection + self.eps) / (intersection + self.alpha * fp + self.beta * fn + self.eps)
        tversky_loss = 1.0 - tversky

        return bce_loss + self.dice_weight * dice_loss + self.tversky_weight * tversky_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.squeeze(-1)
        target = target.squeeze(-1).float()
        p = torch.sigmoid(pred)
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1.0 - p) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * ce
        return loss.sum() if self.reduction == 'sum' else loss.mean()


class P2SGradLoss(nn.Module):
    def __init__(self, scale=30.0, class_weight=None):
        super().__init__()
        self.scale = scale
        if class_weight is not None:
            self.register_buffer('class_weight', class_weight.float())
        else:
            self.class_weight = None

    def forward(self, feat, target, weight_matrix):
        feat_norm = F.normalize(feat, dim=-1)
        weight_norm = F.normalize(weight_matrix, dim=-1)
        scores = torch.matmul(feat_norm, weight_norm.t()) * self.scale
        bsz, num_frames, num_classes = scores.shape
        weight = self.class_weight.to(scores.device) if self.class_weight is not None else None
        return F.cross_entropy(
            scores.reshape(bsz * num_frames, num_classes),
            target.reshape(bsz * num_frames),
            weight=weight,
        )


class FrozenWavLMFrontend(nn.Module):
    def __init__(self, model_name='microsoft/wavlm-large', target_frames=208, finetune=True, local_files_only=True):
        super().__init__()
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.model = WavLMModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.target_frames = target_frames
        self.out_dim = self.model.config.hidden_size
        self.finetune = finetune
        num_hidden_layers = self.model.config.num_hidden_layers + 1  # embeddings + transformer layers
        self.layer_weights = nn.Parameter(torch.zeros(num_hidden_layers))
        for param in self.model.parameters():
            param.requires_grad = finetune
        if not finetune:
            self.model.eval()

    def _match_num_frames(self, features):
        num_frames = features.size(1)
        if num_frames == self.target_frames:
            return features
        if num_frames > self.target_frames:
            return features[:, :self.target_frames, :]
        pad_frames = self.target_frames - num_frames
        pad_tensor = features.new_zeros(features.size(0), pad_frames, features.size(2))
        return torch.cat([features, pad_tensor], dim=1)

    def forward(self, waveform):
        if waveform.dim() == 3:
            waveform = waveform.squeeze(-1)
        if self.finetune:
            outputs = self.model(input_values=waveform, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            stacked = torch.stack(hidden_states, dim=0)
        else:
            with torch.no_grad():
                outputs = self.model(input_values=waveform, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                stacked = torch.stack(hidden_states, dim=0)
        norm_weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        features = (stacked * norm_weights).sum(dim=0)
        features = self._match_num_frames(features)
        return features


class LogMelCNNFrontend(nn.Module):
    def __init__(self, d_model, sample_rate=16000, n_fft=400, hop_length=320, win_length=400, n_mels=128, target_frames=208):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            f_min=20.0, f_max=7600.0, n_mels=n_mels, power=2.0, center=False, normalized=False,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, target_frames))
        self.proj = nn.Linear(64, d_model)

    def forward(self, waveform):
        if waveform.dim() == 3:
            waveform = waveform.squeeze(-1)
        x = self.to_db(self.mel_spec(waveform).clamp_min(1e-5)).unsqueeze(1)
        x = self.cnn(x)
        x = self.freq_pool(x).squeeze(2).transpose(1, 2)
        return self.proj(x)


class BoundaryAwareHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden = max(d_model // 2, 32)
        self.net = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden), nn.SiLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden), nn.SiLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, h):
        return self.net(h.transpose(1, 2)).transpose(1, 2)


class BoundaryControlledMixer(nn.Module):
    def __init__(self, d_model, n_layer, enabled=True, bidirectional=True):
        super().__init__()
        self.enabled = enabled
        self.mixer = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            ssm_cfg={},
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            if_bidirectional=bidirectional,
        )
        self.control = nn.Sequential(nn.Linear(d_model + 1, d_model), nn.SiLU(inplace=True), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, boundary_prob):
        mixed = self.mixer(x)
        if self.enabled:
            gate = self.control(torch.cat([x, boundary_prob], dim=-1))
            out = x + gate * (mixed - x)
        else:
            gate = torch.ones_like(x)
            out = mixed
        return self.norm(out), gate


class BoundaryBiasedConformerMixer(nn.Module):
    """
    Conformer-based sequence mixer with boundary-biased attention.
    
    Key Innovation: Instead of hard gating (which fails in OOD scenarios when
    boundary prediction confidence drops), we inject boundary probability as
    a soft attention bias directly into the Conformer's self-attention mechanism.
    
    This ensures that even when boundary predictions are uncertain, the model
    can still leverage local contrast through attention, avoiding the "gate lock"
    failure mode observed with Mamba's hard gating approach.
    """
    def __init__(self, d_model, n_layer, enabled=True, boundary_bias_scale=1.0, 
                 num_heads=8, conv_kernel_size=31, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.enabled = enabled
        self.d_model = d_model
        self.conformer = ConformerEncoder(
            d_model=d_model,
            num_layers=n_layer,
            num_heads=num_heads,
            conv_kernel_size=conv_kernel_size,
            expansion_factor=expansion_factor,
            dropout=dropout,
            boundary_bias_scale=boundary_bias_scale if enabled else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, boundary_prob):
        if self.enabled:
            out, all_attn_weights = self.conformer(x, boundary_prob)
        else:
            out, all_attn_weights = self.conformer(x, None)
        
        avg_attn = torch.stack(all_attn_weights, dim=0).mean(dim=0).mean(dim=1)
        gate = avg_attn.mean(dim=-1, keepdim=True).unsqueeze(-1).expand_as(x)
        
        return self.norm(out), gate


class BoundaryRoutedCrossStream(nn.Module):
    def __init__(self, d_model, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.time_to_freq = nn.Linear(d_model, d_model)
        self.freq_to_time = nn.Linear(d_model, d_model)
        self.route = nn.Sequential(nn.Linear(d_model * 2 + 1, d_model), nn.SiLU(inplace=True), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.time_norm = nn.LayerNorm(d_model)
        self.freq_norm = nn.LayerNorm(d_model)

    def forward(self, h_time, h_freq, boundary_prob):
        if self.enabled:
            route = self.route(torch.cat([h_time, h_freq, boundary_prob], dim=-1))
            new_time = self.time_norm(h_time + route * self.freq_to_time(h_freq))
            new_freq = self.freq_norm(h_freq + route * self.time_to_freq(new_time))
        else:
            route = torch.zeros_like(h_time)
            new_time = self.time_norm(h_time)
            new_freq = self.freq_norm(h_freq)
        return new_time, new_freq, route


class SoftSegmentParser(nn.Module):
    def __init__(self, d_model, num_segments=4, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.num_segments = num_segments
        self.assign = nn.Linear(d_model + 1, num_segments)
        self.refine = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.SiLU(inplace=True), nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_frame, boundary_prob):
        if self.enabled:
            assign_logits = self.assign(torch.cat([h_frame, boundary_prob], dim=-1))
            assign = F.softmax(assign_logits, dim=-1)
            denom = assign.sum(dim=1, keepdim=True).transpose(1, 2).clamp_min(1e-5)
            segments = torch.matmul(assign.transpose(1, 2), h_frame) / denom
            segment_back = torch.matmul(assign, segments)
            h_segment = self.norm(h_frame + self.refine(torch.cat([h_frame, segment_back], dim=-1)))
            seg_consistency = ((h_frame - segment_back) ** 2).mean()
            seg_entropy = -(assign * (assign.clamp_min(1e-8).log())).sum(dim=-1).mean()
        else:
            batch_size, num_frames, d_model = h_frame.shape
            assign = h_frame.new_zeros(batch_size, num_frames, self.num_segments)
            assign[..., 0] = 1.0
            segments = h_frame.mean(dim=1, keepdim=True).expand(-1, self.num_segments, -1)
            segment_back = h_frame
            h_segment = self.norm(h_frame)
            seg_consistency = h_frame.new_zeros(())
            seg_entropy = h_frame.new_zeros(())
        return h_segment, {'segment_consistency': seg_consistency, 'segment_entropy': seg_entropy, 'assignments': assign, 'segments': segments, 'segment_back': segment_back}


class LocalEdgeRefiner(nn.Module):
    def __init__(self, d_model, kernel_size=5, hidden_scale=1.0, boundary_gate_scale=1.0, detach_boundary_gate=True, local_refine_min_gate=0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('local edge refiner kernel_size must be odd')
        hidden_dim = max(int(round(d_model * hidden_scale)), d_model)
        padding = kernel_size // 2
        self.boundary_gate_scale = boundary_gate_scale
        self.detach_boundary_gate = detach_boundary_gate
        self.local_refine_min_gate = float(local_refine_min_gate)
        self.local_conv = nn.Sequential(
            nn.Conv1d(d_model, hidden_dim, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_dim, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.SiLU(inplace=True),
        )
        self.local_gate = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.SiLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, h, boundary_prob):
        boundary_gate = boundary_prob.detach() if self.detach_boundary_gate else boundary_prob
        local_feat = self.local_conv(h.transpose(1, 2)).transpose(1, 2)
        gate = self.local_gate(torch.cat([h, boundary_gate * self.boundary_gate_scale], dim=-1))
        effective_gate = self.local_refine_min_gate + (1.0 - self.local_refine_min_gate) * gate
        out = h + effective_gate * local_feat
        return self.out_norm(out), effective_gate, local_feat


class HierarchicalAttractorHead(nn.Module):
    def __init__(self, d_model, num_classes=3, num_heads=4, dropout=0.1, num_segments=4, use_soft_segments=True, boundary_temp_strength=0.0, boundary_temp_min=0.5, use_binary_head=True, use_segment_position_head=True, local_refine_enabled=True, local_refine_kernel=5, local_refine_hidden_scale=1.0, boundary_gate_scale=1.0, detach_boundary_gate=True, local_refine_min_gate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.boundary_temp_strength = boundary_temp_strength
        self.boundary_temp_min = boundary_temp_min
        self.use_binary_head = use_binary_head
        self.use_segment_position_head = use_segment_position_head
        self.local_refine_enabled = local_refine_enabled
        self.attractor_tokens = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        self.segment_tokens = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        self.frame_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.segment_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.segment_parser = SoftSegmentParser(d_model=d_model, num_segments=num_segments, enabled=use_soft_segments)
        self.frame_norm = nn.LayerNorm(d_model)
        self.segment_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.SiLU(inplace=True), nn.Linear(d_model, d_model))
        self.fusion_norm = nn.LayerNorm(d_model)
        self.local_refiner = LocalEdgeRefiner(
            d_model=d_model,
            kernel_size=local_refine_kernel,
            hidden_scale=local_refine_hidden_scale,
            boundary_gate_scale=boundary_gate_scale,
            detach_boundary_gate=detach_boundary_gate,
            local_refine_min_gate=local_refine_min_gate,
        ) if local_refine_enabled else None
        self.classifier = nn.Linear(d_model, num_classes)
        self.binary_classifier = nn.Linear(d_model, 2) if use_binary_head else None
        self.segment_position_classifier = nn.Linear(d_model, 5) if use_segment_position_head else None

    def forward(self, h_enhanced, boundary_prob):
        batch_size = h_enhanced.size(0)
        frame_tokens = self.attractor_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        frame_ctx, frame_attn_map = self.frame_attn(query=h_enhanced, key=frame_tokens, value=frame_tokens, need_weights=True, average_attn_weights=False)
        h_frame = self.frame_norm(h_enhanced + frame_ctx)
        stable_weight = (1.0 - boundary_prob).clamp_min(1e-4)
        stable_summary = (h_frame * stable_weight).sum(dim=1, keepdim=True) / stable_weight.sum(dim=1, keepdim=True)
        segment_tokens = self.segment_tokens.unsqueeze(0).expand(batch_size, -1, -1) + stable_summary
        segment_ctx, segment_attn_map = self.segment_attn(query=h_frame, key=segment_tokens, value=segment_tokens, need_weights=True, average_attn_weights=False)
        h_context = self.segment_norm(h_frame + segment_ctx)
        h_segment, aux = self.segment_parser(h_context, boundary_prob)
        h_prime = self.fusion_norm(h_segment + self.fusion(torch.cat([h_frame, h_context, h_segment], dim=-1)))
        if self.local_refiner is not None:
            h_prime_refined, local_refine_gate, local_refine_feat = self.local_refiner(h_prime, boundary_prob)
        else:
            h_prime_refined = h_prime
            local_refine_gate = torch.zeros_like(h_prime)
            local_refine_feat = torch.zeros_like(h_prime)
        logits = self.classifier(h_prime_refined)
        if self.boundary_temp_strength > 0:
            temperature = (1.0 - self.boundary_temp_strength * boundary_prob.detach()).clamp_min(self.boundary_temp_min)
            logits = logits / temperature
        aux['binary_logits'] = self.binary_classifier(h_prime_refined) if self.binary_classifier is not None else None
        aux['segment_position_logits'] = self.segment_position_classifier(h_prime_refined) if self.segment_position_classifier is not None else None
        aux['frame_attn_map'] = frame_attn_map
        aux['segment_attn_map'] = segment_attn_map
        aux['stable_summary'] = stable_summary
        aux['local_refine_gate'] = local_refine_gate
        aux['local_refine_feat'] = local_refine_feat
        return logits, h_prime_refined, aux

    def compute_attractor_repulsion_loss(self):
        if self.num_classes <= 1:
            return self.attractor_tokens.new_zeros(())
        tokens = F.normalize(self.attractor_tokens, p=2, dim=-1)
        sim_matrix = torch.matmul(tokens, tokens.t())
        eye = torch.eye(self.num_classes, device=tokens.device, dtype=tokens.dtype)
        off_diagonal_sim = sim_matrix * (1.0 - eye)
        return F.relu(off_diagonal_sim).sum() / (self.num_classes * (self.num_classes - 1))


class UtteranceAttentionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden = max(d_model // 2, 32)
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, h_frame, boundary_prob):
        spoof_focus = boundary_prob.detach()
        attn_logits = self.attn(h_frame) + spoof_focus
        attn = torch.softmax(attn_logits, dim=1)
        utt_emb = (attn * h_frame).sum(dim=1)
        utt_logit = self.classifier(utt_emb)
        return utt_logit, attn, utt_emb


class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.d_model = getattr(args, 'emb_size', 144)
        self.num_classes = getattr(args, 'num_classes', 3)
        self.num_frames = 208
        self.cached_aux_losses = {}
        self.cached_analysis = {}
        
        # Backbone selection: 'mamba' or 'conformer'
        self.backbone_type = getattr(args, 'backbone_type', 'mamba')
        
        self.use_boundary_control = getattr(args, 'use_boundary_control', True)
        self.use_cross_routing = getattr(args, 'use_cross_routing', True)
        self.use_soft_segments = getattr(args, 'use_soft_segments', True)
        self.mamba_bidirectional = getattr(args, 'mamba_bidirectional', True)
        self.wavlm_local_files_only = getattr(args, 'wavlm_local_files_only', True)
        self.attractor_boundary_temp_strength = getattr(args, 'attractor_boundary_temp_strength', 0.0)
        self.attractor_boundary_temp_min = getattr(args, 'attractor_boundary_temp_min', 0.5)
        self.local_refine_enabled = getattr(args, 'local_refine_enabled', True)
        self.local_refine_kernel = getattr(args, 'local_refine_kernel', 5)
        self.local_refine_hidden_scale = getattr(args, 'local_refine_hidden_scale', 1.0)
        self.boundary_gate_scale = getattr(args, 'boundary_gate_scale', 1.0)
        self.detach_boundary_gate = getattr(args, 'detach_boundary_gate', True)
        self.local_refine_min_gate = getattr(args, 'local_refine_min_gate', 0.0)
        
        # Conformer-specific parameters
        self.boundary_bias_scale = getattr(args, 'boundary_bias_scale', 1.0)
        self.conformer_num_heads = getattr(args, 'conformer_num_heads', 8)
        self.conformer_conv_kernel = getattr(args, 'conformer_conv_kernel', 31)
        self.conformer_expansion_factor = getattr(args, 'conformer_expansion_factor', 4)
        self.conformer_dropout = getattr(args, 'conformer_dropout', 0.1)
        
        num_layers = max(getattr(args, 'num_encoders', 12) // 2, 1)
        num_heads = self._pick_num_heads(self.d_model)
        num_segments = getattr(args, 'num_segments', 4)

        self.ssl_model = FrozenWavLMFrontend(
            model_name='microsoft/wavlm-large',
            target_frames=self.num_frames,
            finetune=getattr(args, 'FT_W2V', True),
            local_files_only=self.wavlm_local_files_only,
        )
        self.time_proj = nn.Linear(self.ssl_model.out_dim, self.d_model)
        self.time_norm = nn.LayerNorm(self.d_model)
        self.freq_stream = LogMelCNNFrontend(d_model=self.d_model, sample_rate=16000, n_fft=400, hop_length=320, win_length=400, n_mels=128, target_frames=self.num_frames)
        self.freq_norm = nn.LayerNorm(self.d_model)
        self.prior_fusion = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model), nn.SiLU(inplace=True), nn.LayerNorm(self.d_model))
        self.boundary_prior_head = BoundaryAwareHead(self.d_model)
        
        # Backbone selection: Mamba or Conformer
        if self.backbone_type == 'conformer':
            self.time_mixer = BoundaryBiasedConformerMixer(
                self.d_model, num_layers, 
                enabled=self.use_boundary_control,
                boundary_bias_scale=self.boundary_bias_scale,
                num_heads=self.conformer_num_heads,
                conv_kernel_size=self.conformer_conv_kernel,
                expansion_factor=self.conformer_expansion_factor,
                dropout=self.conformer_dropout
            )
            self.freq_mixer = BoundaryBiasedConformerMixer(
                self.d_model, num_layers,
                enabled=self.use_boundary_control,
                boundary_bias_scale=self.boundary_bias_scale,
                num_heads=self.conformer_num_heads,
                conv_kernel_size=self.conformer_conv_kernel,
                expansion_factor=self.conformer_expansion_factor,
                dropout=self.conformer_dropout
            )
        else:  # default: mamba
            self.time_mixer = BoundaryControlledMixer(
                self.d_model, num_layers, 
                enabled=self.use_boundary_control, 
                bidirectional=self.mamba_bidirectional
            )
            self.freq_mixer = BoundaryControlledMixer(
                self.d_model, num_layers, 
                enabled=self.use_boundary_control, 
                bidirectional=self.mamba_bidirectional
            )
        
        self.cross_stream = BoundaryRoutedCrossStream(self.d_model, enabled=self.use_cross_routing)
        self.fusion = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model), nn.SiLU(inplace=True), nn.Linear(self.d_model, self.d_model))
        self.fusion_norm = nn.LayerNorm(self.d_model)
        self.boundary_head = BoundaryAwareHead(self.d_model)
        self.attractor_head = HierarchicalAttractorHead(
            d_model=self.d_model,
            num_classes=self.num_classes,
            num_heads=num_heads,
            dropout=0.1,
            num_segments=num_segments,
            use_soft_segments=self.use_soft_segments,
            boundary_temp_strength=self.attractor_boundary_temp_strength,
            boundary_temp_min=self.attractor_boundary_temp_min,
            use_binary_head=getattr(args, 'use_binary_frame_head', True),
            use_segment_position_head=getattr(args, 'use_segment_position_head', True),
            local_refine_enabled=self.local_refine_enabled,
            local_refine_kernel=self.local_refine_kernel,
            local_refine_hidden_scale=self.local_refine_hidden_scale,
            boundary_gate_scale=self.boundary_gate_scale,
            detach_boundary_gate=self.detach_boundary_gate,
            local_refine_min_gate=self.local_refine_min_gate,
        )
        self.utterance_head = UtteranceAttentionHead(self.d_model)

        backbone_name = 'Conformer' if self.backbone_type == 'conformer' else 'Mamba'
        print(f'BAT-{backbone_name}: Boundary-Aware Transformer for Partial Spoofing Detection')
        print(f'Backbone: {self.backbone_type.upper()} | d_model={self.d_model}, num_layers={num_layers}, num_heads={num_heads}, num_segments={num_segments}, num_classes={self.num_classes}')
        print('WavLM finetune: {}'.format(getattr(args, 'FT_W2V', True)))
        print('WavLM local_files_only: {}'.format(self.wavlm_local_files_only))
        
        if self.backbone_type == 'conformer':
            print('Conformer config: num_heads={} conv_kernel={} expansion={} dropout={} boundary_bias_scale={}'.format(
                self.conformer_num_heads, self.conformer_conv_kernel, self.conformer_expansion_factor, 
                self.conformer_dropout, self.boundary_bias_scale
            ))
        else:
            print('Mamba bidirectional: {}'.format(self.mamba_bidirectional))
        
        print('Boundary-aware mechanism: enabled={}'.format(self.use_boundary_control))
        print('Attractor boundary temperature: strength={} min={}'.format(self.attractor_boundary_temp_strength, self.attractor_boundary_temp_min))
        print('Local edge refiner: enabled={} kernel={} hidden_scale={} boundary_gate_scale={} detach_boundary_gate={} min_gate={}'.format(
            self.local_refine_enabled, self.local_refine_kernel, self.local_refine_hidden_scale, self.boundary_gate_scale, self.detach_boundary_gate, self.local_refine_min_gate
        ))
        self.report_acceleration_status()

    def report_acceleration_status(self):
        if self.backbone_type == 'conformer':
            print('[Backbone] Conformer: boundary-biased attention enabled={}'.format(self.use_boundary_control))
        else:
            def mixer_status(name, module):
                mixer = module.mixer
                print(
                    '[Accel] {}: bidirectional={} rms_norm={} fused_add_norm={} residual_in_fp32={}'.format(
                        name,
                        mixer.if_bidirectional,
                        mixer.rms_norm,
                        mixer.fused_add_norm,
                        mixer.residual_in_fp32,
                    )
                )
            mixer_status('time_mixer', self.time_mixer)
            mixer_status('freq_mixer', self.freq_mixer)

    @staticmethod
    def _pick_num_heads(d_model):
        for candidate in [8, 6, 4, 3, 2, 1]:
            if d_model % candidate == 0:
                return candidate
        return 1

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        time_feat = self.time_norm(self.time_proj(self.ssl_model(x)))
        freq_feat = self.freq_norm(self.freq_stream(x))
        prior_feat = self.prior_fusion(torch.cat([time_feat, freq_feat], dim=-1))
        p_bound_prior = torch.sigmoid(self.boundary_prior_head(prior_feat))
        p_bound_prior_ctrl = p_bound_prior.detach()
        h_time, time_gate = self.time_mixer(time_feat, p_bound_prior_ctrl)
        h_freq, freq_gate = self.freq_mixer(freq_feat, p_bound_prior_ctrl)
        h_time, h_freq, route_map = self.cross_stream(h_time, h_freq, p_bound_prior_ctrl)
        h_fusion = self.fusion_norm(self.fusion(torch.cat([h_time, h_freq], dim=-1)))
        p_bound_logits = self.boundary_head(h_fusion)
        p_bound = torch.sigmoid(p_bound_logits)
        p_bound_ctrl = p_bound.detach()
        h_enhanced = h_fusion + (h_fusion * p_bound_ctrl)
        logits_dia, h_prime, aux_losses = self.attractor_head(h_enhanced, p_bound_ctrl)
        utt_logit, utt_attn, utt_emb = self.utterance_head(h_prime, p_bound_ctrl)
        boundary_shift = (p_bound[:, 1:] - p_bound[:, :-1]).abs().mean()
        self.cached_aux_losses = {
            'segment_consistency': aux_losses['segment_consistency'],
            'segment_entropy': aux_losses['segment_entropy'],
            'boundary_sparsity': p_bound.mean(),
            'boundary_sharpness': -boundary_shift,
            'binary_logits': aux_losses.get('binary_logits', None),
            'segment_position_logits': aux_losses.get('segment_position_logits', None),
        }
        self.cached_analysis = {
            'boundary_prior': p_bound_prior.detach(),
            'boundary_final': p_bound.detach(),
            'time_gate': time_gate.detach(),
            'freq_gate': freq_gate.detach(),
            'route_map': route_map.detach(),
            'segment_assignments': aux_losses['assignments'].detach(),
            'segment_centers': aux_losses['segments'].detach(),
            'segment_backproj': aux_losses['segment_back'].detach(),
            'frame_attn_map': aux_losses['frame_attn_map'].detach(),
            'segment_attn_map': aux_losses['segment_attn_map'].detach(),
            'local_refine_gate': aux_losses['local_refine_gate'].detach(),
            'local_refine_feat': aux_losses['local_refine_feat'].detach(),
            'stable_summary': aux_losses['stable_summary'].detach(),
            'utt_attn': utt_attn.detach(),
            'utt_emb': utt_emb.detach(),
        }
        return p_bound_logits, logits_dia, h_prime, utt_logit

    def compute_spoof_ratio(self, logits_dia):
        preds = logits_dia.argmax(dim=-1)
        batch_ratios = []
        for sample_pred in preds:
            class_ratios = {}
            for cls_idx in range(self.num_classes):
                class_ratios[cls_idx] = (sample_pred == cls_idx).float().mean().item() * 100.0
            batch_ratios.append(class_ratios)
        return batch_ratios
