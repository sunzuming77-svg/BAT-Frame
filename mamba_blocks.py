# Copyright (c) 2023, Albert Gu, Tri Dao.
# BAT-Mamba: Modified for frame-level dense prediction
# Key Change: Removed Attention Pooling -> MixerModel now outputs [B, T, D]

import math
from functools import partial
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.models.mixer_seq_simple import Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# BiBlock: dead code, removed.
# BiMixerModel: dead code, removed.
# MixerModel handles bidirectionality via forward_layers + backward_layers + merge_proj.


def create_block(
    d_model, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False,
    residual_in_fp32=False, fused_add_norm=False,
    layer_idx=None, device=None, dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)

    effective_rms_norm = bool(rms_norm and (RMSNorm is not None))
    effective_fused_add_norm = bool(fused_add_norm and (layer_norm_fn is not None) and (rms_norm_fn is not None))

    norm_cls = partial(
        nn.LayerNorm if not effective_rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model, mixer_cls, mlp_cls=nn.Identity,
        norm_cls=norm_cls, fused_add_norm=effective_fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module, n_layer, initializer_range=0.02,
    rescale_prenorm_residual=True, n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# ============================================================
# MixerModel (BAT-Mamba version)
# OUTPUT: frame-level hidden states H_{1:T}  shape: [B, T, D]
# Attention Pooling REMOVED. Dual-head decoder lives in model.py.
# ============================================================
class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        if_bidirectional: bool = True,
        initializer_cfg=None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.if_bidirectional = if_bidirectional
        self.requested_rms_norm = bool(rms_norm)
        self.requested_fused_add_norm = bool(fused_add_norm)
        self.rms_norm = bool(rms_norm and (RMSNorm is not None))
        self.fused_add_norm = bool(fused_add_norm and (layer_norm_fn is not None) and (rms_norm_fn is not None))
        if self.requested_rms_norm and not self.rms_norm:
            print('[MixerModel] RMSNorm Triton kernel unavailable. Falling back to nn.LayerNorm.')
        if self.requested_fused_add_norm and not self.fused_add_norm:
            print('[MixerModel] fused_add_norm kernels unavailable. Falling back to non-fused add+norm.')
        self.forward_layers = nn.ModuleList(
            [create_block(d_model, ssm_cfg=ssm_cfg, norm_epsilon=norm_epsilon,
                          rms_norm=self.rms_norm, residual_in_fp32=residual_in_fp32,
                          fused_add_norm=self.fused_add_norm, layer_idx=i, **factory_kwargs)
             for i in range(n_layer)]
        )
        self.backward_layers = nn.ModuleList(
            [create_block(d_model, ssm_cfg=ssm_cfg, norm_epsilon=norm_epsilon,
                          rms_norm=self.rms_norm, residual_in_fp32=residual_in_fp32,
                          fused_add_norm=self.fused_add_norm, layer_idx=i, **factory_kwargs)
             for i in range(n_layer)]
        )
        self.norm_f = (nn.LayerNorm if not self.rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        # Merge fwd + bwd: [B,T,D*2] -> [B,T,D]
        self.merge_proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.forward_layers)
        }

    def _apply_norm(self, hidden_states, residual):
        """Apply final LayerNorm (fused or non-fused)."""
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_fn(
                hidden_states, self.norm_f.weight, self.norm_f.bias,
                eps=self.norm_f.eps, residual=residual,
                prenorm=False, residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states

    def forward(self, x, inference_params=None):
        # x: [B, T, D]
        hidden_states = self.dropout(x)

        if not self.if_bidirectional:
            residual = None
            for layer in self.forward_layers:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params)
            return self._apply_norm(hidden_states, residual)  # [B, T, D]

        # --- Bidirectional path ---
        f_hidden_states = hidden_states
        b_hidden_states = hidden_states.flip([1])
        f_residual, b_residual = None, None

        for layer in self.forward_layers:
            f_hidden_states, f_residual = layer(
                f_hidden_states, f_residual, inference_params=inference_params
            )
        for layer in self.backward_layers:
            b_hidden_states, b_residual = layer(
                b_hidden_states, b_residual, inference_params=inference_params
            )

        f_hidden_states = self._apply_norm(f_hidden_states, f_residual)
        b_hidden_states = self._apply_norm(b_hidden_states, b_residual)
        b_hidden_states = b_hidden_states.flip([1])

        merged = torch.cat((f_hidden_states, b_hidden_states), dim=-1)
        return self.merge_proj(merged)  # [B, T, D]


# ============================================================
# BiMixerModel removed: dead code.
# Use MixerModel with if_bidirectional=True (default).
# ============================================================
