import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

import eval_metrics as em
from data_utils import load_seglab, load_frame_label_dict, parse_ps_protocol, _load_audio, seglab_to_frame_labels, multiclass_frame_labels_to_tensors, CUT
from model import Model
from main import get_unwrapped_model
from utils import pad


EXPECTED_NUM_CLASSES = {
    'binary': 2,
    'family3': 3,
    'attack19': 20,
}


NUM_FRAMES = 208


def compute_binary_prf(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)

def dilate_binary_events(events, tolerance):
    events = np.asarray(events, dtype=np.int64).reshape(-1)
    if tolerance <= 0 or events.size == 0:
        return events
    kernel = np.ones(2 * tolerance + 1, dtype=np.int64)
    return (np.convolve(events, kernel, mode='same') > 0).astype(np.int64)


def median_filter_1d(x, kernel_size):
    x = np.asarray(x)
    if kernel_size <= 1:
        return x.copy()
    if kernel_size % 2 == 0:
        raise ValueError('median kernel size must be odd')
    pad_width = kernel_size // 2
    padded = np.pad(x, (pad_width, pad_width), mode='edge')
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = np.median(padded[i:i + kernel_size])
    return out


def gaussian_smooth_1d(x, sigma):
    x = np.asarray(x, dtype=np.float32)
    if sigma <= 0:
        return x.copy()
    radius = max(int(round(sigma * 3)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * np.square(offsets / float(sigma))).astype(np.float32)
    kernel /= np.sum(kernel)
    padded = np.pad(x, (radius, radius), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed.astype(np.float32)


def remove_short_positive_runs(mask, min_len):
    mask = np.asarray(mask, dtype=np.int64).copy()
    if min_len <= 1 or mask.size == 0:
        return mask
    start = None
    for i, v in enumerate(mask):
        if v == 1 and start is None:
            start = i
        end_run = (v == 0 or i == mask.size - 1) and start is not None
        if end_run:
            end = i if v == 0 else i + 1
            if end - start < min_len:
                mask[start:end] = 0
            start = None
    return mask


def boundary_peak_nms(prob, threshold, radius):
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    if radius <= 0:
        return (prob >= threshold).astype(np.int64)
    candidates = np.where(prob >= threshold)[0]
    if candidates.size == 0:
        return np.zeros_like(prob, dtype=np.int64)
    order = candidates[np.argsort(prob[candidates])[::-1]]
    keep = []
    suppressed = np.zeros(prob.shape[0], dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        lo = max(0, idx - radius)
        hi = min(prob.shape[0], idx + radius + 1)
        suppressed[lo:hi] = True
    out = np.zeros_like(prob, dtype=np.int64)
    out[np.asarray(keep, dtype=np.int64)] = 1
    return out


def compute_tolerant_boundary_prf(y_true_2d, y_pred_2d, tolerance):
    y_true_2d = np.asarray(y_true_2d, dtype=np.int64)
    y_pred_2d = np.asarray(y_pred_2d, dtype=np.int64)
    if y_true_2d.ndim == 1:
        y_true_2d = y_true_2d.reshape(1, -1)
        y_pred_2d = y_pred_2d.reshape(1, -1)

    tp_precision = 0
    tp_recall = 0
    pred_count = 0
    true_count = 0

    for t_seq, p_seq in zip(y_true_2d, y_pred_2d):
        pred_hits = p_seq & dilate_binary_events(t_seq, tolerance)
        true_hits = t_seq & dilate_binary_events(p_seq, tolerance)
        tp_precision += int(pred_hits.sum())
        tp_recall += int(true_hits.sum())
        pred_count += int(p_seq.sum())
        true_count += int(t_seq.sum())

    precision = 0.0 if pred_count == 0 else tp_precision / pred_count
    recall = 0.0 if true_count == 0 else tp_recall / true_count
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def parse_float_list(s):
    return [float(x.strip()) for x in str(s).split(',') if x.strip()]


def validate_label_mode_and_num_classes(args):
    expected = EXPECTED_NUM_CLASSES[args.ps_label_mode]
    if int(args.num_classes) != expected:
        raise ValueError(
            f"num_classes={args.num_classes} does not match ps_label_mode={args.ps_label_mode}. "
            f"Expected num_classes={expected}."
        )


class EvalWaveDataset(torch.utils.data.Dataset):
    def __init__(self, list_ids, seglab, utt_labels, wav_base_dir, skip_missing=True, label_mode='binary'):
        self.list_ids = list_ids
        self.seglab = seglab
        self.utt_labels = utt_labels if utt_labels is not None else {}
        self.wav_base_dir = Path(wav_base_dir)
        self.skip_missing = skip_missing
        self.label_mode = label_mode

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, idx):
        utt_id = self.list_ids[idx]
        wav_path = self.wav_base_dir / f'{utt_id}.wav'
        if not wav_path.exists():
            if self.skip_missing:
                return {'_error': f'missing wav: {wav_path}', '_idx': idx}
            raise FileNotFoundError(f'missing wav: {wav_path}')

        x = _load_audio(str(wav_path), sr=16000)
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            x = np.zeros(CUT, dtype=np.float32)
        else:
            x = pad(x, CUT).astype(np.float32)
        waveform = torch.from_numpy(x).float().contiguous()

        raw_lab = self.seglab.get(utt_id)
        if raw_lab is None:
            raw_lab = np.ones(NUM_FRAMES, dtype=str) if self.label_mode == 'binary' else np.zeros(NUM_FRAMES, dtype=np.int64)
        if self.label_mode == 'binary':
            frame_labels, boundary_labels = seglab_to_frame_labels(raw_lab, num_frames=NUM_FRAMES)
        else:
            frame_labels, boundary_labels = multiclass_frame_labels_to_tensors(raw_lab, num_frames=NUM_FRAMES)
        utt_label = torch.tensor(self.utt_labels.get(utt_id, int(frame_labels.max().item() > 0)), dtype=torch.long)

        return waveform, frame_labels, boundary_labels, utt_label, utt_id


class SafeEvalDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        try:
            return self.base[idx]
        except Exception as exc:
            return {'_error': str(exc), '_idx': idx}


def load_model(args, device):
    model = Model(args, device).to(device)

    if args.average_model:
        best_dir = Path(args.model_dir) / 'best'
        ckpts = []
        for i in range(args.n_average_model):
            p = best_dir / f'best_{i}.pth'
            if p.exists() and p.stat().st_size > 1000:
                ckpts.append(p)
        if not ckpts:
            raise FileNotFoundError(f'No best_i checkpoints found in: {best_dir}')

        model.load_state_dict(torch.load(str(ckpts[0]), map_location=device))
        sd = model.state_dict()
        for p in ckpts[1:]:
            model.load_state_dict(torch.load(str(p), map_location=device))
            sd2 = model.state_dict()
            for k in sd:
                sd[k] = sd[k] + sd2[k]
        for k in sd:
            sd[k] = sd[k] / len(ckpts)
        model.load_state_dict(sd)
        print(f'[Model] Averaged {len(ckpts)} checkpoints from {best_dir}')
    else:
        ckpt = Path(args.checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
        model.load_state_dict(torch.load(str(ckpt), map_location=device))
        print(f'[Model] Loaded checkpoint: {ckpt}')

    model.eval()
    return model


def build_dataset(args):
    proto_path = Path(args.protocol_path)
    wav_base = Path(args.wav_base_dir)

    ids, utt_labels = parse_ps_protocol(str(proto_path), is_eval=False)
    if args.ps_label_mode == 'binary':
        seglab = load_seglab(str(Path(args.seglab_path)))
    else:
        seglab = load_frame_label_dict(str(Path(args.seglab_path)))

    base_ds = EvalWaveDataset(
        list_ids=ids,
        seglab=seglab,
        utt_labels=utt_labels,
        wav_base_dir=str(wav_base),
        skip_missing=args.skip_missing,
        label_mode=args.ps_label_mode,
    )
    return SafeEvalDataset(base_ds), ids, utt_labels


def parse_args():
    p = argparse.ArgumentParser(description='Standalone eval script for PartialSpoof eval set')

    p.add_argument('--model_dir', type=str, required=True,
                   help='Path to model run dir, e.g. models/BATmamba...')
    p.add_argument('--average_model', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--n_average_model', type=int, default=5)
    p.add_argument('--checkpoint_path', type=str, default='')

    p.add_argument('--protocol_path', type=str,
                   default='/mnt/c/PS_data/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.eval.trl.txt')
    p.add_argument('--seglab_path', type=str,
                   default='/mnt/c/PS_data/multiclass_labels/eval/family3_frames_0.02.npy')
    p.add_argument('--ps_label_mode', type=str, default='family3', choices=['binary', 'family3', 'attack19'])
    p.add_argument('--wav_base_dir', type=str,
                   default='/mnt/c/PS_data/eval/con_wav')

    p.add_argument('--score_output', type=str,
                   default='./Scores/PartialSpoof/eval_family3_scores.txt')
    p.add_argument('--skip_missing', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])

    p.add_argument('--utt_threshold', type=float, default=0.5)
    p.add_argument('--frame_threshold', type=float, default=0.5)
    p.add_argument('--family3_frame_decision', type=str, default='argmax', choices=['argmax', 'spoof_threshold'],
                   help='For family3/attack19 eval: argmax keeps multiclass argmax; spoof_threshold uses 1-P(bonafide) with optional median/min-run postprocess for binary spoof decision.')
    p.add_argument('--frame_score_source', type=str, default='family3', choices=['family3', 'binary_head', 'spl_head'],
                   help='Frame spoof score source: family3=1-P(bona), binary_head=P(spoof), spl_head=sum spoof segment-position states.')
    p.add_argument('--frame_gaussian_sigma', type=float, default=0.0,
                   help='Gaussian smoothing sigma in frames applied to spoof probability before thresholding. 0 disables it.')
    p.add_argument('--frame_median_kernel', type=int, default=1,
                   help='Odd kernel size for median filtering frame spoof probabilities. 1 disables it.')
    p.add_argument('--frame_min_spoof_len', type=int, default=1,
                   help='Remove predicted spoof runs shorter than this many frames. 1 disables it.')
    p.add_argument('--bound_threshold', type=float, default=0.5)
    p.add_argument('--bound_nms_radius', type=int, default=0,
                   help='Boundary peak NMS radius in frames. 0 disables it.')
    p.add_argument('--auto_bound_threshold_sweep', default=False, type=lambda x: str(x).lower() in ['true', 'yes', '1'],
                   help='If true, sweep boundary thresholds and pick best by BoundF1@5.')
    p.add_argument('--bound_sweep_values', type=str, default='0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90',
                   help='Comma-separated boundary thresholds used when auto sweep is enabled.')

    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--emb_size', '--emb-size', dest='emb_size', type=int, default=256)
    p.add_argument('--num_encoders', type=int, default=12)
    p.add_argument('--num_classes', type=int, default=3)
    p.add_argument('--algo', type=int, default=5)
    p.add_argument('--N_f', type=int, default=5)
    p.add_argument('--nBands', type=int, default=5)
    p.add_argument('--minF', type=int, default=20)
    p.add_argument('--maxF', type=int, default=8000)
    p.add_argument('--minBW', type=int, default=100)
    p.add_argument('--maxBW', type=int, default=1000)
    p.add_argument('--minCoeff', type=int, default=10)
    p.add_argument('--maxCoeff', type=int, default=100)
    p.add_argument('--minG', type=int, default=0)
    p.add_argument('--maxG', type=int, default=0)
    p.add_argument('--minBiasLinNonLin', type=int, default=5)
    p.add_argument('--maxBiasLinNonLin', type=int, default=20)
    p.add_argument('--P', type=int, default=10)
    p.add_argument('--g_sd', type=int, default=2)
    p.add_argument('--SNRmin', type=int, default=10)
    p.add_argument('--SNRmax', type=int, default=40)

    p.add_argument('--num_segments', type=int, default=4)
    p.add_argument('--use_boundary_control', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--use_cross_routing', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--use_soft_segments', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--use_binary_frame_head', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--use_segment_position_head', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--FT_W2V', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--wavlm_local_files_only', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--mamba_bidirectional', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--attractor_boundary_temp_strength', type=float, default=0.0)
    p.add_argument('--attractor_boundary_temp_min', type=float, default=0.5)
    p.add_argument('--local_refine_enabled', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--local_refine_kernel', type=int, default=5)
    p.add_argument('--local_refine_hidden_scale', type=float, default=1.0)
    p.add_argument('--boundary_gate_scale', type=float, default=1.0)
    p.add_argument('--detach_boundary_gate', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--local_refine_min_gate', type=float, default=0.0)

    p.add_argument('--track', type=str, default='LA')
    p.add_argument('--loss', type=str, default='WCE')
    p.add_argument('--lr', type=float, default=1e-6)

    return p.parse_args()


def main():
    args = parse_args()
    validate_label_mode_and_num_classes(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.average_model and not args.checkpoint_path:
        args.checkpoint_path = str(Path(args.model_dir) / 'best.pth')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[Device] {device}')

    model = load_model(args, device)
    dataset, _, _ = build_dataset(args)

    all_utt_true, all_utt_scores, all_utt_pred = [], [], []
    all_frame_true, all_frame_pred, all_frame_spoof_prob, all_frame_spoof_prob_raw = [], [], [], []
    all_frame_spoof_prob_family3, all_frame_spoof_prob_binary, all_frame_spoof_prob_spl = [], [], []
    all_bound_true_2d, all_bound_pred_2d, all_bound_prob_2d = [], [], []

    skipped = 0
    processed = 0

    score_output = Path(args.score_output)
    score_output.parent.mkdir(parents=True, exist_ok=True)

    with score_output.open('w', encoding='utf-8') as fh, torch.no_grad():
        for i in tqdm(range(len(dataset)), total=len(dataset)):
            item = dataset[i]
            if isinstance(item, dict) and '_error' in item:
                skipped += 1
                continue

            waveform, frame_labels, boundary_labels, utt_label, utt_id = item

            if waveform.numel() < 1000:
                skipped += 1
                continue

            x = waveform.unsqueeze(0).to(device)
            p_bound_logits, logits_dia, _, utt_logit = model(x)

            frame_prob_tensor = torch.softmax(logits_dia, dim=-1).squeeze(0)
            frame_prob = frame_prob_tensor.detach().cpu().numpy().astype(np.float32)
            family3_spoof_prob = (frame_prob[:, 1] if args.ps_label_mode == 'binary' else 1.0 - frame_prob[:, 0]).astype(np.float32)
            binary_spoof_prob = family3_spoof_prob.copy()
            spl_spoof_prob = family3_spoof_prob.copy()

            aux_losses = getattr(get_unwrapped_model(model), 'cached_aux_losses', {})
            binary_logits = aux_losses.get('binary_logits', None)
            if binary_logits is not None:
                binary_spoof_prob = torch.softmax(binary_logits, dim=-1).squeeze(0)[..., 1].detach().cpu().numpy().astype(np.float32)
            spl_logits = aux_losses.get('segment_position_logits', None)
            if spl_logits is not None:
                spl_prob = torch.softmax(spl_logits, dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
                spl_spoof_prob = spl_prob[:, 1:].sum(axis=-1).astype(np.float32)

            if args.frame_score_source == 'binary_head':
                frame_spoof_prob_raw = binary_spoof_prob.copy()
            elif args.frame_score_source == 'spl_head':
                frame_spoof_prob_raw = spl_spoof_prob.copy()
            else:
                frame_spoof_prob_raw = family3_spoof_prob.copy()

            frame_spoof_prob = frame_spoof_prob_raw.copy()
            if args.frame_gaussian_sigma > 0:
                frame_spoof_prob = gaussian_smooth_1d(frame_spoof_prob, args.frame_gaussian_sigma)
            if args.frame_median_kernel > 1:
                frame_spoof_prob = median_filter_1d(frame_spoof_prob, args.frame_median_kernel).astype(np.float32)

            if args.ps_label_mode != 'binary' and args.family3_frame_decision == 'argmax' and args.frame_score_source == 'family3':
                frame_pred = frame_prob.argmax(axis=-1).astype(np.int64)
            else:
                frame_pred = (frame_spoof_prob >= args.frame_threshold).astype(np.int64)
                frame_pred = remove_short_positive_runs(frame_pred, args.frame_min_spoof_len)
            bound_prob = torch.sigmoid(p_bound_logits).squeeze(-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
            bound_pred = boundary_peak_nms(bound_prob, args.bound_threshold, args.bound_nms_radius)
            utt_score = float(torch.sigmoid(utt_logit).squeeze().item())
            utt_pred = int(utt_score >= args.utt_threshold)

            frame_true = frame_labels.detach().cpu().numpy().astype(np.int64)
            bound_true = boundary_labels.detach().cpu().numpy().astype(np.int64)
            utt_true = int(utt_label.item())

            all_frame_true.append(frame_true)
            all_frame_pred.append(frame_pred)
            all_frame_spoof_prob.append(frame_spoof_prob)
            all_frame_spoof_prob_raw.append(frame_spoof_prob_raw)
            all_frame_spoof_prob_family3.append(family3_spoof_prob)
            all_frame_spoof_prob_binary.append(binary_spoof_prob)
            all_frame_spoof_prob_spl.append(spl_spoof_prob)
            all_bound_true_2d.append(bound_true)
            all_bound_pred_2d.append(bound_pred)
            all_bound_prob_2d.append(bound_prob)
            all_utt_true.append(utt_true)
            all_utt_scores.append(utt_score)
            all_utt_pred.append(utt_pred)

            fh.write(f'{utt_id} {utt_score}\n')
            processed += 1

    # Safety checkpoint: save all raw predictions/labels BEFORE metric computation.
    # This guarantees you can post-compute metrics even if summary stage crashes.
    raw_dump_path = score_output.with_suffix('.raw_eval.npz')
    np.savez_compressed(
        str(raw_dump_path),
        utt_true=np.asarray(all_utt_true, dtype=np.int64),
        utt_scores=np.asarray(all_utt_scores, dtype=np.float64),
        utt_pred=np.asarray(all_utt_pred, dtype=np.int64),
        frame_true=np.concatenate(all_frame_true) if all_frame_true else np.array([], dtype=np.int64),
        frame_pred=np.concatenate(all_frame_pred) if all_frame_pred else np.array([], dtype=np.int64),
        frame_spoof_prob=np.concatenate(all_frame_spoof_prob) if all_frame_spoof_prob else np.array([], dtype=np.float32),
        frame_spoof_prob_raw=np.concatenate(all_frame_spoof_prob_raw) if all_frame_spoof_prob_raw else np.array([], dtype=np.float32),
        frame_spoof_prob_family3=np.concatenate(all_frame_spoof_prob_family3) if all_frame_spoof_prob_family3 else np.array([], dtype=np.float32),
        frame_spoof_prob_binary=np.concatenate(all_frame_spoof_prob_binary) if all_frame_spoof_prob_binary else np.array([], dtype=np.float32),
        frame_spoof_prob_spl=np.concatenate(all_frame_spoof_prob_spl) if all_frame_spoof_prob_spl else np.array([], dtype=np.float32),
        frame_score_source=np.asarray([args.frame_score_source]),
        bound_true_2d=np.vstack(all_bound_true_2d) if all_bound_true_2d else np.empty((0, NUM_FRAMES), dtype=np.int64),
        bound_pred_2d=np.vstack(all_bound_pred_2d) if all_bound_pred_2d else np.empty((0, NUM_FRAMES), dtype=np.int64),
        bound_prob_2d=np.vstack(all_bound_prob_2d) if all_bound_prob_2d else np.empty((0, NUM_FRAMES), dtype=np.float32),
        processed=np.asarray([processed], dtype=np.int64),
        skipped=np.asarray([skipped], dtype=np.int64),
        total=np.asarray([len(dataset)], dtype=np.int64),
    )
    print(f'[SafeDump] Raw eval arrays saved: {raw_dump_path}')

    frame_true = np.concatenate(all_frame_true) if all_frame_true else np.array([], dtype=np.int64)
    frame_pred = np.concatenate(all_frame_pred) if all_frame_pred else np.array([], dtype=np.int64)

    bound_true_2d = np.vstack(all_bound_true_2d) if all_bound_true_2d else np.empty((0, NUM_FRAMES), dtype=np.int64)
    bound_pred_2d = np.vstack(all_bound_pred_2d) if all_bound_pred_2d else np.empty((0, NUM_FRAMES), dtype=np.int64)
    bound_prob_2d = np.vstack(all_bound_prob_2d) if all_bound_prob_2d else np.empty((0, NUM_FRAMES), dtype=np.float32)

    # Optional threshold sweep on boundary probabilities.
    selected_bound_threshold = float(args.bound_threshold)
    if args.auto_bound_threshold_sweep and bound_prob_2d.size > 0:
        sweep_values = parse_float_list(args.bound_sweep_values)
        if len(sweep_values) == 0:
            sweep_values = [selected_bound_threshold]
        best_f1 = -1.0
        best_tuple = None
        for th in sweep_values:
            cand_pred = np.vstack([boundary_peak_nms(p, th, args.bound_nms_radius) for p in bound_prob_2d])
            p5, r5, f5 = compute_tolerant_boundary_prf(bound_true_2d, cand_pred, 5)
            if f5 > best_f1:
                best_f1 = f5
                best_tuple = (th, cand_pred, p5, r5, f5)
        if best_tuple is not None:
            selected_bound_threshold, bound_pred_2d, sweep_p5, sweep_r5, sweep_f5 = best_tuple
            print(f'[Sweep] Selected bound threshold={selected_bound_threshold:.2f} by BoundF1@5={sweep_f5:.4f} (P/R={sweep_p5:.4f}/{sweep_r5:.4f})')

    utt_true = np.asarray(all_utt_true, dtype=np.int64)
    utt_scores = np.asarray(all_utt_scores, dtype=np.float64)
    utt_pred = np.asarray(all_utt_pred, dtype=np.int64)

    frame_score_all = np.concatenate(all_frame_spoof_prob) if all_frame_spoof_prob else np.array([], dtype=np.float32)

    frame_acc = float((frame_true == frame_pred).mean()) if frame_true.size > 0 else 0.0
    class_f1 = []
    for cls_idx in range(args.num_classes):
        cls_true = (frame_true == cls_idx).astype(np.int64)
        cls_pred = (frame_pred == cls_idx).astype(np.int64)
        class_f1.append(float(f1_score(cls_true, cls_pred, zero_division=0)) if frame_true.size > 0 else 0.0)
    macro_f1 = float(np.mean(class_f1)) if class_f1 else 0.0
    present_classes = sorted(np.unique(frame_true).astype(int).tolist()) if frame_true.size > 0 else []
    present_macro_f1 = float(np.mean([class_f1[i] for i in present_classes])) if present_classes else 0.0
    spoof_true = (frame_true > 0).astype(np.int64)
    spoof_pred = (frame_pred > 0).astype(np.int64)
    spoof_f1 = f1_score(spoof_true, spoof_pred, zero_division=0) if frame_true.size > 0 else 0.0
    if frame_score_all.size > 0 and np.any(spoof_true == 0) and np.any(spoof_true == 1):
        frame_eer, frame_thr = em.compute_eer(frame_score_all[spoof_true == 1], frame_score_all[spoof_true == 0])
    else:
        frame_eer, frame_thr = 1.0, 0.0

    bound_p1, bound_r1, bound_f1_at_1 = compute_tolerant_boundary_prf(bound_true_2d, bound_pred_2d, 1)
    bound_p2, bound_r2, bound_f1_at_2 = compute_tolerant_boundary_prf(bound_true_2d, bound_pred_2d, 2)
    bound_p5, bound_r5, bound_f1_at_5 = compute_tolerant_boundary_prf(bound_true_2d, bound_pred_2d, 5)

    utt_f1 = f1_score(utt_true, utt_pred, zero_division=0) if utt_true.size > 0 else 0.0
    bona_scores = utt_scores[utt_true == 0]
    spoof_scores = utt_scores[utt_true == 1]
    if bona_scores.size > 0 and spoof_scores.size > 0:
        utt_eer, utt_thr = em.compute_eer(spoof_scores, bona_scores)
    else:
        utt_eer, utt_thr = 1.0, 0.0

    print('\n===== Eval Summary (Eval Set) =====')
    print(f'Processed: {processed} / {len(dataset)} | Skipped: {skipped}')
    print(f'UttEER: {utt_eer:.4f} | UttF1: {utt_f1:.4f} | UttThr: {utt_thr:.6f}')
    print(f'FrameAcc: {frame_acc:.4f} | MacroF1: {macro_f1:.4f} | PresentMacroF1: {present_macro_f1:.4f} | SpoofF1: {spoof_f1:.4f} | FrameEER: {frame_eer:.4f} | FrameThr(EER): {frame_thr:.6f}')
    print(f'Present classes: {present_classes}')
    print(f'ClassF1: {class_f1}')
    print(f'BoundF1@1: {bound_f1_at_1:.4f} (P/R={bound_p1:.4f}/{bound_r1:.4f})')
    print(f'BoundF1@2: {bound_f1_at_2:.4f} (P/R={bound_p2:.4f}/{bound_r2:.4f})')
    print(f'BoundF1@5: {bound_f1_at_5:.4f} (P/R={bound_p5:.4f}/{bound_r5:.4f})')
    print(f'Score file: {score_output}')

    summary = {
        'processed': int(processed),
        'total': int(len(dataset)),
        'skipped': int(skipped),
        'ps_label_mode': args.ps_label_mode,
        'frame_score_source': str(args.frame_score_source),
        'family3_frame_decision': args.family3_frame_decision,
        'frame_threshold': float(args.frame_threshold),
        'frame_gaussian_sigma': float(args.frame_gaussian_sigma),
        'frame_median_kernel': int(args.frame_median_kernel),
        'frame_min_spoof_len': int(args.frame_min_spoof_len),
        'bound_threshold': float(args.bound_threshold),
        'bound_threshold_selected': float(selected_bound_threshold),
        'bound_nms_radius': int(args.bound_nms_radius),
        'utt_eer': float(utt_eer),
        'utt_f1': float(utt_f1),
        'utt_threshold': float(utt_thr),
        'eval_utt_threshold_arg': float(args.utt_threshold),
        'auto_bound_threshold_sweep': bool(args.auto_bound_threshold_sweep),
        'bound_sweep_values': str(args.bound_sweep_values),
        'frame_acc': float(frame_acc),
        'macro_f1': float(macro_f1),
        'frame_resolution_ms': 20,
        'frame_f1': float(spoof_f1),
        'frame_f1_20ms': float(spoof_f1),
        'frame_eer': float(frame_eer),
        'frame_threshold_eer': float(frame_thr),
        'present_classes': [int(x) for x in present_classes],
        'present_macro_f1': float(present_macro_f1),
        'class_f1': [float(x) for x in class_f1],
        'spoof_f1': float(spoof_f1),
        'bound_f1_at_1': float(bound_f1_at_1),
        'bound_f1_at_2': float(bound_f1_at_2),
        'bound_f1_at_5': float(bound_f1_at_5),
        'bound_precision_at_1': float(bound_p1),
        'bound_recall_at_1': float(bound_r1),
        'bound_precision_at_2': float(bound_p2),
        'bound_recall_at_2': float(bound_r2),
        'bound_precision_at_5': float(bound_p5),
        'bound_recall_at_5': float(bound_r5),
    }
    summary_path = score_output.with_suffix('.summary.json')
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary JSON: {summary_path}')


if __name__ == '__main__':
    main()
