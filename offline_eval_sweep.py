import argparse
import csv
import json
from pathlib import Path

import numpy as np

import eval_metrics as em


def parse_float_list(s):
    return [float(x.strip()) for x in str(s).split(',') if x.strip()]


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


def dilate_binary_events(events, tolerance):
    events = np.asarray(events, dtype=np.int64).reshape(-1)
    if tolerance <= 0 or events.size == 0:
        return events
    kernel = np.ones(2 * tolerance + 1, dtype=np.int64)
    return (np.convolve(events, kernel, mode='same') > 0).astype(np.int64)


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


def aggregate_20ms_to_160ms(scores_2d, labels_2d, group_size=8):
    scores_2d = np.asarray(scores_2d, dtype=np.float32)
    labels_2d = np.asarray(labels_2d, dtype=np.int64)
    if scores_2d.shape != labels_2d.shape:
        raise ValueError('scores_2d and labels_2d must have same shape')
    n, t = scores_2d.shape
    pad = (group_size - (t % group_size)) % group_size
    if pad > 0:
        scores_2d = np.pad(scores_2d, ((0, 0), (0, pad)), mode='edge')
        labels_2d = np.pad(labels_2d, ((0, 0), (0, pad)), mode='edge')
    new_t = scores_2d.shape[1] // group_size
    scores_g = scores_2d.reshape(n, new_t, group_size).mean(axis=-1)
    labels_g = labels_2d.reshape(n, new_t, group_size).max(axis=-1)
    return scores_g.astype(np.float32), labels_g.astype(np.int64)


def main():
    p = argparse.ArgumentParser(description='Fast offline sweep over raw_eval.npz outputs')
    p.add_argument('--raw_eval_npz', type=str, required=True)
    p.add_argument('--csv_output', type=str, required=True)
    p.add_argument('--summary_output', type=str, required=True)
    p.add_argument('--use_raw_frame_probs', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--frame_prob_key', type=str, default='',
                   help='Optional explicit NPZ key for frame probabilities, e.g. frame_spoof_prob_family3, frame_spoof_prob_binary, frame_spoof_prob_spl')
    p.add_argument('--frame_thresholds', type=str, default='0.45,0.50,0.55')
    p.add_argument('--frame_gaussian_sigmas', type=str, default='0,1.0')
    p.add_argument('--frame_median_kernels', type=str, default='1,3,5,7')
    p.add_argument('--frame_min_spoof_lens', type=str, default='1,3,5')
    p.add_argument('--bound_thresholds', type=str, default='0.70,0.75,0.80,0.85,0.90,0.95')
    p.add_argument('--bound_nms_radii', type=str, default='1,2,3')
    p.add_argument('--aggregate_ms', type=int, default=160,
                   help='Aggregation window in milliseconds for coarse frame EER (default: 160ms).')
    args = p.parse_args()

    raw = np.load(args.raw_eval_npz)
    frame_true = raw['frame_true'].astype(np.int64)
    utt_true = raw['utt_true'].astype(np.int64)
    utt_scores = raw['utt_scores'].astype(np.float64)
    bound_true_2d = raw['bound_true_2d'].astype(np.int64)
    bound_prob_2d = raw['bound_prob_2d'].astype(np.float32)
    if args.frame_prob_key:
        if args.frame_prob_key not in raw:
            raise KeyError(f'frame_prob_key not found in npz: {args.frame_prob_key}')
        frame_prob_1d = raw[args.frame_prob_key].astype(np.float32)
    elif args.use_raw_frame_probs and 'frame_spoof_prob_raw' in raw:
        frame_prob_1d = raw['frame_spoof_prob_raw'].astype(np.float32)
    else:
        frame_prob_1d = raw['frame_spoof_prob'].astype(np.float32)

    num_utts, num_frames = bound_true_2d.shape
    frame_prob_2d = frame_prob_1d.reshape(num_utts, num_frames)
    spoof_true_1d = (frame_true > 0).astype(np.int64)
    utt_bona = utt_scores[utt_true == 0]
    utt_spoof = utt_scores[utt_true == 1]
    utt_eer, utt_thr = em.compute_eer(utt_spoof, utt_bona) if utt_bona.size > 0 and utt_spoof.size > 0 else (1.0, 0.0)

    frame_thresholds = parse_float_list(args.frame_thresholds)
    frame_gaussian_sigmas = parse_float_list(args.frame_gaussian_sigmas)
    frame_median_kernels = [int(v) for v in parse_float_list(args.frame_median_kernels)]
    frame_min_spoof_lens = [int(v) for v in parse_float_list(args.frame_min_spoof_lens)]
    bound_thresholds = parse_float_list(args.bound_thresholds)
    bound_nms_radii = [int(v) for v in parse_float_list(args.bound_nms_radii)]

    total_rows = len(frame_gaussian_sigmas) * len(frame_median_kernels) * len(frame_min_spoof_lens) * len(frame_thresholds) * len(bound_nms_radii)
    print(f'[Sweep] total rows to evaluate: {total_rows}')

    rows = []
    done_rows = 0
    for sigma in frame_gaussian_sigmas:
        for mk in frame_median_kernels:
            for msl in frame_min_spoof_lens:
                proc_probs = []
                for seq in frame_prob_2d:
                    pseq = seq.copy()
                    if sigma > 0:
                        pseq = gaussian_smooth_1d(pseq, sigma)
                    if mk > 1:
                        pseq = median_filter_1d(pseq, mk).astype(np.float32)
                    proc_probs.append(pseq)
                proc_probs = np.asarray(proc_probs, dtype=np.float32)
                proc_prob_1d = proc_probs.reshape(-1)
                bona_scores = proc_prob_1d[spoof_true_1d == 0]
                spoof_scores = proc_prob_1d[spoof_true_1d == 1]
                frame_eer, frame_thr = em.compute_eer(spoof_scores, bona_scores) if bona_scores.size > 0 and spoof_scores.size > 0 else (1.0, 0.0)

                agg_group = max(1, int(round(args.aggregate_ms / 20)))
                frame_true_2d_binary = (frame_true.reshape(num_utts, num_frames) > 0).astype(np.int64)
                agg_scores_2d, agg_true_2d = aggregate_20ms_to_160ms(proc_probs, frame_true_2d_binary, group_size=agg_group)
                agg_scores_1d = agg_scores_2d.reshape(-1)
                agg_true_1d = agg_true_2d.reshape(-1)
                agg_bona = agg_scores_1d[agg_true_1d == 0]
                agg_spoof = agg_scores_1d[agg_true_1d == 1]
                frame_eer_agg, frame_thr_agg = em.compute_eer(agg_spoof, agg_bona) if agg_bona.size > 0 and agg_spoof.size > 0 else (1.0, 0.0)

                for ft in frame_thresholds:
                    frame_pred_2d = (proc_probs >= ft).astype(np.int64)
                    if msl > 1:
                        frame_pred_2d = np.asarray([remove_short_positive_runs(seq, msl) for seq in frame_pred_2d], dtype=np.int64)
                    frame_pred_1d = frame_pred_2d.reshape(-1)
                    spoof_precision, spoof_recall, spoof_f1 = compute_binary_prf(spoof_true_1d, frame_pred_1d)
                    frame_acc = float((spoof_true_1d == frame_pred_1d).mean()) if frame_pred_1d.size > 0 else 0.0

                    for nms in bound_nms_radii:
                        best_bound_f1 = -1.0
                        best_bound_th = 0.0
                        best_bound_p = 0.0
                        best_bound_r = 0.0
                        for bth in bound_thresholds:
                            bound_pred_2d = np.vstack([boundary_peak_nms(p, bth, nms) for p in bound_prob_2d])
                            bp, br, bf = compute_tolerant_boundary_prf(bound_true_2d, bound_pred_2d, 5)
                            if bf > best_bound_f1:
                                best_bound_f1 = bf
                                best_bound_th = bth
                                best_bound_p = bp
                                best_bound_r = br

                        rows.append({
                            'sigma': float(sigma),
                            'median_kernel': int(mk),
                            'min_spoof_len': int(msl),
                            'frame_threshold': float(ft),
                            'frame_eer': float(frame_eer),
                            'frame_threshold_eer': float(frame_thr),
                            'frame_eer_160ms': float(frame_eer_agg),
                            'frame_threshold_eer_160ms': float(frame_thr_agg),
                            'frame_f1_20ms': float(spoof_f1),
                            'frame_precision': float(spoof_precision),
                            'frame_recall': float(spoof_recall),
                            'frame_acc': float(frame_acc),
                            'utt_eer': float(utt_eer),
                            'utt_threshold': float(utt_thr),
                            'bound_nms_radius': int(nms),
                            'bound_threshold_selected': float(best_bound_th),
                            'bound_f1_at_5': float(best_bound_f1),
                            'bound_precision_at_5': float(best_bound_p),
                            'bound_recall_at_5': float(best_bound_r),
                        })
                        done_rows += 1
                        if done_rows % 10 == 0 or done_rows == total_rows:
                            print(f'[Sweep] {done_rows}/{total_rows} rows done | sigma={sigma}, mk={mk}, msl={msl}, ft={ft}, nms={nms}')

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_frame_eer = min(rows, key=lambda r: r['frame_eer'])
    best_frame_eer_160ms = min(rows, key=lambda r: r['frame_eer_160ms'])
    best_utt_eer = min(rows, key=lambda r: r['utt_eer'])
    best_bound = max(rows, key=lambda r: r['bound_f1_at_5'])
    summary = {
        'raw_eval_npz': args.raw_eval_npz,
        'frame_prob_key': str(args.frame_prob_key),
        'use_raw_frame_probs': bool(args.use_raw_frame_probs),
        'num_trials': int(num_utts),
        'frame_resolution_ms': 20,
        'aggregated_frame_resolution_ms': int(args.aggregate_ms),
        'best_frame_eer': best_frame_eer,
        'best_frame_eer_160ms': best_frame_eer_160ms,
        'best_utt_eer': best_utt_eer,
        'best_bound_f1_at_5': best_bound,
        'num_rows': len(rows),
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\n=== OFFLINE SWEEP SUMMARY ===')
    print(f'Rows: {len(rows)}')
    print(f'Best FrameEER(20ms): {best_frame_eer}')
    print(f'Best FrameEER({int(args.aggregate_ms)}ms): {best_frame_eer_160ms}')
    print(f'Best UttEER: {best_utt_eer}')
    print(f'Best BoundF1@5: {best_bound}')
    print(f'CSV: {csv_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
