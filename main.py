# BAT-Mamba: main.py
# Multi-task progressive training:
#   Phase 1 (ep 1-5):   L = L_loc                        lambda=(1,0,0)
#   Phase 2 (ep 6-15):  L = L_loc + L_bound              lambda=(1,1,0)
#   Phase 3 (ep 16+):   L = L_loc + L_bound + L_dia      lambda=(1,1,1)

import argparse
import sys
import os
import json
import shutil
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import (
    Dataset_train, Dataset_eval, Dataset_in_the_wild_eval, genSpoof_list,
    Dataset_PartialSpoof_train, Dataset_PartialSpoof_eval,
    Dataset_PartialSpoof_multiclass_train, Dataset_PartialSpoof_multiclass_eval,
    load_seglab, load_frame_label_dict, parse_ps_protocol, NUM_FRAMES
)
from model import Model, WeightedBCELoss, TransitionAwareBoundaryLoss, P2SGradLoss
from partialspoof_multiclass import resolve_cross_platform_path
from utils import reproducibility, read_metadata
import numpy as np
import eval_metrics as em


EXPECTED_NUM_CLASSES = {
    'binary': 2,
    'family3': 3,
    'attack19': 20,
}


def validate_label_mode_and_num_classes(args):
    expected = EXPECTED_NUM_CLASSES[args.ps_label_mode]
    if int(args.num_classes) != expected:
        raise ValueError(
            f"num_classes={args.num_classes} does not match ps_label_mode={args.ps_label_mode}. "
            f"Expected num_classes={expected}."
        )


def _detect_offline_tensor_cache(database_path, algo):
    train_pt_dir = os.path.join(database_path, 'train', f'con_wav_pt_algo{algo}')
    dev_pt_dir = os.path.join(database_path, 'dev', 'con_wav_pt')
    train_ready = os.path.isdir(train_pt_dir)
    dev_ready = os.path.isdir(dev_pt_dir)
    return train_ready and dev_ready, train_pt_dir, dev_pt_dir


def compute_multiclass_f1(y_true, y_pred, num_classes):
    f1_scores = []
    support = []
    pred_count = []
    for cls_idx in range(num_classes):
        tp = np.sum((y_true == cls_idx) & (y_pred == cls_idx))
        fp = np.sum((y_true != cls_idx) & (y_pred == cls_idx))
        fn = np.sum((y_true == cls_idx) & (y_pred != cls_idx))
        denom = 2 * tp + fp + fn
        f1_scores.append(0.0 if denom == 0 else (2.0 * tp) / denom)
        support.append(int(np.sum(y_true == cls_idx)))
        pred_count.append(int(np.sum(y_pred == cls_idx)))
    present = [i for i, n in enumerate(support) if n > 0]
    seen_macro_f1 = float(np.mean([f1_scores[i] for i in present])) if present else 0.0
    return float(np.mean(f1_scores)), f1_scores, seen_macro_f1, present, support, pred_count


def compute_binary_prf(y_true, y_pred):
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


def compute_tolerant_boundary_prf(y_true, y_pred, tolerance):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)

    tp_precision = 0
    tp_recall = 0
    pred_count = 0
    true_count = 0
    for true_seq, pred_seq in zip(y_true, y_pred):
        pred_hits = pred_seq & dilate_binary_events(true_seq, tolerance)
        true_hits = true_seq & dilate_binary_events(pred_seq, tolerance)
        tp_precision += int(pred_hits.sum())
        tp_recall += int(true_hits.sum())
        pred_count += int(pred_seq.sum())
        true_count += int(true_seq.sum())

    precision = 0.0 if pred_count == 0 else tp_precision / pred_count
    recall = 0.0 if true_count == 0 else tp_recall / true_count
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def collapse_multiclass_to_binary(labels):
    labels = np.asarray(labels, dtype=np.int64)
    return (labels > 0).astype(np.int64)


def scan_binary_threshold_prf(y_true, scores, thresholds=None):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best = {'threshold': 0.5, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(np.int64)
        precision, recall, f1 = compute_binary_prf(y_true, y_pred)
        if f1 > best['f1']:
            best = {
                'threshold': float(threshold),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
            }
    return best


def summarize_eval_metrics(frame_true, frame_pred, bound_true, bound_pred, num_classes, utt_true=None, utt_scores=None, utt_pred=None, bound_scores=None, frame_scores=None):
    frame_true = np.asarray(frame_true, dtype=np.int64)
    frame_pred = np.asarray(frame_pred, dtype=np.int64)
    bound_true_seq = np.asarray(bound_true, dtype=np.int64)
    bound_pred_seq = np.asarray(bound_pred, dtype=np.int64)
    bound_true = bound_true_seq.reshape(-1)
    bound_pred = bound_pred_seq.reshape(-1)

    frame_acc = float((frame_true == frame_pred).mean()) if frame_true.size > 0 else 0.0
    macro_f1, class_f1, seen_macro_f1, present_classes, class_support, class_pred_count = compute_multiclass_f1(frame_true, frame_pred, num_classes=num_classes)
    spoof_true = collapse_multiclass_to_binary(frame_true)
    spoof_pred = collapse_multiclass_to_binary(frame_pred)
    spoof_precision, spoof_recall, spoof_f1 = compute_binary_prf(spoof_true, spoof_pred)
    bound_precision, bound_recall, bound_f1 = compute_binary_prf(bound_true, bound_pred)
    tolerant_boundary = {}
    for tolerance in [1, 2, 5]:
        tol_p, tol_r, tol_f1 = compute_tolerant_boundary_prf(bound_true_seq, bound_pred_seq, tolerance)
        tolerant_boundary[f'boundary_precision_at_{tolerance}'] = tol_p
        tolerant_boundary[f'boundary_recall_at_{tolerance}'] = tol_r
        tolerant_boundary[f'boundary_f1_at_{tolerance}'] = tol_f1

    metrics = {
        'frame_acc': frame_acc,
        'macro_f1': macro_f1,
        'seen_macro_f1': seen_macro_f1,
        'present_classes': present_classes,
        'class_support': class_support,
        'class_pred_count': class_pred_count,
        'spoof_precision': spoof_precision,
        'spoof_recall': spoof_recall,
        'spoof_f1': spoof_f1,
        'frame_f1': spoof_f1,
        'boundary_precision': bound_precision,
        'boundary_recall': bound_recall,
        'boundary_f1': bound_f1,
        'class_f1': class_f1,
        **tolerant_boundary,
    }

    if frame_scores is not None:
        frame_scores = np.asarray(frame_scores, dtype=np.float64).reshape(-1)
        bona_scores = frame_scores[spoof_true == 0]
        spoof_scores = frame_scores[spoof_true == 1]
        if bona_scores.size > 0 and spoof_scores.size > 0:
            frame_eer, frame_thr = em.compute_eer(spoof_scores, bona_scores)
        else:
            frame_eer, frame_thr = 1.0, 0.0
        metrics.update({
            'frame_eer': float(frame_eer),
            'frame_threshold_eer': float(frame_thr),
            'frame_resolution_ms': 20,
            'frame_f1_20ms': float(spoof_f1),
        })

    if bound_scores is not None:
        best_bound = scan_binary_threshold_prf(bound_true, bound_scores)
        metrics.update({
            'boundary_best_threshold': best_bound['threshold'],
            'boundary_best_precision': best_bound['precision'],
            'boundary_best_recall': best_bound['recall'],
            'boundary_best_f1': best_bound['f1'],
        })

    if utt_true is not None and utt_scores is not None and utt_pred is not None:
        utt_true = np.asarray(utt_true, dtype=np.int64)
        utt_scores = np.asarray(utt_scores, dtype=np.float64)
        utt_pred = np.asarray(utt_pred, dtype=np.int64)
        utt_acc = float((utt_true == utt_pred).mean()) if utt_true.size > 0 else 0.0
        utt_precision, utt_recall, utt_f1 = compute_binary_prf(utt_true, utt_pred)
        bona_scores = utt_scores[utt_true == 0]
        spoof_scores = utt_scores[utt_true == 1]
        if bona_scores.size > 0 and spoof_scores.size > 0:
            utt_eer, utt_thr = em.compute_eer(spoof_scores, bona_scores)
        else:
            utt_eer, utt_thr = 1.0, 0.0
        utt_pos_rate = float((utt_true == 1).mean()) if utt_true.size > 0 else 0.0
        utt_score_mean_bona = float(bona_scores.mean()) if bona_scores.size > 0 else 0.0
        utt_score_mean_spoof = float(spoof_scores.mean()) if spoof_scores.size > 0 else 0.0
        utt_score_gap = float(utt_score_mean_spoof - utt_score_mean_bona)
        metrics.update({
            'utt_acc': utt_acc,
            'utt_precision': utt_precision,
            'utt_recall': utt_recall,
            'utt_f1': utt_f1,
            'utt_eer': float(utt_eer),
            'utt_threshold': float(utt_thr),
            'utt_pos_rate': utt_pos_rate,
            'utt_score_mean_bona': utt_score_mean_bona,
            'utt_score_mean_spoof': utt_score_mean_spoof,
            'utt_score_gap': utt_score_gap,
        })

    return metrics


def compute_frame_class_weights_from_counts(counts, mode='effective', beta=0.9999, max_weight=5.0, eps=1e-8):
    counts = np.asarray(counts, dtype=np.float64)
    if mode == 'none':
        return None
    present = counts > 0
    weights = np.zeros_like(counts, dtype=np.float64)
    if not present.any():
        return None
    if mode == 'inverse':
        weights[present] = 1.0 / np.maximum(counts[present], eps)
    elif mode == 'sqrt_inverse':
        weights[present] = 1.0 / np.sqrt(np.maximum(counts[present], eps))
    else:
        effective_num = 1.0 - np.power(beta, counts[present])
        weights[present] = (1.0 - beta) / np.maximum(effective_num, eps)
    weights[present] = weights[present] / np.mean(weights[present])
    weights[present] = np.clip(weights[present], 1.0 / max_weight, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


def _check_frame_label_coverage(label_dict, num_classes, split_name, label_mode, require_all_classes=True):
    counts = np.zeros(num_classes, dtype=np.int64)
    for arr in label_dict.values():
        arr = np.asarray(arr, dtype=np.int64).reshape(-1)
        binc = np.bincount(arr, minlength=num_classes)
        counts[:len(binc)] += binc[:num_classes]
    missing = [int(i) for i, c in enumerate(counts.tolist()) if c == 0]
    total = int(counts.sum())
    ratios = [0.0 if total == 0 else float(c / total) for c in counts.tolist()]
    print('[LabelCheck] split={} mode={} class_counts={} class_ratios={} missing={}'.format(
        split_name, label_mode, counts.tolist(), ['%.6f' % r for r in ratios], missing))
    if require_all_classes and missing:
        raise ValueError(
            'Converted labels for split={} mode={} are missing classes {}. '
            'Fix --family_map_path and reconvert labels, or pass --require_all_family3_classes false for a diagnostic run only.'.format(
                split_name, label_mode, missing
            )
        )
    return counts, ratios, missing


def get_experiment_config(args, model_tag):
    return {
        'model_tag': model_tag,
        'comment': args.comment,
        'seed': args.seed,
        'emb_size': args.emb_size,
        'num_encoders': args.num_encoders,
        'num_classes': args.num_classes,
        'ps_label_mode': args.ps_label_mode,
        'ps_label_root': args.ps_label_root,
        'frame_class_weight_mode': args.frame_class_weight_mode,
        'frame_class_weight_beta': args.frame_class_weight_beta,
        'frame_class_weight_max': args.frame_class_weight_max,
        'binary_frame_aux_weight': args.binary_frame_aux_weight,
        'binary_frame_aux_use_class_weights': args.binary_frame_aux_use_class_weights,
        'binary_frame_primary_weight': args.binary_frame_primary_weight,
        'multiclass_frame_aux_weight': args.multiclass_frame_aux_weight,
        'segment_position_aux_weight': args.segment_position_aux_weight,
        'use_binary_frame_head': args.use_binary_frame_head,
        'use_segment_position_head': args.use_segment_position_head,
        'cross_segment_mix_prob': args.cross_segment_mix_prob,
        'cross_segment_mix_min_frames': args.cross_segment_mix_min_frames,
        'cross_segment_mix_max_frames': args.cross_segment_mix_max_frames,
        'dia_class_weight': args.dia_class_weight,
        'boundary_threshold_eval': args.boundary_threshold_eval,
        'require_all_family3_classes': args.require_all_family3_classes,
        'num_segments': args.num_segments,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'grad_accum_steps': args.grad_accum_steps,
        'use_boundary_control': args.use_boundary_control,
        'use_cross_routing': args.use_cross_routing,
        'use_soft_segments': args.use_soft_segments,
        'seg_cons_weight': args.seg_cons_weight,
        'seg_entropy_weight': args.seg_entropy_weight,
        'bound_sparse_weight': args.bound_sparse_weight,
        'bound_sharp_weight': args.bound_sharp_weight,
        'bound_loss_type': args.bound_loss_type,
        'bound_pos_weight': args.bound_pos_weight,
        'bound_soft_sigma': args.bound_soft_sigma,
        'bound_dice_weight': args.bound_dice_weight,
        'bound_tversky_weight': args.bound_tversky_weight,
        'bound_tversky_alpha': args.bound_tversky_alpha,
        'bound_tversky_beta': args.bound_tversky_beta,
        'bound_fp_penalty_weight': args.bound_fp_penalty_weight,
        'bound_fp_margin': args.bound_fp_margin,
        'frame_boundary_cons_weight': args.frame_boundary_cons_weight,
        'utt_loss_weight': args.utt_loss_weight,
        'num_workers': args.num_workers,
        'eval_num_workers': args.eval_num_workers,
        'pin_memory': args.pin_memory,
        'persistent_workers': args.persistent_workers,
        'debug_steps': args.debug_steps,
        'dia_start_epoch': args.dia_start_epoch,
        'dia_full_epoch': args.dia_full_epoch,
        'mamba_bidirectional': args.mamba_bidirectional,
        'attractor_repulsion_weight': args.attractor_repulsion_weight,
        'attractor_boundary_temp_strength': args.attractor_boundary_temp_strength,
        'attractor_boundary_temp_min': args.attractor_boundary_temp_min,
        'wavlm_local_files_only': args.wavlm_local_files_only,
        'use_torch_compile': args.use_torch_compile,
        'torch_compile_mode': args.torch_compile_mode,
        'allow_tf32': args.allow_tf32,
        'cudnn_benchmark_override': args.cudnn_benchmark,
        'keep_last_checkpoints': args.keep_last_checkpoints,
    }


def append_experiment_record(record_path, payload):
    if os.path.exists(record_path):
        with open(record_path, 'r', encoding='utf-8') as fh:
            records = json.load(fh)
    else:
        records = []
    records.append(payload)
    with open(record_path, 'w', encoding='utf-8') as fh:
        json.dump(records, fh, indent=2)


def _to_serializable_config(config):
    serializable = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serializable[key] = value
        elif isinstance(value, (list, tuple)):
            serializable[key] = list(value)
        else:
            serializable[key] = str(value)
    return serializable


def export_analysis_artifacts(dataset, model, device, export_dir, max_batches=1, batch_size=4):
    os.makedirs(export_dir, exist_ok=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    exported = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            if len(batch) == 5:
                batch_x, batch_fl, batch_bl, _, utt_id = batch
            elif len(batch) == 4:
                batch_x, batch_fl, batch_bl, utt_id = batch
            else:
                continue
            batch_x = batch_x.to(device)
            _ = model(batch_x)
            analysis = getattr(get_unwrapped_model(model), 'cached_analysis', {})
            if not analysis:
                continue
            batch_dir = os.path.join(export_dir, 'batch_{:03d}'.format(batch_idx))
            os.makedirs(batch_dir, exist_ok=True)
            meta = {
                'batch_idx': int(batch_idx),
                'utt_id': list(utt_id),
                'frame_labels_shape': list(batch_fl.shape),
                'boundary_labels_shape': list(batch_bl.shape),
                'analysis_keys': sorted(list(analysis.keys())),
            }
            with open(os.path.join(batch_dir, 'meta.json'), 'w', encoding='utf-8') as fh:
                json.dump(meta, fh, indent=2)
            np.save(os.path.join(batch_dir, 'frame_labels.npy'), batch_fl.numpy())
            np.save(os.path.join(batch_dir, 'boundary_labels.npy'), batch_bl.numpy())
            for key, tensor in analysis.items():
                np.save(os.path.join(batch_dir, '{}.npy'.format(key)), tensor.detach().cpu().numpy())
            exported += 1
    print('Exported analysis batches: {} -> {}'.format(exported, export_dir))


# ============================================================
# Helper: get progressive loss weights by epoch
# ============================================================
def get_loss_weights(epoch, dia_start_epoch=5, dia_full_epoch=15):
    """Progressive schedule with configurable dia/P2S start.

    Default matches the original schedule:
      epoch < 5      -> lam3=0.0
      5 <= epoch<15 -> lam3=0.5
      epoch >= 15   -> lam3=1.0
    """
    if epoch < dia_start_epoch:
        return 1.0, 0.5, 0.0
    elif epoch < dia_full_epoch:
        return 1.0, 1.0, 0.5
    else:
        return 1.0, 1.0, 1.0


def get_loss_weights_debug(epoch, dia_start_epoch=5, dia_full_epoch=15):
    """Debug mode follows the same configurable schedule."""
    return get_loss_weights(epoch, dia_start_epoch=dia_start_epoch, dia_full_epoch=dia_full_epoch)


# make_frame_labels removed: PartialSpoof Dataset now returns real frame labels directly.


# ============================================================
# Evaluation helpers
# ============================================================
def evaluate_accuracy(dev_loader, model, device, debug_steps=0, args=None):
    """Validation loop. dev_loader yields (waveform, frame_labels, boundary_labels, utt_label, utt_id)."""
    val_loss = 0.0
    num_total = 0.0
    all_frame_true, all_frame_pred, all_frame_scores = [], [], []
    all_bound_true, all_bound_pred, all_bound_scores = [], [], []
    all_utt_true, all_utt_pred, all_utt_scores = [], [], []
    model.eval()
    criterion_loc = nn.CrossEntropyLoss()
    criterion_utt = nn.BCEWithLogitsLoss()
    binary_frame_aux_weight = getattr(args, 'binary_frame_aux_weight', 0.0) if args is not None else 0.0
    binary_frame_primary_weight = getattr(args, 'binary_frame_primary_weight', 0.0) if args is not None else 0.0
    multiclass_frame_aux_weight = getattr(args, 'multiclass_frame_aux_weight', 1.0) if args is not None else 1.0
    binary_frame_class_weights = getattr(args, 'binary_frame_class_weights', None) if args is not None else None
    if binary_frame_class_weights is not None:
        binary_frame_class_weights = binary_frame_class_weights.to(device)
    criterion_bin_loc = nn.CrossEntropyLoss(weight=binary_frame_class_weights)
    num_batch = len(dev_loader)
    with torch.no_grad():
        for i, batch_data in enumerate(dev_loader):
            if debug_steps > 0 and i >= debug_steps:
                break
            if len(batch_data) == 5:
                batch_x, batch_fl, batch_bl, batch_ul, _ = batch_data
            elif len(batch_data) == 4:
                batch_x, batch_fl, batch_bl, _ = batch_data
                batch_ul = (batch_fl.max(dim=1).values > 0).long()
            else:
                batch_x, batch_fl, batch_bl = batch_data
                batch_ul = (batch_fl.max(dim=1).values > 0).long()

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_fl = batch_fl.to(device)
            batch_bl = batch_bl.to(device)
            batch_ul = batch_ul.to(device).float().unsqueeze(-1)

            p_bound_logits, logits_dia, _, utt_logit = model(batch_x)
            B, T, C = logits_dia.shape
            loss_loc = criterion_loc(
                logits_dia.reshape(B * T, C),
                batch_fl.reshape(B * T)
            )
            base_model = get_unwrapped_model(model)
            aux_losses = getattr(base_model, 'cached_aux_losses', {})
            binary_head_logits = aux_losses.get('binary_logits', None)
            collapsed_binary_logits = collapse_multiclass_logits_to_binary(logits_dia)
            binary_targets = (batch_fl > 0).long()
            loss_bin_primary = criterion_bin_loc(
                binary_head_logits.reshape(B * T, 2),
                binary_targets.reshape(B * T)
            ) if (binary_frame_primary_weight > 0 and binary_head_logits is not None) else torch.tensor(0.0, device=device)
            loss_bin_loc = criterion_bin_loc(
                collapsed_binary_logits.reshape(B * T, 2),
                binary_targets.reshape(B * T)
            ) if binary_frame_aux_weight > 0 else torch.tensor(0.0, device=device)
            loss_utt = criterion_utt(utt_logit, batch_ul)
            loss = multiclass_frame_aux_weight * loss_loc + binary_frame_primary_weight * loss_bin_primary + binary_frame_aux_weight * loss_bin_loc + loss_utt
            val_loss += loss.item() * batch_size

            frame_pred = logits_dia.argmax(dim=-1)
            frame_score = (1.0 - torch.softmax(logits_dia, dim=-1)[..., 0]).detach().cpu()
            bound_score = torch.sigmoid(p_bound_logits).squeeze(-1)
            boundary_threshold = getattr(args, 'boundary_threshold_eval', 0.5) if args is not None else 0.5
            bound_pred = (bound_score >= boundary_threshold).long()
            utt_score = torch.sigmoid(utt_logit).squeeze(-1)
            utt_pred = (utt_score >= 0.5).long()
            all_frame_true.append(batch_fl.detach().cpu().reshape(-1).numpy())
            all_frame_pred.append(frame_pred.detach().cpu().reshape(-1).numpy())
            all_frame_scores.append(frame_score.reshape(-1).numpy())
            all_bound_true.append(batch_bl.detach().cpu().numpy())
            all_bound_pred.append(bound_pred.detach().cpu().numpy())
            all_bound_scores.append(bound_score.detach().cpu().numpy())
            all_utt_true.append(batch_ul.detach().cpu().reshape(-1).numpy())
            all_utt_scores.append(utt_score.detach().cpu().reshape(-1).numpy())
            all_utt_pred.append(utt_pred.detach().cpu().reshape(-1).numpy())
            print("batch %i/%i (val)" % (i + 1, num_batch), end="\r")

    val_loss /= num_total
    metrics = summarize_eval_metrics(
        np.concatenate(all_frame_true) if all_frame_true else np.array([], dtype=np.int64),
        np.concatenate(all_frame_pred) if all_frame_pred else np.array([], dtype=np.int64),
        np.concatenate(all_bound_true, axis=0) if all_bound_true else np.empty((0, NUM_FRAMES), dtype=np.int64),
        np.concatenate(all_bound_pred, axis=0) if all_bound_pred else np.empty((0, NUM_FRAMES), dtype=np.int64),
        num_classes=get_unwrapped_model(model).num_classes,
        utt_true=np.concatenate(all_utt_true) if all_utt_true else np.array([], dtype=np.int64),
        utt_scores=np.concatenate(all_utt_scores) if all_utt_scores else np.array([], dtype=np.float64),
        utt_pred=np.concatenate(all_utt_pred) if all_utt_pred else np.array([], dtype=np.int64),
        bound_scores=np.concatenate(all_bound_scores, axis=0) if all_bound_scores else np.empty((0, NUM_FRAMES), dtype=np.float64),
        frame_scores=np.concatenate(all_frame_scores) if all_frame_scores else np.array([], dtype=np.float64),
    )
    print('Val loss: %.4f | FrameAcc: %.4f | MacroF1: %.4f | SeenMacroF1: %.4f | SpoofF1: %.4f | FrameEER: %.4f | BoundF1: %.4f | BoundF1@1/2/5: %.4f/%.4f/%.4f | UttF1: %.4f | UttEER: %.4f' % (
        val_loss, metrics['frame_acc'], metrics['macro_f1'], metrics['seen_macro_f1'], metrics['spoof_f1'], metrics.get('frame_eer', 1.0), metrics['boundary_f1'],
        metrics.get('boundary_f1_at_1', 0.0), metrics.get('boundary_f1_at_2', 0.0), metrics.get('boundary_f1_at_5', 0.0),
        metrics.get('utt_f1', 0.0), metrics.get('utt_eer', 1.0)))
    print('          SpoofP/R: %.4f / %.4f | BoundP/R: %.4f / %.4f | BoundBest@thr %.2f: P/R/F1 %.4f / %.4f / %.4f | UttP/R: %.4f / %.4f | UttPosRate: %.4f | UttScore(bona/spoof): %.4f / %.4f | UttGap: %.4f | ClassF1=%s' % (
        metrics['spoof_precision'], metrics['spoof_recall'],
        metrics['boundary_precision'], metrics['boundary_recall'],
        metrics.get('boundary_best_threshold', 0.5), metrics.get('boundary_best_precision', 0.0), metrics.get('boundary_best_recall', 0.0), metrics.get('boundary_best_f1', 0.0),
        metrics.get('utt_precision', 0.0), metrics.get('utt_recall', 0.0),
        metrics.get('utt_pos_rate', 0.0),
        metrics.get('utt_score_mean_bona', 0.0), metrics.get('utt_score_mean_spoof', 0.0),
        metrics.get('utt_score_gap', 0.0),
        ','.join(['%.4f' % x for x in metrics['class_f1']])
    ))
    return val_loss, metrics


def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)


def produce_evaluation_file(dataset, model, device, save_path):
    """Evaluation file writer. dataset yields (waveform, frame_labels, boundary_labels, utt_label, utt_id)."""
    data_loader = DataLoader(dataset, batch_size=40, shuffle=False, drop_last=False, collate_fn=collate_skip_none)
    model.eval()
    skipped_missing = 0
    with open(save_path, 'w', encoding='utf-8') as fh, torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            if batch is None:
                skipped_missing += 1
                continue
            if len(batch) == 5:
                batch_x, _, _, _, utt_id = batch
            elif len(batch) == 4:
                batch_x, _, _, utt_id = batch
            else:
                batch_x, utt_id = batch
            batch_x = batch_x.to(device)
            _, _, _, utt_logit = model(batch_x)
            batch_score = utt_logit.squeeze(-1).data.cpu().numpy().ravel()
            for f, cm in zip(utt_id, batch_score.tolist()):
                fh.write('{} {}\n'.format(f, cm))
    if skipped_missing > 0:
        print('Skipped {} missing eval items.'.format(skipped_missing))
    print('Scores saved to {}'.format(save_path))


def compute_boundary_false_positive_penalty(p_bound_logits, target, margin=0.2):
    prob = torch.sigmoid(p_bound_logits).squeeze(-1)
    target = target.squeeze(-1).float()
    non_boundary = 1.0 - target
    return (F.relu(prob - margin).pow(2) * non_boundary).sum() / (non_boundary.sum() + 1e-8)


def compute_frame_boundary_consistency(p_bound_logits, logits_dia):
    p_bound = torch.sigmoid(p_bound_logits).squeeze(-1)
    spoof_prob = 1.0 - torch.softmax(logits_dia, dim=-1)[..., 0]
    frame_delta = torch.zeros_like(spoof_prob)
    frame_delta[:, 1:] = (spoof_prob[:, 1:] - spoof_prob[:, :-1]).abs()
    frame_delta = frame_delta.detach().clamp(0.0, 1.0)
    return nn.functional.binary_cross_entropy(p_bound, frame_delta)


def collapse_multiclass_logits_to_binary(logits_dia):
    bona_logit = logits_dia[..., 0:1]
    spoof_logit = torch.logsumexp(logits_dia[..., 1:], dim=-1, keepdim=True)
    return torch.cat([bona_logit, spoof_logit], dim=-1)


def compute_binary_frame_class_weights_from_counts(class_counts, mode='none', beta=0.9999, max_weight=5.0):
    if class_counts is None or len(class_counts) == 0:
        return None
    if len(class_counts) == 2:
        binary_counts = np.asarray(class_counts, dtype=np.float64)
    else:
        binary_counts = np.asarray([class_counts[0], np.sum(class_counts[1:])], dtype=np.float64)
    return compute_frame_class_weights_from_counts(binary_counts, mode=mode, beta=beta, max_weight=max_weight)


def get_unwrapped_model(model):
    return getattr(model, '_orig_mod', model)


def load_model_state(model, state_dict, strict=True):
    return get_unwrapped_model(model).load_state_dict(state_dict, strict=strict)


def cleanup_step_checkpoints(checkpoint_dir, keep_last=3):
    if keep_last <= 0:
        keep_last = 0
    ckpt_files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_') and f.endswith('.pth')
    ]
    if len(ckpt_files) <= keep_last:
        return
    ckpt_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for old_path in ckpt_files[keep_last:]:
        try:
            os.remove(old_path)
            print('Removed old checkpoint: {}'.format(old_path))
        except OSError:
            pass


# ============================================================
# Training epoch with progressive multi-task loss
# ============================================================
def train_epoch(train_loader, model, optimizer, device, epoch, checkpoint_dir=None, debug_steps=0, args=None):
    """PartialSpoof training epoch.
    train_loader yields: (waveform [B,66800], frame_labels [B,208], boundary_labels [B,208], utt_labels [B])
    checkpoint_dir: if set, saves model every 1000 steps to prevent data loss.
    debug_steps: if > 0, only run this many batches (quick pipeline test).
    """
    model.train()
    num_total = 0.0
    total_loss = 0.0
    class_weights = getattr(args, 'frame_class_weights', None) if args is not None else None
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion_loc = nn.CrossEntropyLoss(weight=class_weights)
    binary_frame_aux_weight = getattr(args, 'binary_frame_aux_weight', 0.0) if args is not None else 0.0
    binary_frame_primary_weight = getattr(args, 'binary_frame_primary_weight', 0.0) if args is not None else 0.0
    multiclass_frame_aux_weight = getattr(args, 'multiclass_frame_aux_weight', 1.0) if args is not None else 1.0
    segment_position_aux_weight = getattr(args, 'segment_position_aux_weight', 0.0) if args is not None else 0.0
    binary_frame_class_weights = getattr(args, 'binary_frame_class_weights', None) if args is not None else None
    if binary_frame_class_weights is not None:
        binary_frame_class_weights = binary_frame_class_weights.to(device)
    criterion_bin_loc = nn.CrossEntropyLoss(weight=binary_frame_class_weights)
    criterion_segpos = nn.CrossEntropyLoss(ignore_index=-100)
    bound_loss_type = getattr(args, 'bound_loss_type', 'transition') if args is not None else 'transition'
    if bound_loss_type == 'weighted':
        criterion_bound = WeightedBCELoss(pos_weight=getattr(args, 'bound_pos_weight', 50.0)).to(device)
    elif bound_loss_type == 'transition':
        criterion_bound = TransitionAwareBoundaryLoss(
            pos_weight=getattr(args, 'bound_pos_weight', 30.0),
            sigma=getattr(args, 'bound_soft_sigma', 1.5),
            dice_weight=getattr(args, 'bound_dice_weight', 0.5),
            tversky_weight=getattr(args, 'bound_tversky_weight', 0.5),
            alpha=getattr(args, 'bound_tversky_alpha', 0.3),
            beta=getattr(args, 'bound_tversky_beta', 0.7),
        ).to(device)
    else:
        criterion_bound = WeightedBCELoss(pos_weight=getattr(args, 'bound_pos_weight', 50.0)).to(device)
    dia_weights = class_weights if getattr(args, 'dia_class_weight', False) else None
    criterion_dia = P2SGradLoss(scale=30.0, class_weight=dia_weights)
    criterion_utt   = nn.BCEWithLogitsLoss()
    seg_cons_weight = getattr(args, 'seg_cons_weight', 0.0) if args is not None else 0.0
    seg_entropy_weight = getattr(args, 'seg_entropy_weight', 0.0) if args is not None else 0.0
    bound_sparse_weight = getattr(args, 'bound_sparse_weight', 0.0) if args is not None else 0.0
    bound_sharp_weight = getattr(args, 'bound_sharp_weight', 0.0) if args is not None else 0.0
    frame_boundary_cons_weight = getattr(args, 'frame_boundary_cons_weight', 0.0) if args is not None else 0.0
    bound_fp_penalty_weight = getattr(args, 'bound_fp_penalty_weight', 0.0) if args is not None else 0.0
    bound_fp_margin = getattr(args, 'bound_fp_margin', 0.2) if args is not None else 0.2
    attractor_repulsion_weight = getattr(args, 'attractor_repulsion_weight', 0.0) if args is not None else 0.0
    utt_loss_weight = getattr(args, 'utt_loss_weight', 1.0) if args is not None else 1.0
    grad_accum_steps = max(1, getattr(args, 'grad_accum_steps', 1) if args is not None else 1)
    dia_start_epoch = getattr(args, 'dia_start_epoch', 5) if args is not None else 5
    dia_full_epoch = getattr(args, 'dia_full_epoch', 15) if args is not None else 15
    lam1, lam2, lam3 = get_loss_weights_debug(epoch, dia_start_epoch, dia_full_epoch) if debug_steps > 0 else get_loss_weights(epoch, dia_start_epoch, dia_full_epoch)
    print('Phase weights: lam1=%.1f  lam2=%.1f  lam3=%.1f  lam_utt=%.1f' % (lam1, lam2, lam3, utt_loss_weight))
    scaler = torch.cuda.amp.GradScaler()  # AMP scaler
    pbar = tqdm(train_loader, total=len(train_loader))
    optimizer.zero_grad(set_to_none=True)
    for step, batch_data in enumerate(pbar):
        if debug_steps > 0 and step >= debug_steps:
            break
        if len(batch_data) == 5:
            batch_x, batch_fl, batch_bl, batch_ul, batch_spl = batch_data
        else:
            batch_x, batch_fl, batch_bl, batch_ul = batch_data
            batch_spl = torch.full_like(batch_fl, -100)
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x  = batch_x.to(device)
        batch_fl = batch_fl.to(device)
        batch_bl = batch_bl.unsqueeze(-1).to(device)
        batch_ul = batch_ul.to(device).float().unsqueeze(-1)
        batch_spl = batch_spl.to(device)

        with torch.cuda.amp.autocast():  # AMP: FP16 forward pass
            p_bound_logits, logits_dia, h_prime, utt_logit = model(batch_x)
            B, T, C = logits_dia.shape

            multiclass_logits = logits_dia
            loss_loc = criterion_loc(
                multiclass_logits.reshape(B * T, C),
                batch_fl.reshape(B * T)
            )
            base_model = get_unwrapped_model(model)
            aux_losses = getattr(base_model, 'cached_aux_losses', {})
            binary_head_logits = aux_losses.get('binary_logits', None)
            collapsed_binary_logits = collapse_multiclass_logits_to_binary(multiclass_logits)
            binary_targets = (batch_fl > 0).long()
            loss_bin_primary = criterion_bin_loc(
                binary_head_logits.reshape(B * T, 2),
                binary_targets.reshape(B * T)
            ) if (binary_frame_primary_weight > 0 and binary_head_logits is not None) else torch.tensor(0.0, device=device)
            loss_bin_loc = criterion_bin_loc(
                collapsed_binary_logits.reshape(B * T, 2),
                binary_targets.reshape(B * T)
            ) if binary_frame_aux_weight > 0 else torch.tensor(0.0, device=device)
            segment_position_logits = aux_losses.get('segment_position_logits', None)
            loss_segpos = criterion_segpos(
                segment_position_logits.reshape(B * T, segment_position_logits.size(-1)),
                batch_spl.reshape(B * T)
            ) if (segment_position_aux_weight > 0 and segment_position_logits is not None) else torch.tensor(0.0, device=device)
            loss_bound = criterion_bound(p_bound_logits.float(), batch_bl.float()) if lam2 > 0 \
                else torch.tensor(0.0, device=device)
            loss_dia = criterion_dia(
                h_prime, batch_fl,
                base_model.attractor_head.attractor_tokens
            ) if lam3 > 0 else torch.tensor(0.0, device=device)
            loss_utt = criterion_utt(utt_logit, batch_ul)

            loss = multiclass_frame_aux_weight * lam1 * loss_loc \
                + binary_frame_primary_weight * loss_bin_primary \
                + binary_frame_aux_weight * loss_bin_loc \
                + lam2 * loss_bound + lam3 * loss_dia + utt_loss_weight * loss_utt \
                + segment_position_aux_weight * loss_segpos

            aux_losses = getattr(base_model, 'cached_aux_losses', {})
            loss_seg_cons = aux_losses.get('segment_consistency', torch.tensor(0.0, device=device))
            loss_seg_entropy = aux_losses.get('segment_entropy', torch.tensor(0.0, device=device))
            loss_bound_sparse = aux_losses.get('boundary_sparsity', torch.tensor(0.0, device=device))
            loss_bound_sharp = aux_losses.get('boundary_sharpness', torch.tensor(0.0, device=device))
            loss_bound_fp = compute_boundary_false_positive_penalty(p_bound_logits, batch_bl.float(), margin=bound_fp_margin) if bound_fp_penalty_weight > 0 else torch.tensor(0.0, device=device)
            loss_attractor_repulsion = base_model.attractor_head.compute_attractor_repulsion_loss() if attractor_repulsion_weight > 0 else torch.tensor(0.0, device=device)
            loss_frame_boundary_cons = compute_frame_boundary_consistency(p_bound_logits, logits_dia) if frame_boundary_cons_weight > 0 else torch.tensor(0.0, device=device)

            loss = loss \
                + seg_cons_weight * loss_seg_cons \
                + seg_entropy_weight * loss_seg_entropy \
                + bound_sparse_weight * loss_bound_sparse \
                + bound_sharp_weight * loss_bound_sharp \
                + bound_fp_penalty_weight * loss_bound_fp \
                + attractor_repulsion_weight * loss_attractor_repulsion \
                + frame_boundary_cons_weight * loss_frame_boundary_cons

        total_loss += loss.item() * batch_size
        loss_to_backward = loss / grad_accum_steps
        scaler.scale(loss_to_backward).backward()   # AMP: scaled backward

        should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(train_loader)) or (debug_steps > 0 and (step + 1) == min(len(train_loader), debug_steps))
        if should_step:
            scaler.step(optimizer)          # AMP: scaled optimizer step
            scaler.update()                 # AMP: update scaler
            optimizer.zero_grad(set_to_none=True)

        pbar.set_postfix({
            'loss': '%.4f' % (total_loss / num_total),
            'loc':  '%.4f' % loss_loc.item(),
            'bnd':  '%.4f' % loss_bound.item(),
            'dia':  '%.4f' % loss_dia.item(),
            'bin':  '%.4f' % loss_bin_loc.item(),
            'bpr':  '%.4f' % loss_bin_primary.item(),
            'spl':  '%.4f' % loss_segpos.item(),
            'utt':  '%.4f' % loss_utt.item(),
            'bfp':  '%.4f' % loss_bound_fp.item(),
            'rep':  '%.4f' % loss_attractor_repulsion.item(),
            'fb':   '%.4f' % loss_frame_boundary_cons.item(),
        })

        if checkpoint_dir is not None and (step + 1) % 1000 == 0:
            ckpt_path = os.path.join(checkpoint_dir,
                'checkpoint_ep{}_step{}.pth'.format(epoch, step + 1))
            torch.save(get_unwrapped_model(model).state_dict(), ckpt_path)
            print('\nCheckpoint saved: {}'.format(ckpt_path))
            cleanup_step_checkpoints(
                checkpoint_dir=checkpoint_dir,
                keep_last=getattr(args, 'keep_last_checkpoints', 3) if args is not None else 3,
            )

    sys.stdout.flush()


# ============================================================
# Main entry
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BAT-Mamba')
    parser.add_argument('--database_path', type=str, default='./data/')
    parser.add_argument('--protocols_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')
    parser.add_argument('--emb-size', type=int, default=256)
    parser.add_argument('--num_encoders', type=int, default=12)
    parser.add_argument('--num_classes', type=int, default=3,
                        help='number of frame-level classes')
    parser.add_argument('--ps_label_mode', type=str, default='family3', choices=['binary', 'family3', 'attack19'],
                        help='frame label scheme for PartialSpoof training/eval')
    parser.add_argument('--ps_label_root', type=str, default='',
                        help='root dir containing converted multiclass labels, e.g. ./multiclass_labels')
    parser.add_argument('--require_all_family3_classes', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']),
                        help='fail fast if family3 train/dev labels miss bonafide, TTS, or VC')
    parser.add_argument('--FT_W2V', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--wavlm_local_files_only', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']),
                        help='load WavLM only from local HuggingFace cache to avoid network retries')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--debug_steps', type=int, default=0,
                        help='If > 0, only run this many batches per epoch/eval (quick pipeline test)')
    parser.add_argument('--comment_eval', type=str, default=None)
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='path to a checkpoint/state_dict to resume fine-tuning from')
    parser.add_argument('--resume_start_epoch', type=int, default=0,
                        help='global epoch offset used for progressive loss schedule when resuming')
    parser.add_argument('--resume_strict', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--train', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--n_mejores_loss', type=int, default=5)
    parser.add_argument('--average_model', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--n_average_model', default=5, type=int)
    parser.add_argument('--algo', type=int, default=5)
    parser.add_argument('--N_f', type=int, default=5)
    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--minF', type=int, default=20)
    parser.add_argument('--maxF', type=int, default=8000)
    parser.add_argument('--minBW', type=int, default=100)
    parser.add_argument('--maxBW', type=int, default=1000)
    parser.add_argument('--minCoeff', type=int, default=10)
    parser.add_argument('--maxCoeff', type=int, default=100)
    parser.add_argument('--minG', type=int, default=0)
    parser.add_argument('--maxG', type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--P', type=int, default=10)
    parser.add_argument('--g_sd', type=int, default=2)
    parser.add_argument('--SNRmin', type=int, default=10)
    parser.add_argument('--SNRmax', type=int, default=40)
    parser.add_argument('--num_segments', type=int, default=4,
                        help='number of latent soft segments for segment parsing')
    parser.add_argument('--seg_cons_weight', type=float, default=0.0,
                        help='weight for segment consistency loss')
    parser.add_argument('--seg_entropy_weight', type=float, default=0.0,
                        help='weight for soft segment assignment entropy')
    parser.add_argument('--bound_sparse_weight', type=float, default=0.0,
                        help='weight for boundary sparsity regularization')
    parser.add_argument('--bound_sharp_weight', type=float, default=0.0,
                        help='weight for boundary sharpness regularization')
    parser.add_argument('--bound_loss_type', type=str, default='transition', choices=['transition', 'weighted'],
                        help='boundary loss type: transition-aware soft boundary loss or weighted BCE')
    parser.add_argument('--bound_pos_weight', type=float, default=30.0,
                        help='positive class weight for boundary BCE terms')
    parser.add_argument('--bound_soft_sigma', type=float, default=1.5,
                        help='sigma in frames for soft transition boundary targets')
    parser.add_argument('--bound_dice_weight', type=float, default=0.5,
                        help='weight for soft Dice term in transition-aware boundary loss')
    parser.add_argument('--bound_tversky_weight', type=float, default=0.5,
                        help='weight for Tversky term in transition-aware boundary loss')
    parser.add_argument('--bound_tversky_alpha', type=float, default=0.3,
                        help='false-positive weight in Tversky boundary loss')
    parser.add_argument('--bound_tversky_beta', type=float, default=0.7,
                        help='false-negative weight in Tversky boundary loss')
    parser.add_argument('--frame_boundary_cons_weight', type=float, default=0.0,
                        help='weight for consistency between boundary probability and spoof-probability temporal changes')
    parser.add_argument('--frame_class_weight_mode', type=str, default='none',
                        choices=['none', 'effective', 'inverse', 'sqrt_inverse'],
                        help='class weighting for frame CE based on train frame-label distribution')
    parser.add_argument('--frame_class_weight_beta', type=float, default=0.9999,
                        help='beta for effective-number class weighting')
    parser.add_argument('--frame_class_weight_max', type=float, default=5.0,
                        help='maximum normalized frame class weight')
    parser.add_argument('--binary_frame_aux_weight', type=float, default=0.0,
                        help='extra weight for binary bonafide/spoof frame CE using collapsed family3 logits')
    parser.add_argument('--binary_frame_aux_use_class_weights', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']),
                        help='reuse collapsed train frame counts to weight binary auxiliary frame loss')
    parser.add_argument('--binary_frame_primary_weight', type=float, default=1.0,
                        help='primary binary spoof frame loss weight using dedicated binary frame head')
    parser.add_argument('--multiclass_frame_aux_weight', type=float, default=0.5,
                        help='auxiliary family3 frame CE weight when using binary-first training')
    parser.add_argument('--segment_position_aux_weight', type=float, default=0.2,
                        help='auxiliary spoof segment position loss weight (SAL-lite)')
    parser.add_argument('--use_binary_frame_head', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']),
                        help='use dedicated binary bonafide/spoof frame head on top of shared representation')
    parser.add_argument('--use_segment_position_head', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']),
                        help='use auxiliary spoof segment position head (start/middle/end/unit)')
    parser.add_argument('--cross_segment_mix_prob', type=float, default=0.0,
                        help='probability of cross-segment mixing augmentation for multiclass train data')
    parser.add_argument('--cross_segment_mix_min_frames', type=int, default=8,
                        help='minimum mixed span length in 20ms frames for cross-segment mixing')
    parser.add_argument('--cross_segment_mix_max_frames', type=int, default=64,
                        help='maximum mixed span length in 20ms frames for cross-segment mixing')
    parser.add_argument('--dia_class_weight', default=False,
                        type=lambda x: (str(x).lower() in ['true','yes','1']),
                        help='reuse frame class weights inside P2S/dia loss')
    parser.add_argument('--boundary_threshold_eval', type=float, default=0.5,
                        help='fixed boundary threshold for reported BoundF1; best threshold is also scanned')
    parser.add_argument('--bound_fp_penalty_weight', type=float, default=0.0,
                        help='extra penalty for high boundary probability on non-boundary frames')
    parser.add_argument('--bound_fp_margin', type=float, default=0.2,
                        help='margin above which non-boundary probabilities are penalized')
    parser.add_argument('--use_boundary_control', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--use_cross_routing', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--use_soft_segments', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--mamba_bidirectional', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']),
                        help='use bidirectional Mamba streams; false is faster but changes model capacity')
    parser.add_argument('--attractor_repulsion_weight', type=float, default=0.0,
                        help='weight for attractor token positive-cosine repulsion loss')
    parser.add_argument('--attractor_boundary_temp_strength', type=float, default=0.0,
                        help='boundary-conditioned logit temperature strength inside attractor head')
    parser.add_argument('--attractor_boundary_temp_min', type=float, default=0.5,
                        help='minimum attractor logit temperature when boundary temperature is enabled')
    parser.add_argument('--local_refine_enabled', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']),
                        help='enable local edge refinement block before frame classifiers')
    parser.add_argument('--local_refine_kernel', type=int, default=5,
                        help='odd kernel size for local edge refinement conv')
    parser.add_argument('--local_refine_hidden_scale', type=float, default=1.0,
                        help='hidden channel multiplier inside local edge refinement block')
    parser.add_argument('--boundary_gate_scale', type=float, default=1.0,
                        help='scale factor for boundary probability injected into local refinement gate')
    parser.add_argument('--detach_boundary_gate', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']),
                        help='detach boundary probability before local refinement gating for stabler training')
    parser.add_argument('--local_refine_min_gate', type=float, default=0.0,
                        help='minimum residual gate for local edge refinement; >0 makes refinement leaky/OOD-robust')
    parser.add_argument('--use_torch_compile', default=False,
                        type=lambda x: (str(x).lower() in ['true','yes','1']),
                        help='optionally compile model with torch.compile after loading/resuming weights')
    parser.add_argument('--torch_compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode when --use_torch_compile true')
    parser.add_argument('--allow_tf32', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']),
                        help='enable TF32 matmul/cudnn on Ampere+ GPUs for speed')
    parser.add_argument('--cudnn_benchmark', default=False,
                        type=lambda x: (str(x).lower() in ['true','yes','1']),
                        help='override cudnn benchmark after reproducibility setup; faster but less deterministic')
    parser.add_argument('--dia_start_epoch', type=int, default=5,
                        help='epoch at which dia/P2S loss starts with weight 0.5')
    parser.add_argument('--dia_full_epoch', type=int, default=15,
                        help='epoch at which dia/P2S loss reaches weight 1.0')
    parser.add_argument('--utt_loss_weight', type=float, default=1.0,
                        help='weight for utterance-level bonafide/spoof loss')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='number of gradient accumulation steps before optimizer update')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of DataLoader workers for training')
    parser.add_argument('--eval_num_workers', type=int, default=2,
                        help='number of DataLoader workers for validation/eval')
    parser.add_argument('--pin_memory', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--persistent_workers', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--export_analysis', default=False,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--analysis_max_batches', type=int, default=1,
                        help='number of evaluation batches to export analysis artifacts for')
    parser.add_argument('--analysis_batch_size', type=int, default=4,
                        help='batch size for analysis export loader')
    parser.add_argument('--keep_last_checkpoints', type=int, default=3,
                        help='number of latest step checkpoints to keep under model dir (<=0 keeps none)')
    
    # Backbone selection: Mamba vs Conformer
    parser.add_argument('--backbone_type', type=str, default='mamba', choices=['mamba', 'conformer'],
                        help='sequence modeling backbone: mamba (state space) or conformer (attention+conv)')
    parser.add_argument('--boundary_bias_scale', type=float, default=1.0,
                        help='[Conformer only] scale factor for boundary-biased attention injection')
    parser.add_argument('--conformer_num_heads', type=int, default=8,
                        help='[Conformer only] number of attention heads in multi-head self-attention')
    parser.add_argument('--conformer_conv_kernel', type=int, default=31,
                        help='[Conformer only] kernel size for depthwise convolution module (must be odd)')
    parser.add_argument('--conformer_expansion_factor', type=int, default=4,
                        help='[Conformer only] expansion factor for feed-forward modules')
    parser.add_argument('--conformer_dropout', type=float, default=0.1,
                        help='[Conformer only] dropout rate for Conformer blocks')

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    validate_label_mode_and_num_classes(args)
    args.database_path = resolve_cross_platform_path(args.database_path)
    args.protocols_path = resolve_cross_platform_path(args.protocols_path)
    args.ps_label_root = resolve_cross_platform_path(args.ps_label_root) if args.ps_label_root else args.ps_label_root
    args.resume_checkpoint = resolve_cross_platform_path(args.resume_checkpoint) if args.resume_checkpoint else args.resume_checkpoint
    if args.wavlm_local_files_only:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    args.track = 'LA'
    print(args)
    reproducibility(args.seed, args)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        if args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        print('TF32 enabled: matmul={} cudnn={}'.format(torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        print('cudnn benchmark active: {}'.format(torch.backends.cudnn.benchmark))

    track = args.track
    n_mejores = args.n_mejores_loss
    assert track in ['LA','DF','In-the-Wild'], 'Invalid track'
    assert args.n_average_model < args.n_mejores_loss + 1

    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    model_tag = 'BATmamba{}_{}_{}_{}_ES{}_NE{}'.format(
        args.algo, track, args.loss, args.lr, args.emb_size, args.num_encoders)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    print('Model tag: ' + model_tag)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    best_save_path = os.path.join(model_save_path, 'best')
    if not os.path.exists(best_save_path):
        os.mkdir(best_save_path)
    experiment_config = get_experiment_config(args, model_tag)
    experiment_config_path = os.path.join(model_save_path, 'experiment_config.json')
    with open(experiment_config_path, 'w', encoding='utf-8') as fh:
        json.dump(_to_serializable_config(experiment_config), fh, indent=2)
    experiment_record_path = os.path.join(model_save_path, 'validation_records.json')
    analysis_export_dir = os.path.join(model_save_path, 'analysis_exports')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    if torch.cuda.is_available():
        print('CUDA device index: {}'.format(torch.cuda.current_device()))
        print('CUDA device name: {}'.format(torch.cuda.get_device_name(torch.cuda.current_device())))

    has_offline_cache, train_pt_dir, dev_pt_dir = _detect_offline_tensor_cache(args.database_path, args.algo)
    if has_offline_cache:
        print('Using offline cached tensors:')
        print('  train -> {}'.format(train_pt_dir))
        print('  dev   -> {}'.format(dev_pt_dir))
    else:
        print('Offline cached tensors not found. Falling back to on-the-fly preprocessing inside Dataset.')

    train_loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': True,
        'drop_last': True,
        'pin_memory': args.pin_memory,
    }
    dev_loader_kwargs = {
        'batch_size': 8,
        'num_workers': args.eval_num_workers,
        'shuffle': False,
        'pin_memory': args.pin_memory,
    }
    if args.num_workers > 0:
        train_loader_kwargs['persistent_workers'] = args.persistent_workers
        train_loader_kwargs['prefetch_factor'] = 2
    if args.eval_num_workers > 0:
        dev_loader_kwargs['persistent_workers'] = args.persistent_workers
        dev_loader_kwargs['prefetch_factor'] = 2

    model = Model(args, device)
    if not args.FT_W2V:
        for param in model.ssl_model.parameters():
            param.requires_grad = False
    model = model.to(device)
    if args.resume_checkpoint:
        if not os.path.exists(args.resume_checkpoint):
            print('ERROR: resume checkpoint not found: {}'.format(args.resume_checkpoint))
            sys.exit(1)
        resume_state = torch.load(args.resume_checkpoint, map_location=device)
        if isinstance(resume_state, dict) and 'state_dict' in resume_state:
            resume_state = resume_state['state_dict']
        load_model_state(model, resume_state, strict=args.resume_strict)
        print('Resumed model weights from: {}'.format(args.resume_checkpoint))
        print('Resume global start epoch for schedule: {}'.format(args.resume_start_epoch))
    if args.use_torch_compile:
        if not hasattr(torch, 'compile'):
            print('WARNING: torch.compile is unavailable in this PyTorch version; continuing without compile.')
        else:
            print('Compiling model with torch.compile(mode={})'.format(args.torch_compile_mode))
            model = torch.compile(model, mode=args.torch_compile_mode)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- In-the-Wild eval only ----
    if args.track == 'In-the-Wild':
        best_save_path = best_save_path.replace(track, 'LA')
        model_save_path = model_save_path.replace(track, 'LA')
        print('######## Eval In-the-Wild ########')
        load_model_state(model, torch.load(
            os.path.join(best_save_path, 'best_0.pth')))
        sd = get_unwrapped_model(model).state_dict()
        for i in range(1, args.n_average_model):
            load_model_state(model, torch.load(
                os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            sd2 = get_unwrapped_model(model).state_dict()
            for key in sd:
                sd[key] = sd[key] + sd2[key]
        for key in sd:
            sd[key] = sd[key] / args.n_average_model
        load_model_state(model, sd)
        file_eval = genSpoof_list(
            dir_meta=os.path.join(args.protocols_path),
            is_train=False, is_eval=True)
        eval_set = Dataset_in_the_wild_eval(
            list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device,
            'Scores/{}/{}.txt'.format(args.track, model_tag))
        sys.exit(0)

    # ---- PartialSpoof Data loaders ----
    # Binary mode uses historical segment_labels/*.npy.
    # family3 / attack19 modes use converted dict labels under ps_label_root/{split}/.

    ps_proto_dir = os.path.join(args.protocols_path,
                                'protocols', 'PartialSpoof_LA_cm_protocols')
    files_id_train, utt_labels_train = parse_ps_protocol(
        os.path.join(ps_proto_dir, 'PartialSpoof.LA.cm.train.trl.txt'))
    files_id_dev, utt_labels_dev   = parse_ps_protocol(
        os.path.join(ps_proto_dir, 'PartialSpoof.LA.cm.dev.trl.txt'))
    print('no. of training trials', len(files_id_train))
    print('no. of validation trials', len(files_id_dev))

    if args.ps_label_mode == 'binary':
        seglab_train = load_seglab(
            os.path.join(args.database_path, 'segment_labels', 'train_seglab_0.02.npy'))
        seglab_dev = load_seglab(
            os.path.join(args.database_path, 'segment_labels', 'dev_seglab_0.02.npy'))
        train_set = Dataset_PartialSpoof_train(
            list_IDs=files_id_train,
            seglab=seglab_train,
            utt_labels=utt_labels_train,
            base_dir=os.path.join(args.database_path, 'train'),
            args=args,
            algo=args.algo,
        )
        dev_set = Dataset_PartialSpoof_eval(
            list_IDs=files_id_dev,
            seglab=seglab_dev,
            utt_labels=utt_labels_dev,
            base_dir=os.path.join(args.database_path, 'dev'),
            skip_missing=True,
        )
    else:
        if not args.ps_label_root:
            args.ps_label_root = os.path.join(os.path.dirname(__file__), 'multiclass_labels')
        label_file_name = 'family3_frames_0.02.npy' if args.ps_label_mode == 'family3' else 'attack19_frames_0.02.npy'
        train_labels = load_frame_label_dict(os.path.join(args.ps_label_root, 'train', label_file_name))
        dev_labels = load_frame_label_dict(os.path.join(args.ps_label_root, 'dev', label_file_name))
        require_all_classes = args.require_all_family3_classes if args.ps_label_mode == 'family3' else False
        train_counts, _, _ = _check_frame_label_coverage(
            train_labels,
            num_classes=args.num_classes,
            split_name='train',
            label_mode=args.ps_label_mode,
            require_all_classes=require_all_classes,
        )
        _check_frame_label_coverage(
            dev_labels,
            num_classes=args.num_classes,
            split_name='dev',
            label_mode=args.ps_label_mode,
            require_all_classes=require_all_classes,
        )
        args.frame_class_weights = compute_frame_class_weights_from_counts(
            train_counts,
            mode=args.frame_class_weight_mode,
            beta=args.frame_class_weight_beta,
            max_weight=args.frame_class_weight_max,
        )
        args.binary_frame_class_weights = compute_binary_frame_class_weights_from_counts(
            train_counts,
            mode=args.frame_class_weight_mode if args.binary_frame_aux_use_class_weights else 'none',
            beta=args.frame_class_weight_beta,
            max_weight=args.frame_class_weight_max,
        )
        if args.frame_class_weights is not None:
            print('[ClassWeight] frame CE weights: {}'.format(['%.4f' % x for x in args.frame_class_weights.tolist()]))
        if args.binary_frame_class_weights is not None:
            print('[ClassWeight] binary frame aux weights: {}'.format(['%.4f' % x for x in args.binary_frame_class_weights.tolist()]))
        train_set = Dataset_PartialSpoof_multiclass_train(
            list_IDs=files_id_train,
            frame_label_dict=train_labels,
            utt_labels=utt_labels_train,
            base_dir=os.path.join(args.database_path, 'train'),
            args=args,
            algo=args.algo,
        )
        dev_set = Dataset_PartialSpoof_multiclass_eval(
            list_IDs=files_id_dev,
            frame_label_dict=dev_labels,
            utt_labels=utt_labels_dev,
            base_dir=os.path.join(args.database_path, 'dev'),
            skip_missing=True,
        )
    train_loader = DataLoader(train_set, **train_loader_kwargs)
    del train_set

    dev_loader = DataLoader(dev_set, **dev_loader_kwargs)
    del dev_set

    # ---- Debug mode: limit batches for quick pipeline test ----
    debug_steps = args.debug_steps  # 0 = disabled, e.g. 5 = only 5 batches
    not_improving = 0
    epoch = 0
    bests = np.ones(n_mejores, dtype=float) * float('inf')
    best_loss = float('inf')

    if args.train:
        # NOTE: Do NOT pre-fill best_*.pth with np.savetxt placeholders
        # (that would overwrite real saved models on restart)
        for local_epoch in range(args.num_epochs):
            epoch = args.resume_start_epoch + local_epoch
            print('######## Epoch {} ########'.format(epoch))
            train_epoch(train_loader, model, optimizer, device, epoch,
                        checkpoint_dir=model_save_path, debug_steps=debug_steps, args=args)
            val_loss, val_metrics = evaluate_accuracy(dev_loader, model, device, debug_steps=debug_steps, args=args)
            val_record = {
                'epoch': int(epoch),
                'val_loss': float(val_loss),
                'best_loss_before_epoch': float(best_loss) if np.isfinite(best_loss) else None,
                'metrics': val_metrics,
            }
            append_experiment_record(experiment_record_path, val_record)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(get_unwrapped_model(model).state_dict(),
                    os.path.join(model_save_path, 'best.pth'))
                print('New best epoch')
                not_improving = 0
            else:
                not_improving += 1
            for i in range(n_mejores):
                if bests[i] > val_loss:
                    for t in range(n_mejores - 1, i, -1):
                        bests[t] = bests[t - 1]
                        src = os.path.join(best_save_path,
                            'best_{}.pth'.format(t - 1))
                        dst = os.path.join(best_save_path,
                            'best_{}.pth'.format(t))
                        if os.path.exists(src):
                            shutil.move(src, dst)
                    bests[i] = val_loss
                    torch.save(get_unwrapped_model(model).state_dict(),
                        os.path.join(best_save_path,
                            'best_{}.pth'.format(i)))
                    break
            print('\n{} - val_loss={:.4f} | macro_f1={:.4f} | seen_macro_f1={:.4f} | spoof_f1={:.4f} | frame_eer={:.4f} | bound_f1={:.4f} | bound_best_f1={:.4f} | utt_f1={:.4f} | utt_eer={:.4f}'.format(
                epoch, val_loss, val_metrics['macro_f1'], val_metrics.get('seen_macro_f1', 0.0), val_metrics['spoof_f1'], val_metrics.get('frame_eer', 1.0), val_metrics['boundary_f1'], val_metrics.get('boundary_best_f1', 0.0), val_metrics.get('utt_f1', 0.0), val_metrics.get('utt_eer', 1.0)))
            cleanup_step_checkpoints(model_save_path, keep_last=args.keep_last_checkpoints)
            print('n-best losses:', bests)
        print('Total epochs: ' + str(args.num_epochs))

    # ---- Final evaluation ----
    print('######## Eval ########')
    if args.average_model:
        actual_n = sum(
            1 for i in range(args.n_average_model)
            if os.path.exists(os.path.join(best_save_path,
                'best_{}.pth'.format(i)))
            and os.path.getsize(os.path.join(best_save_path,
                'best_{}.pth'.format(i))) > 1000)
        n_avg = min(args.n_average_model, actual_n)
        print('Averaging {} best models'.format(n_avg))

        if n_avg == 0:
            # No valid best_*.pth — fall back to best.pth or latest checkpoint
            best_single = os.path.join(model_save_path, 'best.pth')
            checkpoints = sorted(
                [f for f in os.listdir(model_save_path)
                 if f.startswith('checkpoint_') and f.endswith('.pth')],
                key=lambda x: os.path.getmtime(os.path.join(model_save_path, x))
            )
            if os.path.exists(best_single) and os.path.getsize(best_single) > 1000:
                print('Loading best.pth')
                load_model_state(model, torch.load(best_single,
                    map_location=device))
            elif checkpoints:
                latest = os.path.join(model_save_path, checkpoints[-1])
                print('Loading latest checkpoint: {}'.format(latest))
                load_model_state(model, torch.load(latest,
                    map_location=device))
            else:
                print('ERROR: No valid model found. Please train first.')
                sys.exit(1)
        else:
            load_model_state(model, torch.load(
                os.path.join(best_save_path, 'best_0.pth'),
                map_location=device))
            sd = get_unwrapped_model(model).state_dict()
            for i in range(1, n_avg):
                load_model_state(model, torch.load(
                    os.path.join(best_save_path, 'best_{}.pth'.format(i)),
                    map_location=device))
                sd2 = get_unwrapped_model(model).state_dict()
                for key in sd:
                    sd[key] = sd[key] + sd2[key]
            for key in sd:
                sd[key] = sd[key] / n_avg
            load_model_state(model, sd)
    else:
        best_single = os.path.join(model_save_path, 'best.pth')
        checkpoints = sorted(
            [f for f in os.listdir(model_save_path)
             if f.startswith('checkpoint_') and f.endswith('.pth')],
            key=lambda x: os.path.getmtime(os.path.join(model_save_path, x))
        )
        if os.path.exists(best_single) and os.path.getsize(best_single) > 1000:
            load_model_state(model, torch.load(best_single, map_location=device))
        elif checkpoints:
            latest = os.path.join(model_save_path, checkpoints[-1])
            print('Loading latest checkpoint: {}'.format(latest))
            load_model_state(model, torch.load(latest, map_location=device))
        else:
            print('ERROR: No valid model found. Please train first.')
            sys.exit(1)

    tracks = 'LA' if args.algo == 5 else 'DF'
    if args.comment_eval:
        model_tag = model_tag + '_{}'.format(args.comment_eval)
    os.makedirs('./Scores/PartialSpoof', exist_ok=True)
    score_path = './Scores/PartialSpoof/{}.txt'.format(model_tag)
    if not os.path.exists(score_path):
        ps_proto_dir = os.path.join(args.protocols_path,
                                    'protocols', 'PartialSpoof_LA_cm_protocols')
        files_id_eval, utt_labels_eval = parse_ps_protocol(
            os.path.join(ps_proto_dir, 'PartialSpoof.LA.cm.dev.trl.txt'),
            is_eval=True)
        print('no. of eval trials (using dev set)', len(files_id_eval))
        if args.ps_label_mode == 'binary':
            seglab_eval = load_seglab(
                os.path.join(args.database_path, 'segment_labels', 'dev_seglab_0.02.npy'))
            eval_set = Dataset_PartialSpoof_eval(
                list_IDs=files_id_eval,
                seglab=seglab_eval,
                utt_labels=utt_labels_eval,
                base_dir=os.path.join(args.database_path, 'dev'),
            )
        else:
            if not args.ps_label_root:
                args.ps_label_root = os.path.join(os.path.dirname(__file__), 'multiclass_labels')
            label_file_name = 'family3_frames_0.02.npy' if args.ps_label_mode == 'family3' else 'attack19_frames_0.02.npy'
            eval_labels = load_frame_label_dict(os.path.join(args.ps_label_root, 'dev', label_file_name))
            eval_set = Dataset_PartialSpoof_multiclass_eval(
                list_IDs=files_id_eval,
                frame_label_dict=eval_labels,
                utt_labels=utt_labels_eval,
                base_dir=os.path.join(args.database_path, 'dev'),
            )
        produce_evaluation_file(eval_set, model, device, score_path)
        if args.export_analysis:
            export_analysis_artifacts(
                dataset=eval_set,
                model=model,
                device=device,
                export_dir=analysis_export_dir,
                max_batches=args.analysis_max_batches,
                batch_size=args.analysis_batch_size,
            )
    else:
        print('Score file already exists')
        if args.export_analysis:
            ps_proto_dir = os.path.join(args.protocols_path,
                                        'protocols', 'PartialSpoof_LA_cm_protocols')
            files_id_eval, utt_labels_eval = parse_ps_protocol(
                os.path.join(ps_proto_dir, 'PartialSpoof.LA.cm.dev.trl.txt'),
                is_eval=True)
            if args.ps_label_mode == 'binary':
                seglab_eval = load_seglab(
                    os.path.join(args.database_path, 'segment_labels', 'dev_seglab_0.02.npy'))
                eval_set = Dataset_PartialSpoof_eval(
                    list_IDs=files_id_eval,
                    seglab=seglab_eval,
                    utt_labels=utt_labels_eval,
                    base_dir=os.path.join(args.database_path, 'dev'),
                )
            else:
                if not args.ps_label_root:
                    args.ps_label_root = os.path.join(os.path.dirname(__file__), 'multiclass_labels')
                label_file_name = 'family3_frames_0.02.npy' if args.ps_label_mode == 'family3' else 'attack19_frames_0.02.npy'
                eval_labels = load_frame_label_dict(os.path.join(args.ps_label_root, 'dev', label_file_name))
                eval_set = Dataset_PartialSpoof_multiclass_eval(
                    list_IDs=files_id_eval,
                    frame_label_dict=eval_labels,
                    utt_labels=utt_labels_eval,
                    base_dir=os.path.join(args.database_path, 'dev'),
                )
            export_analysis_artifacts(
                dataset=eval_set,
                model=model,
                device=device,
                export_dir=analysis_export_dir,
                max_batches=args.analysis_max_batches,
                batch_size=args.analysis_batch_size,
            )
