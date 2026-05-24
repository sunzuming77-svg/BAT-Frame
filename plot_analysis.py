import argparse
import json
import os
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


HEATMAP_KEYS = [
    'time_gate',
    'freq_gate',
    'route_map',
    'segment_assignments',
    'segment_centers',
    'segment_backproj',
]

ATTN_KEYS = [
    'frame_attn_map',
    'segment_attn_map',
]

LINE_KEYS = [
    'boundary_prior',
    'boundary_final',
]


def load_optional_npy(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    return np.load(path)


def squeeze_sample_axis(array: np.ndarray, sample_idx: int) -> np.ndarray:
    if array.ndim == 0:
        return array
    if array.shape[0] <= sample_idx:
        raise IndexError(f'sample_idx={sample_idx} out of range for shape {array.shape}')
    return array[sample_idx]


def ensure_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array[None, :]
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array.mean(axis=0)
    return array.reshape(array.shape[0], -1)


def ensure_1d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)
    return array.reshape(-1)


def plot_boundary_overview(batch_dir: str, sample_idx: int, output_dir: str, utt_label: str) -> None:
    boundary_prior = load_optional_npy(os.path.join(batch_dir, 'boundary_prior.npy'))
    boundary_final = load_optional_npy(os.path.join(batch_dir, 'boundary_final.npy'))
    frame_labels = load_optional_npy(os.path.join(batch_dir, 'frame_labels.npy'))
    boundary_labels = load_optional_npy(os.path.join(batch_dir, 'boundary_labels.npy'))

    if boundary_prior is None and boundary_final is None:
        return

    plt.figure(figsize=(12, 5))
    x_axis = None

    if frame_labels is not None:
        gt_frames = ensure_1d(squeeze_sample_axis(frame_labels, sample_idx))
        x_axis = np.arange(gt_frames.shape[0])
        plt.step(x_axis, gt_frames, where='mid', label='frame_labels', linewidth=1.5, alpha=0.8)

    if boundary_labels is not None:
        gt_bounds = ensure_1d(squeeze_sample_axis(boundary_labels, sample_idx))
        x_axis = np.arange(gt_bounds.shape[0])
        plt.plot(x_axis, gt_bounds, label='boundary_labels', linewidth=1.2, alpha=0.8)

    if boundary_prior is not None:
        prior = ensure_1d(squeeze_sample_axis(boundary_prior, sample_idx))
        x_axis = np.arange(prior.shape[0])
        plt.plot(x_axis, prior, label='boundary_prior', linewidth=2.0)

    if boundary_final is not None:
        final = ensure_1d(squeeze_sample_axis(boundary_final, sample_idx))
        x_axis = np.arange(final.shape[0])
        plt.plot(x_axis, final, label='boundary_final', linewidth=2.0)

    plt.title(f'Boundary overview - {utt_label}')
    plt.xlabel('Frame index')
    plt.ylabel('Value')
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_{sample_idx:02d}_boundary_overview.png'), dpi=180)
    plt.close()


def plot_heatmap(array: np.ndarray, title: str, output_path: str, xlabel: str = 'Frame index', ylabel: str = 'Channel') -> None:
    plt.figure(figsize=(12, 5))
    plt.imshow(array, aspect='auto', interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_attention_maps(batch_dir: str, sample_idx: int, output_dir: str, utt_label: str) -> None:
    for key in ATTN_KEYS:
        array = load_optional_npy(os.path.join(batch_dir, f'{key}.npy'))
        if array is None:
            continue
        sample = squeeze_sample_axis(array, sample_idx)
        if sample.ndim == 3:
            averaged = sample.mean(axis=0)
            ylabel = 'Token index'
        else:
            averaged = ensure_2d(sample)
            ylabel = 'Channel'
        plot_heatmap(
            averaged,
            title=f'{key} - {utt_label}',
            output_path=os.path.join(output_dir, f'sample_{sample_idx:02d}_{key}.png'),
            xlabel='Source token/frame',
            ylabel=ylabel,
        )


def plot_feature_heatmaps(batch_dir: str, sample_idx: int, output_dir: str, utt_label: str) -> None:
    for key in HEATMAP_KEYS:
        array = load_optional_npy(os.path.join(batch_dir, f'{key}.npy'))
        if array is None:
            continue
        sample = squeeze_sample_axis(array, sample_idx)
        sample_2d = ensure_2d(sample).T
        plot_heatmap(
            sample_2d,
            title=f'{key} - {utt_label}',
            output_path=os.path.join(output_dir, f'sample_{sample_idx:02d}_{key}.png'),
            xlabel='Frame index',
            ylabel='Feature / segment channel',
        )


def plot_stable_summary(batch_dir: str, sample_idx: int, output_dir: str, utt_label: str) -> None:
    array = load_optional_npy(os.path.join(batch_dir, 'stable_summary.npy'))
    if array is None:
        return
    sample = squeeze_sample_axis(array, sample_idx)
    sample_2d = ensure_2d(sample).T
    plot_heatmap(
        sample_2d,
        title=f'stable_summary - {utt_label}',
        output_path=os.path.join(output_dir, f'sample_{sample_idx:02d}_stable_summary.png'),
        xlabel='Summary slot',
        ylabel='Feature channel',
    )


def load_meta(batch_dir: str) -> dict:
    meta_path = os.path.join(batch_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def resolve_sample_indices(meta: dict, requested: Optional[List[int]]) -> List[int]:
    utt_ids = meta.get('utt_id', [])
    total = len(utt_ids)
    if total == 0:
        frame_labels = meta.get('frame_labels_shape', [])
        total = frame_labels[0] if frame_labels else 0
    if requested:
        return [idx for idx in requested if 0 <= idx < total]
    return list(range(total))


def list_batch_dirs(export_root: str) -> List[str]:
    if not os.path.exists(export_root):
        return []
    batch_dirs = []
    for entry in sorted(os.listdir(export_root)):
        full_path = os.path.join(export_root, entry)
        if os.path.isdir(full_path) and entry.startswith('batch_'):
            batch_dirs.append(full_path)
    return batch_dirs


def plot_batch(batch_dir: str, output_root: str, sample_indices: Optional[List[int]] = None) -> None:
    meta = load_meta(batch_dir)
    utt_ids = meta.get('utt_id', [])
    batch_name = os.path.basename(batch_dir)
    batch_output = os.path.join(output_root, batch_name)
    os.makedirs(batch_output, exist_ok=True)

    for sample_idx in resolve_sample_indices(meta, sample_indices):
        utt_label = utt_ids[sample_idx] if sample_idx < len(utt_ids) else f'sample_{sample_idx:02d}'
        sample_output = os.path.join(batch_output, f'sample_{sample_idx:02d}')
        os.makedirs(sample_output, exist_ok=True)
        plot_boundary_overview(batch_dir, sample_idx, sample_output, utt_label)
        plot_feature_heatmaps(batch_dir, sample_idx, sample_output, utt_label)
        plot_attention_maps(batch_dir, sample_idx, sample_output, utt_label)
        plot_stable_summary(batch_dir, sample_idx, sample_output, utt_label)


def parse_sample_indices(raw: str) -> Optional[List[int]]:
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(',') if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot exported BAT-Mamba analysis artifacts')
    parser.add_argument('--export_root', type=str, required=True, help='path to analysis_exports directory')
    parser.add_argument('--output_root', type=str, default='', help='directory to save generated plots')
    parser.add_argument('--batch', type=str, default='', help='specific batch directory name, e.g. batch_000')
    parser.add_argument('--samples', type=str, default='', help='comma-separated sample indices to plot')
    args = parser.parse_args()

    output_root = args.output_root or os.path.join(args.export_root, 'plots')
    os.makedirs(output_root, exist_ok=True)
    sample_indices = parse_sample_indices(args.samples)

    if args.batch:
        batch_dirs = [os.path.join(args.export_root, args.batch)]
    else:
        batch_dirs = list_batch_dirs(args.export_root)

    if not batch_dirs:
        print(f'No batch directories found under {args.export_root}')
        return

    for batch_dir in batch_dirs:
        if not os.path.exists(batch_dir):
            print(f'Skipping missing batch directory: {batch_dir}')
            continue
        plot_batch(batch_dir, output_root, sample_indices=sample_indices)

    print(f'Plots saved to {output_root}')


if __name__ == '__main__':
    main()
