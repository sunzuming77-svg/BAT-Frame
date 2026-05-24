import argparse
import json
import os
from typing import Dict

import numpy as np

from partialspoof_multiclass import (
    ATTACK19_NUM_CLASSES,
    FAMILY3_NUM_CLASSES,
    NUM_FRAMES,
    PartialSpoofLabelConverter,
    collect_vad_files,
    frame_labels_to_boundaries,
    load_vad_file,
    serialize_family_map,
    summarize_frame_dict,
    summarize_attack_ids,
    resolve_cross_platform_path,
)


DEFAULT_PARTIAL_SPOOF_ROOT = r'C:\Partial Spoof'
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'multiclass_labels')


def convert_split(split: str, split_dir: str, converter: PartialSpoofLabelConverter, progress_every: int = 200):
    family3_dict: Dict[str, np.ndarray] = {}
    attack19_dict: Dict[str, np.ndarray] = {}
    family3_boundary: Dict[str, np.ndarray] = {}
    attack19_boundary: Dict[str, np.ndarray] = {}

    vad_files = collect_vad_files(split_dir)
    total_files = len(vad_files)
    print(f'[convert] split={split} files={total_files}')
    for file_idx, vad_path in enumerate(vad_files, start=1):
        utt_id = os.path.splitext(os.path.basename(vad_path))[0]
        segments = load_vad_file(vad_path)
        family3_frames = converter.convert_segments_to_frames(segments, scheme='family3', num_frames=NUM_FRAMES)
        attack19_frames = converter.convert_segments_to_frames(segments, scheme='attack19', num_frames=NUM_FRAMES)

        family3_dict[utt_id] = family3_frames.astype(np.int64)
        attack19_dict[utt_id] = attack19_frames.astype(np.int64)
        family3_boundary[utt_id] = frame_labels_to_boundaries(family3_frames, num_frames=NUM_FRAMES)
        attack19_boundary[utt_id] = frame_labels_to_boundaries(attack19_frames, num_frames=NUM_FRAMES)

        if progress_every > 0 and (file_idx % progress_every == 0 or file_idx == total_files):
            print(f'[convert] split={split} progress={file_idx}/{total_files}')

    return {
        'family3_frames': family3_dict,
        'attack19_frames': attack19_dict,
        'family3_boundary': family3_boundary,
        'attack19_boundary': attack19_boundary,
        'family3_summary': summarize_frame_dict(family3_dict, FAMILY3_NUM_CLASSES),
        'attack19_summary': summarize_frame_dict(attack19_dict, ATTACK19_NUM_CLASSES),
        'attack_id_summary': summarize_attack_ids(attack19_dict),
        'num_vad_files': len(vad_files),
    }


def save_dict_npy(payload: Dict[str, np.ndarray], out_path: str):
    np.save(out_path, payload, allow_pickle=True)


def parse_args():
    p = argparse.ArgumentParser(description='Convert PartialSpoof .vad labels to 208-frame family3/attack19 labels.')
    p.add_argument('--partial_spoof_root', type=str, default=DEFAULT_PARTIAL_SPOOF_ROOT)
    p.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--family_map_path', type=str, default='')
    p.add_argument('--write_default_family_map', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    p.add_argument('--splits', type=str, default='train,dev,eval',
                   help='Comma-separated splits to convert, e.g. dev or train,dev,eval')
    p.add_argument('--progress_every', type=int, default=200,
                   help='Print progress every N files within each split')
    p.add_argument('--strict_family3_coverage', default=False, type=lambda x: str(x).lower() in ['true', 'yes', '1'],
                   help='Fail if a converted split is missing any family3 class. Use after the Axx->TTS/VC map is authoritative.')
    return p.parse_args()


def main():
    args = parse_args()

    partial_spoof_root = resolve_cross_platform_path(args.partial_spoof_root)
    vad_root = os.path.join(partial_spoof_root, 'vad_20sil', 'vad_20sil')
    label2num_path = os.path.join(partial_spoof_root, 'label2num_all')
    output_dir = resolve_cross_platform_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    family_map_path = resolve_cross_platform_path(args.family_map_path) if args.family_map_path else None
    converter = PartialSpoofLabelConverter.from_label2num(label2num_path, family_map_path=family_map_path)

    if args.write_default_family_map and family_map_path is None:
        serialize_family_map(os.path.join(output_dir, 'default_family_map.json'), converter.family_map)

    all_summary = {
        'partial_spoof_root': partial_spoof_root,
        'vad_root': vad_root,
        'label2num_path': label2num_path,
        'output_dir': output_dir,
        'num_frames': NUM_FRAMES,
        'family_map': converter.family_map,
        'splits': {},
    }

    requested_splits = [x.strip() for x in args.splits.split(',') if x.strip()]
    for split in requested_splits:
        split_dir = os.path.join(vad_root, split)
        converted = convert_split(split=split, split_dir=split_dir, converter=converter, progress_every=args.progress_every)
        split_out_dir = os.path.join(output_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)

        save_dict_npy(converted['family3_frames'], os.path.join(split_out_dir, 'family3_frames_0.02.npy'))
        save_dict_npy(converted['attack19_frames'], os.path.join(split_out_dir, 'attack19_frames_0.02.npy'))
        save_dict_npy(converted['family3_boundary'], os.path.join(split_out_dir, 'family3_boundary_0.02.npy'))
        save_dict_npy(converted['attack19_boundary'], os.path.join(split_out_dir, 'attack19_boundary_0.02.npy'))

        split_summary = {
            'num_vad_files': converted['num_vad_files'],
            'family3': converted['family3_summary'],
            'attack19': converted['attack19_summary'],
            'attack_ids': converted['attack_id_summary'],
        }
        missing_family3 = split_summary['family3'].get('missing_classes', [])
        if missing_family3:
            message = (
                f"[convert][WARN] split={split} family3 labels are missing classes {missing_family3}. "
                "Do not use these labels for formal family3 training until the Axx->TTS/VC map is verified."
            )
            print(message)
            if args.strict_family3_coverage:
                raise ValueError(message)
        all_summary['splits'][split] = split_summary

        with open(os.path.join(split_out_dir, 'summary.json'), 'w', encoding='utf-8') as fh:
            json.dump(split_summary, fh, indent=2)

    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(all_summary, fh, indent=2)

    print(json.dumps(all_summary, indent=2))


if __name__ == '__main__':
    main()
