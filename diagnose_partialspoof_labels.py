import argparse
import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Set

from partialspoof_multiclass import (
    ATTACK19_NUM_CLASSES,
    PartialSpoofLabelConverter,
    collect_vad_files,
    load_label2num,
    load_vad_file,
    resolve_cross_platform_path,
)


DEFAULT_PARTIAL_SPOOF_ROOT = r'C:\Partial Spoof'
DEFAULT_PROTOCOL_ROOT = r'C:\PS_data\protocols\PartialSpoof_LA_cm_protocols'
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'multiclass_labels', 'label_diagnostics.json')
DEFAULT_CONVERTED_LABEL_ROOT = os.path.join(os.path.dirname(__file__), 'multiclass_labels')


def load_converted_summary(label_root: str) -> Dict[str, object]:
    summary_path = os.path.join(label_root, 'summary.json')
    if not os.path.exists(summary_path):
        return {'exists': False, 'path': summary_path}
    with open(summary_path, 'r', encoding='utf-8') as fh:
        payload = json.load(fh)
    payload['exists'] = True
    payload['path'] = summary_path
    return payload


def summarize_from_converted_summary(split: str, converted_summary: Dict[str, object], protocol_root: str) -> Dict[str, object]:
    split_payload = converted_summary.get('splits', {}).get(split, {}) if converted_summary.get('exists') else {}
    attack_summary = split_payload.get('attack_ids', {})
    if not attack_summary and split_payload.get('attack19'):
        counts = split_payload['attack19'].get('class_counts', [])
        attack_summary = {
            'present_attack_ids': [f'A{i:02d}' for i in range(1, min(len(counts), ATTACK19_NUM_CLASSES)) if counts[i] > 0],
            'missing_attack_ids': [f'A{i:02d}' for i in range(1, ATTACK19_NUM_CLASSES) if i >= len(counts) or counts[i] == 0],
            'attack_frame_counts': {f'A{i:02d}': int(counts[i]) if i < len(counts) else 0 for i in range(1, ATTACK19_NUM_CLASSES)},
        }
    family3 = split_payload.get('family3', {})
    family_counts = family3.get('class_counts', [])
    missing_family_classes = []
    family_names = ['bonafide', 'tts', 'vc']
    for idx, name in enumerate(family_names):
        if idx >= len(family_counts) or int(family_counts[idx]) == 0:
            missing_family_classes.append(name)

    protocol_path = os.path.join(protocol_root, f'PartialSpoof.LA.cm.{split}.trl.txt')
    protocol = parse_protocol(protocol_path)
    return {
        'split': split,
        'source': 'converted_summary',
        'summary_exists': bool(converted_summary.get('exists')),
        'num_vad_files': int(split_payload.get('num_vad_files', 0) or 0),
        'protocol': protocol,
        'protocol_vad_overlap': {
            'mode': 'not_computed_in_fast_mode',
            'num_protocol_ids': int(protocol.get('num_ids', 0) or 0),
            'num_vad_ids': int(split_payload.get('num_vad_files', 0) or 0),
        },
        'present_attack_ids': attack_summary.get('present_attack_ids', []),
        'missing_attack_ids': attack_summary.get('missing_attack_ids', []),
        'attack_frame_counts': attack_summary.get('attack_frame_counts', {}),
        'family3': family3,
        'missing_family_classes': missing_family_classes,
    }


def parse_protocol(protocol_path: str) -> Dict[str, object]:
    ids: List[str] = []
    key_counts = Counter()
    system_counts = Counter()
    missing_or_short = 0

    if not protocol_path or not os.path.exists(protocol_path):
        return {
            'path': protocol_path,
            'exists': False,
            'num_rows': 0,
            'num_ids': 0,
            'duplicate_ids': [],
            'key_counts': {},
            'system_counts': {},
            'missing_or_short_rows': 0,
        }

    with open(protocol_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                missing_or_short += 1
                continue
            ids.append(parts[1])
            if len(parts) >= 4:
                system_counts[parts[3]] += 1
            if len(parts) >= 5:
                key_counts[parts[4]] += 1

    id_counts = Counter(ids)
    duplicate_ids = sorted([utt_id for utt_id, count in id_counts.items() if count > 1])
    return {
        'path': protocol_path,
        'exists': True,
        'num_rows': len(ids),
        'num_ids': len(set(ids)),
        'duplicate_ids': duplicate_ids[:50],
        'num_duplicate_ids': len(duplicate_ids),
        'key_counts': dict(sorted(key_counts.items())),
        'system_counts': dict(sorted(system_counts.items())),
        'missing_or_short_rows': missing_or_short,
    }


def labels_from_segments(converter: PartialSpoofLabelConverter, segments) -> Set[str]:
    labels = set()
    for seg in segments:
        labels.add(converter.label_name(seg.label_id))
    return labels


def attack_tags_from_labels(converter: PartialSpoofLabelConverter, labels: Iterable[str]) -> Set[str]:
    tags = set()
    for label_name in labels:
        tag = converter.attack_tag_from_label(label_name)
        if tag is not None:
            tags.add(tag)
    return tags


def summarize_split(
    split: str,
    vad_root: str,
    protocol_root: str,
    converter: PartialSpoofLabelConverter,
    max_files: int = 0,
    progress_every: int = 1000,
) -> Dict[str, object]:
    split_dir = os.path.join(vad_root, split)
    vad_files = collect_vad_files(split_dir) if os.path.isdir(split_dir) else []
    total_vad_files = len(vad_files)
    if max_files and max_files > 0:
        vad_files = vad_files[:max_files]

    label_file_counts = Counter()
    attack_file_counts = Counter()
    attack_frame_counts = Counter()
    family_file_counts = Counter()
    family_frame_counts = Counter()
    label_id_counts = Counter()
    bad_files: List[Dict[str, str]] = []

    for file_idx, vad_path in enumerate(vad_files, start=1):
        if progress_every > 0 and (file_idx == 1 or file_idx % progress_every == 0 or file_idx == len(vad_files)):
            print(f'[diagnose] split={split} scan_vad progress={file_idx}/{len(vad_files)}', flush=True)
        utt_id = os.path.splitext(os.path.basename(vad_path))[0]
        try:
            segments = load_vad_file(vad_path)
            labels = labels_from_segments(converter, segments)
        except Exception as exc:
            bad_files.append({'utt_id': utt_id, 'path': vad_path, 'error': str(exc)})
            continue

        for seg in segments:
            label_id_counts[int(seg.label_id)] += 1

        for label_name in labels:
            label_file_counts[label_name] += 1

        attacks = attack_tags_from_labels(converter, labels)
        for attack_tag in attacks:
            attack_file_counts[attack_tag] += 1
            family = converter.family_map.get(attack_tag)
            if family:
                family_file_counts[family] += 1

        for seg in segments:
            label_name = converter.label_name(seg.label_id)
            attack_tag = converter.attack_tag_from_label(label_name)
            frame_estimate = max(1, int(round((seg.end - seg.start) / 0.02))) if seg.end > seg.start else 0
            if attack_tag is not None:
                attack_frame_counts[attack_tag] += frame_estimate
                family = converter.family_map.get(attack_tag)
                if family:
                    family_frame_counts[family] += frame_estimate
            elif label_name in ('bonafide', 'nonbona'):
                family_frame_counts['bonafide'] += frame_estimate

    present_attack_ids = [f'A{i:02d}' for i in range(1, ATTACK19_NUM_CLASSES) if attack_frame_counts[f'A{i:02d}'] > 0]
    missing_attack_ids = [f'A{i:02d}' for i in range(1, ATTACK19_NUM_CLASSES) if attack_frame_counts[f'A{i:02d}'] == 0]
    missing_family_classes = [name for name in ['bonafide', 'tts', 'vc'] if family_frame_counts[name] == 0]

    protocol_path = os.path.join(protocol_root, f'PartialSpoof.LA.cm.{split}.trl.txt')
    protocol = parse_protocol(protocol_path)
    vad_ids = {os.path.splitext(os.path.basename(path))[0] for path in vad_files}
    proto_ids = set()
    if protocol.get('exists'):
        with open(protocol_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 2:
                    proto_ids.add(parts[1])

    return {
        'split': split,
        'vad_dir': split_dir,
        'num_vad_files_total': total_vad_files,
        'num_vad_files_scanned': len(vad_files),
        'bad_files': bad_files[:50],
        'num_bad_files': len(bad_files),
        'protocol': protocol,
        'protocol_vad_overlap': {
            'num_protocol_ids': len(proto_ids),
            'num_vad_ids': len(vad_ids),
            'num_intersection': len(proto_ids & vad_ids),
            'protocol_not_in_vad_count': len(proto_ids - vad_ids),
            'vad_not_in_protocol_count': len(vad_ids - proto_ids),
            'protocol_not_in_vad_examples': sorted(list(proto_ids - vad_ids))[:20],
            'vad_not_in_protocol_examples': sorted(list(vad_ids - proto_ids))[:20],
        },
        'label_file_counts': dict(sorted(label_file_counts.items())),
        'label_id_segment_counts': {str(k): int(v) for k, v in sorted(label_id_counts.items())},
        'present_attack_ids': present_attack_ids,
        'missing_attack_ids': missing_attack_ids,
        'attack_file_counts': {f'A{i:02d}': int(attack_file_counts[f'A{i:02d}']) for i in range(1, ATTACK19_NUM_CLASSES)},
        'attack_frame_counts': {f'A{i:02d}': int(attack_frame_counts[f'A{i:02d}']) for i in range(1, ATTACK19_NUM_CLASSES)},
        'family_file_counts': {name: int(family_file_counts[name]) for name in ['tts', 'vc']},
        'family_frame_counts': {name: int(family_frame_counts[name]) for name in ['bonafide', 'tts', 'vc']},
        'missing_family_classes': missing_family_classes,
        'family_map_used': converter.family_map,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Diagnose PartialSpoof .vad labels and Axx->family3 mapping coverage.')
    parser.add_argument('--partial_spoof_root', type=str, default=DEFAULT_PARTIAL_SPOOF_ROOT)
    parser.add_argument('--protocol_root', type=str, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument('--family_map_path', type=str, default='')
    parser.add_argument('--splits', type=str, default='train,dev,eval')
    parser.add_argument('--max_files_per_split', type=int, default=0,
                        help='If >0, scan only the first N vad files per split for quick diagnostics.')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--converted_label_root', type=str, default=DEFAULT_CONVERTED_LABEL_ROOT,
                        help='Root containing converted summary.json. Used by default for fast diagnostics.')
    parser.add_argument('--scan_vad', default=False, type=lambda x: str(x).lower() in ['true', 'yes', '1'],
                        help='If true, scan raw .vad files. This is slow on /mnt/c with many small files.')
    parser.add_argument('--progress_every', type=int, default=1000,
                        help='Print progress every N vad files when --scan_vad true.')
    return parser.parse_args()


def main():
    args = parse_args()
    partial_spoof_root = resolve_cross_platform_path(args.partial_spoof_root)
    vad_root = os.path.join(partial_spoof_root, 'vad_20sil', 'vad_20sil')
    label2num_path = os.path.join(partial_spoof_root, 'label2num_all')
    protocol_root = resolve_cross_platform_path(args.protocol_root)
    converted_label_root = resolve_cross_platform_path(args.converted_label_root)
    family_map_path = resolve_cross_platform_path(args.family_map_path) if args.family_map_path else None

    if not os.path.exists(label2num_path):
        raise FileNotFoundError(
            f'label2num_all not found: {label2num_path}. '
            'When using --scan_vad true with a Linux path (e.g. ~/PartialSpoof), '
            'please ensure both vad_20sil/vad_20sil/* and label2num_all are copied there.'
        )
    label_to_id, id_to_label = load_label2num(label2num_path)
    family_map = None
    if family_map_path:
        with open(family_map_path, 'r', encoding='utf-8') as fh:
            family_map = json.load(fh)
    converter = PartialSpoofLabelConverter(id_to_label=id_to_label, family_map=family_map)

    splits = [x.strip() for x in args.splits.split(',') if x.strip()]
    payload = {
        'partial_spoof_root': partial_spoof_root,
        'vad_root': vad_root,
        'label2num_path': label2num_path,
        'protocol_root': protocol_root,
        'converted_label_root': converted_label_root,
        'family_map_path': family_map_path,
        'scan_vad': bool(args.scan_vad),
        'label2num_duplicate_sensitive_note': 'label2num_all contains duplicated A06 lines in the provided copy; identical duplicate IDs are harmless but should be documented.',
        'label_to_id': label_to_id,
        'id_to_label': {str(k): v for k, v in sorted(id_to_label.items())},
        'splits': {},
    }

    converted_summary = load_converted_summary(converted_label_root)
    if not args.scan_vad:
        payload['converted_summary_path'] = converted_summary.get('path')
        payload['converted_summary_exists'] = bool(converted_summary.get('exists'))
        if not converted_summary.get('exists'):
            print(f"[diagnose][WARN] converted summary not found: {converted_summary.get('path')}", flush=True)

    for split in splits:
        print(f'[diagnose] split={split}', flush=True)
        if args.scan_vad:
            split_summary = summarize_split(
                split,
                vad_root,
                protocol_root,
                converter,
                max_files=args.max_files_per_split,
                progress_every=args.progress_every,
            )
        else:
            split_summary = summarize_from_converted_summary(split, converted_summary, protocol_root)
        payload['splits'][split] = split_summary
        if args.scan_vad:
            print(
                '[diagnose] split={} vad={} protocol_overlap={}/{} present_attacks={} missing_family={}'.format(
                    split,
                    split_summary.get('num_vad_files_scanned', split_summary.get('num_vad_files', 0)),
                    split_summary.get('protocol_vad_overlap', {}).get('num_intersection', 0),
                    split_summary.get('protocol_vad_overlap', {}).get('num_protocol_ids', split_summary.get('protocol', {}).get('num_ids', 0)),
                    ','.join(split_summary.get('present_attack_ids', [])),
                    ','.join(split_summary.get('missing_family_classes', [])),
                )
            )
        else:
            print(
                '[diagnose] split={} fast_mode source=summary present_attacks={} missing_family={}'.format(
                    split,
                    ','.join(split_summary.get('present_attack_ids', [])),
                    ','.join(split_summary.get('missing_family_classes', [])),
                )
            )

    output_path = resolve_cross_platform_path(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    print(f'[diagnose] wrote {output_path}')


if __name__ == '__main__':
    main()
