import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SR = 16000
CUT = 66800
STRIDE = 320
NUM_FRAMES = CUT // STRIDE
FRAME_DUR = STRIDE / SR

FAMILY3_LABELS = {
    'bonafide': 0,
    'tts': 1,
    'vc': 2,
}

ATTACK19_NUM_CLASSES = 20
FAMILY3_NUM_CLASSES = 3


def resolve_cross_platform_path(path: str) -> str:
    """Resolve Windows-style paths when scripts are executed inside WSL2.

    Examples:
        C:\\Partial Spoof -> /mnt/c/Partial Spoof on Linux/WSL
        /mnt/c/Partial Spoof -> /mnt/c/Partial Spoof unchanged
    """
    if path is None:
        return path
    path = str(path).strip()
    if not path:
        return path
    match = re.match(r'^([A-Za-z]):[\\/](.*)$', path)
    if os.name != 'nt' and match:
        drive = match.group(1).lower()
        rest = match.group(2).replace('\\', '/')
        return os.path.normpath(f'/mnt/{drive}/{rest}')
    return os.path.normpath(path)


# Default A01-A19 family grouping.
# Source note:
#   PartialSpoof's detailed .vad files directly expose A01-A19 IDs, but the
#   local annotation package does not include an authoritative Axx -> TTS/VC
#   table.  This default is a diagnostic placeholder based on a common but
#   not guaranteed contiguous grouping assumption, and it must be verified by
#   checking per-split class coverage after conversion.
#   The table is intentionally centralized and can be overridden with a JSON
#   file via `family_map_path` in the conversion script / loaders.
#   If train/dev/eval show a missing TTS or VC class, do not use those labels
#   for family3 training until the mapping is corrected.
DEFAULT_A19_TO_FAMILY3 = {
    'A01': 'vc',
    'A02': 'vc',
    'A03': 'vc',
    'A04': 'vc',
    'A05': 'vc',
    'A06': 'vc',
    'A07': 'tts',
    'A08': 'tts',
    'A09': 'tts',
    'A10': 'tts',
    'A11': 'tts',
    'A12': 'tts',
    'A13': 'tts',
    'A14': 'tts',
    'A15': 'tts',
    'A16': 'tts',
    'A17': 'tts',
    'A18': 'tts',
    'A19': 'tts',
}


@dataclass(frozen=True)
class VADSegment:
    start: float
    end: float
    label_id: int


class PartialSpoofLabelConverter:
    def __init__(self, id_to_label: Dict[int, str], family_map: Optional[Dict[str, str]] = None):
        self.id_to_label = dict(id_to_label)
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.family_map = self._normalize_family_map(family_map or DEFAULT_A19_TO_FAMILY3)

    @staticmethod
    def _normalize_family_map(family_map: Dict[str, str]) -> Dict[str, str]:
        normalized = {}
        for attack_tag, family in family_map.items():
            attack_tag = str(attack_tag).strip().upper()
            family = str(family).strip().lower()
            if not attack_tag.startswith('A'):
                raise ValueError(f'Invalid attack tag in family map: {attack_tag}')
            if family not in FAMILY3_LABELS or family == 'bonafide':
                raise ValueError(f'Invalid family for {attack_tag}: {family}. Expected one of tts/vc.')
            normalized[attack_tag] = family
        expected = {f'A{i:02d}' for i in range(1, 20)}
        missing = sorted(expected - set(normalized.keys()))
        if missing:
            raise ValueError(f'Family map missing attack IDs: {missing}')
        return normalized

    @classmethod
    def from_label2num(cls, label2num_path: str, family_map_path: Optional[str] = None):
        id_to_label = load_label2num(label2num_path)[1]
        family_map = None
        if family_map_path:
            with open(family_map_path, 'r', encoding='utf-8') as fh:
                family_map = json.load(fh)
        return cls(id_to_label=id_to_label, family_map=family_map)

    def label_name(self, label_id: int) -> str:
        if label_id not in self.id_to_label:
            raise KeyError(f'Unknown label id: {label_id}')
        return self.id_to_label[label_id]

    def attack_tag_from_label(self, label_name: str) -> Optional[str]:
        label_name = str(label_name).strip()
        if label_name.startswith('A') and len(label_name) == 3:
            return label_name
        if label_name.startswith('nonA') and len(label_name) == 6:
            return 'A' + label_name[-2:]
        return None

    def label_to_internal_class(self, label_id: int, scheme: str) -> Optional[int]:
        label_name = self.label_name(label_id)
        attack_tag = self.attack_tag_from_label(label_name)
        scheme = str(scheme).strip().lower()

        if scheme == 'family3':
            if label_name in ('bonafide', 'nonbona'):
                return FAMILY3_LABELS['bonafide']
            if attack_tag is not None:
                family_name = self.family_map[attack_tag]
                return FAMILY3_LABELS[family_name]
            if label_name in ('nonspeech', 'nonmix'):
                return None
            raise ValueError(f'Unsupported label for family3 conversion: {label_name} (id={label_id})')

        if scheme == 'attack19':
            if label_name in ('bonafide', 'nonbona'):
                return 0
            if attack_tag is not None:
                return int(attack_tag[1:])
            if label_name in ('nonspeech', 'nonmix'):
                return None
            raise ValueError(f'Unsupported label for attack19 conversion: {label_name} (id={label_id})')

        raise ValueError(f'Unknown scheme: {scheme}')

    def convert_segments_to_frames(
        self,
        segments: Sequence[VADSegment],
        scheme: str,
        num_frames: int = NUM_FRAMES,
        frame_dur: float = FRAME_DUR,
    ) -> np.ndarray:
        raw = np.full(num_frames, -1, dtype=np.int64)
        best_overlap = np.full(num_frames, -1.0, dtype=np.float32)
        timeline_end = num_frames * frame_dur

        for seg in segments:
            cls = self.label_to_internal_class(seg.label_id, scheme=scheme)
            if cls is None:
                continue

            start_idx = max(0, int(np.floor(seg.start / frame_dur)))
            end_idx = min(num_frames - 1, int(np.ceil(seg.end / frame_dur)) - 1)
            if end_idx < start_idx:
                continue

            for frame_idx in range(start_idx, end_idx + 1):
                frame_start = frame_idx * frame_dur
                frame_end = frame_start + frame_dur
                overlap = min(frame_end, seg.end) - max(frame_start, seg.start)
                if overlap <= 0:
                    continue
                if overlap > best_overlap[frame_idx]:
                    best_overlap[frame_idx] = overlap
                    raw[frame_idx] = cls

        known_indices = np.where(raw >= 0)[0]
        if known_indices.size == 0:
            return np.zeros(num_frames, dtype=np.int64)

        first_known = int(known_indices[0])
        raw[:first_known] = raw[first_known]
        prev_idx = first_known
        for idx in known_indices[1:]:
            idx = int(idx)
            if idx - prev_idx > 1:
                fill_val = raw[prev_idx]
                raw[prev_idx + 1:idx] = fill_val
            prev_idx = idx
        raw[prev_idx + 1:] = raw[prev_idx]

        if segments:
            last_end = max(seg.end for seg in segments)
            if last_end < timeline_end:
                last_valid_frame = min(num_frames - 1, int(np.floor(max(last_end - 1e-9, 0.0) / frame_dur)))
                raw[last_valid_frame + 1:] = raw[last_valid_frame]

        raw[raw < 0] = 0
        return raw.astype(np.int64)


def load_label2num(label2num_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    label_to_id: Dict[str, int] = {}
    id_to_label: Dict[int, str] = {}
    with open(label2num_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f'Invalid label2num line: {line}')
            label, idx = parts[0], int(parts[1])
            label_to_id[label] = idx
            id_to_label[idx] = label
    return label_to_id, id_to_label


def load_vad_file(vad_path: str) -> List[VADSegment]:
    segments: List[VADSegment] = []
    with open(vad_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f'Invalid VAD line in {vad_path}: {line}')
            start, end, label_id = float(parts[0]), float(parts[1]), int(float(parts[2]))
            if end < start:
                raise ValueError(f'Negative-duration segment in {vad_path}: {line}')
            segments.append(VADSegment(start=start, end=end, label_id=label_id))
    segments.sort(key=lambda x: (x.start, x.end, x.label_id))
    return segments


def frame_labels_to_boundaries(frame_labels: Sequence[int], num_frames: int = NUM_FRAMES) -> np.ndarray:
    arr = np.asarray(frame_labels, dtype=np.int64).reshape(-1)
    if arr.size < num_frames:
        pad_val = int(arr[-1]) if arr.size > 0 else 0
        arr = np.concatenate([arr, np.full(num_frames - arr.size, pad_val, dtype=np.int64)])
    elif arr.size > num_frames:
        arr = arr[:num_frames]
    boundary = np.zeros(num_frames, dtype=np.float32)
    if num_frames > 1:
        boundary[1:] = (arr[1:] != arr[:-1]).astype(np.float32)
    return boundary


def serialize_family_map(path: str, family_map: Optional[Dict[str, str]] = None):
    payload = family_map or DEFAULT_A19_TO_FAMILY3
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def collect_vad_files(split_dir: str) -> List[str]:
    file_names = [x for x in os.listdir(split_dir) if x.lower().endswith('.vad')]
    file_names.sort()
    return [os.path.join(split_dir, x) for x in file_names]


def summarize_frame_dict(frame_dict: Dict[str, np.ndarray], num_classes: int) -> Dict[str, object]:
    counts = np.zeros(num_classes, dtype=np.int64)
    utt_positive = 0
    for arr in frame_dict.values():
        arr = np.asarray(arr, dtype=np.int64).reshape(-1)
        binc = np.bincount(arr, minlength=num_classes)
        counts[:len(binc)] += binc[:num_classes]
        utt_positive += int(np.any(arr > 0))
    total_frames = int(counts.sum())
    class_ratio = [0.0 if total_frames == 0 else float(c / total_frames) for c in counts.tolist()]
    missing_classes = [int(i) for i, c in enumerate(counts.tolist()) if c == 0]
    return {
        'num_utts': int(len(frame_dict)),
        'total_frames': total_frames,
        'class_counts': counts.tolist(),
        'class_ratios': class_ratio,
        'missing_classes': missing_classes,
        'spoof_utt_count': int(utt_positive),
        'bonafide_utt_count': int(len(frame_dict) - utt_positive),
    }


def summarize_attack_ids(frame_dict: Dict[str, np.ndarray]) -> Dict[str, object]:
    counts = np.zeros(ATTACK19_NUM_CLASSES, dtype=np.int64)
    for arr in frame_dict.values():
        arr = np.asarray(arr, dtype=np.int64).reshape(-1)
        binc = np.bincount(arr, minlength=ATTACK19_NUM_CLASSES)
        counts[:len(binc)] += binc[:ATTACK19_NUM_CLASSES]
    present = [f'A{i:02d}' for i in range(1, ATTACK19_NUM_CLASSES) if counts[i] > 0]
    missing = [f'A{i:02d}' for i in range(1, ATTACK19_NUM_CLASSES) if counts[i] == 0]
    return {
        'present_attack_ids': present,
        'missing_attack_ids': missing,
        'attack_frame_counts': {f'A{i:02d}': int(counts[i]) for i in range(1, ATTACK19_NUM_CLASSES)},
    }
