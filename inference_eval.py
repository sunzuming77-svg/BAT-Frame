import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import eval_metrics as em
from data_utils import Dataset_PartialSpoof_eval, Dataset_PartialSpoof_multiclass_eval, load_seglab, load_frame_label_dict, parse_ps_protocol
from model import Model


EXPECTED_NUM_CLASSES = {
    'binary': 2,
    'family3': 3,
    'attack19': 20,
}


def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)


def validate_label_mode_and_num_classes(args):
    expected = EXPECTED_NUM_CLASSES[args.ps_label_mode]
    if int(args.num_classes) != expected:
        raise ValueError(
            f"num_classes={args.num_classes} does not match ps_label_mode={args.ps_label_mode}. "
            f"Expected num_classes={expected}."
        )


def load_model(args, device):
    model = Model(args, device).to(device)
    if args.average_model:
        best_dir = Path(args.model_dir) / 'best'
        checkpoints = []
        for i in range(args.n_average_model):
            p = best_dir / f'best_{i}.pth'
            if p.exists() and p.stat().st_size > 1000:
                checkpoints.append(p)
        if len(checkpoints) == 0:
            raise FileNotFoundError(f'No best_i checkpoints found in {best_dir}')

        model.load_state_dict(torch.load(str(checkpoints[0]), map_location=device))
        sd = model.state_dict()
        for p in checkpoints[1:]:
            model.load_state_dict(torch.load(str(p), map_location=device))
            sd2 = model.state_dict()
            for k in sd:
                sd[k] = sd[k] + sd2[k]
        for k in sd:
            sd[k] = sd[k] / len(checkpoints)
        model.load_state_dict(sd)
        print(f'Loaded averaged model from {len(checkpoints)} checkpoints.')
    else:
        ckpt = Path(args.model_dir) / args.checkpoint_name
        if not ckpt.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
        model.load_state_dict(torch.load(str(ckpt), map_location=device))
        print(f'Loaded single model: {ckpt}')

    model.eval()
    return model


def build_eval_dataset(args):
    split = args.eval_split
    proto = Path(args.protocols_path) / 'protocols' / 'PartialSpoof_LA_cm_protocols' / f'PartialSpoof.LA.cm.{split}.trl.txt'
    ids, utt_labels = parse_ps_protocol(str(proto), is_eval=(split == 'eval'))
    base_dir = Path(args.database_path) / split / 'con_wav'

    if args.ps_label_mode == 'binary':
        seglab_path = Path(args.database_path) / 'segment_labels' / f'{split}_seglab_0.02.npy'
        seglab = load_seglab(str(seglab_path)) if seglab_path.exists() else {}
        ds = Dataset_PartialSpoof_eval(
            list_IDs=ids,
            seglab=seglab,
            utt_labels=utt_labels,
            base_dir=str(base_dir),
            skip_missing=args.skip_missing,
        )
    else:
        if not args.ps_label_root:
            raise ValueError('ps_label_root is required for family3/attack19 inference.')
        label_file_name = 'family3_frames_0.02.npy' if args.ps_label_mode == 'family3' else 'attack19_frames_0.02.npy'
        label_path = Path(args.ps_label_root) / split / label_file_name
        if not label_path.exists():
            raise FileNotFoundError(f'Multiclass label file not found: {label_path}')
        frame_label_dict = load_frame_label_dict(str(label_path))
        ds = Dataset_PartialSpoof_multiclass_eval(
            list_IDs=ids,
            frame_label_dict=frame_label_dict,
            utt_labels=utt_labels,
            base_dir=str(base_dir),
            skip_missing=args.skip_missing,
        )

    print(f'Eval split={split} trials={len(ids)} label_mode={args.ps_label_mode} num_classes={args.num_classes}')
    return ds, ids, utt_labels


def run_scoring(dataset, model, device, out_path, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_skip_none)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skipped_batches = 0
    with out_path.open('w', encoding='utf-8') as fh, torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            if batch is None:
                skipped_batches += 1
                continue
            if len(batch) == 5:
                x, _, _, _, utt_ids = batch
            elif len(batch) == 4:
                x, _, _, utt_ids = batch
            else:
                x, utt_ids = batch
            x = x.to(device)
            _, _, _, utt_logit = model(x)
            scores = utt_logit.squeeze(-1).detach().cpu().numpy().ravel().tolist()
            for utt_id, score in zip(utt_ids, scores):
                fh.write(f'{utt_id} {score}\n')

    if skipped_batches > 0:
        print(f'Skipped {skipped_batches} empty batches due to missing items.')
    print(f'Scores saved to: {out_path}')


def compute_eer_from_score(score_path, protocol_path):
    scores = {}
    with open(score_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                scores[parts[0]] = float(parts[1])

    bona, spoof = [], []
    missing = 0
    total = 0
    with open(protocol_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            total += 1
            utt_id, label = parts[1], parts[4]
            if utt_id not in scores:
                missing += 1
                continue
            if label == 'bonafide':
                bona.append(scores[utt_id])
            else:
                spoof.append(scores[utt_id])

    bona = np.asarray(bona, dtype=np.float64)
    spoof = np.asarray(spoof, dtype=np.float64)
    if bona.size == 0 or spoof.size == 0:
        raise RuntimeError('Cannot compute EER: missing bonafide or spoof scores.')

    eer, thr = em.compute_eer(spoof, bona)
    print(f'total_trials={total} matched={len(bona) + len(spoof)} missing={missing}')
    print(f'bonafide={len(bona)} spoof={len(spoof)}')
    print(f'EER={eer * 100:.4f}% threshold={thr:.6f}')


def parse_args():
    parser = argparse.ArgumentParser(description='Standalone inference/eval scoring for PartialSpoof.')
    parser.add_argument('--database_path', type=str, default='/mnt/c/PS_data')
    parser.add_argument('--protocols_path', type=str, default='/mnt/c/PS_data')
    parser.add_argument('--ps_label_mode', type=str, default='family3', choices=['binary', 'family3', 'attack19'])
    parser.add_argument('--ps_label_root', type=str, default='', help='root dir containing converted multiclass labels')
    parser.add_argument('--model_dir', type=str, required=True, help='e.g., models/BATmamba...')
    parser.add_argument('--checkpoint_name', type=str, default='best.pth')
    parser.add_argument('--average_model', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    parser.add_argument('--n_average_model', type=int, default=5)
    parser.add_argument('--eval_split', type=str, default='dev', choices=['dev', 'eval'])
    parser.add_argument('--skip_missing', default=False, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    parser.add_argument('--score_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--compute_eer', default=False, type=lambda x: str(x).lower() in ['true', 'yes', '1'])

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--emb_size', type=int, default=144)
    parser.add_argument('--num_encoders', type=int, default=12)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--algo', type=int, default=5)
    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--first_conv', type=int, default=128)
    parser.add_argument('--filts', type=list, default=[70, [1, 32], [32, 32], [32, 64], [64, 64]])
    parser.add_argument('--d_args', type=list, default=[0.5, 0.5, 0.5])
    parser.add_argument('--gru_node', type=int, default=1024)
    parser.add_argument('--nb_gru_layer', type=int, default=1)
    parser.add_argument('--nb_fc_node', type=int, default=1024)
    parser.add_argument('--num_segments', type=int, default=4)
    parser.add_argument('--use_boundary_control', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    parser.add_argument('--use_cross_routing', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    parser.add_argument('--use_soft_segments', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    parser.add_argument('--FT_W2V', default=True, type=lambda x: str(x).lower() in ['true', 'yes', '1'])
    parser.add_argument('--track', type=str, default='LA')
    parser.add_argument('--loss', type=str, default='WCE')
    parser.add_argument('--lr', type=float, default=1e-6)

    return parser.parse_args()


def main():
    args = parse_args()
    validate_label_mode_and_num_classes(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(args, device)
    dataset, _, _ = build_eval_dataset(args)

    if args.score_path:
        score_path = args.score_path
    else:
        model_tag = Path(args.model_dir).name
        score_path = f'./Scores/PartialSpoof/{model_tag}_{args.eval_split}_inference.txt'

    run_scoring(dataset, model, device, score_path, args.batch_size)

    if args.compute_eer:
        proto = Path(args.protocols_path) / 'protocols' / 'PartialSpoof_LA_cm_protocols' / f'PartialSpoof.LA.cm.{args.eval_split}.trl.txt'
        compute_eer_from_score(score_path, str(proto))


if __name__ == '__main__':
    main()
