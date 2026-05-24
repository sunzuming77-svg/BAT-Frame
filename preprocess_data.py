import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from RawBoost import process_Rawboost_feature
from utils import pad
from data_utils import _load_audio, SR, CUT


TRAIN_SRC = r"C:\PS_data\train\con_wav"
TRAIN_DST = r"C:\PS_data\train\con_wav_pt_algo5"
DEV_SRC = r"C:\PS_data\dev\con_wav"
DEV_DST = r"C:\PS_data\dev\con_wav_pt"
EVAL_SRC = r"C:\PS_data\eval\con_wav"
EVAL_DST = r"C:\PS_data\eval\con_wav_pt"
DEFAULT_ALGO = 5


def build_rawboost_args():
    return SimpleNamespace(
        N_f=5,
        nBands=5,
        minF=20,
        maxF=8000,
        minBW=100,
        maxBW=1000,
        minCoeff=10,
        maxCoeff=100,
        minG=0,
        maxG=0,
        minBiasLinNonLin=5,
        maxBiasLinNonLin=20,
        P=10,
        g_sd=2,
        SNRmin=10,
        SNRmax=40,
    )


RAWBOOST_ARGS = build_rawboost_args()


def _seed_from_path(path: str) -> int:
    digest = hashlib.md5(path.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _save_tensor(tensor: torch.Tensor, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(tensor.contiguous(), save_path)


def _process_train_file(wav_path: str, out_dir: str, algo: int):
    np.random.seed(_seed_from_path(wav_path))
    x = _load_audio(wav_path, sr=SR)
    y = process_Rawboost_feature(x, SR, RAWBOOST_ARGS, algo=algo)
    y_pad = pad(np.asarray(y, dtype=np.float32), CUT)
    tensor = torch.tensor(y_pad, dtype=torch.float32)
    utt_id = os.path.splitext(os.path.basename(wav_path))[0]
    save_path = os.path.join(out_dir, utt_id + ".pt")
    _save_tensor(tensor, save_path)
    return save_path


def _process_dev_file(wav_path: str, out_dir: str):
    x = _load_audio(wav_path, sr=SR)
    x_pad = pad(np.asarray(x, dtype=np.float32), CUT)
    tensor = torch.tensor(x_pad, dtype=torch.float32)
    utt_id = os.path.splitext(os.path.basename(wav_path))[0]
    save_path = os.path.join(out_dir, utt_id + ".pt")
    _save_tensor(tensor, save_path)
    return save_path


def _list_wavs(src_dir: str):
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    wavs = [
        os.path.join(src_dir, name)
        for name in os.listdir(src_dir)
        if name.lower().endswith(".wav")
    ]
    wavs.sort()
    return wavs


def run_parallel(file_paths, worker_fn, desc: str, max_workers: int, *worker_args):
    if not file_paths:
        print(f"{desc}: no wav files found.")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_fn, path, *worker_args) for path in file_paths]
        with tqdm(total=len(futures), desc=desc, unit="file") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline preprocess PartialSpoof waveform tensors.")
    parser.add_argument("--train_src", type=str, default=TRAIN_SRC)
    parser.add_argument("--train_dst", type=str, default=TRAIN_DST)
    parser.add_argument("--dev_src", type=str, default=DEV_SRC)
    parser.add_argument("--dev_dst", type=str, default=DEV_DST)
    parser.add_argument("--algo", type=int, default=DEFAULT_ALGO)
    parser.add_argument("--eval_src", type=str, default=EVAL_SRC)
    parser.add_argument("--eval_dst", type=str, default=EVAL_DST)
    parser.add_argument("--skip_eval", default=False,
                        type=lambda x: (str(x).lower() in ["true", "yes", "1"]))
    parser.add_argument("--max_workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.train_dst, exist_ok=True)
    os.makedirs(args.dev_dst, exist_ok=True)
    if not args.skip_eval:
        os.makedirs(args.eval_dst, exist_ok=True)

    train_files = _list_wavs(args.train_src)
    dev_files = _list_wavs(args.dev_src)
    eval_files = _list_wavs(args.eval_src) if not args.skip_eval else []

    print(f"Train wav files: {len(train_files)}")
    print(f"Dev wav files: {len(dev_files)}")
    if not args.skip_eval:
        print(f"Eval wav files: {len(eval_files)}")
    print(f"Max workers: {args.max_workers}")

    run_parallel(train_files, _process_train_file, "Preprocess train", args.max_workers, args.train_dst, args.algo)
    run_parallel(dev_files, _process_dev_file, "Preprocess dev", args.max_workers, args.dev_dst)
    if not args.skip_eval:
        run_parallel(eval_files, _process_dev_file, "Preprocess eval", args.max_workers, args.eval_dst)

    print("Done.")
    print(f"Train tensors saved to: {args.train_dst}")
    print(f"Dev tensors saved to: {args.dev_dst}")
    if not args.skip_eval:
        print(f"Eval tensors saved to: {args.eval_dst}")


if __name__ == "__main__":
    main()
