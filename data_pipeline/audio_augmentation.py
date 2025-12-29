"""
audiomentations-based augmentation pipeline tailored for song aesthetic evaluation tasks.

Features:
- Safe parameter ranges (won't heavily change melody/structure)
- File-based augmentation: augment_file(...)
- HuggingFace datasets integration: augment_batch_hf(...)
- Multi-view crops helper
"""

import os
import math
import random
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
from audiomentations import Compose, AddGaussianNoise, AddGaussianSNR, TimeStretch, PitchShift, Shift, \
    HighPassFilter, LowPassFilter, Gain, SevenBandParametricEQ
import traceback


# ---------------------
# Config / safe params
# ---------------------
SAFE_SR = 24000  # or 48000 depending on your dataset
SNR_MIN = 40.0  # minimal SNR for added noise (so noise is subtle)
PITCH_CENTS = 10  # +/- cents for micro pitch shift (±10 cents)
TIME_STRETCH_RANGE = (0.99, 1.01)  # small time stretch
GAIN_DB = (-2.0, 2.0)
LOUDNESS_DB = (-2.0, 2.0)  # alternative loudness augmentation

# ---------------------
# Make augmenter
# ---------------------
def make_augmenter():
    """Return an audiomentations Compose object with conservative transforms."""
    transforms = [
        # Small random gain (loudness)
        Gain(min_gain_db=GAIN_DB[0], max_gain_db=GAIN_DB[1], p=0.9),

        # Add mild Gaussian noise (very high SNR)
        AddGaussianSNR(min_snr_db=30, max_snr_db=50, p=0.25),

        # Small pitch shift: audiomentations' PitchShift uses semitones.
        # We'll convert cents to semitones: cents/100.
        PitchShift(min_semitones=-PITCH_CENTS/100.0, max_semitones=PITCH_CENTS/100.0, p=0.5),

        # Small time stretch
        TimeStretch(min_rate=TIME_STRETCH_RANGE[0], max_rate=TIME_STRETCH_RANGE[1], p=0.4),

        # Small shift in time (circular or pad)
        Shift(min_shift=-0.5, max_shift=0.5, shift_unit='seconds', p=0.4),

        # Mild equalization via bandpass / high/low pass (small chance)
        HighPassFilter(min_cutoff_freq=80.0, max_cutoff_freq=120.0, p=0.3),
        LowPassFilter(min_cutoff_freq=15000.0, max_cutoff_freq=18000.0, p=0.3),

        SevenBandParametricEQ(min_gain_db=-3.0, max_gain_db=+3.0, p=0.3),
    ]

    augmenter = Compose(transforms, p=1.0)
    return augmenter

# ---------------------
# util: load / save
# ---------------------
def load_audio(path: str, sr: int = SAFE_SR) -> Tuple[np.ndarray, int]:
    """Load audio to mono float32 at target sample rate."""
    wav, sr = librosa.load(path, sr=sr)
    return wav.astype(np.float32), sr

def save_audio(path: str, samples: np.ndarray, sr: int = SAFE_SR):
    """Save float32 waveform to file (wav)."""
    sf.write(path, samples.astype(np.float32), samplerate=sr, subtype="PCM_24")

# ---------------------
# file-level augmentation
# ---------------------
def augment_file(input_path: str,
                 output_dir: str,
                 n_augmentations: int = 5,
                 sample_rate: int = SAFE_SR,
                 augmenter=None,
                 segment_sec: Optional[float] = None,
                 seed: Optional[int] = None):
    """
    Augment a single audio file and save augmented copies.
    - Keeps utt ID (filename) and appends suffix _augN
    - segment_sec: if provided, crop random segment of this length from the audio before augmenting.
    """
    try:
        base = os.path.splitext(os.path.basename(input_path))[0]
        wav, sr = load_audio(input_path, sr=sample_rate)
        # save_audio(os.path.join(output_dir, base+'.wav'), wav, sr)
        if augmenter is None:
            augmenter = make_augmenter()

        rng = np.random.RandomState(seed)

        total_len = len(wav)
        seg_samples = None
        if segment_sec is not None:
            seg_samples = int(segment_sec * sr)

        for i in range(n_augmentations):
            if seg_samples is not None and seg_samples < total_len:
                start = rng.randint(0, total_len - seg_samples)
                chunk = wav[start:start + seg_samples].copy()
            else:
                chunk = wav.copy()

            # audiomentations expects shape (samples,) for mono
            augmented = augmenter(samples=chunk, sample_rate=sr)
            out_name = f"{base}_aug{i+1}.wav"
            out_path = os.path.join(output_dir, out_name)
            save_audio(out_path, augmented, sr)
            # print(f"Saved {out_path}")
        return True, out_path
    except Exception as e:
        print(traceback.format_exc())
        return False, None


if __name__ == "__main__":
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', type=str, required=True)
    parser.add_argument('--output_txt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='origin_song_eval_dataset/mp3_aug')
    parser.add_argument('--n_augmentations', type=int, default=3)
    parser.add_argument('--sample_rate', type=int, default=SAFE_SR)
    parser.add_argument('--seed', type=int, default=random.randint(6, 666888))
    args = parser.parse_args()
    with open(args.input_txt, 'r') as f:
        mp3_list = [x.strip() for x in f.readlines()]
    os.makedirs(args.output_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=32) as excutor, \
        open(args.output_txt, 'w') as fw:
        futures = []
        for input_mp3_path in tqdm(mp3_list):
            futures.append(excutor.submit(augment_file, input_mp3_path, args.output_dir, n_augmentations=args.n_augmentations, sample_rate=args.sample_rate, seed=args.seed))
        for future in tqdm(as_completed(futures), total=len(futures), desc='processing audio augmentation'):
            try:
                success, out_path = future.result()
                if success:
                    fw.write(f"{out_path}\n")
            except Exception as e:
                print(traceback.format_exc())
                continue
