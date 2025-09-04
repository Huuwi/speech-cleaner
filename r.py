"""
Optimized preprocessing pipeline for lower WER.
Dependencies:
  pip install numpy librosa soundfile scipy webrtcvad noisereduce setuptools
"""
import math
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import scipy.signal as signal
import webrtcvad
import noisereduce as nr
from scipy.ndimage import binary_closing

INPUT_AUDIO_PATH  = "./6541392_input.wav"
OUTPUT_AUDIO_PATH = "./6541392.wav"


# ---------- Config ----------
TARGET_SAMPLE_RATE = 16000
HIGHPASS_CUTOFF_HZ = 80
LOWPASS_CUTOFF_HZ = 7600      # allow more speech harmonics (telephone ~ 4k, but ASR can benefit up to 7-8k)
VAD_MODE_INITIAL = 1          # conservative initial scan (detect speech regions roughly)
VAD_MODE_FINAL = 2            # stricter or more sensitive final pass
FRAME_MS = 30

# Denoise params
N_FFT = 2048
HOP_LENGTH = 512
DENOISE_PROP_STAGE1 = 0.95
# DENOISE_PROP_STAGE2 = 0.85

MIN_UTTERANCE_SEC = 0.25
MIN_GAP_SEC = 0.10
PAD_AROUND_SPEECH_SEC = 0.12
MIN_RMS_DBFS = -42.0

# Silence removal (inner)
SILENCE_DB_THRESHOLD = -50.0
SILENCE_WIN_MS = 30

# output normalization
TARGET_RMS_DBFS = -20.0

# ---------- util ----------
def dbfs(samples: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(samples)) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)

def rms_normalize(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    current_db = dbfs(audio)
    gain_db = target_db - current_db
    factor = 10.0 ** (gain_db / 20.0)
    return audio * factor

def pre_emphasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    return np.append(x[0], x[1:] - coef * x[:-1])

def de_emphasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = coef * y[i - 1] + x[i]
    return y

def load_mono_resample(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.99
    return audio.astype(np.float32), target_sr

def save_wav_pcm16(path: str, audio: np.ndarray, sample_rate: int):
    audio = np.asarray(audio, dtype=np.float32)
    # avoid clipping
    peak = np.max(np.abs(audio)) + 1e-9
    audio = (audio / peak) * 0.98
    sf.write(path, audio, sample_rate, subtype="PCM_16")

# ---------- filters ----------
def butter_bandpass(audio: np.ndarray, sr: int, lowcut: Optional[float], highcut: Optional[float], order=6):
    nyq = 0.5 * sr
    sos = []
    if lowcut and lowcut > 0:
        sos.append(signal.butter(order, lowcut / nyq, btype='highpass', output='sos'))
    if highcut and highcut < nyq:
        sos.append(signal.butter(order, highcut / nyq, btype='lowpass', output='sos'))
    out = audio.copy()
    for s in sos:
        out = signal.sosfiltfilt(s, out).astype(np.float32)
    return out

# ---------- VAD helpers ----------
def webrtcvad_mask(audio: np.ndarray, sr: int, frame_ms: int = FRAME_MS, mode: int = 2, pad_ms: int = 150) -> np.ndarray:
    vad = webrtcvad.Vad(mode)
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len
    nframes = math.ceil(len(audio) / hop)
    def to_pcm16(fr):
        fr = np.clip(fr, -1.0, 1.0)
        return (fr * 32767).astype(np.int16).tobytes()
    flags = np.zeros(nframes, dtype=bool)
    for i in range(nframes):
        s = i * hop
        e = min(s + frame_len, len(audio))
        frame = audio[s:e]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        try:
            flags[i] = vad.is_speech(to_pcm16(frame), sr)
        except Exception:
            flags[i] = False
    # pad
    pad_frames = max(1, int(pad_ms / frame_ms))
    padded = flags.copy()
    for i, v in enumerate(flags):
        if v:
            padded[max(0, i - pad_frames): i + pad_frames + 1] = True
    # expand to sample mask
    mask = np.zeros(len(audio), dtype=bool)
    for i, v in enumerate(padded):
        s = i * hop
        e = min(s + hop, len(audio))
        if v:
            mask[s:e] = True
    # morphological close to avoid choppy mask
    frame_mask = padded.astype(np.uint8)
    closed = binary_closing(frame_mask, structure=np.ones(3)).astype(bool)
    mask2 = np.zeros(len(audio), dtype=bool)
    for i, v in enumerate(closed):
        s = i * hop
        e = min(s + hop, len(audio))
        if v:
            mask2[s:e] = True
    return mask2

# ---------- noise extraction ----------
def collect_noise_profile(audio: np.ndarray, mask: np.ndarray, sr: int, max_total_sec: float = 4.0) -> Optional[np.ndarray]:
    # take multiple small non-speech windows spread across file
    nonspeech_idx = np.nonzero(~mask)[0]
    if nonspeech_idx.size < int(0.2 * sr):
        return None
    # sample up to max_total_sec seconds evenly
    step = max(1, int(len(audio) / max(1, int(max_total_sec * sr))))
    picks = nonspeech_idx[::step][:int(max_total_sec * sr)]
    if picks.size == 0:
        return None
    return audio[picks]

# ---------- segmentation ----------
def mask_to_segments(mask: np.ndarray, sr: int, min_speech_sec: float = MIN_UTTERANCE_SEC,
                     min_gap_sec: float = MIN_GAP_SEC, pad_sec: float = PAD_AROUND_SPEECH_SEC) -> List[Tuple[int,int]]:
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            in_seg = True
            start = i
        if not v and in_seg:
            in_seg = False
            segments.append((start, i))
    if in_seg:
        segments.append((start, len(mask)))
    # merge small gaps
    merged = []
    if not segments:
        return merged
    cur_s, cur_e = segments[0]
    min_gap_samples = int(min_gap_sec * sr)
    for s, e in segments[1:]:
        if s - cur_e <= min_gap_samples:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    # pad and filter by length
    out = []
    pad = int(pad_sec * sr)
    min_len = int(min_speech_sec * sr)
    for s, e in merged:
        s2 = max(0, s - pad)
        e2 = min(len(mask), e + pad)
        if e2 - s2 >= min_len:
            out.append((s2, e2))
    return out

# ---------- inner silence removal ----------
def remove_inner_silences(segment: np.ndarray, sr: int, win_ms: int = SILENCE_WIN_MS, thr_db: float = SILENCE_DB_THRESHOLD) -> np.ndarray:
    # simple frame-based gate to remove very quiet frames inside a speech segment
    hop = int(sr * win_ms / 1000)
    if hop <= 0 or len(segment) < hop:
        return segment
    keep = np.zeros(len(segment), dtype=bool)
    for s in range(0, len(segment), hop):
        frame = segment[s:s+hop]
        if dbfs(frame) > thr_db:
            keep[s:s+hop] = True
    if keep.sum() == 0:
        return segment
    return segment[keep]

# ---------- concat ----------
def join_chunks(chunks: List[np.ndarray], sr: int, gap_sec: float = 0.05, fade_ms: int = 6):
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(sr * gap_sec), dtype=np.float32)
    def fade(x):
        if len(x) == 0 or fade_ms <= 0:
            return x
        L = int(sr * fade_ms / 1000)
        L = min(L, len(x)//2)
        if L <= 0:
            return x
        win_in = np.linspace(0.0, 1.0, L)
        win_out = np.linspace(1.0, 0.0, L)
        y = x.copy()
        y[:L] *= win_in
        y[-L:] *= win_out
        return y
    out = fade(chunks[0])
    for c in chunks[1:]:
        out = np.concatenate([out, gap, fade(c)], axis=0)
    return out

# ---------- main process ----------
def process_file_optimized(in_path: str, out_path: str):
    audio, sr = load_mono_resample(in_path, TARGET_SAMPLE_RATE)
    # highpass + lowpass
    audio = butter_bandpass(audio, sr, HIGHPASS_CUTOFF_HZ, LOWPASS_CUTOFF_HZ, order=6)

    # initial VAD to find long non-speech regions for noise profiling
    initial_mask = webrtcvad_mask(audio, sr, frame_ms=30, mode=VAD_MODE_INITIAL, pad_ms=150)
    noise_clip = collect_noise_profile(audio, initial_mask, sr, max_total_sec=4.0)

    # pre-emphasis to boost high-frequency speech harmonics (can help ASR)
    audio_emph = pre_emphasis(audio, coef=0.97)

    # spectral denoise stage 1 (conservative)
    try:
        den1 = nr.reduce_noise(y=audio_emph, sr=sr, y_noise=noise_clip, stationary=False,
                                prop_decrease=DENOISE_PROP_STAGE1, n_fft=N_FFT, hop_length=HOP_LENGTH)
    except Exception:
        den1 = audio_emph

    # final VAD on denoised signal (more accurate)
    final_mask = webrtcvad_mask(den1, sr, frame_ms=30, mode=VAD_MODE_FINAL, pad_ms=180)

    # segments from mask with padding/merge
    segments = mask_to_segments(final_mask, sr, min_speech_sec=MIN_UTTERANCE_SEC,
                                min_gap_sec=MIN_GAP_SEC, pad_sec=PAD_AROUND_SPEECH_SEC)

    # second denoise pass focused on voiced segments (stronger)
    den2 = den1
    if noise_clip is not None:
        try:
            den2 = nr.reduce_noise(y=den1, sr=sr, y_noise=noise_clip, stationary=False,
                                    prop_decrease=DENOISE_PROP_STAGE2, n_fft=N_FFT, hop_length=HOP_LENGTH)
        except Exception:
            den2 = den1

    # cut, remove very quiet segments and inner silences
    chunks = []
    for s, e in segments:
        seg = den2[s:e]
        # remove internal tiny silences (frame gate)
        seg = remove_inner_silences(seg, sr, win_ms=SILENCE_WIN_MS, thr_db=SILENCE_DB_THRESHOLD)
        # fallback: if too short after inner-silence removal, try librosa split on original denoised segment
        if len(seg) < int(MIN_UTTERANCE_SEC * sr):
            parts = librosa.effects.split(den2[s:e], top_db=30)  # top_db can be tuned
            for a, b in parts:
                part = den2[s + a: s + b]
                if dbfs(part) >= MIN_RMS_DBFS:
                    chunks.append(part)
        else:
            if dbfs(seg) >= MIN_RMS_DBFS:
                chunks.append(seg)

    # If no chunks found, fallback to whole denoised audio trimmed by energy
    if not chunks:
        candid = librosa.effects.split(den2, top_db=30)
        for a, b in candid:
            part = den2[a:b]
            if dbfs(part) >= MIN_RMS_DBFS:
                chunks.append(part)

    # de-emphasize each chunk and normalize
    chunks = [de_emphasis(c, coef=0.97) for c in chunks]
    chunks = [rms_normalize(c, TARGET_RMS_DBFS) for c in chunks]

    # join chunks with small gap
    out = join_chunks(chunks, sr, gap_sec=0.06, fade_ms=6)

    # final normalization & ensure not empty
    if out.size == 0:
        out = np.zeros(int(0.5 * sr), dtype=np.float32)
    out = rms_normalize(out, TARGET_RMS_DBFS)

    save_wav_pcm16(out_path, out, sr)
    print("Saved:", out_path)

process_file_optimized(INPUT_AUDIO_PATH, OUTPUT_AUDIO_PATH)
