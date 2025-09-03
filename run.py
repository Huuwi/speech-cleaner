"""
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

INPUT_AUDIO_PATH  = "./data.wav"
OUTPUT_AUDIO_PATH = "./data_output.wav"

# ====== DEFAULT PARAMETERS ======
TARGET_SAMPLE_RATE              = 16000
HIGHPASS_CUTOFF_HZ              = 60        # high-pass cutoff (Hz)
LOWPASS_CUTOFF_HZ               = 3800      # low-pass cutoff  (Hz) ~ telephone band
VAD_AGGRESSIVENESS_MODE         = 2         # 0..3 (WebRTC); higher = stricter

MIN_UTTERANCE_DURATION_SEC      = 0.20      # min duration of a kept utterance (s)
MIN_GAP_BETWEEN_UTTERANCES_SEC  = 0.12      # min gap to split utterances (s)
PAD_AROUND_SPEECH_SEC           = 0.05      # pad around detected speech (s)

DENOISE_PROP_DECREASE_STAGE1    = 0.90      # denoise strength (stage 1)
DENOISE_PROP_DECREASE_STAGE2    = 0.80      # denoise strength (stage 2)

CROSSFADE_DURATION_MS           = 10        # crossfade length if no delay is inserted (ms)
FADE_DURATION_MS                = 6         # small fade-in/out for delayed concatenation (ms)

DELAY_BETWEEN_UTTERANCES_SEC    = 0.05      # insert fixed silence between utterances (seconds)

# ---------- Utils ----------
def dbfs(samples: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(samples)) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)

def load_mono_resample(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:  # load audio data => mix down to mono => resample
    audio, sample_rate = librosa.load(path, sr=target_sr, mono=True)
    if np.max(np.abs(audio)) > 1e-6:
        audio = audio / np.max(np.abs(audio)) * 0.99
    return audio.astype(np.float32), target_sr

def save_wav_pcm16(path: str, audio: np.ndarray, sample_rate: int):
    audio = np.asarray(audio, dtype=np.float32)
    peak = np.max(np.abs(audio)) + 1e-9
    audio = (audio / peak) * 0.98
    sf.write(path, audio, sample_rate, subtype="PCM_16")

def butter_filter(audio: np.ndarray, sample_rate: int,  # Butterworth filter by frequency
                  lowcut: Optional[float] = None,
                  highcut: Optional[float] = None,
                  order: int = 4) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    sos_sections: List[np.ndarray] = []
    if lowcut and lowcut > 0:
        sos_sections.append(signal.butter(order, lowcut / nyquist, btype='highpass', output='sos'))
    if highcut and highcut < nyquist:
        sos_sections.append(signal.butter(order, highcut / nyquist, btype='lowpass', output='sos'))
    filtered_audio = audio.copy()
    for sos in sos_sections:
        filtered_audio = signal.sosfiltfilt(sos, filtered_audio).astype(np.float32)
    return filtered_audio


# ---------- VAD ----------
def vad_mask_webrtc(audio: np.ndarray, sample_rate: int, frame_ms: int = 30, #check noise , return array noise flag
                    mode: int = 2, pad_ms: int = 150) -> np.ndarray:
    vad_detector = webrtcvad.Vad(mode)
    frame_len = int(sample_rate * frame_ms / 1000)
    hop_length = frame_len
    num_frames = math.ceil(len(audio) / hop_length)

    def float_to_pcm16(frame_f32: np.ndarray) -> bytes:
        clipped = np.clip(frame_f32, -1.0, 1.0)
        int16 = (clipped * 32767.0).astype(np.int16)
        return int16.tobytes()

    speech_flags = np.zeros(num_frames, dtype=bool)
    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        end = min(start + frame_len, len(audio))
        frame = audio[start:end]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        speech_flags[frame_idx] = vad_detector.is_speech(float_to_pcm16(frame), sample_rate)

    pad_frames = max(1, int(pad_ms / frame_ms))
    padded_flags = speech_flags.copy()
    for idx, is_speech in enumerate(speech_flags):
        if is_speech:
            padded_flags[max(0, idx - pad_frames):min(num_frames, idx + pad_frames + 1)] = True

    mask = np.zeros(len(audio), dtype=bool)
    for idx, flag in enumerate(padded_flags):
        start = idx * hop_length
        end = min(start + hop_length, len(audio))
        if flag:
            mask[start:end] = True
    return mask

# ---------- Segmentation & Noise Reduction ----------
def extract_noise_clip(audio: np.ndarray, mask: np.ndarray,
                       sample_rate: int, max_noise_sec: float = 3.0) -> Optional[np.ndarray]:
    nonspeech_indices = np.nonzero(~mask)[0]
    if nonspeech_indices.size < int(0.2 * sample_rate):
        return None
    take_indices = nonspeech_indices[:min(len(nonspeech_indices), int(max_noise_sec * sample_rate))]
    if take_indices.size == 0:
        return None
    return audio[take_indices]

def denoise_gating(audio: np.ndarray, sample_rate: int,
                   noise_clip: Optional[np.ndarray],
                   prop_decrease: float = 0.9) -> np.ndarray:
    return nr.reduce_noise(y=audio, sr=sample_rate, y_noise=noise_clip,
                           stationary=False, prop_decrease=prop_decrease)

def mask_to_segments(mask: np.ndarray, sample_rate: int,
                     min_speech_sec: float = 0.2,
                     min_gap_sec: float = 0.12,
                     pad_sec: float = 0.05) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    in_segment = False
    seg_start = 0
    for idx, is_speech in enumerate(mask):
        if is_speech and not in_segment:
            in_segment = True
            seg_start = idx
        if not is_speech and in_segment:
            in_segment = False
            segments.append((seg_start, idx))
    if in_segment:
        segments.append((seg_start, len(mask)))

    # merge short gaps
    merged_segments: List[Tuple[int, int]] = []
    if not segments:
        return merged_segments
    current_start, current_end = segments[0]
    min_gap_samples = int(min_gap_sec * sample_rate)
    for start, end in segments[1:]:
        if start - current_end <= min_gap_samples:
            current_end = end
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end
    merged_segments.append((current_start, current_end))

    # pad and filter by minimum length
    pad_samples = int(pad_sec * sample_rate)
    output_segments: List[Tuple[int, int]] = []
    min_len_samples = int(min_speech_sec * sample_rate)
    for start, end in merged_segments:
        start_padded = max(0, start - pad_samples)
        end_padded = min(len(mask), end + pad_samples)
        if end_padded - start_padded >= min_len_samples:
            output_segments.append((start_padded, end_padded))
    return output_segments

def filter_segments_by_rms(audio: np.ndarray, segments: List[Tuple[int, int]],
                           min_rms_dbfs: float = -40.0) -> List[Tuple[int, int]]:
    kept_segments: List[Tuple[int, int]] = []
    for start, end in segments:
        segment_audio = audio[start:end]
        if dbfs(segment_audio) >= min_rms_dbfs:
            kept_segments.append((start, end))
    return kept_segments


# ---------- Silence tools ----------
def remove_silence(audio: np.ndarray, sample_rate: int,
                   win_ms: int = 30, thr_db: float = -45.0) -> np.ndarray:
    """Remove very quiet frames in a chunk (simple RMS gate)."""
    hop_length = int(sample_rate * win_ms / 1000)
    frame_len = hop_length
    keep_mask = np.zeros(len(audio), dtype=bool)
    for start in range(0, len(audio), hop_length):
        frame = audio[start:start + frame_len]
        if frame.size == 0:
            continue
        if dbfs(frame) > thr_db:
            keep_mask[start:start + frame_len] = True
    if keep_mask.sum() == 0:
        return audio
    return audio[keep_mask]

def apply_fade(audio: np.ndarray, sample_rate: int, fade_ms: int = 6) -> np.ndarray:
    """Apply a short fade-in/out to a chunk to avoid clicks."""
    if audio.size == 0 or fade_ms <= 0:
        return audio
    n = len(audio)
    fade_len = int(sample_rate * fade_ms / 1000)
    fade_len = min(fade_len, n // 2)
    if fade_len <= 0:
        return audio
    faded_audio = audio.copy()
    fade_in_curve = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out_curve = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    faded_audio[:fade_len] *= fade_in_curve
    faded_audio[-fade_len:] *= fade_out_curve
    return faded_audio


# ---------- Concatenation ----------
def join_with_delay_or_crossfade(chunks: List[np.ndarray], sample_rate: int,
                                 delay_sec: float = 0.0,
                                 xf_ms: int = 10,
                                 fade_ms: int = 6) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)

    if delay_sec > 0:
        silence_gap = np.zeros(int(sample_rate * delay_sec), dtype=np.float32)
        output_audio = apply_fade(chunks[0], sample_rate, fade_ms)
        for seg in chunks[1:]:
            output_audio = np.concatenate([output_audio, silence_gap, apply_fade(seg, sample_rate, fade_ms)], axis=0)
        return output_audio

    # else: crossfade
    xf_samples = int(sample_rate * xf_ms / 1000)
    output_audio = chunks[0].copy()
    for seg in chunks[1:]:
        if xf_samples > 0 and len(output_audio) > xf_samples and len(seg) > xf_samples:
            fade_out_curve = np.linspace(1.0, 0.0, xf_samples, dtype=np.float32)
            fade_in_curve = np.linspace(0.0, 1.0, xf_samples, dtype=np.float32)
            tail = output_audio[-xf_samples:] * fade_out_curve + seg[:xf_samples] * fade_in_curve
            output_audio = np.concatenate([output_audio[:-xf_samples], tail, seg[xf_samples:]], axis=0)
        else:
            output_audio = np.concatenate([output_audio, seg], axis=0)
    return output_audio


# ---------- Processing ----------
def process_file(in_path: str, out_path: str):
    # 1) Load & bandpass
    audio, sample_rate = load_mono_resample(in_path, TARGET_SAMPLE_RATE)
    audio = butter_filter(audio, sample_rate, lowcut=HIGHPASS_CUTOFF_HZ, highcut=LOWPASS_CUTOFF_HZ, order=6)

    # 2) mask audio and extract noise clip
    pre_vad_mask = vad_mask_webrtc(audio, sample_rate, frame_ms=30, mode=VAD_AGGRESSIVENESS_MODE, pad_ms=150)
    noise_clip = extract_noise_clip(audio, pre_vad_mask, sample_rate, max_noise_sec=3.0)

    # 3 + 4) Denoise
    audio_denoised = denoise_gating(audio, sample_rate, noise_clip, prop_decrease=DENOISE_PROP_DECREASE_STAGE1)
    audio_denoised = denoise_gating(audio_denoised, sample_rate, noise_clip, prop_decrease=DENOISE_PROP_DECREASE_STAGE2) if DENOISE_PROP_DECREASE_STAGE2 > 0 else audio_denoised

    # 5) Main VAD on denoised signal
    speech_mask = vad_mask_webrtc(audio_denoised, sample_rate, frame_ms=30, mode=VAD_AGGRESSIVENESS_MODE, pad_ms=150)

    # 6+ 7) filter by RMS (remove too-quiet/unclear speech)
    segments = mask_to_segments(speech_mask, sample_rate, min_speech_sec=MIN_UTTERANCE_DURATION_SEC,
                                min_gap_sec=MIN_GAP_BETWEEN_UTTERANCES_SEC, pad_sec=PAD_AROUND_SPEECH_SEC)
    segments = filter_segments_by_rms(audio_denoised, segments, min_rms_dbfs=-40.0)

    # 8) Cut chunks
    chunks = [audio_denoised[start:end] for (start, end) in segments]

    # 9) Remove inner-silence inside each chunk
    chunks = [remove_silence(c, sample_rate, win_ms=30, thr_db=-60.0) for c in chunks]

    # 10) Join with either delay or crossfade
    output_audio = join_with_delay_or_crossfade(
        chunks, sample_rate,
        delay_sec=DELAY_BETWEEN_UTTERANCES_SEC,
        xf_ms=CROSSFADE_DURATION_MS,
        fade_ms=FADE_DURATION_MS
    )

    if output_audio.size == 0:
        # avoid empty output file
        output_audio = np.zeros(int(0.5 * sample_rate), dtype=np.float32)

    # 11) Save
    save_wav_pcm16(out_path, output_audio, sample_rate)
    print(f" Saved processed file: {out_path}")


if __name__ == "__main__":
    process_file(INPUT_AUDIO_PATH, OUTPUT_AUDIO_PATH)
    print(" Done:", OUTPUT_AUDIO_PATH)
    