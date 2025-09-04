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

INPUT_AUDIO_PATH  = "./6541392_input.wav"
OUTPUT_AUDIO_PATH = "./6541392.wav"

# ====== DEFAULT PARAMETERS (WER-friendly) ======
TARGET_SAMPLE_RATE               = 16000
HIGHPASS_CUTOFF_HZ               = 60         # HPF để bỏ ù/rumble
LOWPASS_CUTOFF_HZ                = 7900       # giữ sibilants cho 16kHz
BASE_VAD_AGGRESSIVENESS_MODE     = 1          # sẽ tự điều chỉnh theo SNR

MIN_UTTERANCE_DURATION_SEC       = 0.20
MIN_GAP_BETWEEN_UTTERANCES_SEC   = 0.12
PAD_AROUND_SPEECH_SEC            = 0.20       # rộng hơn để bảo toàn phụ âm biên

DENOISE_PROP_DECREASE_STAGE1     = 0.65       # vừa phải để tránh méo formant
DENOISE_PROP_DECREASE_STAGE2     = 0.0        # tắt stage 2 (tránh over-denoise)

# Ghép phát ngôn: giữ time tự nhiên -> không chèn delay/crossfade
CROSSFADE_DURATION_MS            = 0
FADE_DURATION_MS                 = 6          # chống click khi nối
DELAY_BETWEEN_UTTERANCES_SEC     = 0.0

# Chuẩn hoá loudness (RMS dBFS mục tiêu và giới hạn peak)
TARGET_RMS_DBFS                  = -23.0
PEAK_LIMIT_DBFS                  = -1.0

# ---------- Utils ----------
def dbfs(samples: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(samples)) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)

def load_mono_resample(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    audio, sample_rate = librosa.load(path, sr=target_sr, mono=True)
    # remove DC offset
    audio = audio - np.mean(audio)
    # peak protect (giữ biên an toàn khi vào pipeline)
    peak = np.max(np.abs(audio)) + 1e-9
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32), target_sr

def save_wav_pcm16(path: str, audio: np.ndarray, sample_rate: int):
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(path, audio, sample_rate, subtype="PCM_16")

def butter_filter(audio: np.ndarray, sample_rate: int,
                  lowcut: Optional[float] = None,
                  highcut: Optional[float] = None,
                  order: int = 6) -> np.ndarray:
    nyq = 0.5 * sample_rate
    sos_sections: List[np.ndarray] = []
    if lowcut and lowcut > 0:
        sos_sections.append(signal.butter(order, lowcut / nyq, btype='highpass', output='sos'))
    if highcut and highcut < nyq:
        sos_sections.append(signal.butter(order, highcut / nyq, btype='lowpass', output='sos'))
    y = audio.copy()
    for sos in sos_sections:
        y = signal.sosfiltfilt(sos, y).astype(np.float32)  # zero-phase
    return y

def normalize_rms_dbfs(audio: np.ndarray,
                       target_rms_dbfs: float = -23.0,
                       peak_limit_dbfs: float = -1.0) -> np.ndarray:
    # RMS normalize
    cur_rms = np.sqrt(np.mean(np.square(audio)) + 1e-12)
    cur_dbfs = 20*np.log10(cur_rms + 1e-12)
    gain_db = target_rms_dbfs - cur_dbfs
    gain = 10**(gain_db/20.0)
    y = audio * gain
    # Peak headroom
    peak_limit_lin = 10**(peak_limit_dbfs/20.0)  # -1 dBFS ~ 0.891
    cur_peak = np.max(np.abs(y)) + 1e-12
    if cur_peak > peak_limit_lin:
        y = y * (peak_limit_lin / cur_peak)
    return y.astype(np.float32)

# ---------- VAD ----------
def vad_mask_webrtc(audio: np.ndarray, sample_rate: int,
                    frame_ms: int = 20, mode: int = 1,
                    pad_ms: int = 200,
                    min_speech_ms: int = 120,
                    min_gap_ms: int = 120) -> np.ndarray:
    """
    WebRTC VAD + smoothing (closing/opening) để tránh "bể" phụ âm.
    """
    vad_detector = webrtcvad.Vad(mode)
    frame_len = int(sample_rate * frame_ms / 1000)
    hop_length = frame_len  # non-overlap, đúng yêu cầu WebRTC (10/20/30 ms)
    num_frames = math.ceil(len(audio) / hop_length)

    def float_to_pcm16(frame_f32: np.ndarray) -> bytes:
        clipped = np.clip(frame_f32, -1.0, 1.0)
        int16 = (clipped * 32767.0).astype(np.int16)
        return int16.tobytes()

    flags = np.zeros(num_frames, dtype=bool)
    for i in range(num_frames):
        s = i * hop_length
        e = min(s + frame_len, len(audio))
        frame = audio[s:e]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        flags[i] = vad_detector.is_speech(float_to_pcm16(frame), sample_rate)

    # Dilate/Pad quanh speech để giữ phụ âm biên
    pad_frames = max(1, int(pad_ms / frame_ms))
    padded = flags.copy()
    for i, f in enumerate(flags):
        if f:
            padded[max(0, i - pad_frames):min(num_frames, i + pad_frames + 1)] = True

    # Smoothing: loại đảo nhỏ & lấp khe nhỏ
    def smooth_binary(arr: np.ndarray, min_true: int, min_false: int) -> np.ndarray:
        # merge short gaps (closing)
        out = arr.copy()
        run_start = 0
        while run_start < len(out):
            run_val = out[run_start]
            run_end = run_start + 1
            while run_end < len(out) and out[run_end] == run_val:
                run_end += 1
            run_len = run_end - run_start
            if not run_val and run_len < min_false:
                out[run_start:run_end] = True
            run_start = run_end
        # remove short islands (opening)
        run_start = 0
        while run_start < len(out):
            run_val = out[run_start]
            run_end = run_start + 1
            while run_end < len(out) and out[run_end] == run_val:
                run_end += 1
            run_len = run_end - run_start
            if run_val and run_len < min_true:
                out[run_start:run_end] = False
            run_start = run_end
        return out

    min_true_frames  = max(1, int(min_speech_ms / frame_ms))
    min_false_frames = max(1, int(min_gap_ms / frame_ms))
    smoothed = smooth_binary(padded, min_true_frames, min_false_frames)

    # về mask theo sample
    mask = np.zeros(len(audio), dtype=bool)
    for i, f in enumerate(smoothed):
        s = i * hop_length
        e = min(s + hop_length, len(audio))
        if f:
            mask[s:e] = True
    return mask

# ---------- Noise profiling & NR ----------
def robust_noise_clip(audio: np.ndarray,
                      mask: np.ndarray,
                      sample_rate: int,
                      max_noise_sec: float = 3.0) -> Optional[np.ndarray]:
    """
    Lấy noise từ các frame năng lượng thấp nhất (20th percentile)
    + ưu tiên vùng non-speech theo VAD; nối đến tối đa max_noise_sec.
    """
    win = int(0.03 * sample_rate)
    hop = win
    energies = []
    starts = []
    for s in range(0, len(audio), hop):
        e = audio[s:s+win]
        if e.size == 0:
            continue
        energies.append(np.mean(e*e))
        starts.append(s)
    if not energies:
        return None
    energies = np.asarray(energies)
    starts = np.asarray(starts)
    thr = np.percentile(energies, 20.0)

    # ưu tiên frame: non-speech & dưới ngưỡng
    non_speech_by_vad = []
    for s in starts[energies <= thr]:
        frame_mask = mask[s:s+win]
        if frame_mask.size == 0 or not frame_mask.any():  # non-speech
            non_speech_by_vad.append(audio[s:s+win])
    pool = non_speech_by_vad if non_speech_by_vad else [audio[s:s+win] for s in starts[energies <= thr]]

    if not pool:
        return None
    clip = np.concatenate(pool, axis=0)
    clip = clip[:int(max_noise_sec * sample_rate)]
    if clip.size < int(0.2 * sample_rate):
        return None
    return clip.astype(np.float32)

def denoise_gating(audio: np.ndarray, sample_rate: int,
                   noise_clip: Optional[np.ndarray],
                   prop_decrease: float = 0.65) -> np.ndarray:
    if noise_clip is None:
        # fallback nhẹ nhàng nếu không đủ noise
        return audio
    y = nr.reduce_noise(y=audio, sr=sample_rate, y_noise=noise_clip,
                        stationary=False, prop_decrease=prop_decrease)
    return y.astype(np.float32)

# ---------- Segmentation helpers ----------
def mask_to_segments(mask: np.ndarray, sample_rate: int,
                     min_speech_sec: float = 0.2,
                     min_gap_sec: float = 0.12,
                     pad_sec: float = 0.20) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    in_seg = False
    st = 0
    for i, f in enumerate(mask):
        if f and not in_seg:
            in_seg = True
            st = i
        if (not f) and in_seg:
            in_seg = False
            segs.append((st, i))
    if in_seg:
        segs.append((st, len(mask)))

    # merge short gaps
    merged: List[Tuple[int, int]] = []
    if not segs:
        return merged
    cur_s, cur_e = segs[0]
    min_gap = int(min_gap_sec * sample_rate)
    for s, e in segs[1:]:
        if s - cur_e <= min_gap:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # pad & filter by min len
    pad = int(pad_sec * sample_rate)
    out: List[Tuple[int, int]] = []
    min_len = int(min_speech_sec * sample_rate)
    n = len(mask)
    for s, e in merged:
        ps = max(0, s - pad)
        pe = min(n, e + pad)
        if pe - ps >= min_len:
            out.append((ps, pe))
    return out

def filter_segments_by_rms(audio: np.ndarray, segs: List[Tuple[int, int]],
                           min_rms_dbfs: float = -44.0) -> List[Tuple[int, int]]:
    kept = []
    for s, e in segs:
        if dbfs(audio[s:e]) >= min_rms_dbfs:
            kept.append((s, e))
    return kept

# ---------- Silence handling (WER-preserving) ----------
def attenuate_quiet_frames(audio: np.ndarray, sample_rate: int,
                           win_ms: int = 30, thr_db: float = -60.0,
                           atten_db: float = -12.0) -> np.ndarray:
    """
    Thay vì xoá frame yên lặng bên trong chunk (làm gãy nhịp),
    ta *giảm âm* (attenuate) để giữ thời lượng tự nhiên.
    """
    hop = int(sample_rate * win_ms / 1000)
    win = hop
    y = audio.copy()
    atten = 10**(atten_db/20.0)
    for s in range(0, len(audio), hop):
        f = audio[s:s+win]
        if f.size == 0:
            continue
        if dbfs(f) < thr_db:
            y[s:s+win] = f * atten
    return y.astype(np.float32)

def apply_fade(audio: np.ndarray, sample_rate: int, fade_ms: int = 6) -> np.ndarray:
    if audio.size == 0 or fade_ms <= 0:
        return audio
    n = len(audio)
    fl = int(sample_rate * fade_ms / 1000)
    fl = min(fl, n // 2)
    if fl <= 0:
        return audio
    y = audio.copy()
    fade_in = np.linspace(0.0, 1.0, fl, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fl, dtype=np.float32)
    y[:fl] *= fade_in
    y[-fl:] *= fade_out
    return y

# ---------- Concatenation ----------
def join_chunks(chunks: List[np.ndarray], sample_rate: int,
                delay_sec: float = 0.0,
                xf_ms: int = 0,
                fade_ms: int = 6) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if delay_sec > 0:
        gap = np.zeros(int(sample_rate * delay_sec), dtype=np.float32)
        out = apply_fade(chunks[0], sample_rate, fade_ms)
        for c in chunks[1:]:
            out = np.concatenate([out, gap, apply_fade(c, sample_rate, fade_ms)], axis=0)
        return out
    # no delay, no crossfade -> chỉ nối; áp fade nhẹ để chống click
    out = apply_fade(chunks[0], sample_rate, fade_ms)
    for c in chunks[1:]:
        out = np.concatenate([out, apply_fade(c, sample_rate, fade_ms)], axis=0)
    return out

# ---------- SNR & adaptive VAD ----------
def estimate_snr_db(audio: np.ndarray, speech_mask: np.ndarray) -> float:
    if speech_mask.sum() < int(0.2 * TARGET_SAMPLE_RATE) or (~speech_mask).sum() < int(0.2 * TARGET_SAMPLE_RATE):
        return 10.0  # fallback
    sig = audio[speech_mask]
    noi = audio[~speech_mask]
    p_sig = np.mean(sig*sig) + 1e-12
    p_noi = np.mean(noi*noi) + 1e-12
    return 10.0 * np.log10(p_sig / p_noi)

# ---------- Processing ----------
def process_file(in_path: str, out_path: str):
    # 1) Load & bandpass
    audio, sr = load_mono_resample(in_path, TARGET_SAMPLE_RATE)
    audio = butter_filter(audio, sr, lowcut=HIGHPASS_CUTOFF_HZ, highcut=LOWPASS_CUTOFF_HZ, order=6)

    # 2) Thăm dò VAD ban đầu (mode cơ bản) để ước lượng noise/SNR
    pre_mask = vad_mask_webrtc(audio, sr, frame_ms=20, mode=BASE_VAD_AGGRESSIVENESS_MODE,
                               pad_ms=int(PAD_AROUND_SPEECH_SEC*1000),
                               min_speech_ms=int(MIN_UTTERANCE_DURATION_SEC*1000),
                               min_gap_ms=int(MIN_GAP_BETWEEN_UTTERANCES_SEC*1000))

    # 3) Noise profile (robust, đa điểm)
    noise_clip = robust_noise_clip(audio, pre_mask, sr, max_noise_sec=3.0)

    # 4) Denoise (nhẹ nhàng, tránh méo)
    audio_dn = denoise_gating(audio, sr, noise_clip, prop_decrease=DENOISE_PROP_DECREASE_STAGE1)

    # 5) VAD chính xác hơn + thích nghi theo SNR
    #    Nếu SNR thấp -> tăng aggressiveness một chút; cao -> giữ thấp để tránh miss speech
    tmp_mask = vad_mask_webrtc(audio_dn, sr, frame_ms=20, mode=BASE_VAD_AGGRESSIVENESS_MODE,
                               pad_ms=int(PAD_AROUND_SPEECH_SEC*1000),
                               min_speech_ms=int(MIN_UTTERANCE_DURATION_SEC*1000),
                               min_gap_ms=int(MIN_GAP_BETWEEN_UTTERANCES_SEC*1000))
    snr = estimate_snr_db(audio_dn, tmp_mask)
    mode = BASE_VAD_AGGRESSIVENESS_MODE
    if snr < 5:   mode = min(3, mode + 2)
    elif snr < 10: mode = min(3, mode + 1)

    speech_mask = vad_mask_webrtc(audio_dn, sr, frame_ms=20, mode=mode,
                                  pad_ms=int(PAD_AROUND_SPEECH_SEC*1000),
                                  min_speech_ms=int(MIN_UTTERANCE_DURATION_SEC*1000),
                                  min_gap_ms=int(MIN_GAP_BETWEEN_UTTERANCES_SEC*1000))

    # 6) Segments + lọc RMS
    segments = mask_to_segments(speech_mask, sr,
                                min_speech_sec=MIN_UTTERANCE_DURATION_SEC,
                                min_gap_sec=MIN_GAP_BETWEEN_UTTERANCES_SEC,
                                pad_sec=PAD_AROUND_SPEECH_SEC)
    segments = filter_segments_by_rms(audio_dn, segments, min_rms_dbfs=-44.0)

    # 7) Cắt chunks & xử lý yên lặng bên trong (attenuate, không xoá)
    chunks = []
    for (s, e) in segments:
        c = audio_dn[s:e]
        c = attenuate_quiet_frames(c, sr, win_ms=30, thr_db=-60.0, atten_db=-12.0)
        chunks.append(c)

    # 8) Nối (không delay, không crossfade)
    out = join_chunks(chunks, sr,
                      delay_sec=DELAY_BETWEEN_UTTERANCES_SEC,
                      xf_ms=CROSSFADE_DURATION_MS,
                      fade_ms=FADE_DURATION_MS)

    # 9) Nếu rỗng, tạo 0.5s im lặng
    if out.size == 0:
        out = np.zeros(int(0.5 * sr), dtype=np.float32)

    # 10) Chuẩn hoá loudness (RMS & peak headroom)
    out = normalize_rms_dbfs(out, TARGET_RMS_DBFS, PEAK_LIMIT_DBFS)

    # 11) Save
    save_wav_pcm16(out_path, out, sr)
    print(f"Saved processed file: {out_path} | VAD mode={mode} | est SNR={snr:.1f} dB")

if __name__ == "__main__":
    process_file(INPUT_AUDIO_PATH, OUTPUT_AUDIO_PATH)
    print("Done:", OUTPUT_AUDIO_PATH)
