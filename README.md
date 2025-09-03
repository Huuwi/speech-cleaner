# Hướng dẫn sử dụng

## 1. Cài đặt thư viện

Mở terminal tại thư mục dự án và chạy lệnh sau để cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## 2. Chuẩn bị dữ liệu

- Đặt file âm thanh đầu vào tên `data.wav` vào cùng thư mục với mã nguồn.

## 3. Chạy chương trình

Chạy script xử lý bằng lệnh:

```bash
python run.py
```

- Kết quả sẽ được lưu vào file `data_output.wav` trong cùng thư mục.

## 4. Mô tả các tham số cấu hình (`run.py`)

Bạn có thể chỉnh sửa các tham số xử lý trong file `run.py` để phù hợp với nhu cầu:

| Tham số                        | Ý nghĩa                                                                 | Giá trị mặc định |
|------------------------------- |------------------------------------------------------------------------|------------------|
| `TARGET_SAMPLE_RATE`           | Tần số lấy mẫu đầu ra (Hz)                                             | 16000            |
| `HIGHPASS_CUTOFF_HZ`           | Tần số cắt của bộ lọc thông cao (Hz)                                   | 60               |
| `LOWPASS_CUTOFF_HZ`            | Tần số cắt của bộ lọc thông thấp (Hz)                                  | 3800             |
| `VAD_AGGRESSIVENESS_MODE`      | Độ nhạy phát hiện tiếng nói (0-3, càng cao càng nghiêm ngặt)           | 2                |
| `MIN_UTTERANCE_DURATION_SEC`   | Thời lượng tối thiểu của một đoạn tiếng nói được giữ lại (giây)        | 0.20             |
| `MIN_GAP_BETWEEN_UTTERANCES_SEC`| Khoảng cách tối thiểu để tách các đoạn tiếng nói (giây)                | 0.12             |
| `PAD_AROUND_SPEECH_SEC`        | Thời gian đệm quanh đoạn tiếng nói (giây)                              | 0.05             |
| `DENOISE_PROP_DECREASE_STAGE1` | Mức độ khử nhiễu giai đoạn 1 (0-1, càng cao càng mạnh)                 | 0.90             |
| `DENOISE_PROP_DECREASE_STAGE2` | Mức độ khử nhiễu giai đoạn 2 (0-1, càng cao càng mạnh)                 | 0.80             |
| `CROSSFADE_DURATION_MS`        | Độ dài crossfade khi nối các đoạn (ms)                                 | 10               |
| `FADE_DURATION_MS`             | Độ dài fade-in/out khi nối đoạn có chèn khoảng lặng (ms)               | 6                |
| `DELAY_BETWEEN_UTTERANCES_SEC` | Thời gian chèn khoảng lặng giữa các đoạn tiếng nói (giây)              | 0.05             |
