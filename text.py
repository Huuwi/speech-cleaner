import json

# Đọc file JSON
with open("0_3_6532555.json", "r", encoding="utf-8") as f:
    data = json.load(f)

transcriptions = data[0]["transcriptions"]

# Lọc text theo speaker
speaker00_texts = [t["text"] for t in transcriptions if t["speaker"] == "SPEAKER_00"]

# Ghép thành 1 đoạn dài
final_text = " ".join(speaker00_texts)

# Ghi ra file
with open("speaker00.txt", "w", encoding="utf-8") as f:
    f.write(final_text)

print("✅ Đã ghi text của SPEAKER_00 vào speaker00.txt")
