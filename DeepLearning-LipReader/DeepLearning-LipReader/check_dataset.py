import os

data_dir = "processed_data"
words = sorted(os.listdir(data_dir))
total = 0

print("Dataset Summary:")
print("=" * 50)
for w in words:
    word_path = os.path.join(data_dir, w)
    count = len([f for f in os.listdir(word_path) if f.endswith(".npy")])
    total += count
    print(f"{w:15s}: {count:3d} samples")

print("=" * 50)
print(f"{'Total':15s}: {total:3d} samples")
print(f"{'Words':15s}: {len(words)} classes")
print(f"{'Avg per class':15s}: {total // len(words) if len(words) > 0 else 0} samples")

