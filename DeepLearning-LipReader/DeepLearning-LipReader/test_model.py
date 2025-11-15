import numpy as np
import tensorflow as tf
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, PROJECT_ROOT)

# Load model
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "lip_reader_3dcnn.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Load training data
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
words = sorted(os.listdir(PROCESSED_DATA_DIR))
word_to_index = {word: i for i, word in enumerate(words)}

X_test = []
y_test = []

print("Loading test data from training set...")
for word in words:
    word_path = os.path.join(PROCESSED_DATA_DIR, word)
    for take_file in sorted(os.listdir(word_path)):
        if take_file.endswith(".npy"):
            filepath = os.path.join(word_path, take_file)
            frames = np.load(filepath)
            if frames.shape == (22, 80, 112):
                frames = np.expand_dims(frames, axis=-1)
                X_test.append(frames)
                y_test.append(word_to_index[word])

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"Loaded {len(X_test)} samples")
print(f"Testing model on training data...\n")

# Make predictions
predictions = model.predict(X_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
confidences = np.max(predictions, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == y_test)
print(f"Accuracy on training data: {accuracy*100:.2f}%")

# Show per-class results
print("\nPer-class results:")
for i, word in enumerate(words):
    mask = y_test == i
    if np.sum(mask) > 0:
        class_acc = np.mean(predicted_classes[mask] == y_test[mask])
        avg_conf = np.mean(confidences[mask])
        print(f"  {word:10s}: Accuracy={class_acc*100:6.2f}%, Avg Confidence={avg_conf*100:6.2f}%")

# Show some example predictions
print("\nExample predictions (first 10):")
for i in range(min(10, len(X_test))):
    pred_word = words[predicted_classes[i]]
    actual_word = words[y_test[i]]
    conf = confidences[i] * 100
    match = "[OK]" if pred_word == actual_word else "[X]"
    print(f"  {match} Actual: {actual_word:10s}, Predicted: {pred_word:10s}, Confidence: {conf:5.2f}%")

