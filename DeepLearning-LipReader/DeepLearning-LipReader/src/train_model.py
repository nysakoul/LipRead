
import numpy as np
import tensorflow as tf
import os
import time
import sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Progress bar

# Get the project root directory (parent of src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ==== MODEL HYPERPARAMETERS ====
BATCH_SIZE = 8  # Increased batch size for better training
EPOCHS = 200  # Even more epochs for better convergence
LEARNING_RATE = 0.0005  # Balanced learning rate
INPUT_SHAPE = (22, 80, 112, 1)  # (Frames, Height, Width, Channels)

# ==== LOAD DATA ====
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
if not os.path.exists(PROCESSED_DATA_DIR):
    print(f"[ERROR] {PROCESSED_DATA_DIR} does not exist. Please run preprocess.py first.")
    sys.exit(1)
words = sorted(os.listdir(PROCESSED_DATA_DIR))
if len(words) == 0:
    print(f"[ERROR] No words found in {PROCESSED_DATA_DIR}. Please collect and preprocess data first.")
    sys.exit(1)
word_to_index = {word: i for i, word in enumerate(words)}

X, y = [], []

print("\nLoading data...")

for word in words:
    word_path = os.path.join(PROCESSED_DATA_DIR, word)
    
    for take_file in sorted(os.listdir(word_path)):
        if take_file.endswith(".npy"):
            filepath = os.path.join(word_path, take_file)
            frames = np.load(filepath)

            if frames.shape == (22, 80, 112):  # Ensure correct shape
                frames = np.expand_dims(frames, axis=-1)  # Add channel dimension
                X.append(frames)
                y.append(word_to_index[word])

X = np.array(X)
y = np.array(y)

print(f"[OK] Loaded {len(X)} samples across {len(words)} words.")
print(f"[INFO] Dataset size: {len(X)} samples, {len(words)} classes")
print(f"[INFO] Samples per class: {len(X) // len(words)}")

# For very small datasets, use a different strategy
if len(X) < 20:
    # For very small datasets, use leave-one-out or minimal validation
    # Ensure at least 1 sample per class in validation
    min_val_samples = len(words)  # At least one per class
    if len(X) <= min_val_samples * 2:
        # Use all data for training, create a small validation set manually
        print(f"[INFO] Very small dataset - using all data for training")
        print(f"[INFO] Will use training data for validation (not ideal but necessary)")
        X_train, X_val = X, X
        y_train, y_val = y, y
    else:
        # Use minimal validation split without stratify for very small datasets
        test_size = min_val_samples / len(X)
        print(f"[INFO] Using {int((1-test_size)*100)}% training, {int(test_size*100)}% validation split (no stratification)")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=None, random_state=42)
else:
    test_size = 0.2
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

print(f"[INFO] Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Convert labels to one-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(words))
y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(words))

# ==== DATA AUGMENTATION FUNCTION FOR 3D DATA ====
def augment_3d_data(X_batch, y_batch):
    """Apply augmentation to 3D video sequences"""
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X_batch)):
        video = X_batch[i]
        label = y_batch[i]
        
        # Original sample
        augmented_X.append(video)
        augmented_y.append(label)
        
        # Augmentation 1: Brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        bright_video = np.clip(video * brightness_factor, 0, 1)
        augmented_X.append(bright_video)
        augmented_y.append(label)
        
        # Augmentation 2: Add small noise
        noise = np.random.normal(0, 0.05, video.shape)
        noisy_video = np.clip(video + noise, 0, 1)
        augmented_X.append(noisy_video)
        augmented_y.append(label)
        
        # Augmentation 3: Temporal shift (circular shift frames)
        if len(video) > 1:
            shift = np.random.randint(-2, 3)
            shifted_video = np.roll(video, shift, axis=0)
            augmented_X.append(shifted_video)
            augmented_y.append(label)
    
    return np.array(augmented_X), np.array(augmented_y)

# Apply augmentation to training data
print("\n[INFO] Applying data augmentation...")
X_train_aug, y_train_aug = augment_3d_data(X_train, y_train_onehot)
print(f"[INFO] Augmented training samples: {len(X_train_aug)} (from {len(X_train)})")
y_train_aug_onehot = y_train_aug  # Already one-hot
# ==== BUILD 3D CNN MODEL ====
def build_3d_cnn(input_shape, num_classes):
    # Simplified model for small dataset - reduce complexity to prevent overfitting
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.GlobalAveragePooling3D(),  # Use GlobalAveragePooling instead of Flatten to reduce parameters
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create model
model = build_3d_cnn(INPUT_SHAPE, len(words))

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])

# Callbacks for better training
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",  # Monitor accuracy instead of loss
    patience=20,  # More patience
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Model checkpoint to save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(PROJECT_ROOT, "model", "lip_reader_3dcnn_best.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ==== TRAIN THE MODEL ====
print("\nTraining model...\n")
print(f"[INFO] Model parameters: {model.count_params():,}")

history = model.fit(
    X_train_aug, y_train_aug_onehot,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val_onehot),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Save model
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "model", "lip_reader_3dcnn.h5")
if not os.path.exists(os.path.join(PROJECT_ROOT, "model")):
    os.makedirs(os.path.join(PROJECT_ROOT, "model"))
model.save(MODEL_SAVE_PATH)
print(f"\n[OK] Model saved to {MODEL_SAVE_PATH}")

# ==== EVALUATE MODEL ====
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_val, y_val_onehot)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Final Test Precision: {test_precision:.4f}")
print(f"Final Test Recall: {test_recall:.4f}")

# ==== PLOT TRAINING PERFORMANCE ====
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(8, 8))

axs[0].plot(history.history['loss'], label='Training Loss')
axs[0].plot(history.history['val_loss'], label='Validation Loss')
axs[0].legend(loc='upper right')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training and Validation Loss')

axs[1].plot(history.history['accuracy'], label='Training Accuracy')
axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axs[1].legend(loc='lower right')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Training and Validation Accuracy')

plt.xlabel('Epoch')
plt.show()

def compute_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-7)

# Extract logged metrics from training history
train_precision = history.history['precision']
val_precision = history.history['val_precision']
train_recall = history.history['recall']
val_recall = history.history['val_recall']

# Compute F1 scores epoch-wise
train_f1 = [compute_f1(p, r) for p, r in zip(train_precision, train_recall)]
val_f1 = [compute_f1(p, r) for p, r in zip(val_precision, val_recall)]

epochs = range(1, EPOCHS + 1)

# Create subplots for precision, recall, and F1 score
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Precision Plot
axs[0].plot(epochs, train_precision, label="Train Precision")
axs[0].plot(epochs, val_precision, label="Validation Precision")
axs[0].set_title("Precision Over Epochs")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Precision")
axs[0].legend()

# Recall Plot
axs[1].plot(epochs, train_recall, label="Train Recall")
axs[1].plot(epochs, val_recall, label="Validation Recall")
axs[1].set_title("Recall Over Epochs")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Recall")
axs[1].legend()

# F1 Score Plot
axs[2].plot(epochs, train_f1, label="Train F1 Score")
axs[2].plot(epochs, val_f1, label="Validation F1 Score")
axs[2].set_title("F1 Score Over Epochs")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("F1 Score")
axs[2].legend()

plt.tight_layout()
plt.show()