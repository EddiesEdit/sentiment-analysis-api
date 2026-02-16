print("Loading app/train.py...")

# -------------------------------
# Imports
# -------------------------------
from data import load_dataset, prepare_data
from model import build_model_v3
import tensorflow as tf
import json
import os

# -------------------------------
# Load and Prepare Data
# -------------------------------
df = load_dataset()
X_train, X_test, y_train, y_test, tokenizer, vocab_size = prepare_data(df)

# -------------------------------
# Build and Compile Model
# -------------------------------
model = build_model_v3(vocab_size)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=3,
    batch_size=128,
    verbose=1
)

# -------------------------------
# Save Model and Tokenizer
# -------------------------------
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(save_dir, exist_ok=True)

# Save trained model in new Keras format
model_path = os.path.join(save_dir, "cnn_sentiment_v3.keras")
model.save(model_path)

# Save tokenizer
tokenizer_json = tokenizer.to_json()
tokenizer_path = os.path.join(save_dir, "tokenizer.json")
with open(tokenizer_path, "w", encoding="utf-8") as f:
    f.write(tokenizer_json)

print(f"✅ Model saved to: {model_path}")
print(f"✅ Tokenizer saved to: {tokenizer_path}")
print("Training complete!")
