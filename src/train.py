from data_loader import get_generators
from model import build_cnn_model
import matplotlib.pyplot as plt

# Load data
train_gen, val_gen = get_generators()

# Build model
model = build_cnn_model(num_classes=len(train_gen.class_indices))
model.summary()

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20
)

# Save model
model.save('plant_seedlings_cnn.h5')
print("Model saved!")

# Plot accuracy/loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()
