import numpy as np
from tensorflow.keras.models import load_model
from data_loader import get_generators
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
model = load_model('plant_seedlings_cnn.h5')

# Load validation data
_, val_gen = get_generators()
val_gen.reset()

# Predict
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

# Classification report
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion matrix (needed to calculate per-class accuracy)
cm = confusion_matrix(y_true, y_pred)

# Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(12,6))
sns.barplot(x=class_labels, y=per_class_acc)
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.title("Per-Class Accuracy")
plt.show()
