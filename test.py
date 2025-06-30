import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from src.model import build_unet
from src.losses import bce_dice_loss

# ============ Load Data ============
data_dir = "data"
X_test = np.load(os.path.join(data_dir, "test_images.npy"))
Y_test = np.load(os.path.join(data_dir, "test_masks.npy"))

X_test = X_test.astype("float32") / 255.0
Y_test = Y_test.astype("float32")

# ============ Load Model ============
model_path = os.path.join("models", "unet_model.h5")
model = build_unet(input_shape=(256, 256, 6))
model.compile(optimizer="adam", loss=bce_dice_loss, metrics=["accuracy"])
model.load_weights(model_path)

# ============ Evaluate ============
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# ============ Visualize Sample Prediction ============
idx = 0
sample_image = X_test[idx]
sample_mask = Y_test[idx]
pred_mask = model.predict(np.expand_dims(sample_image, axis=0))[0]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(sample_image[:, :, :3])  # pre-image

plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(sample_mask[:, :, 0], cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow((pred_mask[:, :, 0] > 0.5).astype("uint8"), cmap="gray")

plt.tight_layout()
plt.savefig(os.path.join("results", "sample_prediction.png"))
