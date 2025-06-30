import os
import numpy as np
import tensorflow as tf
from src.model import build_unet
from src.losses import bce_dice_loss
import matplotlib.pyplot as plt

data_dir = "data"
X_train = np.load(os.path.join(data_dir, "train_images.npy"))
Y_train = np.load(os.path.join(data_dir, "train_masks.npy"))

X_train = X_train.astype("float32") / 255.0
Y_train = Y_train.astype("float32")

model = build_unet(input_shape=(256, 256, 6))
model.compile(optimizer="adam", loss=bce_dice_loss, metrics=["accuracy"])

checkpoint_path = os.path.join("models", "unet_model.h5")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss", mode="min"),
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]

history = model.fit(
    X_train, Y_train,
    validation_split=0.1,
    batch_size=8,
    epochs=50,
    callbacks=callbacks
)

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig(os.path.join("results", "loss_curve.png"))
