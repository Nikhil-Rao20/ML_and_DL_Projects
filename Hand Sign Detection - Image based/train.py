# Usage: python train.py

from pyimagesearch.utils import prepare_batch_dataset
from pyimagesearch.utils import callbacks
from pyimagesearch import config
from pyimagesearch.network import MobileNet
from matplotlib import pyplot as plt
import tensorflow as tf
import os

print("[INFO] building the training and Validation dataset...")
train_ds = prepare_batch_dataset(
    config.TEST_DATA_PATH, config.IMAGE_SIZE, config.BATCH_SIZE
)
val_ds = prepare_batch_dataset(
    config.VALID_DATA_PATH, config.IMAGE_SIZE, config.BATCH_SIZE
)

if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)

print("[INFO] compiling model...")
callbacks = callbacks()
model = MobileNet.build(
    width = config.IMAGE_SIZE,
    height= config.IMAGE_SIZE,
    depth=config.CHANNELS,
    classes=config.N_CLASSES
)
optimizer = tf.keras.optimizers.Adam(learning_rate=config.LR_INIT)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer = optimizer,
    loss=loss,
    metrics=['accuracy']
)

(initial_loss, initial_accuracy) = model.evaluate(val_ds)
print("initial loss: {:.2f}".format(initial_loss))
print("initial accuracy: {:.2f}".format(initial_accuracy))

print("[INFO] training network...")
history = model.fit(
    train_ds,
    epochs=config.NUM_EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
)

print("[INFO] serializing network....")
model.save(config.TRAINED_MODEL_PATH)

plt.style.use("ggplot")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig(config.ACCURACY_LOSS_PLOT_PATH)