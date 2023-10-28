# Usage: python test.py

from pyimagesearch.utils import prepare_batch_dataset
from pyimagesearch.utils import callbacks
from pyimagesearch import config
from pyimagesearch.network import MobileNet
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import os

print("[INFO] building the test dataset with and without shuffles...")
test_ds_wo_shuffle = prepare_batch_dataset(
    config.TEST_DATA_PATH,
    config.IMAGE_SIZE,
    config.BATCH_SIZE,
    shuffle=False
)
test_ds_shuffle = prepare_batch_dataset(
    config.TEST_DATA_PATH,
    config.IMAGE_SIZE,
    config.BATCH_SIZE,
    shuffle=True
)

model = tf.keras.models.load_model(config.TRAINED_MODEL_PATH)
print(model.summary())

loss, accuracy = model.evaluate(test_ds_wo_shuffle)
print("Test accuracy : ", accuracy)

class_names = test_ds_wo_shuffle.class_names

test_pred = model.predict(test_ds_wo_shuffle)
test_pred = tf.nn.softmax(test_pred)
test_pred = tf.argmax(test_pred, axis=1)
test_true_labels = tf.concat(
    [label for _, label in test_ds_wo_shuffle], axis=0
)
print(
    classification_report(
        test_true_labels, test_pred, target_names=class_names
    )
)

image_batch,label_batch = test_ds_shuffle.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)

score = tf.nn.softmax(predictions)
print("Score shape is : ",score.shape)

output_mapping = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E',
    6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'K',
    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}



plt.figure(figsize=(10, 10))
for i in range(9):
	ax = plt.subplot(3, 3, i + 1)
	plt.imshow(image_batch[i].astype("uint8"))
	title = output_mapping[int(class_names[np.argmax(score[i])])]
	plt.title(title)
	plt.axis("off")
plt.savefig(config.TEST_PREDICTION_OUTPUT)
