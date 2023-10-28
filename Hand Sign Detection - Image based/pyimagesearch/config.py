import os

BASE_DATASET_PATH = "captured_images"
TRAIN_DATA_PATH = os.path.join(BASE_DATASET_PATH, "train")
VALID_DATA_PATH = os.path.join(BASE_DATASET_PATH, "valid")
TEST_DATA_PATH = os.path.join(BASE_DATASET_PATH, "test")
OUTPUT_PATH = "output"

IMAGE_SIZE = 224
BATCH_SIZE = 32
CHANNELS = 3
LR_INIT = 0.0001
NUM_EPOCHS = 50
N_CLASSES = 24

ACCURACY_LOSS_PLOT_PATH = os.path.join("output", "accuracy_loss_plot.png")
TRAINED_MODEL_PATH = os.path.join("output", "hand_sign_classifier")
TEST_PREDICTION_OUTPUT = os.path.join("output", "test_prediction.png")