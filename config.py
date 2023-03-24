import torch

DBG = False;
HIDDEN_DIM = 128;
BATCH_SIZE = 128;
LR = 1e-4;
TEMPERATURE = 0.07;
WEIGHT_DECAY = 1e-4;
MAX_EPOCHS = 500;
TRAIN_CONTRASTIVE = False;
IMAGE_SIZE = 96

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/tutorial17"
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = 2
MAX_DROPS = 10;

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", DEVICE)