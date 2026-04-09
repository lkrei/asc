from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "Dataset_1" / "architectural-styles-dataset"
RESULTS_DIR = CODE_DIR / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 25
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

LEARNING_RATE = 0.001
NUM_EPOCHS = 15
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

MODEL_NAME = "resnet50"
PRETRAINED = True
FREEZE_BACKBONE = True

SAVE_BEST_MODEL = True
SAVE_LAST_MODEL = True
MODEL_CHECKPOINT_NAME = "best_model_resnet50.pth"

CLASS_NAMES = [
    "Achaemenid architecture",
    "American craftsman style",
    "American Foursquare architecture",
    "Ancient Egyptian architecture",
    "Art Deco architecture",
    "Art Nouveau architecture",
    "Baroque architecture",
    "Bauhaus architecture",
    "Beaux-Arts architecture",
    "Byzantine architecture",
    "Chicago school architecture",
    "Colonial architecture",
    "Deconstructivism",
    "Edwardian architecture",
    "Georgian architecture",
    "Gothic architecture",
    "Greek Revival architecture",
    "International style",
    "Novelty architecture",
    "Palladian architecture",
    "Postmodern architecture",
    "Queen Anne architecture",
    "Romanesque architecture",
    "Russian Revival architecture",
    "Tudor Revival architecture"
]
