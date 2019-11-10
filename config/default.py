from yacs.config import CfgNode as CN

_C = CN()
_C.CONFIG_NAME = "FinalParamsLarge"
# ------------------------------------------------------------
# MODEL PARAMETERS
# ------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DROPOUT = 0.25
_C.MODEL.RECURRENT_DROPOUT = 0.0
_C.MODEL.USE_RNN = True
_C.MODEL.RNN = "lstm"  # either 'lstm' or 'gru'
_C.MODEL.RNN_SIZE = 300
_C.MODEL.USE_CNN = True
_C.MODEL.CNN_LAYERS = 2  # must be greater than 0
_C.MODEL.NUM_FILTERS = 200
_C.MODEL.FILTER_SIZE = 3
_C.MODEL.MAX_POOL_SIZE = 2  # if None or False, do not use MaxPooling
_C.MODEL.CLASSIFIER = "crf"  # either 'softmax' or 'crf' (by Philipp Gross)
_C.MODEL.EMBEDDING_SIZE = 300
# ------------------------------------------------------------
# TRAINING PARAMETERS
# ------------------------------------------------------------
_C.TRAINING = CN()
# same as the language
# ["english", "italian", "basque", "dutch", "manipuri", "french"]
_C.TRAINING.DATASETS = ["french"]
_C.TRAINING.FEATURE_NAMES = ["tokens"]
_C.TRAINING.MODEL_SAVE_PATH = (
    "models/[ModelName]_[Epoch]_[DevScore]_[TestScore].h5"
)
# used to get mean and standard deviation of model
_C.TRAINING.TRAINING_REPEATS = 1  # just trains the model once
_C.TRAINING.EPOCHS = 120
_C.TRAINING.MINI_BATCH_SIZE = 64
_C.TRAINING.EARLY_STOPPING = 10
# ------------------------------------------------------------
# OPTIMIZATION PARAMETERS
# ------------------------------------------------------------
_C.OPTIMIZER = CN()
_C.OPTIMIZER.OPTIMIZER = "adam"
_C.OPTIMIZER.CLIP_NORM = 0.0  # must be >= 0.0
_C.OPTIMIZER.CLIP_VALUE = 0.0

# SEE IF DROPOUT WORKS WITH LSTM


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    return _C.clone()
