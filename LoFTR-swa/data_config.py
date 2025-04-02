from configs.data.base import cfg

TRAIN_BASE_PATH = "dataa"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "Custom"
cfg.DATASET.TRAIN_DATA_ROOT = "dataa/feature_matching"
cfg.DATASET.VAL_DATA_ROOT = "dataa/feature_matching_val"
cfg.DATASET.TEST_DATA_ROOT = "dataa/feature_matching_val"  # Default test data to be the same as validation data
cfg.DATASET.TRAIN_PKL_FILE = "dataa/feature_matching/data.pkl"
cfg.DATASET.VAL_PKL_FILE = "dataa/feature_matching_val/data.pkl"
cfg.DATASET.TEST_PKL_FILE = "dataa/feature_matching_val/data.pkl"

cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0   # for both test and val
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0
cfg.DATASET.AUGMENTATION_TYPE = None
