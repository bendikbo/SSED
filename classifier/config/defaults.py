from yacs.config import CfgNode as CN

cfg = CN()


#Model setup
cfg.MODEL = CN()
# Set any record containing THRESHOLD or more of a class as positive
cfg.MODEL.THRESHOLD = 0.25
cfg.MODEL.NUM_CLASSES = 5

# Hard negative mining configurations
cfg.MODEL.NEG_POS_RATIO = 3

# ---------------------------------------------------------------------------- #
# Model name
# ---------------------------------------------------------------------------- #

cfg.MODEL.NAME = 'effnetv2m'

# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
#Dataset setup
cfg.INPUT = CN()
cfg.INPUT.RECORD_LENGTH = 65536
cfg.INPUT.SAMPLE_FREQ = 16000
cfg.INPUT.NAME = "kauto5cls"
#Used in case of spectrogram
cfg.INPUT.IMAGE_SIZE = [224, 224]

#Basic stuff
cfg.INPUT.TRANSFORM = CN()
cfg.INPUT.TRANSFORM.SAMPLE_COORDS = CN()
cfg.INPUT.TRANSFORM.SAMPLE_COORDS.ACTIVE = True
cfg.INPUT.TRANSFORM.CROP = CN()
cfg.INPUT.TRANSFORM.CROP.ACTIVE = True
cfg.INPUT.TRANSFORM.LENGTH = 65536

#Alright config for spectrogram
cfg.INPUT.TRANSFORM.SPECTROGRAM = CN()
cfg.INPUT.TRANSFORM.SPECTROGRAM.ACTIVE = True
cfg.INPUT.TRANSFORM.SPECTROGRAM.RESOLUTION = [224, 450]#OUTPUT RESOLUTION
cfg.INPUT.TRANSFORM.SPECTROGRAM.FREQ_CROP = [12, 224] #[BOTTOM_CROP, HEIGHT]
cfg.INPUT.TRANSFORM.SPECTROGRAM.CHANNELS = ["normal", "mel", "log"]

#Random gaussian noise with random intensity
cfg.INPUT.TRANSFORM.RAND_GAUSS = CN()
cfg.INPUT.TRANSFORM.RAND_GAUSS.ACTIVE = False
cfg.INPUT.TRANSFORM.RAND_GAUSS.INTENSITY = 2.0
cfg.INPUT.TRANSFORM.RAND_GAUSS.RAND = True
cfg.INPUT.TRANSFORM.CHANCE = 1

#Random "flip"
cfg.INPUT.TRANSFORM.RAND_FLIP = CN()
cfg.INPUT.TRANSFORM.RAND_FLIP.ACTIVE = False
cfg.INPUT.TRANSFORM.RAND_FLIP.CHANCE = 0.5

#Random amplification/attenuation
cfg.INPUT.TRANSFORM.RAND_AMP_ATT = CN()
cfg.INPUT.TRANSFORM.RAND_AMP_ATT.ACTIVE = False
cfg.INPUT.TRANSFORM.RAND_AMP_ATT.FACTOR = 5
cfg.INPUT.TRANSFORM.RAND_AMP_ATT.CHANCE = 1.0

#Random "contrast"
cfg.INPUT.TRANSFORM.RAND_CONTRAST = CN()
cfg.INPUT.TRANSFORM.RAND_CONTRAST.ACTIVE = False
cfg.INPUT.TRANSFORM.RAND_CONTRAST.CHANCE = 1.0
cfg.INPUT.TRANSFORM.RAND_CONTRAST.ENHANCE = 50

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
cfg.DATA_LOADER = CN()
cfg.DATA_LOADER.NUM_WORKERS = 32
cfg.DATA_LOADER.PIN_MEMORY = False

# ---------------------------------------------------------------------------- #
# Solver - The same as optimizer
# ---------------------------------------------------------------------------- #
cfg.TRAINER = CN()
cfg.TRAINER.EPOCHS = 20
cfg.TRAINER.SCHEDULER = "multistep"
cfg.TRAINER.LR_STEPS = [12, 16, 18]
cfg.TRAINER.GAMMA = 0.1
cfg.TRAINER.BATCH_SIZE = 128
cfg.TRAINER.EVAL_STEP = 1
cfg.TRAINER.OPTIMIZER = "sgd"
cfg.TRAINER.LR = 1e-3
cfg.TRAINER.MOMENTUM = 0.9
cfg.TRAINER.WEIGHT_DECAY = 5e-4
cfg.TRAINER.ACTIVATION = "sigmoid"

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
cfg.TEST = CN()
cfg.TEST.NMS_THRESHOLD = 0.5
cfg.TEST.CONFIDENCE_THRESHOLD = 0.01
cfg.TEST.BATCH_SIZE = 128
cfg.LOG_STEP = 10
cfg.OUTPUT_DIR = "outputs"
cfg.DATASET_DIR = "datasets"
