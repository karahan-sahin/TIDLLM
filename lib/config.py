
# Why to use dictionary instead of global variables?
# When needed, all configurations can be easily saved

from types import SimpleNamespace

GLOBAL_CONFIG = {
    ################################################################################
    # Please visit:
    #   https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md
    ################################################################################
    "MEDIAPIPE_STATIC_IMAGE_MODE" : False,      # If True, person detection runs every input image. Default to False.
    "MEDIAPIPE_MODEL_COMPLEXITY" : 1,           # 0, 1, 2. Default to 1
    "MEDIAPIPE_SMOOTH_LANDMARKS" : True,        # Default to True
    "MEDIAPIPE_ENABLE_SEGMENTATION" : False,    # Default to False
    "MEDIAPIPE_SMOOTH_SEGMENTATION" : True,     # Default to True
    "MEDIAPIPE_REFINE_FACE_LANDMARKS" : False,  # Default to False
    "MEDIAPIPE_MIN_DETECTION_CONFIDENCE" : 0.5, # Between [0, 1]
    "MEDIAPIPE_MIN_TRACKING_CONFIDENCE" : 0.5,  # Between [0, 1]
    ################################################################################
    "MODEL_ENCODER_INPUT_DIM" : 225,
    "MODEL_ENCODER_HIDDEN_DIM" : 256,
    "MODEL_ENCODER_OUTPUT_DIM" : 768,
    "MODEL_VQ_NUM_EMBS" : 1000,
    "MODEL_VQ_EMBED_DIM" : 768,
    "MODEL_VQ_COMMITMENT_COST" : 0.25,
    "MODEL_DECODER_INPUT_DIM" : 768,
    "MODEL_DECODER_HIDDEN_DIM" : 256,
    "MODEL_DECODER_OUTPUT_DIM" : 225,
    "NUM_EPOCHS" : 100,
    "BATCH_SIZE" : 8,
    "LEARNING_RATE" : 1e-5,
}

GLOBAL_CONFIG = SimpleNamespace(**GLOBAL_CONFIG)

def write_global_config_to_file(filepath: str):

    """ Writes GLOBAL_CONFIG to given filepath. """
    
    assert type(filepath) is str, f"{filepath} type must be 'str' ."
    assert filepath.endswith(".pkl"), f"{filepath} must ends with '.pkl' ."
    
    import pickle

    with open(filepath, 'wb') as fp:
        global GLOBAL_CONFIG
        pickle.dump(GLOBAL_CONFIG, fp)

        print(f'GLOBAL_CONFIG saved successfully to {filepath}')

def read_global_config_from_file(filepath: str) -> any:
    """ Reads given filepath and stores it as GLOBAL_CONFIG. """

    import os
    import pickle

    assert type(filepath) is str, f"{filepath} type must be 'str' ."
    assert filepath.endswith(".pkl"), f"{filepath} must ends with '.pkl' ."
    assert os.path.exists(filepath), f"{filepath} does not exist !"

    with open(filepath, 'rb') as fp:
        config = pickle.load(fp)
        print(f'GLOBAL_CONFIG restored succesfully from {filepath}.')
        return config