
# Why to use dictionary instead of global variables?
# When needed, all configurations can be easily saved

GLOBAL_CONFIG = {
    "MEDIAPIPE_MIN_DETECTION_CONFIDENCE" : 0.01,
    "MEDIAPIPE_MIN_TRACKING_CONFIDENCE" : 0.01,
    "MODEL_ENCODER_INPUT_DIM" : 225,
    "MODEL_ENCODER_HIDDEN_DIM" : 256,
    "MODEL_ENCODER_OUTPUT_DIM" : 768,
    "MODEL_VQ_NUM_EMBS" : 10_000,
    "MODEL_VQ_EMBED_DIM" : 768,
    "MODEL_VQ_COMMITMENT_COST" : 0.25,
    "MODEL_DECODER_INPUT_DIM" : 768,
    "MODEL_DECODER_HIDDEN_DIM" : 256,
    "MODEL_DECODER_OUTPUT_DIM" : 225,
    "NUM_EPOCHS" : 100,
    "BATCH_SIZE" : 32,
    "LEARNING_RATE" : 1e-5,
}

def write_global_config_to_file(filepath: str):

    """ Writes GLOBAL_CONFIG to given filepath. """
    
    assert type(filepath) is str, f"{filepath} type must be 'str' ."
    assert filepath.endswith(".pkl"), f"{filepath} must ends with '.pkl' ."
    
    import pickle

    with open(filepath, 'wb') as fp:
        global GLOBAL_CONFIG
        pickle.dump(GLOBAL_CONFIG, fp)

        print(f'GLOBAL_CONFIG saved successfully to {filepath}')

def read_global_config_from_file(filepath: str):
    """ Reads given filepath and stores it as GLOBAL_CONFIG. """

    import os
    import pickle

    assert type(filepath) is str, f"{filepath} type must be 'str' ."
    assert filepath.endswith(".pkl"), f"{filepath} must ends with '.pkl' ."
    assert os.path.exists(filepath), f"{filepath} does not exist !"

    with open(filepath, 'rb') as fp:
        global GLOBAL_CONFIG
        GLOBAL_CONFIG = pickle.load(fp)

        print(f'GLOBAL_CONFIG restored succesfully from {filepath}.')