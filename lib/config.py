# Why to use dictionary instead of global variables?
# When needed, all configurations can be easily saved

from types import SimpleNamespace

GLOBAL_CONFIG = {
    ################################################################################
    # Please visit:
    #   https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md
    ################################################################################
    "MEDIAPIPE_STATIC_IMAGE_MODE": False,  # If True, person detection runs every input image. Default to False.
    "MEDIAPIPE_MODEL_COMPLEXITY": 1,  # 0, 1, 2. Default to 1
    "MEDIAPIPE_SMOOTH_LANDMARKS": True,  # Default to True
    "MEDIAPIPE_ENABLE_SEGMENTATION": False,  # Default to False
    "MEDIAPIPE_SMOOTH_SEGMENTATION": True,  # Default to True
    "MEDIAPIPE_REFINE_FACE_LANDMARKS": False,  # Default to False
    "MEDIAPIPE_MIN_DETECTION_CONFIDENCE": 0.5,  # Between [0, 1]
    "MEDIAPIPE_MIN_TRACKING_CONFIDENCE": 0.5,  # Between [0, 1]
    ################################################################################
    # 3D-CNN Model Configurations
    "INPUT_CHANNELS": 2,  # Discard z-axis
    "FRAME_WINDOW": 25,  # timesteps to consider
    "INPUT_DIM": (
        1,  # Batch Size
        None,  # Channels (INPUT_CHANNELS)
        None,  # Depth (NUM FRAMES)
        75,  # Height (Joint Count)
        75,  # Width (Joint Count)
    ),
    # Encoder Configurations
    "MODEL_ENCODER_NAME": "POSE_3D_ENCODER",
    "MODEL_ENCODER_OUT_CHANNEL": 5,
    "MODEL_ENCODER_CONVOLUTIONAL_LAYERS": [
        {
            # Convolutional Layer 1
            # CNN Layer 1: 1x10x10 -> 10x10x10
            "in_channels": None,
            "out_channels": 10,
            "kernel_size": (1, 10, 10),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            # Batch Normalization
            "batch_norm": True,
            # Activation Function
            "activation": "gelu",
            # Pooling Layer 1: 10x10x10 -> 10x5x5
            "pooling": "max",
            "pooling_kernel_size": (1, 2, 2),
            "pooling_stride": 1,
        },
        {
            # Convolutional Layer 2
            # CNN Layer 1: 1x10x10 -> 10x10x10
            "in_channels": 10,
            "out_channels": 5,
            "kernel_size": (1, 10, 10),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            # Batch Normalization
            "batch_norm": True,
            # Activation Function
            "activation": "gelu",
            # Pooling Layer 1: 10x10x10 -> 10x5x5
            "pooling": "max",
            "pooling_kernel_size": (1, 2, 2),
            "pooling_stride": 1,
        },
        {
            # Convolutional Layer 3
            # CNN Layer 1: 1x10x10 -> 10x10x10
            "in_channels": 5,
            "out_channels": 2,
            "kernel_size": (1, 10, 10),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            # Batch Normalization
            "batch_norm": True,
            # Activation Function
            "activation": "gelu",
            # Pooling Layer 1: 10x10x10 -> 10x5x5
            "pooling": "max",
            "pooling_kernel_size": (1, 2, 2),
            "pooling_stride": 1,
        },
    ],
    "MODEL_ENCODER_LINEAR_LAYERS": [
        # Linear Layer 1: 10x5x5 -> 512
        {"in_features": 512, "out_features": 512, "activation": "gelu", "dropout": 0.5},
        # ....
        # Linear Layer 2: 512 -> 512 (Output Layer) (Should be equal to MODEL_VQ_EMBED_DIM)
        {"in_features": 512, "out_features": 768, "activation": "gelu", "dropout": 0.5},
    ],
    # VQ-VAE Configurations
    "MODEL_VQ_NAME": "VQ_VAE",
    "MODEL_VQ_CODEBOOK": 2,  # NUM OF RESIDUAL BLOCKS (CODEBOOKS) (NUM_QUANDRANTS = 2^CODEBOOK)
    "MODEL_VQ_EMBED_DIM": 512,
    "MODEL_VQ_COMMITMENT_COST": 0.25,
    "MODEL_VQ_VOCAB": 512,
    # Decoder Configurations
    "MODEL_DECODER_NAME": "POSE_3D_DECODER",
    "MODEL_DECODER_INPUT_DIM": 512,
    "MODEL_DECODER_CONVOLUTIONAL_LAYERS": [
        {
            # Convolutional Layer 1: 10x10x10 -> 10x10x10
            "in_channels": None,
            "out_channels": 3,
            "kernel_size": (3, 5, 5),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            "output_padding": (1, 1, 1),
            # Activation Function
            "activation": "gelu",
            # Pooling Layer 1: 10x10x10 -> 10x5x5
            "pooling": "avg",
            "pooling_kernel_size": (2, 2, 2),
            "pooling_stride": 1,
        },
        {
            # Convolutional Layer 2: 10x10x10 -> 10x10x10
            "in_channels": 3,
            "out_channels": 3,
            "kernel_size": (4, 5, 5),
            "stride": (1, 2, 2),
            "padding": (1, 1, 1),
            "output_padding": (1, 1, 1),
            # Activation Function
            "activation": "gelu",
            # Pooling Layer 2: 10x10x10 -> 10x5x5
            "pooling_kernel_size": (2, 2, 2),
            "pooling_stride": 1,
        },
        {
            # Convolutional Layer 3: 10x10x10 -> 10x10x10
            "in_channels": 3,
            "out_channels": None,
            "kernel_size": (3, 15, 15),
            "stride": (1, 2, 2),
            "padding": (1, 1, 1),
            "output_padding": (1, 1, 1),
            # Activation Function
            "activation": "gelu",
            # Pooling Layer 3: 10x10x10 -> 10x5x5
            "pooling_kernel_size": (2, 2, 2),
            "pooling_stride": 1,
        },
    ],
    "MODEL_DECODER_LINEAR_LAYERS": [
        # Linear Layer 1: 10x5x5 -> 512
        {
            "in_features": None,
            "out_features": 512,
            "activation": "gelu",
            "dropout": 0.5,
        },
        # ....
        # Linear Layer 2: 512 -> 512 (Output Layer) (Should be equal to MODEL_VQ_EMBED_DIM)
        {
            "in_features": 512,
            "out_features": None,
            "activation": "gelu",
            "dropout": 0.5,
        },
    ],
    # Training Configurations
    "NUM_EPOCHS": 100,
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 1e-5,
    "STEP_SIZE": 10,
    "GAMMA": 0.1,
    "DEVICE": "cuda:0",
    "MODEL_OUT_DIR": "",
    "MODEL_LOG_DIR": "",
    ############################################################################
    # FFNN Model Configurations
    "MODEL_ENCODER_INPUT_DIM": 225,
    "MODEL_ENCODER_HIDDEN_DIM": 256,
    "MODEL_ENCODER_OUTPUT_DIM": 768,
    "MODEL_VQ_NUM_EMBS": 1000,
    "MODEL_VQ_EMBED_DIM": 768,
    "MODEL_VQ_COMMITMENT_COST": 0.25,
    "MODEL_DECODER_INPUT_DIM": 768,
    "MODEL_DECODER_HIDDEN_DIM": 256,
    "MODEL_DECODER_OUTPUT_DIM": 225,
    ############################################################################
    # Training Configurations
    "NUM_EPOCHS": 100,
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 1e-5,
}

GLOBAL_CONFIG = SimpleNamespace(**GLOBAL_CONFIG)


def write_global_config_to_file(filepath: str):
    """Writes GLOBAL_CONFIG to given filepath."""

    assert type(filepath) is str, f"{filepath} type must be 'str' ."
    assert filepath.endswith(".pkl"), f"{filepath} must ends with '.pkl' ."

    import pickle

    with open(filepath, "wb") as fp:
        global GLOBAL_CONFIG
        pickle.dump(GLOBAL_CONFIG, fp)

        print(f"GLOBAL_CONFIG saved successfully to {filepath}")


def read_global_config_from_file(filepath: str) -> any:
    """Reads given filepath and stores it as GLOBAL_CONFIG."""

    import os
    import pickle

    assert type(filepath) is str, f"{filepath} type must be 'str' ."
    assert filepath.endswith(".pkl"), f"{filepath} must ends with '.pkl' ."
    assert os.path.exists(filepath), f"{filepath} does not exist !"

    with open(filepath, "rb") as fp:
        config = pickle.load(fp)
        print(f"GLOBAL_CONFIG restored succesfully from {filepath}.")
        return config
