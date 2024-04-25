import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.pooling import Pooling
from lib.utils.activation import Activation

class CNN3dEncoder(nn.Module):
    def __init__(
        self, 
        model_name, 
        input_size, 
        conv_layers, 
        linear_layers, 
        output_size, 
        out_channels=3,
        channel_size=2,
        depth_size=25,
        log=False
    ):

        super(CNN3dEncoder, self).__init__()

        self.model_name = model_name
        self.depth_size = depth_size
        self.channel_size = channel_size

        self.input_size = (1, channel_size, depth_size, input_size[3], input_size[4])
        self.log = log

        conv_layers[0]['in_channels'] = self.input_size[1]
        conv_layers[-1]['out_channels'] = out_channels
        ###############
        # CONV LAYERS #
        ###############

        self.conv_layers = []
        for idx, layer in enumerate(conv_layers):

            self.conv_layers.append(self.make_conv_layer(layer))
            self.register_module(f"ENCODER_CONV_{idx}", self.conv_layers[-1])

            print(f'encoder_conv_{idx}', layer)

        self.output_size = self.calculate_output_size(self.input_size)

        #################
        # LINEAR LAYERS #
        #################

        linear_layers[0]["in_features"] = (
            self.output_size[1]
            * self.output_size[2]
            * self.output_size[3]
            * self.output_size[4]
        )

        self.linear_layers = []
        for idx, layer in enumerate(linear_layers):
            self.linear_layers.append(self.make_linear_layer(layer))
            self.register_module(f"ENCODER_Linear_{idx}", self.linear_layers[-1])

            print(f"encoder_linear_{idx}", layer)

        self.output_layer = nn.Linear(
            in_features=linear_layers[-1]["out_features"],
            out_features=output_size,
        )

    def initialize_weights(self):
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv3d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in self.linear_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def calculate_output_size(self, input_size):
        rand_input = torch.rand(input_size)
        for layer in self.conv_layers:
            rand_input = layer(rand_input)
        return rand_input.size()

    def make_linear_layer(self, layer):
        return nn.Sequential(
            nn.Linear(
                in_features=layer["in_features"], out_features=layer["out_features"]
            ),
            nn.BatchNorm1d(layer["out_features"]),
            Activation(layer["activation"]),
        )

    def make_conv_layer(self, layer):
        return nn.Sequential(
            nn.Conv3d(
                in_channels=layer["in_channels"],
                out_channels=layer["out_channels"],
                kernel_size=layer["kernel_size"],
                stride=layer["stride"],
                padding=layer["padding"],
            ),
            nn.BatchNorm3d(layer["out_channels"]),
            Activation(layer["activation"]),
            Pooling[layer['pooling']](
                kernel_size=layer["pooling_kernel_size"], stride=layer["pooling_stride"]
            ),
        )

    def forward(self, x):

        for idx, layer in enumerate(self.conv_layers):
            x = layer(x)
            if self.log: print(f"Encoder_Conv_{idx} layer output shape:", x.shape)

        x = x.reshape(
            -1,
            self.output_size[1]
            * self.output_size[2]
            * self.output_size[3]
            * self.output_size[4],
        )

        for idx, layer in enumerate(self.linear_layers):
            x = layer(x)
            if self.log: print(f"Encoder_Linear_{idx} layer output shape:", x.shape)

        x = self.output_layer(x)
        return x
    

class CNN3dDecoder(nn.Module):
    def __init__(
        self,
        model_name,
        input_size,
        in_channels,
        linear_input,
        conv_transpose_layers,
        linear_layers,
        output_size,
        log=False,
        upsampling_mode="trilinear",
    ):
        """
        3D Convolutional Decoder with Convolutional Transpose Layers

        Args:
            model_name (str): model name
            input_size (tuple): input size
            conv_transpose_layers (list): convolutional transpose layers configuration
            linear_layers (list): linear layers configuration
            output_size (int): output size
            log (bool, optional): log the output shape. Defaults to False.
            upsampling_mode (str, optional): upsampling mode. Defaults to "trilinear".
        
        Example:
        ```
        model = CNN3dDecoder(
            model_name="CNN3D_DECODER",
            input_size=(1, 10, 10, 10),
            conv_transpose_layers=[
                {
                    "in_channels": None,
                    "out_channels": 3,
                    "kernel_size": (3, 5, 5),
                    "stride": (1, 1, 1),
                    "padding": (1, 1, 1),
                    "activation": "relu",
                },
                {
                    "in_channels": 3,
                    "out_channels": 3,
                    "kernel_size": (4, 5, 5),
                    "stride": (1, 2, 2),
                    "padding": (1, 1, 1),
                    "activation": "relu",
                },
            ],
            linear_layers=[
                {"in_features": None, "out_features": 512, "activation": "relu"},
                {"in_features": 512, "out_features": None, "activation": "relu"},
            ],
            output_size=10,
            log=True,
        )
        ```
        """
        
        super(CNN3dDecoder, self).__init__()

        self.model_name = model_name
        self.log = log


        #################
        # LINEAR LAYERS #
        #################
        self.input_size = input_size

        self.linear_layers = []
        # Update the last linear layer out_features
        linear_layers[0]['in_features'] = linear_input
        linear_layers[-1]["out_features"] = (
            self.input_size[1]
            * self.input_size[2]
            * self.input_size[3]
            * self.input_size[4]
        )

        for idx, layer in enumerate(linear_layers):
            print(f'decoder_linear_out_{idx}:',layer)
            self.linear_layers.append(self.make_linear_layer(layer))
            self.register_module(f"DECODER_LINEAR_{idx}", self.linear_layers[-1])

        #########################
        # CONV TRANSPOSE LAYERS #
        #########################

        # Update the first and last conv transpose layers
        conv_transpose_layers[0]["in_channels"] = in_channels
        conv_transpose_layers[-1]["out_channels"] = output_size[0]

        self.conv_transpose_layers = []
        for idx, layer in enumerate(conv_transpose_layers):
            print(f'decoder_conv_out_{idx}:',layer)
            self.conv_transpose_layers.append(self.make_conv_transpose_layer(layer))
            self.register_module(
                f"DECODER_CONVTRANSPOSE_{idx}", self.conv_transpose_layers[-1]
            )

        # Final output to match the encoder input size for reconstruction
        # MODE should be differentiable
        self.upsample = nn.Upsample(
            output_size[1:], mode=upsampling_mode, align_corners=True
        )

    def forward(self, x):

        for idx, layer in enumerate(self.linear_layers):
            x = layer(x)
            if self.log:
                print(f"Decoder_Linear_{idx} layer output shape:", x.shape)

        x = x.reshape(
            -1,
            self.input_size[1],
            self.input_size[2],
            self.input_size[3],
            self.input_size[4],
        )
        for idx, layer in enumerate(self.conv_transpose_layers):
            x = layer(x)
            if self.log: print(f"Decoder_Conv_{idx} layer output shape:", x.shape)

        x = self.upsample(x)
        if self.log: print("Upsample layer output shape:", x.shape)

        return x

    def make_linear_layer(self, layer) -> nn.Module:
        """Make a linear layer with batch normalization and activation function

        Args:
            layer (dict): layer configuration

        Returns:
            nn.Module: linear layer
        """
        return nn.Sequential(
            nn.Linear(
                in_features=layer["in_features"], out_features=layer["out_features"]
            ),
            nn.BatchNorm1d(layer["out_features"]),
            Activation(layer["activation"]),
        )

    def make_conv_transpose_layer(self, layer) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=layer["in_channels"],
                out_channels=layer["out_channels"],
                kernel_size=layer["kernel_size"],
                stride=layer["stride"],
                padding=layer["padding"],
            ),
            nn.BatchNorm3d(layer["out_channels"]),
            Activation(layer["activation"]),
        )