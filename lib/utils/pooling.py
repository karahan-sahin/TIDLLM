import torch
import torch.nn as nn
import torch.nn.functional as F        

Pooling = {
    'max' : nn.MaxPool3d,
    'avg' : nn.AvgPool3d,
    'adaptive_max' : nn.AdaptiveAvgPool3d,
    'adaptive_avg' : nn.AdaptiveAvgPool3d,
    'max_unpool' : nn.MaxUnpool3d,
}
    
