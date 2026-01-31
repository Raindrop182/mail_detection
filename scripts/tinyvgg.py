import torch.nn as nn
import torch

def conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

class TinyVGG(nn.Module):
    """
  Model architecture copying TinyVGG from: 
  https://poloclub.github.io/cnn-explainer/

  Args:
    input_shape (int): Number of input channels.
    hidden_units (int): Number of channels produced by the convolutions
    output_shape (int): Number of classes.

  Input:
    x: Tensor of shape (batch_size, input_shape, 28, 28)
    """
    def __init__(self, input_shape: int, hidden_units:int, output_shape:int):
        super().__init__()        
        self.block_1 = conv_block(input_shape, hidden_units)
        self.block_2 = conv_block(hidden_units, hidden_units)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape))
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x
