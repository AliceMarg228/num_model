import torch
from torch import nn

HIDDEN_UNITS = 100


class LetterModelV0(nn.Module):
  def __init__(self, class_number=10):
    super().__init__()
    self.conv_layer_1 = nn.Sequential(
        nn.Conv2d(in_channels=3,
                  out_channels=HIDDEN_UNITS,
                  kernel_size=3)
        ,
        nn.ReLU()
        ,
        nn.Conv2d(in_channels=HIDDEN_UNITS,
                  out_channels=HIDDEN_UNITS,
                  kernel_size=3)
        ,
        nn.ReLU()
        ,
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_layer_2 = nn.Sequential(
        nn.Conv2d(in_channels=HIDDEN_UNITS,
                  out_channels=HIDDEN_UNITS,
                  kernel_size=3)
        ,
        nn.ReLU()
        ,
        nn.Conv2d(in_channels=HIDDEN_UNITS,
                  out_channels=HIDDEN_UNITS,
                  kernel_size=3)
        ,
        nn.ReLU()
        ,
        nn.MaxPool2d(kernel_size=2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten()
        ,
        nn.Linear(in_features=HIDDEN_UNITS*2209,
                  out_features=class_number)
    )
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classifier(self.conv_layer_2(self.conv_layer_1(x)))
