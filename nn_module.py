from turtle import forward
import torch
import torch.nn as nn


class Satuo(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, input):
        output = input + 1 
        return output

Satuo = Satuo()
x = torch.tensor(1.0)
output = Satuo(x)
print(output)