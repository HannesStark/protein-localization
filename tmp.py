from models import *
model = FirstAttention()



for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.numel())
