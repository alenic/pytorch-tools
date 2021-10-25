import pytorchtools as pt
import torch
import timm

model = timm.create_model("resnet50", pretrained=False)

monitor = pt.ForwardMonitor(model, verbose=False)
monitor.add_layer("global_pool.flatten", alias="features")

input = torch.rand((8, 3, 224, 224))
output = model(input)

print(monitor.get_layer("features").size())

"""
output: torch.Size([8, 2048])
"""