import re
import sys
import torch
import torchvision
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import torch.utils.data
import numpy as np
import copy
import torch.functional as F
from adabelief_pytorch import AdaBelief
import matplotlib
from matplotlib import pyplot as plt
from mnist_randomized_priors import *
plt.rcParams["figure.figsize"] = (8,6)

PATH = sys.argv[1]
BATCH_SIZE = 1
# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dev = torch.device("cpu")
x = torch.randn(BATCH_SIZE, 1, 28, 28, requires_grad=False)
x.to(dev)

print(f'Running on {dev}')
saved_models = torch.load(PATH, map_location=dev)
classifiers = nn.ModuleList()
for model_name in saved_models.keys():
    if not re.match(r'classifier_\d+', model_name):
        continue

    trainable_model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    trainable_model.to(dev)

    prior_model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    prior_model.to(dev)

    cls = RandomizedPriorNetwork(prior_model, trainable_model)
    cls.load_state_dict(saved_models[model_name])
    classifiers.append(cls)

    # Export the prior
    # torch.onnx.export(cls.prior_net,               # model being run
    #                   x.to(dev),  # model input (or a tuple for multiple inputs)
    #                   "models/onnx/" + model_name + "_prior.onnx",   # where to save the model (can be a file or file-like object)
    #                   export_params=True,        # store the trained parameter weights inside the model file
    #                   opset_version=10,          # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names = ['input'],   # the model's input names
    #                   output_names = ['output'], # the model's output names
    #                   dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
    #                                 'output' : {0 : 'batch_size'}})

classifiers.to(dev)
combined_model = VotingNetwork(classifiers)
combined_model.to(dev)

# Export the prior
torch.onnx.export(combined_model,               # model being run
                  x.to(dev),                         # model input (or a tuple for multiple inputs)
                  "models/onnx/combined_classifier.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
