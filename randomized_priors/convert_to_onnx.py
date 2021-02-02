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
import mnist_randomized_priors
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
print(list(saved_models.keys()))
for model_name in saved_models.keys():
    if not re.match(r'classifier_\d+', model_name):
        continue

    trainable_model = TrainableNetwork()
    trainable_model.to(dev)

    prior_model = PriorNetwork()
    prior_model.to(dev)

    cls = RandomizedPriorNetwork(prior_model, trainable_model, beta=mnist_randomized_priors.BETA)
    cls.load_state_dict(saved_models[model_name])
    classifiers.append(cls)

    # Export the prior and posterior
    torch.onnx.export(cls.prior_net,               # model being run
                      x.to(dev),  # model input (or a tuple for multiple inputs)
                      "models/onnx/" + model_name + "_prior.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}})

    torch.onnx.export(cls,  # model being run
                      x.to(dev),  # model input (or a tuple for multiple inputs)
                      "models/onnx/" + model_name + "_posterior.onnx",
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

classifiers.to(dev)
combined_model = CombinedPosteriorNetwork(classifiers)
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


voting_model = VotingNetwork(classifiers)
voting_model.to(dev)

# Export the prior
torch.onnx.export(voting_model,               # model being run
                  x.to(dev),                         # model input (or a tuple for multiple inputs)
                  "models/onnx/voting_classifier.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})


standard_model = TrainableNetwork()
standard_model.load_state_dict(saved_models['standard_classifier'])
standard_model.to(dev)
standard_model = torch.nn.Sequential(standard_model, torch.nn.Softmax(dim=1))
torch.onnx.export(standard_model,               # model being run
                  x.to(dev),                         # model input (or a tuple for multiple inputs)
                  "models/onnx/standard_classifier.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
