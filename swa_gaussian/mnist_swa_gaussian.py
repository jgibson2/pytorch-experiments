import functools
import operator
import re

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import copy
import torch.functional as F
from adabelief_pytorch import AdaBelief
import matplotlib
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (8,6)

TRAIN_EPOCHS = 100
VAL_PERCENTAGE = 25
LEARNING_RATE = 1E-2
SWA_LEARNING_RATE = 1E-2
SWA_TRAIN_EPOCHS = 25
SWA_TEST_DRAWS = 12
SWA_K = 10
BATCH_SIZE = 1024
BOOTSTRAP_PERCENTAGE = 0.9
NUM_CLASSIFIERS = 12
LOAD_MODEL = True
BASE_PATH = "models/mnist_swa_gaussian"
SEED = 12345
EPS = 1e-5

torch.manual_seed(SEED)
np.random.seed(SEED)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class TrainableNetwork(nn.Sequential):
    def __init__(self):
        super(TrainableNetwork, self).__init__()
        self.add_module('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv2', nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv3', nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1))
        self.add_module('relu3', nn.ReLU())
        self.add_module('avgpool1', nn.AdaptiveAvgPool2d(1))
        self.add_module('flatten1', nn.Flatten())
        # self.add_module('softmax1', nn.Softmax(dim=1))


class CombinedPosteriorNetwork(nn.Module):
    def __init__(self, networks):
        super().__init__()
        self.networks = networks

    def forward(self, x):
        # sm = [torch.softmax(n(x), dim=1) for n in self.networks]
        sm = [n(x) for n in self.networks]
        cat = torch.stack(sm, dim=2)
        res = torch.sum(cat, dim=2)
        return torch.div(res, len(self.networks))


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, checkpoints, model, device, loss_func, opt, train_dl, valid_dl, burnin=10, patience=5):
    chks = {}
    val_min = np.inf
    its_no_improvement = 0
    best_params = None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb.to(device), yb.to(device)) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(f'\t\tEpoch: {epoch} Validation loss: {val_loss.item()}')
        if val_loss < val_min or val_min == np.inf:
            its_no_improvement = 0
            if val_loss.item() < val_min:
                val_min = val_loss.item()
                best_params = copy.deepcopy(model.state_dict())
        elif epoch >= burnin:
            its_no_improvement += 1
            if its_no_improvement > patience:
                print(f'Validation loss: {val_loss.item()}')
                print(f'Stopped after {epoch} epochs')
                break

    model.load_state_dict(best_params)
    return model


def train_standard_classifier(
        epochs,
        trainable_net,
        device,
        loss,
        train_dataset,
        val_dataset):
    trainable_net.to(device)
    opt = AdaBelief(trainable_net.parameters(), lr=LEARNING_RATE, eps=1e-8, betas=(0.9, 0.999), weight_decouple=True,
                          rectify=False, print_change_log=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    trainable_net = fit(epochs, [], trainable_net, device, loss, opt, train_loader, val_loader)
    return trainable_net


def train_swa_gaussian_parameters(
        epochs,
        trainable_net,
        device,
        loss,
        train_dataset
):
    opt = optim.SGD(trainable_net.parameters(), lr=SWA_LEARNING_RATE)
    train_subset = torch.utils.data.Subset(train_dataset,
                                           indices=np.random.choice(
                                               np.arange(0, len(train_dataset)),
                                               size=int(round(BOOTSTRAP_PERCENTAGE / 100.0 * len(train_dataset))),
                                               replace=False))
    train_loader = torch.utils.data.DataLoader(train_subset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               pin_memory=True)
    weights = []
    weights.append(np.hstack([l.weight.detach().cpu().numpy().ravel() for l in trainable_net if hasattr(l, 'weight')]))
    for i in range(epochs):
        trainable_net.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            ls = loss(trainable_net(xb), yb)
            ls.backward()
            opt.step()
        weights.append(np.hstack([l.weight.detach().cpu().numpy().ravel() for l in trainable_net if hasattr(l, 'weight')]))
    weights = np.vstack(weights)
    mean_weights = np.mean(weights, axis=0)
    sigma_diag_weights = np.maximum(np.mean(np.square(weights), axis=0) - np.square(mean_weights), 0)
    D = weights[-SWA_K:, :] - mean_weights[np.newaxis, :]
    return mean_weights, sigma_diag_weights, D


def predict_swa_gaussian_classifier(model, batch, train_dataset, mean_weights, sigma_diag_weights, D, return_all=False):
    outputs = []
    C = np.linalg.cholesky(np.diag((0.5 * sigma_diag_weights + EPS)) + ((D.T @ D) / (2 * (SWA_K - 1))))
    for i in range(SWA_TEST_DRAWS):
        weights = mean_weights + (np.random.normal(size=mean_weights.shape) @ C)
        n = 0
        with torch.no_grad():
            model.to(torch.device('cpu'))
            for layer in model:
                if hasattr(layer, 'weight'):
                    layer_size = layer.weight.size()
                    layer.weight = torch.nn.Parameter(torch.from_numpy(weights[n:n+layer_size.numel()].reshape(tuple(layer_size))).float())
                    n += layer_size.numel()
            model.to(batch.device)
            model.train()
            train_subset = torch.utils.data.Subset(train_dataset,
                                                   indices=np.random.choice(
                                                       np.arange(0, len(train_dataset)),
                                                       size=int(
                                                           round(BOOTSTRAP_PERCENTAGE / 100.0 * len(train_dataset))),
                                                       replace=False))
            train_loader = torch.utils.data.DataLoader(train_subset,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True,
                                                       pin_memory=True)
            for xb, _ in train_loader:
                # update the batch statistics by making a forward pass,
                # but not calling backward
                # TODO: update these directly
                model(xb.to(batch.device))
            model.to(batch.device)
            model.eval()
            outputs.append(model(batch))
    if return_all:
        return outputs
    return functools.reduce(operator.add, outputs) / SWA_TEST_DRAWS


def build_swa_gaussian_classifier(model, dev, train_dataset, mean_weights, sigma_diag_weights, D):
    C = np.linalg.cholesky(np.diag((0.5 * sigma_diag_weights + EPS)) + ((D.T @ D) / (2 * (SWA_K - 1))))
    weights = mean_weights + (np.random.normal(size=mean_weights.shape) @ C)
    n = 0
    with torch.no_grad():
        model.to(torch.device('cpu'))
        for layer in model:
            if hasattr(layer, 'weight'):
                layer_size = layer.weight.size()
                layer.weight = torch.nn.Parameter(torch.from_numpy(weights[n:n+layer_size.numel()].reshape(tuple(layer_size))).float())
                n += layer_size.numel()
        model.to(dev)
        model.train()
        train_subset = torch.utils.data.Subset(train_dataset,
                                               indices=np.random.choice(
                                                   np.arange(0, len(train_dataset)),
                                                   size=int(
                                                       round(BOOTSTRAP_PERCENTAGE / 100.0 * len(train_dataset))),
                                                   replace=False))
        train_loader = torch.utils.data.DataLoader(train_subset,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   pin_memory=True)
        for xb, _ in train_loader:
            # update the batch statistics by making a forward pass,
            # but not calling backward
            # TODO: update these directly
            model(xb.to(dev))
        model.eval()
    return copy.deepcopy(model)



def display_incorrect_classifications(models, device, test_loader, display_all=False, figure_limit=None):
    confidence_correct = []
    confidence_incorrect = []
    for m in models:
        m.eval()
    figs = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = functools.reduce(operator.add, [m(xb) for m in models]).detach().cpu().numpy() / len(models)
            for i in range(yb.size(0)):
                if np.argmax(preds[i, :]) != yb[i]:
                    if display_all and (figure_limit is None or figs < figure_limit):
                        plt.clf()
                        plt.subplot(1, 2, 1)
                        plt.title(f'Predicted: {np.argmax(preds[i, :])} Actual: {yb[i].item()}')
                        plt.imshow(xb[i, :, :].cpu().detach().numpy().reshape((xb.size(2), xb.size(3))))
                        plt.subplot(1, 2, 2)
                        plt.bar(np.arange(0, 10) + 0.5, preds[i, :], tick_label=[str(n) for n in np.arange(0, 10)])
                        plt.show(block=True)
                        figs += 1


if __name__ == '__main__':
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Running on {dev}')
    loss = nn.CrossEntropyLoss()

    train_dataset = torchvision.datasets.MNIST('data', train=True, download=True,
                                               transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST('data', train=False, download=True,
                                              transform=torchvision.transforms.ToTensor())
    train_val_set = torch.utils.data.random_split(
        train_dataset,
        [
            len(train_dataset) - int(((VAL_PERCENTAGE / 100) * len(train_dataset))),
            int((VAL_PERCENTAGE / 100) * len(train_dataset))
        ])
    train_dataset, val_dataset = train_val_set[0], train_val_set[1]

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    standard_cls = None
    if not LOAD_MODEL:
        checkpts = []
        standard_trainable_model = TrainableNetwork()
        standard_cls = train_standard_classifier(
            TRAIN_EPOCHS,
            standard_trainable_model,
            dev,
            loss,
            train_dataset,
            val_dataset
        )
        # torch.save(
        #     standard_cls.state_dict(),
        #     BASE_PATH + ".pt")
    else:
        saved_model = torch.load(BASE_PATH + ".pt", map_location=dev)
        standard_cls = TrainableNetwork()
        standard_cls.load_state_dict(saved_model['standard_classifier'])
        standard_cls.to(dev)

    with torch.no_grad():
        losses, nums = zip(
            *[(torch.sum(
                torch.eq(torch.argmax(standard_cls(xb.to(dev)), dim=1).cpu(), yb).long()),
               yb.size(0)) for xb, yb in test_loader]
        )
    test_loss = np.sum(losses) / np.sum(nums)
    print(f'Standard classifier accuracy (post-train): {test_loss * 100.0}% ({np.sum(losses)}/{np.sum(nums)})')

    mw, sw, D = train_swa_gaussian_parameters(SWA_TRAIN_EPOCHS, standard_cls, dev, loss, train_dataset)
    models = [build_swa_gaussian_classifier(standard_cls, dev, train_dataset, mw, sw, D) for _ in range(SWA_TEST_DRAWS)]
    combined_model = CombinedPosteriorNetwork(models)
    combined_model.to(dev)

    with torch.no_grad():
        losses, nums = zip(
            *[(torch.sum(
                torch.eq(torch.argmax(combined_model(xb.to(dev)).cpu(), dim=1), yb).long()),
               yb.size(0)) for xb, yb in test_loader]
        )
    test_loss = np.sum(losses) / np.sum(nums)
    print(f'Combined test accuracy (post-train): {test_loss * 100.0}% ({np.sum(losses)}/{np.sum(nums)})')

    states = {'classifier_{}'.format(i): cls.state_dict() for i, cls in enumerate(models)}
    states['standard_classifier'] = standard_cls.state_dict()
    torch.save(
        states,
        BASE_PATH + ".pt")
    # display_incorrect_classifications(models, dev, test_loader, display_all=True, figure_limit=10)
