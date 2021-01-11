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

EPOCHS = 20
VAL_PERCENTAGE = 25
LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 32
BOOTSTRAP_PERCENTAGE = 90
NUM_CLASSIFIERS = 10
LOAD_MODEL = True
PATH = "models/bootstrapped_random_priors_CNN_MNIST.pt"


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class RandomizedPriorNetwork(nn.Module):
    def __init__(self, prior_net, trainable_net, beta=1.0):
        super().__init__()
        self.prior_net = prior_net
        for param in self.prior_net.parameters():
            param.requires_grad = False
        self.trainable_net = trainable_net
        self.beta = beta

    def forward(self, x):
        return torch.add(
            self.trainable_net(x),
            torch.mul(
                self.prior_net(x),
                self.beta))


class VotingNetwork(nn.Module):
    def __init__(self, networks):
        super().__init__()
        self.networks = networks

    def forward(self, x):
        votes = torch.cat([torch.argmax(n(x), dim=1, keepdim=True) for n in self.networks], dim=1)
        return votes


class CombinedPosteriorNetwork(nn.Module):
    def __init__(self, networks):
        super().__init__()
        self.networks = networks

    def forward(self, x):
        sm = [torch.softmax(n(x), dim=1) for n in self.networks]
        cat = torch.stack(sm, dim=2)
        res = torch.sum(cat, dim=2)
        return res


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, device, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
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

        print(epoch, val_loss)


def train_bootstrapped_random_prior_classifier(
        epochs,
        prior_net,
        trainable_net,
        device,
        loss,
        train_dataset,
        val_dataset,
        beta=1.0):
    model = RandomizedPriorNetwork(prior_net, trainable_net, beta=beta)
    model.to(device)
    # opt = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    opt = AdaBelief(model.parameters(), lr=1e-3, eps=1e-8, betas=(0.9, 0.999), weight_decouple=True,
                          rectify=False, print_change_log=False)
    train_subset = torch.utils.data.Subset(train_dataset,
                                           indices=np.random.choice(
                                               np.arange(0, len(train_dataset)),
                                               size=int(round(BOOTSTRAP_PERCENTAGE / 100.0 * len(train_dataset))),
                                               replace=False))
    train_loader = torch.utils.data.DataLoader(train_subset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    fit(epochs, model, device, loss, opt, train_loader, val_loader)
    return model

def predict_bootstrapped_random_prior_classifier(models, batch):
    for m in models:
        m.eval()
    votes = torch.cat([torch.argmax(m(batch), dim=1, keepdim=True) for m in models], dim=1)
    return torch.mode(votes, dim=1).values


def display_incorrect_classifications(models, device, test_loader, display_all=False, figure_limit=None):
    for m in models:
        m.eval()
    confidence_correct = []
    confidence_incorrect = []
    figs = 0
    for xb, yb in test_loader:
        xb = xb.to(device)
        votes = torch.cat([torch.argmax(m(xb), dim=1, keepdim=True) for m in models], dim=1).cpu()
        modes = torch.mode(votes, dim=1)
        preds = modes.values.cpu()
        for i in range(yb.size(0)):
            uniq, inv, counts = torch.unique(votes[i, :], return_counts=True, return_inverse=True)
            mode_count = counts[inv[modes.indices[i]]]
            if preds[i] != yb[i]:
                if display_all and (figure_limit is None or figs < figure_limit):
                    plt.clf()
                    plt.subplot(1, 2, 1)
                    plt.title(f'Predicted: {preds[i].item()} Actual: {yb[i].item()}')
                    plt.imshow(xb[i, :, :].cpu().detach().numpy().reshape((xb.size(2), xb.size(3))))
                    plt.subplot(1, 2, 2)
                    n, bins, patches = plt.hist(votes[i, :].detach().numpy().ravel(), bins=10, label="Votes", rwidth=0.5, range=(0, 9))
                    plt.xticks(np.array(bins[:-1]) + 0.5, [str(n) for n in np.arange(0, 10)])
                    plt.show(block=True)
                    figs += 1
                confidence_incorrect.append(torch.div(mode_count.float(), torch.sum(counts)).item())
            else:
                confidence_correct.append(torch.div(mode_count.float(),torch.sum(counts)).item())
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.title(f'Confidences in correct guesses')
    n, bins, patches = plt.hist(confidence_correct, bins=10, label="Confidences", range=(0.0, 1.0))
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.title(f'Confidences in incorrect guesses')
    n, bins, patches = plt.hist(confidence_incorrect, bins=10, label="Confidences", range=(0.0, 1.0))
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.show(block=True)


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

    classifiers = nn.ModuleList()
    if not LOAD_MODEL:
        for i in range(NUM_CLASSIFIERS):
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

            prior_model = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(16, 10, kernel_size=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            for layer in prior_model:
                if hasattr(layer, 'weight'):
                    torch.nn.init.xavier_normal_(layer.weight)
                if hasattr(layer, 'bias'):
                    torch.nn.init.normal_(layer.bias)

            cls = train_bootstrapped_random_prior_classifier(
                EPOCHS,
                prior_model,
                trainable_model,
                dev,
                loss,
                train_dataset,
                val_dataset
            )
            classifiers.append(cls)
            print(f'Trained classifier {i}', flush=True)
        torch.save(
            {'classifier_{}'.format(i): cls.state_dict() for i, cls in enumerate(classifiers)},
            PATH)
    else:
        saved_models = torch.load(PATH, map_location=dev)
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

            prior_model = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(16, 10, kernel_size=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )

            cls = RandomizedPriorNetwork(prior_model, trainable_model)
            cls.load_state_dict(saved_models[model_name])
            cls.to(dev)
            classifiers.append(cls)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    voting_classifier = nn.Sequential(
            VotingNetwork(classifiers),
            Lambda(lambda x: torch.mode(x, dim=1).values)
        )
    voting_classifier.to(dev)

    with torch.no_grad():
        losses, nums = zip(
            *[(torch.sum(
               torch.eq(voting_classifier(xb.to(dev)).cpu(), yb).long()),
               yb.size(0)) for xb, yb in test_loader]
        )
    test_loss = np.sum(losses) / np.sum(nums)
    print(f'Voting test accuracy (post-train): {test_loss * 100.0}% ({np.sum(losses)}/{np.sum(nums)})')

    display_incorrect_classifications(classifiers, dev, test_loader, display_all=True, figure_limit=10)

    combined_classifier = CombinedPosteriorNetwork(classifiers)
    combined_classifier.to(dev)

    with torch.no_grad():
        losses, nums = zip(
            *[(torch.sum(
                torch.eq(torch.argmax(combined_classifier(xb.to(dev)).cpu(), dim=1), yb).long()),
               yb.size(0)) for xb, yb in test_loader]
        )
    test_loss = np.sum(losses) / np.sum(nums)
    print(f'Combined test accuracy (post-train): {test_loss * 100.0}% ({np.sum(losses)}/{np.sum(nums)})')
