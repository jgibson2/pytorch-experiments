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

# TODO save at specific checkpoints

EPOCHS = 25
CHECKPOINTS = [2, 25]
VAL_PERCENTAGE = 25
LEARNING_RATE = 1E-2
BATCH_SIZE = 32
BOOTSTRAP_PERCENTAGE = 90
NUM_CLASSIFIERS = 12
BETA = 10.0
LOAD_MODEL = False
BASE_PATH = "models/bootstrapped_random_priors_CNN_MNIST"
SEED = 12345

torch.manual_seed(SEED)
np.random.seed(SEED)

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


class PriorNetwork(nn.Sequential):
    def __init__(self):
        super(PriorNetwork, self).__init__()
        self.add_module('conv1', nn.Conv2d(1, 16, kernel_size=3))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv2', nn.Conv2d(16, 10, kernel_size=3))
        self.add_module('relu2', nn.ReLU())
        self.add_module('avgpool1', nn.AdaptiveAvgPool2d(1))
        self.add_module('flatten1', nn.Flatten())
        # self.add_module('softmax1', nn.Softmax(dim=1))


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
        # sm = [n(x) for n in self.networks]
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


def fit(epochs, checkpoints, model, device, loss_func, opt, train_dl, valid_dl):
    chks = {}
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

        print(epoch, val_loss)

        if epoch in set(checkpoints):
            chks[epoch] = copy.deepcopy(model.state_dict())
    return chks


def train_bootstrapped_random_prior_classifier(
        epochs,
        checkpoints,
        prior_net,
        trainable_net,
        device,
        loss,
        train_dataset,
        val_dataset,
        beta=1.0):
    model = RandomizedPriorNetwork(prior_net, trainable_net, beta=beta)
    model.to(device)
    opt = AdaBelief(model.parameters(), lr=LEARNING_RATE, eps=1e-8, betas=(0.9, 0.999), weight_decouple=True,
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
    chkpts = fit(epochs, checkpoints, model, device, loss, opt, train_loader, val_loader)
    return model, chkpts

def train_standard_classifier(
        epochs,
        checkpoints,
        trainable_net,
        device,
        loss,
        train_dataset,
        val_dataset):
    trainable_net.to(device)
    opt = AdaBelief(trainable_net.parameters(), lr=LEARNING_RATE, eps=1e-8, betas=(0.9, 0.999), weight_decouple=True,
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
    chkpts = fit(epochs, checkpoints, trainable_net, device, loss, opt, train_loader, val_loader)
    return trainable_net, chkpts


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

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    classifiers = nn.ModuleList()
    standard_cls, standard_checkpoints = None, []
    if not LOAD_MODEL:
        checkpts = []
        standard_trainable_model = TrainableNetwork()
        standard_cls, standard_checkpoints = train_standard_classifier(
            EPOCHS,
            CHECKPOINTS,
            standard_trainable_model,
            dev,
            loss,
            train_dataset,
            val_dataset
        )
        for i in range(NUM_CLASSIFIERS):
            trainable_model = TrainableNetwork()
            prior_model = PriorNetwork()
            for layer in prior_model:
                if hasattr(layer, 'weight'):
                    torch.nn.init.xavier_normal_(layer.weight)
                if hasattr(layer, 'bias'):
                    torch.nn.init.normal_(layer.bias)

            cls, chks = train_bootstrapped_random_prior_classifier(
                EPOCHS,
                CHECKPOINTS,
                prior_model,
                trainable_model,
                dev,
                loss,
                train_dataset,
                val_dataset,
                beta=BETA
            )
            classifiers.append(cls)
            checkpts.append(chks)
            print(f'Trained classifier {i}', flush=True)
        states = {'classifier_{}'.format(i): cls.state_dict() for i, cls in enumerate(classifiers)}
        states['standard_classifier'] = standard_cls.state_dict()
        torch.save(
            states,
            BASE_PATH + ".pt")
        for ep in CHECKPOINTS:
            ch_states = {'classifier_{}'.format(i): chpt[ep] for i, chpt in enumerate(checkpts)}
            ch_states['standard_classifier'] = standard_checkpoints[ep]
            torch.save(
                ch_states,
                BASE_PATH + f'_EPOCH_{ep}.pt')
    else:
        saved_models = torch.load(BASE_PATH + ".pt", map_location=dev)
        for model_name in saved_models.keys():
            if not re.match(r'^classifier_\d+$', model_name, re.MULTILINE):
                continue

            trainable_model = TrainableNetwork()
            prior_model = PriorNetwork()
            cls = RandomizedPriorNetwork(prior_model, trainable_model, beta=BETA)
            cls.load_state_dict(saved_models[model_name])
            cls.to(dev)
            classifiers.append(cls)
        standard_cls = TrainableNetwork()
        standard_cls.load_state_dict(saved_models['standard_classifier'])
        standard_cls.to(dev)

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
                torch.eq(torch.argmax(standard_cls(xb.to(dev)), dim=1).cpu(), yb).long()),
               yb.size(0)) for xb, yb in test_loader]
        )
    test_loss = np.sum(losses) / np.sum(nums)
    print(f'Standard classifier accuracy (post-train): {test_loss * 100.0}% ({np.sum(losses)}/{np.sum(nums)})')

    with torch.no_grad():
        losses, nums = zip(
            *[(torch.sum(
               torch.eq(voting_classifier(xb.to(dev)).cpu(), yb).long()),
               yb.size(0)) for xb, yb in test_loader]
        )
    test_loss = np.sum(losses) / np.sum(nums)
    print(f'Voting test accuracy (post-train): {test_loss * 100.0}% ({np.sum(losses)}/{np.sum(nums)})')

    # display_incorrect_classifications(classifiers, dev, test_loader, display_all=True, figure_limit=10)

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
