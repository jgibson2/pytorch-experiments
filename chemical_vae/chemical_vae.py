import functools
import os
import sys
import time

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import copy
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import ShowMol
from torch import nn
import torch.utils.data
import selfies as sf

LEARNING_RATE = 1e-3
EPOCHS = 200
VAL_PERCENTAGE = 20
BATCH_SIZE = 2
EMBEDDING_DIM = 50
HIDDEN_DIMS = (250, 200, 100)
KL_WEIGHT = 0.05


class SMILESDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename, skip_header=True):
        with open(filename, 'r') as f:
            self.selfies = [sf.encoder(l.strip()) for l in f.readlines()[1 if skip_header else 0:]]
            self.max_selfies_length = max([sf.len_selfies(s) for s in self.selfies])
            alphabet = sf.get_alphabet_from_selfies(self.selfies)
            alphabet.add('[nop]')  # '[nop]' is a special padding symbol
            alphabet = list(sorted(alphabet))
            self.feature_vector_length = len(alphabet)
            self.symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
            self.idx_to_symbol = {i: s for i, s in enumerate(alphabet)}

    def __iter__(self):
        return iter(map(lambda s: torch.Tensor(sf.selfies_to_encoding(s,
                                                                                    vocab_stoi=self.symbol_to_idx,
                                                                                    pad_to_len=self.max_selfies_length,
                                                                                    enc_type='one_hot')),
                        self.selfies))

    def __getitem__(self, idx):
        return torch.Tensor(sf.selfies_to_encoding(self.selfies[idx],
                                                                 vocab_stoi=self.symbol_to_idx,
                                                                 pad_to_len=self.max_selfies_length,
                                                                 enc_type='one_hot'))

    def __len__(self):
        return len(self.selfies)


class LargeFeatureEncoder(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim=4, hidden_dims=(50, 25, 10)):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        super(LargeFeatureEncoder, self).__init__()
        self.add_module('linearIn', torch.nn.Linear(input_dim, hidden_dims[0]))
        self.add_module('batchnormIn', torch.nn.BatchNorm1d(hidden_dims[0]))
        self.add_module('reluIn', torch.nn.LeakyReLU())
        for i in range(len(hidden_dims) - 1):
            self.add_module(f'linear{i}', torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.add_module(f'batchnorm{i}', torch.nn.BatchNorm1d(hidden_dims[i + 1]))
            self.add_module(f'relu{i}', torch.nn.LeakyReLU())
        self.add_module('linearOut', torch.nn.Linear(hidden_dims[-1], output_dim))

        for layer in self:
            if layer is torch.nn.Linear:
                if hasattr(layer, 'weight'):
                    torch.nn.init.xavier_normal_(layer.weight)
                if hasattr(layer, 'bias'):
                    torch.nn.init.normal_(layer.bias)


class LargeFeatureDecoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=50, num_layers=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        super(LargeFeatureDecoder, self).__init__()

        # Simple Decoder
        self.decode_RNN = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True)

        self.decode_FC = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, batch_size,
                                self.hidden_dim)

    def forward(self, z, hidden):
        """
        A forward pass throught the entire model.
        """

        # Decode
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)  # fully connected layer

        return decoded, hidden


class LargeFeatureVAE(torch.nn.Module):
    def __init__(self, input_dim, seq_length, output_dim, hidden_dims=(100, 50, 25, 10)):
        super(LargeFeatureVAE, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.encoder = LargeFeatureEncoder(input_dim, hidden_dims[-1], hidden_dims=hidden_dims[:-1])
        self.decoder = LargeFeatureDecoder(output_dim, input_dim // seq_length, hidden_dim=hidden_dims[-1])

        self.fc_mu = torch.nn.Linear(hidden_dims[-1], output_dim)
        torch.nn.init.xavier_normal_(self.fc_mu.weight)
        torch.nn.init.normal_(self.fc_mu.bias)
        self.fc_logvar = torch.nn.Linear(hidden_dims[-1], output_dim)
        torch.nn.init.xavier_normal_(self.fc_logvar.weight)
        torch.nn.init.normal_(self.fc_logvar.bias)

    def encode(self, x):
        h1 = F.relu(self.encoder(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        hidden = self.decoder.init_hidden(z.size(0))
        z = z.unsqueeze(1)
        out = torch.zeros((z.size(0), self.seq_length, self.input_dim // self.seq_length)).to(z.device)
        for i in range(self.seq_length):
            q, hidden = self.decoder(z, hidden)
            q = torch.argmax(q, dim=2).squeeze()
            q = F.one_hot(q, self.input_dim // self.seq_length)
            out[:, i, :] = q
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def calculate_loss(self, out, mu, logvar, inputs):
        target = torch.argmax(inputs, dim=2)
        # TODO: Fix loss (reshape, get labels)
        recon_loss = F.cross_entropy(out.reshape(inputs.size()).transpose(1,2), target)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        loss = recon_loss + (KL_WEIGHT * kl_loss)
        return loss


def train_vae(train_dataloader, val_dataloader, model,
              patience=5, burnin=10, fname='models/vae.pt', train=True, dataset=None):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if train:
        model = model.to(dev)
        best_params = None
        opt = AdaBelief(
            model.parameters(),
            lr=LEARNING_RATE,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decouple=False,
            weight_decay=5e-4,
            rectify=False,
            fixed_decay=False,
            amsgrad=False,
            print_change_log=False)
        val_min = np.inf
        its_no_improvement = 0
        mol = None
        for i in range(EPOCHS):
            model.train()
            for batch_x in train_dataloader:
                batch_x = batch_x.to(dev)
                inp = torch.flatten(batch_x, 1)
                opt.zero_grad()
                out, mu, logvar = model(inp)
                loss = model.calculate_loss(out, mu, logvar, batch_x)
                loss.backward()
                opt.step()
            model.eval()
            val_loss = torch.Tensor([0])
            for batch_x in val_dataloader:
                batch_x = batch_x.to(dev)
                inp = torch.flatten(batch_x, 1)
                _, val_mu, val_logvar = model(inp)
                val_out = model.decode(val_mu)
                val_loss += model.calculate_loss(val_out, val_mu, val_logvar, batch_x).detach().cpu() * batch_x.size(0)
                mol = batch_x[0].detach()
            val_loss /= len(val_dataloader.dataset)
            print(f'\t\tValidation loss: {val_loss.item()}')
            if val_loss.item() < val_min or val_min == np.inf:
                its_no_improvement = 0
                if val_loss.item() < val_min:
                    val_min = val_loss.item()
                    best_params = copy.deepcopy(model.state_dict())
            elif i >= burnin:
                its_no_improvement += 1
                if its_no_improvement > patience:
                    print(f'Validation loss: {val_loss.item()}')
                    print(f'Stopped after {i} epochs')
                    break
            if dataset:
                inp = torch.argmax(mol, dim=1).cpu().numpy().ravel()
                ShowMol(MolFromSmiles(sf.decoder(
                    sf.encoding_to_selfies(inp, dataset.idx_to_symbol,
                                           enc_type='label'))))
                print(f'Input: {inp}')
                enc, logvar = model.encode(torch.unsqueeze(torch.flatten(mol), 0))
                print(f'Encoded: {enc.cpu().detach().numpy()}')
                dec = model.decode(enc)
                dec = torch.argmax(dec.reshape((dataset.max_selfies_length, dataset.feature_vector_length)),
                                   dim=1).cpu().numpy().ravel()
                print(f'Decoded: {dec}')
                ShowMol(MolFromSmiles(sf.decoder(
                    sf.encoding_to_selfies(dec, dataset.idx_to_symbol,
                                           enc_type='label'))))

        model.load_state_dict(best_params)
        torch.save(model.state_dict(), fname)
    else:
        model.load_state_dict(torch.load(fname))

    return model


def main():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = SMILESDataset('data/250K_ZINC.txt')
    print('Max SELFIES Length: {}'.format(dataset.max_selfies_length))
    print('Feature Vector Length: {}'.format(dataset.feature_vector_length))
    train_val_set = torch.utils.data.random_split(
        dataset,
        [
            len(dataset) - int(((VAL_PERCENTAGE / 100) * len(dataset))),
            int((VAL_PERCENTAGE / 100) * len(dataset))
        ])
    train_dataset, val_dataset = train_val_set[0], train_val_set[1]

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    vae = LargeFeatureVAE(dataset.max_selfies_length * dataset.feature_vector_length, dataset.max_selfies_length, EMBEDDING_DIM,
                          hidden_dims=HIDDEN_DIMS)
    vae.to(dev)

    vae = train_vae(train_loader, val_loader, vae, train=True, fname='models/vae.pt', patience=10, burnin=50, dataset=None)

    vae.eval()
    for mol in val_loader:
        input = torch.argmax(mol[0], dim=1).cpu().numpy().ravel()
        ShowMol(MolFromSmiles(sf.decoder(
            sf.encoding_to_selfies(input, dataset.idx_to_symbol,
                                   enc_type='label'))))
        print(f'Input: {input}')
        mol = mol.to(dev)
        enc, logvar = vae.encode(torch.unsqueeze(torch.flatten(mol[0]), 0))
        print(f'Encoded: {enc.cpu().detach().numpy()}')
        dec = vae.decode(enc)
        dec = torch.argmax(dec.reshape((dataset.max_selfies_length, dataset.feature_vector_length)),
                           dim=1).cpu().numpy().ravel()
        print(f'Decoded: {dec}')
        ShowMol(MolFromSmiles(sf.decoder(
            sf.encoding_to_selfies(dec, dataset.idx_to_symbol,
                                   enc_type='label'))))


if __name__ == '__main__':
    main()
