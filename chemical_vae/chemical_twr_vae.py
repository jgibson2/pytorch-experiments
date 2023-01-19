import functools
import itertools
import os
import sys
import time

import torch
import math
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
EPOCHS = 250
VAL_PERCENTAGE = 20
BATCH_SIZE = 2
EMBEDDING_DIM = 512
HIDDEN_DIM = 256
Z_DIM = 64
DROPOUT = 0.5


class SMILESDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename, skip_header=True, pad=True):
        with open(filename, 'r') as f:
            self.selfies = [sf.encoder(l.strip()) for l in f.readlines()[1 if skip_header else 0:]]
            self.max_selfies_length = max([sf.len_selfies(s) for s in self.selfies])
            alphabet = sf.get_alphabet_from_selfies(self.selfies)
            alphabet.add('[nop]')  # '[nop]' is a special padding symbol
            alphabet = list(sorted(alphabet))
            self.feature_vector_length = len(alphabet)
            self.symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
            self.idx_to_symbol = {i: s for i, s in enumerate(alphabet)}
            self.pad = self.max_selfies_length if pad else -1

    def __iter__(self):
        return iter(map(lambda s: torch.Tensor(sf.selfies_to_encoding(s,
                                                                      vocab_stoi=self.symbol_to_idx,
                                                                      pad_to_len=self.pad,
                                                                      enc_type='label')).long(), self.selfies))

    def __getitem__(self, idx):
        return torch.Tensor(sf.selfies_to_encoding(self.selfies[idx],
                                                   vocab_stoi=self.symbol_to_idx,
                                                   pad_to_len=self.pad,
                                                   enc_type='label')).long()

    def __len__(self):
        return len(self.selfies)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, z_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.layer_dim = hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,
                           dropout=dropout if n_layers > 1 else 0, bidirectional=False, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.linear_mu = nn.Linear(self.layer_dim, z_dim)
        self.linear_var = nn.Linear(self.layer_dim, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def embed(self, src):
        embedded = self.dropout(self.embedding(src))

        output, (Hs, Cs) = self.rnn(embedded, None)
        mus = torch.stack([self.linear_mu(h) for h in Hs], dim=1)
        logvars = torch.stack([self.linear_var(h) for h in Hs], dim=1)

        return mus, logvars, (Hs, Cs)

    def forward(self, src):
        mus, logvars, states = self.embed(src)
        z = self.reparameterize(mus[:, -1].squeeze(), logvars[:, -1].squeeze())

        return z, mus, logvars, states

    def loss(self, mus, logvars):
        # TODO: verify
        KLD = -0.5 * torch.sum(1.0 + logvars - mus.pow(2) - logvars.exp())
        KLD = KLD / mus.size(0)
        return KLD


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, z_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.layer_dim = hid_dim
        self.rnn = nn.LSTM(z_dim, hid_dim, n_layers,
                           dropout=dropout if n_layers > 1 else 0, bidirectional=False, batch_first=True)
        self.out = nn.Linear(z_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_z, states):
        # TODO: build sequence
        z = torch.unsqueeze(enc_z, dim=1)
        seq, _ = self.rnn(z, states)
        return seq


    def loss(self, prod, target, weight=1.0):
        recon_loss = F.cross_entropy(
            prod.view(-1, prod.shape[2]), target[1:].view(-1),
            ignore_index=0, reduction="sum")
        return weight * recon_loss


class TWRVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(TWRVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder.forward(x)

    def decode(self, z, states):
        return self.decoder.forward(z, states)

    def calculate_loss(self, out, mu, logvar, inp):
        kl_loss = self.encoder.loss(mu, logvar)
        recon_loss = self.decoder.loss(out, inp)
        return kl_loss + recon_loss


def train_vae(train_dataloader, val_dataloader, model,
              patience=5, burnin=10, fname='models/twr_vae.pt', train=True, dataset=None):
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
                opt.zero_grad()
                z, mu, logvar, states = model.encode(batch_x)
                out = model.decode(z, states)
                loss = model.calculate_loss(out, mu, logvar, batch_x)
                loss.backward()
                opt.step()
            model.eval()
            val_loss = torch.Tensor([0])
            with torch.no_grad():
                for batch_x in val_dataloader:
                    batch_x = batch_x.to(dev)
                    val_z, val_mu, val_logvar, val_states = model.encode(batch_x)
                    val_out = model.decode(val_mu, val_states)
                    val_loss += model.calculate_loss(val_out, val_mu, val_logvar, batch_x).detach().cpu()
                    mol = batch_x[0].detach()
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
    print(f"Using device {dev}")
    dataset = SMILESDataset('data/1K_ZINC.txt')
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

    encoder = Encoder(
        dataset.feature_vector_length, EMBEDDING_DIM, HIDDEN_DIM, Z_DIM, dataset.max_selfies_length, DROPOUT,
    )
    decoder = Decoder(
        dataset.feature_vector_length, EMBEDDING_DIM, HIDDEN_DIM, Z_DIM, dataset.max_selfies_length, DROPOUT,
    )
    vae = TWRVAE(encoder, decoder)
    vae.to(dev)

    vae = train_vae(train_loader, val_loader, vae, train=True, fname='models/twr_vae.pt', patience=10, burnin=50,
                    dataset=None)

    vae.eval()
    with torch.no_grad():
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
