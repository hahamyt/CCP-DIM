import os
import random

import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as T
from models_ import Encoder, GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ColorJitter
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import statistics as stats
import argparse
import visdom

# viz = visdom.Visdom()


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.6, gamma=0.1):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 26, 26)

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepInfomax pytorch')
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size

    imagenet_train_dt = datasets.ImageFolder(
        './data/ILSVRC2015_64/',
        T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
        ]
    ))
    cifar_10_train_l = DataLoader(imagenet_train_dt, batch_size=batch_size, shuffle=True, drop_last=True,
                                  pin_memory=torch.cuda.is_available())

    encoder = Encoder().to(device)
    loss_fn = DeepInfoMaxLoss().to(device)
    optim = torch.optim.SGD(encoder.parameters(), lr=1e-3, momentum=0.99)
    loss_optim = torch.optim.SGD(loss_fn.parameters(), lr=1e-3, momentum=0.99)
    loss_fn_s = torch.nn.MSELoss()
    epoch_restart = 76
    root = Path(r'./models/run5/')

    if epoch_restart is not None and root is not None:
        enc_file = root / Path('512encoder' + str(epoch_restart) + '.wgt')
        loss_file = root / Path('512loss' + str(epoch_restart) + '.wgt')
        encoder.load_state_dict(torch.load(str(enc_file)))
        loss_fn.load_state_dict(torch.load(str(loss_file)))

    for epoch in range(epoch_restart + 1, 10000):
        batch = tqdm(cifar_10_train_l, total=len(imagenet_train_dt) // batch_size)
        train_loss = []
        DIM_loss = []
        S_loss = []
        V_loss = []
        for x, target in batch:
            x = x.to(device)
            target = target.to(device)

            x_s = T.ColorJitter(brightness=random.random(), contrast=random.random(), saturation=random.random())(x)
            x_s.data += torch.normal(torch.zeros_like(x_s.data), std=0.01)
            x_ = torch.vstack([x, x_s])

            optim.zero_grad()
            loss_optim.zero_grad()
            y_, M_ = encoder(x_)

            y = y_[:batch_size, :]
            M = M_[:batch_size, :, :, :]

            y_s = y_[batch_size:, :]
            # rotate images to create pairs for comparison
            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
            loss_dim = loss_fn(y, M, M_prime)

            uni_target = target.unique().float().to(device)
            mus = torch.zeros(len(uni_target), y.shape[1]).to(device)
            # Calculating centers for different classes
            for i in range(len(uni_target)):
                mus[i, :] = y[target==uni_target[i]].mean(0)
            #
            Swis = []
            for i in range(len(uni_target)):
                y_i = y[target == uni_target[i]]
                Swi = 0
                for j in range(len(y_i)):
                    Swi += (y_i[None, j, :] - mus[None, i, :]).mm((y_i[None, j, :] - mus[None, i, :]).T)
                Swis.append(Swi)

            # All data mean
            mu_overall = mus.mean(0).unsqueeze(0)
            # Probs
            probs = torch.zeros_like(uni_target)
            for i in range(len(uni_target)):
                probs[i] = y[target == uni_target[i]].shape[0]/y.shape[0]
            # Inter-class variance G: The bigger, the better
            G = 0
            for i in range(len(mus)):
                G += probs[i]*(mus[None, i]-mu_overall).mm((mus[None, i]-mu_overall).T)
            # Intra-class variance: The smaller, the better
            S = 0
            for i in range(len(mus)):
                S += probs[i]*Swis[i]
            # Optimization: G/S-> G*S^-1
            loss_supervised = S / (G + 1)

            loss_s = loss_fn_s(y, y_s)

            loss = loss_dim + 1e-3*loss_supervised + loss_s

            train_loss.append(loss.item())
            DIM_loss.append(loss_dim.item())
            S_loss.append(loss_s.item())
            V_loss.append(loss_supervised.item())

            batch.set_description(str(epoch) + ' Loss: ' + str(stats.mean(train_loss[-20:]))[:7]
                                  +" DIM: "+str(stats.mean(DIM_loss[-20:]))[:7]
                                  +" S: "+str(stats.mean(S_loss[-20:]))[:7]
                                  +" L: "+str(stats.mean(V_loss[-20:]))[:7])

            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 10)
            loss.backward()

            optim.step()
            loss_optim.step()

        if epoch % 1 == 0:
            root = Path(r'./models/run5')
            enc_file = root / Path('512encoder' + str(epoch) + '.wgt')
            loss_file = root / Path('512loss' + str(epoch) + '.wgt')
            enc_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), str(enc_file))
            torch.save(loss_fn.state_dict(), str(loss_file))
