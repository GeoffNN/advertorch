# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function

import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from robustness.datasets import CIFAR
from robustness.datasets import cifar_models

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch_examples.utils import TRAINED_MODEL_PATH



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CIFAR')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default="cln", help="cln | adv | add_rr")
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model', default='resnet18', type=str, help='resnet50 | resnet18 | vgg13')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--stop_prob', default=1. / 7, type=float, help='The probability of stopping the Russian roulette estimator.')
    parser.add_argument('--inner_steps', default=7, type=int, help="The number of steps for the inner PGD.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    print("Running on: ", torch.cuda.get_device_name(args.device))
    if args.mode == "cln":
        flag_advtrain = False
        model_filename = f"cifar10_{args.model}_clntrained.pt"
        log_path = f"runs/cifar10/{args.model}/clntrained/"
    elif args.mode == "adv":
        flag_advtrain = True
        russian_roulette = False
        model_filename = f"cifar10_{args.model}_advtrained.pt"
        log_path = f"runs/cifar10/{args.model}/advtrained/"
    elif args.mode == "adv_rr":
        flag_advtrain = True
        russian_roulette = True
        model_filename = f"cifar10_{args.model}_adv_rrtrained.pt"
        log_path = f"runs/cifar10/{args.model}/advRRtrained/"
    else:
        raise


    dataset = CIFAR()
    train_loader, test_loader = dataset.make_loaders(batch_size=128, workers=20)

    if args.model == "resnet50":
        model = cifar_models.ResNet50()
    elif args.model == "resnet18":
        model = cifar_models.ResNet18()
    elif args.model == "vgg13":
        model = cifar_models.VGG13()

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    now = datetime.now()
    writer = SummaryWriter(log_path + str(now))

    example_images, labels = iter(train_loader).next()
    writer.add_graph(model, example_images.to(device))

    if flag_advtrain:
        from advertorch.attacks import LinfPGDAttack, PGDAttackRussianRoulette
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
            nb_iter=args.inner_steps, eps_iter=0.01, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=False)
        if russian_roulette:
            RRadversary = PGDAttackRussianRoulette(
                model, ord=np.inf, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                stop_prob=args.stop_prob, eps_iter=0.01, rand_init=True, clip_min=0.0,
                clip_max=1.0, targeted=False)


    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if flag_advtrain:
                # when performing attack, the model needs to be in eval mode
                # also the parameters should NOT be accumulating gradients
                with ctx_noparamgrad_and_eval(model):
                    if russian_roulette:
                        data = RRadversary.perturb(data, target)
                    else:
                        data = adversary.perturb(data, target)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(
                output, target, reduction='mean')
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)

        model.eval()
        test_clnloss = 0
        clncorrect = 0

        test_advloss = 0
        advcorrect = 0

        for clndata, target in test_loader:
            clndata, target = clndata.to(device), target.to(device)
            with torch.no_grad():
                output = model(clndata)
            test_clnloss += F.cross_entropy(
                output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            clncorrect += pred.eq(target.view_as(pred)).sum().item()

            advdata = adversary.perturb(clndata, target)
            with torch.no_grad():
                output = model(advdata)
            test_advloss += F.cross_entropy(
                output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            advcorrect += pred.eq(target.view_as(pred)).sum().item()

        test_clnloss /= len(test_loader.dataset)
        print('\nTest set: avg cln loss: {:.4f},'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                  test_clnloss, clncorrect, len(test_loader.dataset),
                  100. * clncorrect / len(test_loader.dataset)))
        writer.add_scalar("Loss/test", test_clnloss, epoch)
        writer.add_scalar("Acc/test", 100. * clncorrect / len(test_loader.dataset), epoch)

        test_advloss /= len(test_loader.dataset)
        print('Test set: avg adv loss: {:.4f},'
                ' adv acc: {}/{} ({:.0f}%)\n'.format(
                    test_advloss, advcorrect, len(test_loader.dataset),
                    100. * advcorrect / len(test_loader.dataset)))
        writer.add_scalar("AdvLoss/test", test_advloss, epoch)
        writer.add_scalar("AdvAcc/test", 100. * advcorrect / len(test_loader.dataset), epoch)

    writer.flush()
    torch.save(
        model.state_dict(),
        os.path.join(TRAINED_MODEL_PATH, model_filename))