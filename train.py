"""
Frontend for training various models

author: William Tong (wlt2115@columbia.edu)
date: 11/5/2020
"""
import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from model.vae import VAE
from dataset.celeba import build_datasets


def _build_parser():
    parser = argparse.ArgumentParser(description="Train various models")
    parser.add_argument('path', type=Path, 
        help='Path to directory of .jpg images')
    parser.add_argument('--save', type=Path, default=Path('save/'),
        help='Path to model save directory. Default=save/')
    parser.add_argument('--model', type=str, choices=['vae', 'hm'], default='vae',
        help='Type of model to train, either vae for variational autoencoder or hm for helmholtz machine. ' +\
             'Defaults to vae')
    parser.add_argument('--epochs', type=int, default=20,
        help='Number of epochs to train for. Defaults to 20')
    parser.add_argument('--batch_size', type=int, default=32,
        help='Size of each batch fed to the model. Defaults to 32')
    parser.add_argument('--workers', type=int, default=4,
        help='Number of workers used to retrieve data. Defaults to 4')

    return parser

def _eval(model, test_data, device, n_samples=1000):
    size = len(test_data)
    idxs = np.random.choice(np.arange(size), n_samples, replace=False)
    x = torch.stack([test_data[i] for i in idxs]).to(device)

    with torch.no_grad():
        reco_params = model(x)
        loss = model.loss_function(*reco_params)
    
    return loss


def main():
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = None
    save_path = args.save
    if args.model == 'vae':
        model = VAE()
        save_path = save_path / 'vae'
    else:
        logging.critical('model unimplemented: %s' % args.model)
        return
    
    if not save_path.exists():
        save_path.mkdir(parents=True)
    
    model = model.double()
    model.to(device)
    optimizer = optim.Adam(model.parameters())

    train_ds, test_ds = build_datasets(args.path)

    losses = []
    for e in range(args.epochs):
        logging.info('epoch: %d of %d' % (e+1, args.epochs))

        loader = DataLoader(train_ds, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.workers,
                            pin_memory=torch.cuda.is_available())
        total_batches = len(train_ds) // args.batch_size

        log_every = total_batches // 4
        save_every = 1   # hardcoded for now
        for i, x in enumerate(loader):
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            total_loss = model.loss_function(*output)['loss']
            total_loss.backward()
            optimizer.step()

            if i % log_every == 0:
                model.eval()
                loss = _eval(model, test_ds, device)
                model.train()

                print_params = (i+1, total_batches, loss['loss'], loss['mse'], loss['kld'])
                logging.info('[batch %d/%d] loss: %f, mse: %f, kld: %f' % print_params)
                losses.append({'iter': i, 'epoch': e, 'loss': loss})
            
        if e % save_every == 0:
            torch.save({
                'epoch': e+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path / ('epoch_%d.pt' % str(e+1))

    model.eval()
    loss = _eval(model, test_ds, device)
    model.train()

    print_params = (loss['loss'], loss['mse'], loss['kld'])
    logging.info('final loss: %f, mse: %f, kld: %f' % print_params)
    losses.append({'iter': 0, 'epoch': e+1, 'loss': loss})

    with open(save_path / 'loss.pk', 'wb') as pkf:
        pickle.dump(losses, pkf)

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path / 'final.pt')
    print('done!')


if __name__ == '__main__':
    main()