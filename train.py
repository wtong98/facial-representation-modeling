"""
Frontend for training various models

author: William Tong (wlt2115@columbia.edu)
date: 11/5/2020
"""
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

# from model.hm import HM
from model.hm_binary import HM
from model.ae import AE
from model.vae import VAE
from model.vae_gm import GMVAE
from dataset.synthetic import build_datasets
# from dataset.celeba import build_datasets
# from dataset.celeba_single import build_datasets
# from dataset.mnist import build_datasets


def _build_parser():
    parser = argparse.ArgumentParser(description="Train various models")
    parser.add_argument('path', type=Path,
                        help='Path to directory of .jpg images')
    parser.add_argument('--save', type=Path, default=Path('save/'),
                        help='Path to model save directory. Default=save/')
    parser.add_argument('--model', type=str, choices=['vae', 'ae', 'hm', 'gmvae'], default='vae',
                        help='Type of model to train, either vae for variational autoencoder or hm for helmholtz machine. ' +
                        'Defaults to hm'),
    parser.add_argument('--color', action='store_true',
                        help='Specify this toggle to use the full-color HM model'),
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train for. Defaults to 20')
    parser.add_argument('--dim', type=int, default=40,
                        help='Latent dimension of the model (VAE only)')
    # TODO: add params for GMVAE
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of each batch fed to the model. Defaults to 32')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers used to retrieve data. Defaults to 4')

    return parser

# TODO: generalize for flip-flop wake sleep


def _eval(model, test_data, device, n_samples=100):
    size = len(test_data)
    idxs = np.random.choice(np.arange(size), n_samples, replace=False)
    x = torch.stack([test_data[i] for i in idxs]).to(device)

    with torch.no_grad():
        reco_params = model(x)
        loss = model.loss_function(reco_params)

    # with torch.no_grad():
    #     reco_params = model(x)
    #     loss += model.loss_function(*reco_params)
    #     reco2 = model(x)
    #     loss += model.loss_function(*reco2)

    # print('GEN_LOSS', reco2[2])
    # print('REC_LOSS', reco_params[3])

    # with torch.no_grad():
    #     reco_params = model(x)
    #     loss = model.loss_function(*reco_params)

    # print('GEN_LOSS', reco_params[2])
    # print('REC_LOSS', reco_params[3])
    # print('BIAS_MU', model.g_bias)
    # print('BIAS_LOGVAR', model.g_bias_logvar)
    # print('MU', model.g(2))
    # with torch.no_grad():
    #     print('MU', [torch.sum(model.g(i).mu) for i in range(model.num_layers)])
    #     print('LOGVAR', [torch.sum(model.g(i).log_var) for i in range(model.num_layers)])

    return loss


def main():
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = None
    save_path = args.save
    if args.model == 'vae':
        model = VAE(args.dim)
        logging.info('training VAE with dims: {}'.format(args.dim))
    elif args.model == 'ae':
        model = AE(args.dim)
        logging.info('training AE with dims: {}'.format(args.dim))
    elif args.model == 'hm':
        model = HM(args.color)
    elif args.model == 'gmvae':
        model = GMVAE()
    else:
        logging.critical('model unimplemented: %s' % args.model)
        return

    if not save_path.exists():
        save_path.mkdir(parents=True)

    model = model.float()
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

        log_every = total_batches // 50 + 1
        save_every = 1   # hardcoded for now
        for i, x in enumerate(loader):
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            total_loss = model.loss_function(output)
            if type(total_loss) is dict:  # TODO: generalize loss handling
                total_loss = total_loss['loss']

            total_loss.backward()
            optimizer.step()

            if i % log_every == 0:
                model.eval()
                loss = _eval(model, test_ds, device)
                model.train()

                logging.info('[batch %d/%d] ' %
                             (i+1, total_batches) + model.print_loss(loss))
                # TODO: generalize printing
                # print_params = (i+1, total_batches, loss['loss'], loss['mse'], loss['kld'])
                # logging.info('[batch %d/%d] loss: %f, mse: %f, kld: %f' % print_params)
                # print_params = (i+1, total_batches, loss)
                # logging.info('[batch %d/%d] loss: %f' % print_params)
                losses.append({'iter': i, 'epoch': e, 'loss': loss})

        if e % save_every == 0:
            torch.save({
                'epoch': e+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path / ('epoch_%d.pt' % (e+1)))

    model.eval()
    loss = _eval(model, test_ds, device)
    model.train()

    logging.info('final loss: %s' % loss)
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
