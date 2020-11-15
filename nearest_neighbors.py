"""
Produces sample / nearest-neighbor pair to validate quality of trained model

author: William Tong (wlt2115@columbia.edu)
date: 11/10/2020
"""
import argparse
import logging
import pickle
from pathlib import Path

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.vae import VAE
from dataset.celeba import build_datasets, IM_DIMS, TOTAL_IMAGES

def _build_parser():
    parser = argparse.ArgumentParser(description="Train various models")
    parser.add_argument('save_path', type=Path, 
        help='Path to saved model')
    parser.add_argument('im_path', type=Path, 
        help='Path to training images')
    # TODO: save model type with .pt files
    parser.add_argument('--model', type=str, choices=['vae', 'hm'], default='vae',
        help='Type of model to train, either vae for variational autoencoder or hm for helmholtz machine. ' +\
             'Defaults to vae')
    parser.add_argument('--samples', type=int, default=25,
        help='number of samples to compare. Defaults to 25')
    parser.add_argument('--batch', type=int, default=1000,
        help='number of images to hold in memory at one time. Defaults to 1000.')
    parser.add_argument('--out', type=Path, default='out/nearest_neighbors.npy',
        help='output path nearest neighbor pairs. Defaults to out/nearest_neighbors.npy')
    parser.add_argument('--workers', type=int, default=4,
        help='Number of workers used to retrieve data. Defaults to 4')

    return parser


def _init_record(samps):
    record = {
        'pair': np.zeros((2, samps.shape[0], 3, *IM_DIMS)),
        'distance': np.zeros(samps.shape[0])
    }    

    record['pair'][0] = samps
    record['distance'][:] = np.inf
    return record


def _update_winner(chunk, record, pbar):
    for im in chunk:
        for i, samp in enumerate(record['pair'][0]):
            curr_dist = np.linalg.norm(im - samp)
            if curr_dist < record['distance'][i]:
                record['pair'][1][i] = im
                record['distance'][i] = curr_dist
        
        pbar.update(1)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = None
    if args.model == 'vae':
        model = VAE().double().to(device)
    else:
        logging.critical('model unimplemented: %s' % args.model)
        return

    if not args.out.parent.exists():
        args.out.parent.mkdir()

    _, test_ds = build_datasets(args.im_path, train_test_split=1, total=10)

    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with torch.no_grad():
        samps = model.sample(args.samples)

    loader = DataLoader(test_ds, 
                        batch_size=args.batch, 
                        num_workers=args.workers,
                        pin_memory=torch.cuda.is_available())

    record = _init_record(samps)
    with tqdm(total=TOTAL_IMAGES) as pbar:
        for chunk in loader:
            _update_winner(chunk, record, pbar)
    
    np.save(args.out, record['pair'])
    


if __name__ == '__main__':
    main()