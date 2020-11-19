"""
Generates sample directories for FID scoring
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
from dataset.celeba import build_datasets, IM_DIMS


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
    parser.add_argument('--samples', type=int, default=10000,
        help='number of samples to generate. Defaults to 10000')
    parser.add_argument('--batch', type=int, default=1000,
        help='number of samples to hold in memory at one time. Defaults to 1000.')
    parser.add_argument('--out', type=Path, default='out/',
        help='output directory for samples and true images. Defaults to out/')

    return parser

# TODO: debug FID with small dataset size
def _sample_images(model, batch_size, num_images, dataset, out_dir, workers=4):
    batch_size = min(batch_size, num_images)
    total = 0

    samp_dir = out_dir / 'sample'
    im_dir = out_dir / 'image'
    
    if not samp_dir.exists():
        samp_dir.mkdir()

    if not im_dir.exists():
        im_dir.mkdir()

    pbar = tqdm(total=num_images)
    with torch.no_grad():
        loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        num_workers=workers,
                        pin_memory=torch.cuda.is_available())
        
        for images in loader:
            samp = model.reconstruct(images)
            for i, reco in enumerate(samp):
                idx = pbar.n
                samp_path = samp_dir / ('%d.png' % idx)
                im_path = im_dir / ('%d.png' % idx)

                real_im = images[i].reshape(*IM_DIMS, 3).numpy()
                reco = reco.reshape(*IM_DIMS, 3). numpy()

                imageio.imwrite(samp_path, _to_rgb(reco))
                imageio.imwrite(im_path, _to_rgb(real_im))
                pbar.update(1)

                if pbar.n == pbar.total:
                    pbar.close()
                    return


# def _sample_images(model, batch_size, num_images, dataset, out_dir):
#     batch_size = min(batch_size, num_images)
#     total = 0

#     samp_dir = out_dir / 'sample'
#     im_dir = out_dir / 'image'
    
#     if not samp_dir.exists():
#         samp_dir.mkdir()

#     if not im_dir.exists():
#         im_dir.mkdir()

#     pbar = tqdm(total=num_images)
#     with torch.no_grad():
#         while total < num_images:
#             samp = model.sample(batch_size).reshape(batch_size, *(IM_DIMS), 3)
#             samp = samp.numpy()

#             for i, image in enumerate(samp):
#                 idx = total + i
#                 samp_path = samp_dir / ('%d.png' % idx)
#                 im_path = im_dir / ('%d.png' % idx)

#                 real_im = dataset[idx].reshape(*(IM_DIMS), 3)
#                 real_im = real_im.numpy()

#                 imageio.imwrite(samp_path, _to_rgb(image))
#                 imageio.imwrite(im_path, _to_rgb(real_im))
#                 pbar.update(1)
            
#             total += batch_size
#             batch_size = min(batch_size, num_images - total)
    
#     pbar.close()

def _to_rgb(im):
    return np.uint8(im * 255)
            

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
    
    if not args.out.exists():
        args.out.mkdir(parents=True)

    _, test_ds = build_datasets(args.im_path, train_test_split=1)

    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _sample_images(model, args.batch, args.samples, test_ds, args.out)


if __name__ == '__main__':
    main()