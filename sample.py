"""
Generates sample directories for FID scoring
"""

import argparse
import logging
import pickle
from pathlib import Path


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