"""
Converts CelebA faces into numpy arrays

All images are 178 x 218 px with 3 (RGB) channels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path('data/img')
OUT_DIR = Path('data/npy')

def process(im):
    return im.astype('double') / 255

if __name__ == '__main__':
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True)
    
    for name in DATA_DIR.iterdir():
        im = plt.imread(name)
        im = process(im)
        out_path = OUT_DIR / (name.stem + '.npy')
        np.save(out_path, im)
