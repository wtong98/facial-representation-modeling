"""
Quick-n-dirty script for rendering nearest_neighbors.py output

author: William Tong
date 11/27/2020
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm

# <codecell>
def data_gen():
    data = np.load('vae_save/nearest_neighbors.npy')
    data = data[:,:5]

    col = 0
    row = 0

    while row < data.shape[1]:
        print('yielding (%d, %d)' % (col, row))
        yield(data[col, row])

        if col == 0:
            col = 1
        else:
            col = 0
            row += 1


# <codecell>
samp_im = data_gen()

fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 2),
                 axes_pad=0.1,
                 )

for ax, im in tqdm(zip(grid, samp_im), total=10):
    ax.imshow(im)

fig.suptitle('Sampled images and their nearest neighbor')
# plt.show()
plt.savefig('nn_fig.png')