# Human Facial Representations
This project seeks to model face representations, with an aim to better
understand the computations underlying human face cognition.

For more information, see [this post.](www.google.com)

## Running the code
Before starting, ensure you have:
* Python 3.6+
* Python virtualenv
* Nvidia GPU with >= 2 GB vRAM (optional)
* CUDA >= 10.2 (optional)

To set up the virtual environment
```sh
python -m venv venv_face
source venv_face/bin/activate   # or whichever script matches your shell
pip install -r requirements.txt
```

After installation completes, you're all set to run any of the scripts within `scratch/`, or to train a new model!

For more information about training new models, run
```
python train.py --help
```

## Layout
A brief tour of the source files:
* `train.py`: script that trains and manages new models
* `model/`: collection of various model families including autoencoders, variational autencoders, and Helmholtz machines
* `dataset/`: PyTorch wrappers for various datasets, including CelebA, Chicago Face Database, MNIST, UTKFace, and synthetic data
* `util.py`: various helpful utilities, mostly used by routines in `scratch/`
* `scratch/`: miscellaneous directory that contains mostly one-off scripts and utilities. These include plot-generating scripts, various analysis routines, and small experiments

