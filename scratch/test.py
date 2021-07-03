# <codecell>
from pathlib import Path
import sys
sys.path.append('../')
from dataset.synthetic import build_datasets

train, test = build_datasets(Path('../save/syn'))

# <codecell>
print(train[1]) # TODO: something wrong with saving model
