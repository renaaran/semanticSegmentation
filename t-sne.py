import os
import glob
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from dataset import FacadesDataset
from sklearn import datasets
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from random import shuffle

#plt.style.use('seaborn-whitegrid')

class BaseDataset(Dataset):
    def __init__(self, root_dir, root='train', mask='*.jpg'):
        self.root_dir = root_dir
        self.root = root
        self.files_count = 0
        self.files = []
        for file_path in glob.glob(os.path.join(self.root_dir, root, mask)):
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            self.files.append(file_name)
        shuffle(self.files)
        self.files_count = len(self.files)

    def __len__(self):
        return self.files_count

    def _read(self, index, ext):
        file_name = f'{self.files[index]}.{ext}'
        img_path = os.path.join(self.root_dir, self.root, file_name)
        return Image.open(img_path)

class FacadesDataset(BaseDataset):

    def __init__(self, root_dir, root='train', mask='*.jpg'):
        super().__init__(root_dir, root)
        self.preprocess = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.4749, 0.4510, 0.4152],
                                   std=[0.2217, 0.2175, 0.2189]),
         ])

    def __getitem__(self, index):
        img = np.array(super()._read(index, 'jpg'))
        return self.preprocess(img).flatten()

N = 800
dt = FacadesDataset('./datasets/geo', root='train')
train_loader = DataLoader(dt, shuffle=True, batch_size=N)
X1 = next(iter(train_loader))
dt = FacadesDataset('./datasets/geo.SPADE.150_epochs_no_sem',
                    root='train', mask='*SPADE.jpg')
#dt = FacadesDataset('./datasets/facades_new', root='train')
train_loader = DataLoader(dt, shuffle=True, batch_size=N)
X2 = next(iter(train_loader))
X = np.concatenate((X1, X2), axis=0)
y = np.array([0]*len(X))
y[N:] = 1
print(X.shape, y.shape)
tsne = TSNE(n_components=2, random_state=0)
# Fit and transform with a TSNE
X_2d = tsne.fit_transform(X)
print(X_2d.shape)
# Visualize the data
plt.figure(figsize=(10, 10))
colors = 'r', 'g'
for i, c, l, m in zip([0, 1], colors, ['Regular', 'Augmented'],
                      ['^', 'o']):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=l, marker=m,
                alpha=0.5, s=70)
plt.legend()
plt.show()
