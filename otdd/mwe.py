#MWE

    df = pd.read_csv(f'../../{args.set}_{args.set_type}/{args.dataset}_{args.set_type}/labels.csv')
    le = preprocessing.LabelEncoder()
    le.fit(df['CATEGORY'])
    df['encoded_labels'] = le.transform(df['CATEGORY'])
    n_classes = len(df['encoded_labels'].unique())
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=32,pin_memory=True)



dist = DatasetDistance(train_queue, train_queue,
                        inner_ot_method = 'exact',
                        debiased_loss = True,
                        feature_cost = feature_cost,
                        sqrt_method = 'spectral',
                        sqrt_niters=10,
                        precision='single',
                        p = 2, entreg = 1e-1,
                        device='cpu')


class CustomImageDataset(Dataset):
    def __init__(self, annotations_df, img_dir, transform=1, target_transform=None):
        self.img_labels = annotations_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.targets = torch.tensor(self.img_labels.iloc[:, 1].tolist())
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = image.float()
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import os

class CustomImageDataset(Dataset):
    def __init__(self, annotations_df, img_dir, transform=1, target_transform=None):
        self.img_labels = annotations_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.targets = torch.tensor(self.img_labels.iloc[:, 1].tolist())  # Convert targets to a tensor
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = image.float()
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    @property
    def classes(self):
        return torch.sort(torch.unique(self.targets))[0].tolist()


train_queue = torch.utils.data.DataLoader(train_data, batch_size=32,pin_memory=True)


