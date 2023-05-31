#MWE
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.utils.data as torchdata
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Dataset
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import torch
from torchvision.models import resnet18
import pandas as pd
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance, FeatureCost

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


def check_folders_exist(folders):
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True

def dataset_loader(dataset_name, CustomImageDataset=CustomImageDataset):
    df = pd.read_csv(f'{dataset_name}/labels.csv')
    le = preprocessing.LabelEncoder()
    le.fit(df['CATEGORY'])
    df['encoded_labels'] = le.transform(df['CATEGORY'])
    n_classes = len(df['encoded_labels'].unique())
    train_data = CustomImageDataset(annotations_df = df[['FILE_NAME', 'encoded_labels']], img_dir = f'{dataset_name}/images')
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=32,pin_memory=True)
    return train_queue

def calculate_similarity(datasets):
    embedder = resnet18(pretrained=True).eval()
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False
    feature_cost = FeatureCost(src_embedding = embedder,
                           src_dim = (3,128,128),
                           tgt_embedding = embedder,
                           tgt_dim = (3,128,128),
                           p = 2,
                           device='cpu')

    similarity_matrix = {}
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            d1 = dataset_loader(datasets[i])
            d2 = dataset_loader(datasets[j])
            
            # Check if the similarity has already been calculated
            if (d1, d2) in similarity_matrix:
                similarity = similarity_matrix[(d1, d2)]
            else:
                # Calculate the similarity using your existing 'dist' function
                similarity =  DatasetDistance(train_queue, train_queue,
                                                inner_ot_method = 'exact',
                                                debiased_loss = True,
                                                feature_cost = feature_cost,
                                                sqrt_method = 'spectral',
                                                sqrt_niters=10,
                                                precision='single',
                                                p = 2, entreg = 1e-1,
                                                device='cuda')
                d = similarity.distance(maxsamples = 1000)

                # Store the calculated similarity in the matrix for both directions
                similarity_matrix[(d1, d2)] = similarity
                similarity_matrix[(d2, d1)] = similarity
            
            # Output the similarity for the pair (d1, d2)
            print(f"Similarity between d{i+1} and d{j+1}: {similarity}")

    return similarity_matrix

def save_similarity_matrix(similarity_matrix, dataset_names, filename):
    # Create a list of dataset names for the DataFrame index and columns
    index = dataset_names
    columns = dataset_names
    
    # Convert the similarity matrix to a DataFrame
    df = pd.DataFrame(similarity_matrix, index=index, columns=columns)
    
    # Save the DataFrame as a CSV file
    df.to_csv(filename)


set0_names = ["BCT_Micro","PLK_Micro","FLW_Micro","SPT_Micro",
"BRD_Micro","PLT_VIL_Micro","TEX_Micro",
"CRS_Micro","RESISC_Micro"]
set1_names = ["ACT_40_Micro","INS_2_Micro","PLT_NET_Micro","TEX_DTD_Micro","APL_Micro",
"PNU_Micro","DOG_Micro","MED_LF_Micro","RSICB_Micro"]
set2_names = ["ACT_410_Micro","FNG_Micro","PLT_DOC_Micro","TEX_ALOT_Micro","AWA_Micro",
              "INS_Micro","RSD_Micro","PRT_Micro","BTS_Micro"]
dataset_list = set0_names+set1_names+set2_names


similarity_matrix = calculate_similarity(dataset_list)

save_similarity_matrix(similarity_matrix, dataset_list, "similarity_matrix.csv")









# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_df, img_dir, transform=1, target_transform=None):
#         self.img_labels = annotations_df
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform
#         self.targets = torch.tensor(self.img_labels.iloc[:, 1].tolist())
#     def __len__(self):
#         return len(self.img_labels)
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = image.float()
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label







