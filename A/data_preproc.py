import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit


def data_preprocessing(spec_dir, train_ratio=0.7, seed=37):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = datasets.ImageFolder(root=spec_dir, transform=transform)

    class_names = dataset.classes
    labels = np.array([sample[1] for sample in dataset])

    # Obtaining Training sample indexes with stratified splitting
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(labels)), labels))
    temp_labels = labels[temp_idx]

    # Now Obtaining both Validation and Test sample indexes from the Temp Indexes
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(sss2.split(np.zeros(len(temp_labels)), temp_labels))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    # Actually partitioning the dataset into train, validation and test subsets
    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)

    # Setting up Pytorch Dataloaders with batchsize 32
    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return class_names, train_loader, val_loader, test_loader








