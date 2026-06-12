from torch.utils.data import DataLoader
import torch
import os
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import  transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

class WaferDataset(Dataset):

    def __init__(self, images, labels, transform=None):
        # В статье используется нормализация [0, 1]
        self.images = torch.tensor(images, dtype=torch.float32) / 255.0 
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx] # Исходный размер (1, H, W)
  
        x = x.repeat(3, 1, 1) 
        
        if self.transform:
            x = self.transform(x)
            
        return x, self.labels[idx].float()

class DataLoaderWafer:

    def __init__(self, dataset_path, vis=False):

        self.dataset_path = dataset_path
        self.is_vis = vis
        
        # Список возможных деффектов
        self.label_keys = [
            "Center", "Donut", "Edge_Loc", "Edge_Ring",
            "Loc", "Near_Full", "Scratch", "Random"
        ]
        
    def load_data(self):

        data = np.load(self.dataset_path)

        images = data["arr_0"]
        labels = data["arr_1"]
        
        images_flat = images.reshape(images.shape[0], -1)
        _, unique_indices = np.unique(images_flat, axis=0, return_index=True)
        
        self.images = images[unique_indices]
        self.labels = labels[unique_indices]

        print("MixedWM38:", len(self.labels), "wafer loaded")
    
    def prep_data(self):
        # Разделение на трейн, тест и валидацию с сохранением пропорций
        x_train, x_test, y_train, y_test = train_test_split(
            self.images,
            self.labels,
            test_size=0.2,
            random_state=42,
            stratify=self.labels
        )
        
        x_train, x_val, y_train, y_val = train_test_split(
            self.images,
            self.labels,
            test_size=0.2,
            random_state=42,
            stratify=self.labels
        )
        
        # Для работы модели нужны картинки 224х224
        train_transform = transforms.Compose([
            transforms.Resize((56, 56)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((56, 56)),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((56, 56)),
        ])

        self.train_dataset = WaferDataset(x_train, y_train, transform=train_transform)
        self.val_dataset = WaferDataset(x_val, y_val, transform=val_transform)
        self.test_dataset = WaferDataset(x_test, y_test, transform=test_transform)
        
        
    def get_loaders(self, batch_size=64):

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader
    
    def read_label(self, label):

        if np.sum(label) == 0:
            return "Normal wafer"

        defect_types = ""

        for i in range(len(label)):
            if label[i] == 1:
                defect_types += self.label_keys[i] + ", "

        return defect_types
    
    def see_wafer(self, wafer_num):

        defect_types = self.read_label(self.labels[wafer_num])


        print("Defect types =", defect_types)
        print("Label:", self.labels[wafer_num])

    def prepare(self):

        self.load_data()
        self.prep_data()

def image_transform(args):
    if args.dataset=='cifar100':
        mean_statistics = (0.5071, 0.4867, 0.4408)
        std_statistics = (0.2675, 0.2565, 0.2761)
        max_values = (1.0, 1.0, 1.0)
        min_values = (0.0, 0.0, 0.0)
        args.num_classes=100
    elif args.dataset=='cifar10':
        mean_statistics = (0.4914, 0.4822, 0.4465)
        std_statistics = (0.2470, 0.2435, 0.2616)
        max_values = (1.0, 1.0, 1.0)
        min_values = (0.0, 0.0, 0.0)
        args.num_classes=100
    offset = [0.5 * (min_val + max_val) for min_val, max_val in zip(min_values, max_values)]
    scale = [(max_val - min_val) / 2 for max_val, min_val in zip(max_values, min_values)]
    normalize = transforms.Normalize(mean=offset, std=scale)
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=args.input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return train_transform, test_transform



def get_torch_dataset(args):
    #data_path = os.path.join(args.datadir, args.dataset)
    #print(os.listdir("/kaggle/input/datasets/romkaaa/wafer-dataset"))
    data_path =  "/kaggle/input/datasets/romkaaa/wafer-dataset/Wafer_Map_Datasets.npz"
    #train_transform, test_transform = image_transform(args)
    if args.dataset == "cifar10":
        train_set = CIFAR10(data_path, train=True, transform=train_transform, download=True)
        test_set = CIFAR10(data_path, train=False, transform=test_transform, download=True)
    elif args.dataset == "cifar100":
        #train_set = CIFAR100(data_path, train=True, transform=train_transform, download=True)
        #test_set = CIFAR100(data_path, train=False, transform=test_transform, download=True)
        train_set = CIFAR100(data_path, train=True, transform=train_transform, download=False)
        test_set = CIFAR100(data_path, train=False, transform=test_transform, download=False)
    elif args.dataset == "mydataset":
        loader = DataLoaderWafer(data_path)
        loader.prepare()
        train_loader, _, test_loader = loader.get_loaders(batch_size=32)

    else:
        raise NotImplementedError(f"{args.dataset} not supported")
    #train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    #test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    #print(f'Dataset information: {args.dataset}\t {len(train_set)} images for training \t {len(test_set)} images for testing\t')
    return train_loader, test_loader

