import torch
from torch.utils.data import Dataset
import torchvision.transforms as Transforms

import os
from PIL import Image

import util

class CelebA(Dataset):
    def __init__(self, img_dir, ann_path, transform, mode='train'):
        ann_file = open(ann_path, 'r').readlines()
        attr_name = ann_file[1].split()
        attr_val = ann_file[2:]
        
        male = os.listdir('data/test/0')
        female = os.listdir('data/test/1')

        for i in range(len(male)):
            num, _ = os.path.splitext(male[i])
            num = int(num)
            male[i] = num-1
        
        for i in range(len(female)):
            num, _ = os.path.splitext(female[i])
            num = int(num)
            female[i] = num-1

        test_num = male + female

        if mode == 'train':
            attr_val = [x for i, x in enumerate(attr_val) if i not in test_num]
        elif mode == 'test':
            attr_val = [attr_val[i] for i in test_num]

        self.img_dir = img_dir
        self.transform = transform
        self.data_set = []
        
        target_attr_name = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        target_attr_idx = []

        for target in target_attr_name:
            target_attr_idx.append(attr_name.index(target))

        for idx, attr in enumerate(attr_val):
            attr = attr.split()
            img_path = attr[0]
            
            for i in range(1, len(attr)):
                if attr[i] == '-1':
                    attr[i] = 0
                elif attr[i] == '1':
                    attr[i] = 1
                    
            labels = [attr[i + 1] for i in target_attr_idx]
            labels = util.list2tensor(labels)
                
            self.data_set.append([img_path, labels])

    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        img_name, labels = self.data_set[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path)
        img = self.transform(img)

        return img, labels
    
def data_loader(img_dir, ann_path, batch_size=16, shuffle=True, mode='train'):
    transform = Transforms.Compose([Transforms.CenterCrop((178, 178)), 
                                    Transforms.Scale((128, 128)), 
                                    Transforms.RandomHorizontalFlip(),
                                    Transforms.ToTensor(), 
                                    Transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                                         std=(0.5, 0.5, 0.5))
                                   ])
    dset = CelebA(img_dir, ann_path, transform, mode=mode)
    dloader = torch.utils.data.DataLoader(dset, 
                                          batch_size=batch_size, 
                                          shuffle=shuffle, 
                                          num_workers=0)
    dlen = len(dset)
    
    return dloader, dlen
