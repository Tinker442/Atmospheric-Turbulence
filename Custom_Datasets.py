import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import glob

class KineticsImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.annotations = pd.read_csv(annotations_file,index_col=0)
        self.img_paths = self.annotations['img'].tolist()
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.annotations.iloc[idx,0]
        image = read_image(img_path)
        label = self.annotations.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class KineticsVideoDataset(Dataset):
    def __init__(self, annotations_file, clip_length=10, transform=None, target_transform=None):
        annotations = pd.read_csv(annotations_file,index_col=0)
        vid_paths =annotations['vid'].tolist()
        self.clip_paths=[]
        # self.label_paths=[]
        for vid in vid_paths:
            frames = glob.glob(vid+'/*')
            for i in range(0,len(frames),clip_length):
                if i+clip_length>=len(frames):
                    break
                self.clip_paths.append(frames[i:i+clip_length])
                # self.clip_paths.append(frames[i:i+clip_length-1])
                # self.label_paths.append(frames[i+clip_length])

        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        clip_path= self.clip_paths[idx]
        # label_path=self.label_paths[idx]
        image_list=[]
        for img_path in clip_path:
            image = read_image(img_path)
            image_list.append(image)

        
        clip=torch.stack(image_list)

        # label = read_image(label_path)

        if self.transform:
            clip = self.transform(clip)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return clip#, label