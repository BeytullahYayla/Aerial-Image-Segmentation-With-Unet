import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self,images,masks,isTrain=True):
        self.images=images
        self.masks=masks
        self.isTrain=isTrain
        
    def __getitem__(self, index):
        tfx=[A.RandomCrop(96, 96)]
        train_fx=[A.HorizontalFlip(p=0.5),A.VerticalFlip(p=0.5)]
        if self.isTrain:
            tfx.extend(train_fx)
            
        tfx.append(ToTensorV2(transpose_mask=(True)))
        
        transformer=A.Compose(tfx)
        data=transformer(image=self.images,mask=self.masks)
        
        return data["image"],data["mask"]
    def __len__(self):
        return len(self.images)
    
    
