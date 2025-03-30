import os
import torch
from torch.utils.data import Dataset
import cv2 

# Creating a custom class 
class CustomDataset(Dataset): 

    def __init__(self, root_dir, transforms = None):
        self.root_dir   = root_dir
        self.transforms = transforms
        self.classes    = os.listdir(root_dir)
        self.classes.sort()
        self.images     = []
        print("classes name ", self.classes)
        
        for clc in self.classes: 
            image_names  = os.listdir(self.root_dir + "/" + clc)
            self.images +=\
                  [self.root_dir + "/" + clc + "/" + img_name for img_name in image_names] 
        # print(image_names)
        # print(self.images)


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        image      = cv2.imread(image_path)
        image      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:

            image = self.transforms(image)
        
        image_class_name = image_path.split("/")[-2]
        label            = torch.tensor(self.classes.index(image_class_name))
        return image, label
    
