# learnPytorch
Following the tutorial at [Udemy](https://www.udemy.com/course/deep-learning-image-classification-in-pytorch-20)


## [Google Colab](https://colab.research.google.com/)
Creater a new noteboot and choose Python3 runtime and T4 GPU which can be accessed for free

Some usefule commands 
`!nvidia-smi`
- nvidia-smi is a command-line tool provided by NVIDIA that shows the status of your GPU(s).
- The ! tells the notebook to run it as a shell command instead of Python code.
![image](https://github.com/user-attachments/assets/51ae6790-7aa7-404b-92fd-bab53a31a994)

## The Data Set 
The data set is a collection of images with 7 classes. With dedicated training and testing classes. 

![image](https://github.com/user-attachments/assets/3c5574ba-5bdf-4b15-997a-a4595f14a211)

We have 2 versions just to have a smaller data set for training and debugging purposes. 


## Accessing your google drive from colab 
```
from google.colab import drive
drive.mount('/content/drive/')
```

## Data Processing 
There are two ways to load  the images 
- ImageFolder: `torchvision.datasets.imageFolder(root,transform,target_transform,loader,is_valid_file)
  - root (string): root directory  --> Mandotory 
  - transform: function to take PIL Image nad transform
  - target_transform: a funcion that takes target and tramsform it  
  - loader: function to load 
  - is_valid_file: sees if its a valid file 

Eg: 
```
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dataset_path = "../data/version2/train"
test_dataset_path  = "../data/version2/test"

train_dataset      = datasets.ImageFolder(train_dataset_path)
```





- Custom Class 
