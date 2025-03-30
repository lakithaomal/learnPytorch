import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image 
from dataProcessing.data_loader_with_custon_data import CustomDataset



def load_data(data_path, image_size, batch_size = 16):
    train_dataset_path = data_path + "/train"
    test_dataset_path  = data_path + "/test"

    transformers_train = transforms.Compose([\
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ColorJitter(\
                    brightness=.5,
                    contrast=.5,
                    saturation=.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=.1),
        transforms.RandomRotation(degrees= 30),
        transforms.ToTensor()
    ])
    transformers_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(\
                        train_dataset_path,\
                        transforms= transformers_train\
                        )
    test_dataset = CustomDataset(\
                        test_dataset_path,\
                        transforms= transformers_test\
                        )
    
    print("No of samples in train data set: ",  len(train_dataset))
    print("No of samples in test data set : ",  len(test_dataset))

    train_loader = DataLoader(
                    train_dataset,\
                          batch_size=batch_size,\
                            shuffle=True)

    test_loader = DataLoader(
                    test_dataset,\
                          batch_size=batch_size,\
                            shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    root_path = "data/version2"
    train_data, test_data = load_data(root_path,\
                                      (224,224),\
                                            32)

    for data in train_data: 
        images, labels = data 
        print("Train Images Shape: ", images.shape)
        print("Train Labels ", labels)
        save_image(images,"images_train.jpg")
        break

    for data in test_data: 
        images, labels = data 
        print("Test Images Shape: ", images.shape)
        print("Test Labels ", labels)
        save_image(images,"images_test.jpg")
        break 


    
            