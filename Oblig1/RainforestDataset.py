import torch
from torch.utils.data import Dataset
import PIL.Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import csv
import numpy as np

torch.manual_seed(0)

def get_classes_list():
    classes = ['clear', 'cloudy', 'haze', 'partly_cloudy',
               'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
               'blow_down', 'conventional_mine', 'cultivation', 'habitation',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    return classes, len(classes)

class ChannelSelect(torch.nn.Module):
    """This class is to be used in transforms.Compose when you want to use selected channels. e.g only RGB.
    It works only for a tensor, not PIL object.
    Args:
        channels (list or int): The channels you want to select from the original image (4-channel).

    Returns: img
    """
    def __init__(self, channels=[0, 1, 2]):
        super().__init__()
        self.channels = channels

    def forward(self, img):
        """
        Args:
            img (Tensor): Image
        Returns:
            Tensor: Selected channels from the image.
        """
        return img[self.channels, ...]

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RainforestDataset(Dataset):
    def __init__(self, root_dir, trvaltest, transform):

        self.root_dir = root_dir
        self.name_label_file = root_dir + 'train_v2.csv'
        self.image_dir = root_dir + 'train-tif-v2/'
        
        self.transform = transform
        self.imgfilenames=[]
        self.labels=[]  

        classes, num_classes = get_classes_list()

        with open(self.name_label_file, 'r') as file: # open file
            reader = csv.reader(file) # make reader object
            next(reader) # Skip header line
            for row in reader:
                self.imgfilenames.append(row[0] + '.tif')
                self.labels.append(str.split(row[1]))        

        self.mlb = MultiLabelBinarizer(classes=classes) # Initzialize ML Binarizer
        self.binarized_labels = self.mlb.fit_transform(self.labels)# Binarize the labels


        # Test/Train split
        self.imgfilenames_train, self.imgfilenames_test, self.binarized_labels_train,\
            self.binarized_labels_test = train_test_split(self.imgfilenames,
                self.binarized_labels, test_size=0.33, random_state=0)

        if trvaltest==0: # train
            self.imgfilenames = self.imgfilenames_train
            self.binarized_labels = self.binarized_labels_train
        if trvaltest==1: # val
            self.imgfilenames = self.imgfilenames_test
            self.binarized_labels = self.binarized_labels_test

    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.image_dir + self.imgfilenames[idx])

        if self.transform:
            img = self.transform(img)

        label = self.binarized_labels[idx].astype(np.float32)

        sample = {'image': img,
                  'label': label,
                  'filename': self.imgfilenames[idx]}
        return sample