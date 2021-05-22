import torch

from torch.utils.data import Dataset
from PIL import Image


Image.MAX_IMAGE_PIXELS = 900000000

import numpy as np

def load_image(filename):
    # h, w= shape
    image_array = Image.open(str(filename))
    image_array=image_array.convert("RGB")
    # image_array = image_array.resize((h,w))
    # image_array = np.asarray(image_array)/255.0
    return image_array

class PerchWideDataset(Dataset):
    def __init__(self,data,label_var=None,path_var='resized_path',transforms=None):
        self.data=data
        self.label_var=label_var
        self.transforms=transforms
        self.path_var=path_var

    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, item):
        row=self.data.iloc[item,:]
        filepath=row[self.path_var]
        image=load_image(filepath)
        # image=torch.from_numpy(image)

        if self.transforms is not None:
            image=self.transforms(image)
        if self.label_var:
            label=row[self.label_var]
            return image,label
        return image


class PerchLongDataset(Dataset):
    def __init__(self,data,label_var=None,path_var='resized_path',transforms=None,sample_by=None):
        self.data=data
        self.label_var=label_var
        self.transforms=transforms
        self.sample_by=sample_by
        self.path_var=path_var
        if sample_by is not None:
            self.unique_ids=data[sample_by].unique()

    def __len__(self):
        if self.sample_by:
            return len(self.unique_ids)
        return self.data.shape[0]
    def __getitem__(self, item):
        if self.sample_by:
            row=self.data.loc[self.data[self.sample_by]==self.unique_ids[item],:].sample(1).iloc[0,:]
        else:
            row=self.data.iloc[item,:]
        filepath=row[self.path_var]
        image=load_image(filepath)
        rev=row['reviewer']

        if self.transforms is not None:
            image=self.transforms(image)
        if self.label_var:
            label=row[self.label_var]
            return image,rev,label
        return image,rev