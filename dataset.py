import os
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import os


class Dataset(BaseDataset):
    CLASSES = ['aegirine', 'aegirine augite', 'alkaline feldspar', 'apatite', 'biotite', 'calcite', 'chloritic pyroxene', 
               'clinopyroxene', 'feldspar', 'hornblende', 'hypersthene', 'ilmenite', 'mica', 'muscovite', 'nepheline',
                  'nosean', 'olivine', 'orthoclase', 'orthopyroxene', 'perilla pyroxene', 'phlogopite', 'plagioclase',
                   'pyroxene', 'quartz', 'quartz cryptocrystalline', 'sanidine', 'sodalite', 'sodium iron amphibole',
                     'spinel', 'titanaugite' , 'zircon']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        mask = cv2.imread(self.masks_fps[i], 0)
        
        masks = [(mask == (v+1)) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate((mask, background), axis=-1)
        

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)