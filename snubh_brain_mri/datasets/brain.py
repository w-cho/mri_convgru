import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
from glob import glob

from utils.transforms import augment_imgs


class BrainDataset(Dataset):
    def __init__(self, opt, is_Train=True, augmentation=False, return_id=True):
        super(BrainDataset, self).__init__()

        self.pat_list = glob(os.path.join(opt.data_root, '*-*'))
        self.annots = pd.read_csv(opt.csv_path)
        self.return_id = return_id

        n_total = len(self.pat_list)
        if is_Train:
            train_patients = self.annots[self.annots['train'] == 1]['patno'].unique()
            self.pat_list = [path for path in self.pat_list 
                            if int(os.path.basename(path).split('-')[0]) in train_patients]
            print("Split-%d Training dataset (N=%d)" % (opt.split, len(self.pat_list)))

        else:
            valid_patients = self.annots[self.annots['train'] == 0]['patno'].unique()
            self.pat_list = [path for path in self.pat_list 
                            if int(os.path.basename(path).split('-')[0]) in valid_patients]
            print("Split-%d Validation dataset (N=%d)" % (opt.split, len(self.pat_list)))

        # Error cases
        self.pat_list = [path for path in self.pat_list if \
            ('042-3' not in path) and ('002-4' not in path) and ('031-1' not in path) and ('014-2') not in path and ('088-6') not in path]

        self.len = len(self.pat_list)

        self.augmentation = augmentation
        self.rot_factor = opt.rot_factor
        self.scale_factor = opt.scale_factor
        self.flip = opt.flip
        self.trans_factor = opt.trans_factor

        # Add volume feature
        self.with_vol = opt.with_vol

        self.is_Train = is_Train
        self.opt = opt

    def __getitem__(self, index):
        # Patient Info
        patDir = self.pat_list[index]
        patno, tumorno = list(map(int, patDir.split(os.sep)[-1].split('-')))

        # Load images
        img_list = sorted(glob(os.path.join(patDir, '*', 'image.npy')))
        imgs = [np.load(path).astype(np.float32) for path in img_list]
        
        # Load masks
        mask_list = sorted(glob(os.path.join(patDir, '*', 'label.npy')))
        masks = [np.load(path).astype(np.float32) for path in mask_list]

        # Concat img and mask
        if self.opt.in_dim == 2:
            imgs = [np.concatenate([img, mask], 0) for img, mask in zip(imgs, masks)]
        elif self.opt.in_dim == 3:
            imgs = [np.concatenate([img, mask], 0)[:,None] for img, mask in zip(imgs, masks)]

        if self.augmentation and np.ndim(imgs[0])==2:
            imgs = augment_imgs(imgs, self.rot_factor, self.scale_factor, self.trans_factor, self.flip)

        # Load ground truth label
        annot_row = self.annots[(self.annots.patno == patno) &(self.annots.tumorno == tumorno)]
        label = annot_row.label
        label = label.values[0].astype(np.float32)
        
        if self.with_vol:
            vols = annot_row[['pre', 'tre1', 'tre2', 'tre3']].values[0].astype(np.float32)
        else:
            vols = -1.
                    
        if self.return_id:
            return imgs[0], imgs[1], imgs[2], imgs[3], vols, label, '%d-%d' % (patno, tumorno) 

        return imgs[0], imgs[1], imgs[2], imgs[3], vols, label

    def __len__(self):
        return self.len