import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
from datasets.brain import BrainDataset

def get_dataloader(opt):
    if opt.in_dim == 2:
        opt.data_root = './brain_data/crop_data/'
    elif opt.in_dim == 3:
        opt.data_root = './brain_data/crop_data2/'
    
    opt.csv_path = './brain_data/brain_data_rev2_split%d.csv' % opt.split
    print("Load Dataset from %s" % opt.csv_path)
    
    trn_dataset = BrainDataset(opt, is_Train=True, augmentation=opt.augmentation)
    val_dataset = BrainDataset(opt, is_Train=False, augmentation=False)

    train_dataloader = DataLoader(trn_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.workers)

    valid_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.workers)
    
    return train_dataloader, valid_dataloader