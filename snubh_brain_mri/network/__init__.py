import os
import torch
import torch.nn as nn
from network.conv_rnn import ConvRNN

def create_model(opt):
    # Load network
    net = ConvRNN(opt)

    # GPU settings
    if opt.use_gpu:
        net.cuda()
        if opt.ngpu > 1:
            net = torch.nn.DataParallel(net)
    
    if opt.resume:
        if os.path.isfile(opt.resume):
            pretrained_dict = torch.load(opt.resume, map_location=torch.device('cpu'))
            net.load_state_dict(pretrained_dict)
            print("=> Successfully loaded weights from %s" % (opt.resume,))

        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    return net