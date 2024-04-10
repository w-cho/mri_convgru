import torch
import torch.nn as nn

from torchvision.models import resnet34
from network.resnet3d import BasicBlock
from network.resnet3d import ResNet as resnet_3d
from network.rnn import RNNModel, LSTMModel, GRUModel, GRUModel_atn

class ConvRNN(nn.Module):
    def __init__(self, opt):
        super(ConvRNN, self).__init__()
        if opt.in_dim == 2:
            self.backbone = resnet34()
            self.backbone.conv1 = nn.Conv2d(
                opt.in_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        
        elif opt.in_dim == 3:
            self.backbone = resnet_3d(BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512], n_input_channels=128)

        self.backbone.fc = nn.Identity()

        if opt.classifier.lower() == 'rnn':
            self.rnn = RNNModel(512, hidden_dim=256, layer_dim=1, output_dim=opt.n_classes, dropout_prob=0.1)
        elif opt.classifier.lower() == 'lstm':
            self.rnn = LSTMModel(512, hidden_dim=256, layer_dim=1, output_dim=opt.n_classes, dropout_prob=0.1)
        elif opt.classifier.lower() == 'gru':
            self.rnn = GRUModel(512, hidden_dim=256, layer_dim=1, output_dim=opt.n_classes, dropout_prob=0.1)
        elif opt.classifier.lower() == 'gru2':
            self.rnn = GRUModel_atn(512, hidden_dim=256, layer_dim=1, output_dim=opt.n_classes, dropout_prob=0.1)
        elif opt.classifier.lower() == 'fc':
            fc_in_channels = 512*4 + 4 if opt.with_vol else 512*4
            self.classifier = nn.Sequential(
                nn.Linear(fc_in_channels, 128),
                nn.Dropout(0.3),
                nn.Linear(128, opt.n_classes)
            )
        
        self.opt = opt
    
    def forward(self, img1, img2=None, img3=None, img4=None, vols=None):
        # For CAM code
        if img2 is None and self.opt.cam:
            img1, img2, img3, img4 = img1

        feat1 = self.backbone(img1)
        feats = [feat1]

        if img2 is not None:
            feat2 = self.backbone(img2)
            feats.append(feat2)

        if img3 is not None:
            feat3 = self.backbone(img3)
            feats.append(feat3)

        if img4 is not None:
            feat4 = self.backbone(img4)
            feats.append(feat4)
        
        if self.opt.with_vol:
            feats.append(vols)

        if self.opt.classifier == 'fc':
            feats = torch.cat(feats, dim=1)
            out = self.classifier(feats)

        else:
            feats = torch.stack(feats, dim=1)
            out = self.rnn(feats)

        return out