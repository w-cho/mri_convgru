import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

import torch
from torch.autograd import Variable

from utils import AverageMeter
from utils.metrics import DiceCoef, cutoff_youdens_j
from utils.transforms import decode_preds

def train(net, dataset_trn, optimizer, criterion, epoch, opt):
    print("Start Training...")
    net.train()

    losses = AverageMeter()
    pred_probs, gt_labels = [], []

    for it, (img1, img2, img3, img4, vols, label, _) in enumerate(dataset_trn):
        # Optimizer
        optimizer.zero_grad()

        # Load Data
        img1, img2, img3, img4, label = [torch.Tensor(t).float() for t in [img1, img2, img3, img4, label]]
        if opt.use_gpu:
            img1, img2, img3, img4, label = [t.cuda(non_blocking=True) for t in [img1, img2, img3, img4, label]]
        
        # 1: Pre-RT only | 2: Pre+1st | 3: Pre~2nd | 4: Pre~3rd
        if opt.setting == 1:
            img2, img3, img4 = None, None, None
        if opt.setting == 2:
            img3, img4 = None, None
        if opt.setting == 3:
            img4 = None


        if opt.with_vol:
            vols = torch.Tensor(vols).float()

            if opt.use_gpu:
                vols = vols.cuda(non_blocking=True)

        label = label[:,None]

        # Predict
        pred = net(img1, img2, img3, img4, vols)

        # Loss Calculation
        loss = criterion(pred, label)

        # Backward and step
        loss.backward()
        optimizer.step()

        # Stack Results
        losses.update(loss.item(), img1.size(0))

        # Calculate accuracy
        pred_label = list(pred.sigmoid().detach().cpu().numpy()[:,0])
        pred_probs.extend(pred_label)
        gt_labels.extend(label.int().detach().cpu().numpy()[:,0])

        print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f'
            % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg), end='\r')

    auc = roc_auc_score(gt_labels, pred_probs)
    print("\n>>> Epoch[%3d/%3d] | Training Loss: %.4f | AUC: %.4f\n"
        % (epoch+1, opt.max_epoch, losses.avg, auc))


def validate(dataset_val, net, criterion, optimizer, epoch, opt, best_results):
    print("Start Evaluation...")
    net.eval()

    losses = AverageMeter()
    pred_probs, gt_labels = [], []

    for it, (img1, img2, img3, img4, vols, label, _) in enumerate(dataset_val):
        # Load Data
        img1, img2, img3, img4, label = [torch.Tensor(t).float() for t in [img1, img2, img3, img4, label]]
        if opt.use_gpu:
            img1, img2, img3, img4, label = [t.cuda(non_blocking=True) for t in [img1, img2, img3, img4, label]]

        # 1: Pre-RT only | 2: Pre+1st | 3: Pre~2nd | 4: Pre~3rd
        if opt.setting == 1:
            img2, img3, img4 = None, None, None
        if opt.setting == 2:
            img3, img4 = None, None
        if opt.setting == 3:
            img4 = None

        if opt.with_vol:
            vols = torch.Tensor(vols).float()

            if opt.use_gpu:
                vols = vols.cuda(non_blocking=True)

        label = label[:,None]

        # Predict
        with torch.no_grad():
            pred = net(img1, img2, img3, img4, vols)

        # Loss Calculation
        loss = criterion(pred, label)

        # Stack Results
        losses.update(loss.item(), img1.size(0))

        # Calculate accuracy
        pred_label = list(pred.sigmoid().detach().cpu().numpy()[:,0])
        pred_probs.extend(pred_label)
        gt_labels.extend(label.int().detach().cpu().numpy()[:,0])

    auc = roc_auc_score(gt_labels, pred_probs)
    print(">>> Epoch[%3d/%3d] | Validation Loss: %.4f | AUC %.4f"
        % (epoch+1, opt.max_epoch, losses.avg, auc))

    cutoff = cutoff_youdens_j(gt_labels, pred_probs)
    pred_labels = [int(prob>cutoff) for prob in pred_probs]

    tn, fp, fn, tp = confusion_matrix(gt_labels, pred_labels).ravel()
    print(">>> Confusion matrix: TP %d | FP %d| TN %d | FN %d\n" % (tp, fp, tn, fn))

    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    print(">>> Specificity %.4f | Sensitivity %.4f\n" % (specificity, sensitivity))

    # Update Result
    if auc > best_results[0]:
        print('Best Score Updated...')
        best_results = auc, epoch, specificity, sensitivity

        # Remove previous weights pth files
        for path in glob('%s/*.pth' % opt.exp):
            os.remove(path)

        model_filename = '%s/epoch_%04d_auc%.4f_spec%.4f_sens%.4f_loss%.8f.pth' %\
                (opt.exp, epoch+1, best_results[0], best_results[2], best_results[3], losses.avg)

        # Single GPU
        if opt.ngpu == 1:
            torch.save(net.state_dict(), model_filename)
        # Multi GPU
        else:
            torch.save(net.module.state_dict(), model_filename)

    print('>>> Current best: AUC %.4f | Spec %.4f | Sens %.4f in %3d epoch\n' %
            (best_results[0], best_results[2], best_results[3], best_results[1]+1))
    
    return best_results


def evaluate(dataset_val, net, opt):
    print("Start Evaluation...")
    net.eval()

    losses = AverageMeter()
    pred_probs, gt_labels, tumorids = [], [], []

    for it, (img1, img2, img3, img4, vols, label, tumorid) in enumerate(dataset_val):
        # Load Data
        img1, img2, img3, img4, label = [torch.Tensor(t).float() for t in [img1, img2, img3, img4, label]]
        if opt.use_gpu:
            img1, img2, img3, img4, label = [t.cuda(non_blocking=True) for t in [img1, img2, img3, img4, label]]

            
        if opt.setting == 1:
            img2, img3, img4 = None, None, None
        if opt.setting == 2:
            img3, img4 = None, None
        if opt.setting == 3:
            img4 = None
            
        if opt.with_vol:
            vols = torch.Tensor(vols).float()

            if opt.use_gpu:
                vols = vols.cuda(non_blocking=True)

        label = label[:,None]

        # Predict
        with torch.no_grad():
            pred = net(img1, img2, img3, img4, vols)

        # Calculate accuracy
        pred_label = list(pred.sigmoid().detach().cpu().numpy()[:,0])
        pred_probs.extend(pred_label)
        gt_labels.extend(label.int().detach().cpu().numpy()[:,0])
        tumorids.extend(tumorid)

    print("Cutoff threshold using 0.5")
    auc = roc_auc_score(gt_labels, pred_probs)
    print(">>> Evalution result | AUC %.4f" % auc)

    pred_labels = [int(prob>0.5) for prob in pred_probs]

    tn, fp, fn, tp = confusion_matrix(gt_labels, pred_labels).ravel()
    print(">>> Confusion matrix: TP %d | FP %d| TN %d | FN %d" % (tp, fp, tn, fn))

    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    print(">>> Specificity %.4f | Sensitivity %.4f\n" % (specificity, sensitivity))
        
    print("Cutoff threshold using Youden's J statistic")
    auc = roc_auc_score(gt_labels, pred_probs)
    print(">>> Evalution result | AUC %.4f" % auc)

    cutoff = cutoff_youdens_j(gt_labels, pred_probs)
    pred_labels = [int(prob>cutoff) for prob in pred_probs]

    tn, fp, fn, tp = confusion_matrix(gt_labels, pred_labels).ravel()
    print(">>> Confusion matrix: TP %d | FP %d | TN %d | FN %d" % (tp, fp, tn, fn))

    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    print(">>> Specificity %.4f | Sensitivity %.4f\n" % (specificity, sensitivity))


    return tumorids, gt_labels, pred_labels
    
    