import os
import cv2
import argparse
import numpy as np


def compute_hist(pred, gt, cl):
    valid = (gt >= 0) & (gt < cl)
    return np.bincount(cl * gt[k].astype(int) + pred[k], minlength=cl**2).reshape(cl, cl)

def evalute(pred_dir, gt_dir, cl):
    lst = os.listdir(pred_dir)
    hist = np.zeros((cl, cl))
    for x in lst:
        pred = cv2.imread(pred_dir + x, cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(gt_dir + x, cv2.IMREAD_UNCHANGED)
        hist += compute_hist(pred, gt, cl)
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print('overall accuracy', acc)
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print('mean accuracy', np.nanmean(acc))
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('cat IU:', iu)
    print('mean IU', np.nanmean(iu))
    freq = hist.sum(1) / hist.sum()
    print('fwavacc', (freq[freq > 0] * iu[freq > 0]).sum())
    return hist
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--pred_dir', type=str, required=True, help='The directory where pred store')
    parser.add_argument('--gt_dir', type=str, required=True, help='The directory where gt store')
    parser.add_argument('--num_class', type=int, default=5, help='The number of  classes')

    args = parser.parse_args()
    evalute(args.pred_dir, args.gt_dir, args.num_class)
