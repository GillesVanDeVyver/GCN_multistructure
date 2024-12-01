from medpy.metric.binary import hd,dc
import numpy as np

def calc_dice(seg, ref, k=1):
    return dc(seg==k, ref==k)

def calc_dice_per_segment(seg, ref, classes=None):
    if classes is None:
        classes = [1, 2, 3]
    dices = []
    for k in classes:
        dice_score = calc_dice(seg, ref, k)
        dices.append(dice_score)
    return dices

def calc_hausdorf_per_segment(seg, ref, classes=None, voxelspacing=None):
    if classes is None:
        classes = [1, 2, 3]
    hds = []
    for k in classes:
        hd_score = calc_hausdorf(seg, ref, k, voxelspacing)
        hds.append(hd_score)
    return hds

def calc_hausdorf(seg,gt,k=1, voxelspacing=None):
    return hd(seg==k,gt == k, voxelspacing)

def calc_multi_class_dice(seg, ref, classes=None): # all segments
    if classes is None:
        classes = [1, 2, 3]
    intersection = 0
    union = 0
    for k in classes:
        intersection += np.sum(seg[ref == k] == k) * 2.0
        union += (np.sum(seg[seg == k] == k) + np.sum(ref[ref == k] == k))
    dice = intersection/union
    return dice


