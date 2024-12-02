import pickle
import numpy as np
import os
import nibabel as nib
import cv2
from utils.utils_eval import keypoints_to_segmentation,  distances_to_kpts_lv, merge_masks
from metrics import calc_dice_per_segment, calc_hausdorf_per_segment, calc_multi_class_dice
from tqdm import tqdm

def resize_label_wise(label, new_shape,classes=None):
    '''
    Resize a label image label-wise to avoid artifacts due to interpolation
    :param label: reference image
    :param new_shape: new shape
    :param classes: list of classes to resize
    '''
    if classes is None:
        classes = [0, 1,2,3]
    result = np.zeros(new_shape)
    for i in classes:
        label_mask = label == i
        resized_label_mask = cv2.resize(np.uint8(label_mask), (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST)
        result[resized_label_mask == 1] = i
    return result


def pred_to_segmentation(prediction, displacement_model=True):
    kpts_pred = prediction["keypoints_prediction"]
    kpts_lv = kpts_pred[:43, :]
    kpts_la = np.concatenate((np.array(kpts_pred[42:64, :]),
                              np.expand_dims(np.array(kpts_pred[0, :]), 0)),
                             0)  # also includes the first annulus points (0 and 42)
    if displacement_model:
        displacements = prediction["distances_pred"]
        kpts_ep = distances_to_kpts_lv(kpts_lv, displacements, as_np_array=True,
                                       transpose=False)
    else:
        kpts_ep = kpts_pred[64:, :]

    # also plot pixelwise segmentation
    mask_lv, mask_la, mask_ep = keypoints_to_segmentation(256, kpts_lv, kpts_la, kpts_ep)
    return merge_masks(mask_lv, mask_la, mask_ep)



def evaluate_predictions(predictions_file_path,camus_path, displacement_model=True):

    ovarall_dices = []
    dices_per_seg = []
    hausdorfs_per_seg_mm = []

    predictions = pickle.load(open(predictions_file_path, 'rb'))
    print(f"Comparing predictions in {predictions_file_path} to CAMUS reference...")
    # evaluate predictions
    for key in tqdm(predictions):
        pred = predictions[key]
        seg_pred = pred_to_segmentation(pred, displacement_model)

        # key is a string of the form path_to_preprocessed_files/patientXXXX/patientXXXX_view_phase.npy
        rel_path = '/'.join(key.split('/')[-2:]) # rel_path is of the form patientXXXX/patientXXXX_view_phase.npy
        # change extension to nii.gz
        rel_path = rel_path.replace('.npy','_gt.nii.gz')
        # get the path to the corresponding ground truth file in the CAMUS dataset
        ref_path = os.path.join(camus_path,rel_path)
        # load the reference segmentation
        ref = nib.load(ref_path)
        spacing = ref.header.get_zooms()
        ref_data = ref.get_fdata().T

        classes = [1,2,3] # lv, myo, la

        resized_ref = resize_label_wise(ref_data, seg_pred.shape, classes=classes)
        resized_spacing = (spacing[1]*ref_data.shape[1]/seg_pred.shape[1], spacing[0]*ref_data.shape[0]/seg_pred.shape[0])

        # calculate metrics
        dice = calc_multi_class_dice(seg_pred, resized_ref, classes=classes)
        ovarall_dices.append(dice)
        hausdorf_per_seg_mm = calc_hausdorf_per_segment(seg_pred, resized_ref, classes=classes, voxelspacing=resized_spacing)
        hausdorfs_per_seg_mm.append(hausdorf_per_seg_mm)
        dice_per_seg = calc_dice_per_segment(seg_pred, resized_ref, classes=classes)
        dices_per_seg.append(dice_per_seg)


    print("Overall dice: ", np.mean(ovarall_dices), "±", np.std(ovarall_dices))
    print("Hausdorf per segment (left ventricle, myocardium, left atrium) in millimeters: ",
          np.mean(hausdorfs_per_seg_mm, axis=0), "±", np.std(hausdorfs_per_seg_mm, axis=0))
    print("Overall dice per segment (left ventricle, myocardium, left atrium): ",
          np.mean(dices_per_seg, axis=0), "±", np.std(dices_per_seg, axis=0))


if __name__ == '__main__':
    # change these to paths on your system
    # predictins_path is the path to the 'predictions.pkl' file generated when running eval.py
    predictions_path = '../experiments/logs/CAMUS_displacement_cv_1/GCN_multi_displacement_small/mobilenet2/trained/weights_CAMUS_displacement_cv_1_GCN_multi_displacement_small_best_loss_eval_on_CAMUS_displacement_cv_1/predictions.pkl'
    # camus_path is the path to the CAMUS dataset as downloaded from https://www.creatis.insa-lyon.fr/Challenge/camus/
    camus_path = '../data/local_data/database_nifti'

    evaluate_predictions(predictions_path,camus_path)

