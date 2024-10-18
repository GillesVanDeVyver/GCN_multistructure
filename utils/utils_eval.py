import cv2
import numpy as np
import scipy.interpolate as interpolate
import math

from sympy import transpose


def get_point_in_dir(x0, y0, angle, distance):
    x = x0 + distance * math.cos(angle)
    y = y0 + distance * math.sin(angle)
    return [x, y]


def get_normal(p1, p2):
    if p1[0] == p2[0]:
        normal_slope = 0
    else:
        slope = (p1[1] - p2[1]) / (p1[0] - p2[0])
        normal_slope = -1 / slope
    if p1[1] < p2[1]:
        dir = -1
    else:
        dir = 1

    return normal_slope, dir


def distances_to_kpts_lv(contour_LV, distances_ep, as_np_array=True, transpose=False):
    if transpose:
        contour_LV = np.transpose(contour_LV)
    result = []
    x0, y0 = contour_LV[0]
    x1, y1 = contour_LV[-1]
    angle_base = math.atan((y1 - y0) / (x1 - x0))
    ep_point = get_point_in_dir(x0, y0, angle_base + math.pi, distances_ep[0])
    result.append(ep_point)
    prev2_lv_point = contour_LV[0]
    prev_lv_point = contour_LV[1]
    for lv_point, idx in zip(contour_LV[2:], range(len(distances_ep[1:-1]))):
        r, dir = get_normal(lv_point, prev2_lv_point)
        angle = math.atan(r)
        if dir == -1:
            angle += math.pi
        ep_point = get_point_in_dir(prev_lv_point[0], prev_lv_point[1], angle, distances_ep[idx])
        result.append(ep_point)
        prev2_lv_point = prev_lv_point
        prev_lv_point = lv_point
    ep_point = get_point_in_dir(x1, y1, angle_base, distances_ep[-1])
    result.append(ep_point)
    if as_np_array:
        result = np.array(result)
    return result

def denormalise_kpts(kpts,img_size,transpose=True):
    # check if img_size is an array
    if isinstance(img_size, tuple):
        img_size_0 = img_size[0]
        img_size_1 = img_size[1]
    else:
        img_size_0 = img_size
        img_size_1 = img_size
    if transpose:
        for p in kpts.T:
            p[0] = int(np.round(p[0] * img_size_0))
            p[1] = int(np.round(p[1] * img_size_1))
    else:
        for p in kpts:
            p[0] = int(np.round(p[0] * img_size_0))
            p[1] = int(np.round(p[1] * img_size_1))

def distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def draw_line(p1, p2, interpolation_step_size):
    nb_points = int(distance(p1, p2) / interpolation_step_size)
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)
    return np.array([[int(np.round(p1[0] + i * x_spacing)), int(np.round(p1[1] +  i * y_spacing))]
            for i in range(1, nb_points+1)]).T

def interpolate_part(keypoints,k,interpolation_step_size):
    pred_tck, _ = interpolate.splprep([keypoints[:, 0], keypoints[:, 1]], s=0,k=k)
    interpol = np.arange(0, 1.00, interpolation_step_size)
    contour = np.array(interpolate.splev(interpol, pred_tck))
    return contour

def keypoints_to_segmentation(img_size, keypoints_lv, keypoints_la=None, keypoints_ep=None,
                              interpolation_step_size=0.001, denormalise=True,
                              img_sizey=None):
    if img_sizey is None:
        img_sizey = img_size
    interpolation_step_size_line = interpolation_step_size * 50
    mask_lv = np.zeros((img_size, img_sizey))
    if keypoints_la is not None:
        mask_la = np.zeros((img_size, img_sizey))
    else:
        mask_la = None
    if keypoints_ep is not None:
        mask_ep = np.zeros((img_size, img_sizey))
    else:
        mask_ep = None

    pred_tck_lv, _ = interpolate.splprep([keypoints_lv[:, 0], keypoints_lv[:, 1]], s=0)
    interpol_lv = np.arange(0, 1.00, interpolation_step_size)
    contour_lv = np.array(interpolate.splev(interpol_lv, pred_tck_lv))
    if denormalise:
        denormalise_kpts(contour_lv, img_size)
    base_left = contour_lv[:, 0]
    base_right = contour_lv[:, -1]
    base = draw_line(base_left, base_right, interpolation_step_size_line)
    contour_lv_closed = np.concatenate((contour_lv, base), 1)
    cv2.fillPoly(mask_lv, [contour_lv_closed.T.astype(np.int32)], color=(255, 0, 0))
    if keypoints_la is not None:
        pred_tck_la, _ = interpolate.splprep([keypoints_la[:, 0], keypoints_la[:, 1]], s=0)
        interpol_la = np.arange(0, 1.00, interpolation_step_size)
        contour_la = np.array(interpolate.splev(interpol_la, pred_tck_la))
        if denormalise:
            denormalise_kpts(contour_la, img_size)
        contour_la = np.concatenate((contour_la, base), 1)
        cv2.fillPoly(mask_la, [contour_la.T.astype(np.int32)], color=(255, 0, 0))
    if keypoints_ep is not None:
        nb_p = len(keypoints_ep)

        contour_ep_part1 = interpolate_part(keypoints_ep[:4], 1, interpolation_step_size * (5 / nb_p))
        contour_ep_part2 = interpolate_part(keypoints_ep[4:nb_p - 4], 3, interpolation_step_size * (33 / nb_p))
        contour_ep_part3 = interpolate_part(keypoints_ep[nb_p - 4:], 1, interpolation_step_size * (5 / nb_p))
        contour_ep = np.concatenate((contour_ep_part1, contour_ep_part2, contour_ep_part3), axis=1)

        if denormalise:
            denormalise_kpts(contour_ep, img_size)
        conn1_p1 = contour_lv[:, 0]
        conn1_p2 = contour_ep[:, 0]
        conn1 = draw_line(conn1_p1, conn1_p2, interpolation_step_size_line)
        conn2_p1 = contour_ep[:, -1]
        conn2_p2 = contour_lv[:, -1]
        conn2 = draw_line(conn2_p1, conn2_p2, interpolation_step_size_line)
        contour_ep = np.concatenate((conn1, contour_ep, conn2, np.flip(contour_lv, 1)), 1)
        cv2.fillPoly(mask_ep, [contour_ep.T.astype(np.int32)], color=(255, 0, 0))
    if mask_lv is not None:
        mask_lv = mask_lv.astype(np.int32)
    if mask_la is not None:
        mask_la = mask_la.astype(np.int32)
    if mask_ep is not None:
        mask_ep = mask_ep.astype(np.int32)

    return mask_lv, mask_la, mask_ep

def merge_masks(mask_lv,mask_la,mask_ep,one_hot_val=255):
    merged_mask=1*(mask_lv==one_hot_val)+\
                3*((mask_la==one_hot_val) & (mask_lv==0))+\
                2*((mask_ep==one_hot_val) & (mask_lv==0) & (mask_la==0))
    return merged_mask