import math
from math import floor
import numpy as np
from skimage.measure import find_contours

def write_log_entry(log_entry, logfile_object,print_too=False):
    logfile_object.write(log_entry+'\n')
    if print_too:
        print(log_entry)

def markersize(size,points):
    return [size for n in range(len(points))]

def merge_base_points(contour_LV,contour_LA,contour_EP=None):
    for index in [0,-1]: # first and last points in contour are BaseSep and BaseLat
        base_LV = contour_LV[index]
        base_LA = contour_LA[index]
        merged_base = [int(round((base_LV[0]+base_LA[0])/2)), # take average position
                           int(round((base_LV[1]+base_LA[1])/2))]
        contour_LV[index] = merged_base # Store merged landmarks in both contours
        contour_LA[index] = merged_base
        if contour_EP is not None:
            contour_EP[index] = merged_base
    if contour_EP is not None:
        return contour_LV,contour_LA,contour_EP
    else:
        return contour_LV,contour_LA

def get_epicardium_landmarks(segmentation,landmarks_LV):
    landmark_dict = dict()
    contour_ep = find_contours(segmentation == 2, level=0.5)[0]
    contour_ep_ar = []
    for point in contour_ep:
        x = int(round(point[1]))
        y = int(round(point[0]))
        contour_ep_ar.append([x, y])
    for landmark in ['BaseSep','BaseLat','ApexEpi']:
        distances = []
        for contour_point_ep in contour_ep_ar:
            distances.append(np.linalg.norm(landmarks_LV[landmark] - contour_point_ep))
        closest_index = np.argmin(distances)
        closest_contour_point = contour_ep_ar[closest_index]
        landmark_dict[landmark] = closest_contour_point
    landmark_dict['Apex']=landmark_dict['ApexEpi']
    return landmark_dict


def line(p1,p2,img_width=256,img_height=256,cont=True):
        x0, y0 = p1
        x1, y1 = p2
        steep = abs(y1 - y0) > abs(x1 - x0) #abs(r) > 0
        width = img_width
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            width=img_height
        flip = False
        if x0 > x1:
            x0, y0, x1, y1  = x1, y1, x0, y0
            flip =True
        r = (y1 - y0) / (x1 - x0)

        deltax = x1 - x0
        deltay = abs(y1 - y0)
        error = -deltax / 2

        if cont:
            y = y0-r*x0
            x0 = 0
        else:
            y = y0

        if y < y1:
            ystep = 1
        else:
            ystep = -1


        line = []
        if cont:
            it_range = range(0,width)
        else:
            it_range = range(x0, x1 + 1)

        for x in it_range:
            if steep:
                line.append((int(y), int(x)))
            else:
                line.append((int(x), int(y)))
            error = error + deltay
            if error > 0:
                y = y + ystep
                error = error - deltax
        if flip:
            line = np.flip(line,axis=0)
        return line

def line_single_point(p,r,img_width=256,dir='both'):
    x0, y0 = p
    if dir=='left' or dir=='both':
        y = y0 - r * x0
        x0 = 0
    else:
        y = y0
    line = []

    if dir=='right' or dir=='both':
        it_range = range(x0, img_width)
    else:
        it_range = range(x0, p[0])
    line.append((int(x0), int(y0)))
    for x in it_range:
        y = y+r
        point = (int(round(x)),int(round(y)))
        if not point in line:
            line.append(point)
    return line

def iteration_range_cetered(center,vicinity):
    iter_range = [*range(center - vicinity, center + vicinity + 1)]
    return sorted(iter_range, key=lambda x: -abs(x - center))


def intersection(curve1,curve2,hor_vicinity=0,ver_vicinity=0,strict=False):
    result = []
    for p in curve1:
        x1,y1 = p
        iter_range_hor = iteration_range_cetered(x1,hor_vicinity) # avoid bias, append points closer to
        iter_range_ver = iteration_range_cetered(y1,ver_vicinity) # actual line last
        for x2 in iter_range_hor:
            for y2 in iter_range_ver:
                if [x2,y2] in curve2:
                    if strict:
                        if not [x1,x2] in result:
                            result.append((x1, y1))
                    else:
                        if not [x2,y2] in result:
                            result.append((x2, y2))
    return np.array(result)


def get_contour(segmentation,label):
    contour = find_contours(segmentation == label, level=0.5)[0]
    contour_ar = []
    for point in contour:
        x = int(round(point[1]))
        y = int(round(point[0]))
        contour_ar.append([x, y])
    return contour_ar

def get_epicardium_landmarks_baseline_def(segmentation,landmarks_LV,ver_vicinity=0,x_size=256,
                                          y_size=256):
    landmark_dict = dict()
    contour_ep_ar = get_contour(segmentation, 2)
    base_line = line(landmarks_LV['BaseSep'],landmarks_LV['BaseLat'],img_width=x_size,img_height=y_size)
    vic = ver_vicinity
    landmarks_found = False
    while not landmarks_found:
        seg_base_intersec = intersection(base_line,contour_ep_ar,ver_vicinity=vic,strict=False)
        if vic>10:
            raise Exception("Could not find epicardium landmarks")
        if len(seg_base_intersec)<2:
            vic+=1
        elif seg_base_intersec[0][0] < landmarks_LV['BaseMid'][0] \
            and seg_base_intersec[-1][0] > landmarks_LV['BaseMid'][0]:
            landmark_dict['BaseSep'] = seg_base_intersec[0]
            landmark_dict['BaseLat'] = seg_base_intersec[-1]
            landmarks_found = True
        else:
            vic+=1

    distances = []
    for contour_point_ep in contour_ep_ar:
        distances.append(np.linalg.norm(landmarks_LV['ApexEpi'] - contour_point_ep))
    closest_index = np.argmin(distances)
    closest_contour_point = contour_ep_ar[closest_index]
    landmark_dict['Apex'] = closest_contour_point
    return landmark_dict

def extract_contour(segmentation, landmarks, label,nb_points):
    if label == 1:
        # Because the landmarks of lv are defined with flipping, we have to undo the flipping here
        segmentation = np.flipud(segmentation)
    contours = find_contours(segmentation == label, level=0.5) # Counter-clockwise
    lens = [len(cont) for cont in contours]
    contour = contours[np.argmax(lens)]
    # First, find left and right side..
    i = 0
    leftSide = False
    rightSide = False
    leftSideDone = False
    rightSideDone = False
    left_points = []
    right_points = []
    nb_points_prcessed = 0
    while not leftSideDone or not rightSideDone:
        point = contour[i]
        x = int(round(point[1]))
        if label == 1:
            # Because the landmarks of lv are defined with flipping, we have to undo the flipping here
            y = segmentation.shape[0] - int(round(point[0])) - 1  # Flip due to flipping on find_contours
        else:
            y = int(round(point[0]))
        current = np.array([x, y])

        # TODO: better structure code
        if label ==2:
            # now the apex is above the base points and we are going clockwise,
            # the left side starts at the BaseSep point and ends in the apex
            # the right side starts in the apex and ends in the BaseLat point
            if leftSide and not leftSideDone:
                if np.all(current == landmarks['Apex']):
                    leftSideDone = True
                else:
                    if len(left_points) > 0:
                        if np.linalg.norm(current - left_points[-1]) >= 1:
                            left_points.append(current)
                    else:
                        left_points.append(current)

            if rightSide and not rightSideDone:
                if np.all(current == landmarks['BaseLat']):
                    rightSideDone = True
                else:
                    if len(right_points) > 0:
                        if np.linalg.norm(current - right_points[-1]) >= 1:
                            right_points.append(current)
                    else:
                        right_points.append(current)

            if np.all(current == landmarks['BaseSep']):
                leftSide = True
            if np.all(current == landmarks['Apex']):
                rightSide = True
        else:
            # the left side starts in the apex and ends in the BaseSep point
            # the right side starts at the BaseLat point and ends in the apex
            if rightSide and not rightSideDone:
                if np.all(current == landmarks['Apex']):
                    rightSideDone = True
                else:
                    if len(right_points) > 0:
                        if np.linalg.norm(current - right_points[-1]) >= 1:
                            right_points.append(current)
                    else:
                        right_points.append(current)

            if leftSide and not leftSideDone:
                if np.all(current == landmarks['BaseSep']):
                    leftSideDone = True
                else:
                    if len(left_points) > 0:
                        if np.linalg.norm(current - left_points[-1]) >= 1:
                            left_points.append(current)
                    else:
                        left_points.append(current)

            if np.all(current == landmarks['BaseLat']):
                rightSide = True
            if np.all(current == landmarks['Apex']):
                leftSide = True
        i = (i + 1) % len(contour)
        if nb_points_prcessed > 5*len(contour):
            # this can happen when for example the segmentation contains two areas instead of one.
            return None
        nb_points_prcessed += 1

    right_points = np.array(right_points)
    left_points = np.array(left_points)

    N = nb_points[label-1]
    resampled_contour = []
    resampled_contour.append(landmarks['BaseLat'])
    for i in range(1, N+1):
        resampled_contour.append(right_points[floor((right_points.shape[0]/(N+1))*i), :]) # this can throw error on bad contours
    resampled_contour.append(landmarks['Apex'])
    for i in range(1, N+1):
        resampled_contour.append(left_points[floor((left_points.shape[0]/(N+1))*i), :])
    resampled_contour.append(landmarks['BaseSep'])

    resampled_contour.reverse()
    return np.array(resampled_contour)

def mirror_struct(struct,img_width,img_height):
    new_struct = []
    for point in struct:
        x,y = point
        new_struct.append([img_width-y,img_height-x])
    return new_struct

def get_normal(p1,p2):
    if p1[0] == p2[0]:
        normal_slope = 0
    else:
        slope = (p1[1] - p2[1]) / (p1[0] - p2[0])
        normal_slope = -1 / slope
    if p1[1] < p2[1]:
        dir = -1
    else:
        dir = 1
    return normal_slope,dir


def excract_normal_points_ep(contour_points_LV,seg):
    prev2_kpt = contour_points_LV[0]
    prev_kpt = contour_points_LV[1]
    result_points = []
    result_distances = []
    for kpt_lv in contour_points_LV[2:]:
        normal_slope, dir = get_normal(kpt_lv,prev2_kpt)
        x, y = prev_kpt
        prevx,prevy = x,y
        inside_ep = True
        while inside_ep:
            if abs(normal_slope) >1:
                dir2 = np.sign(normal_slope)*dir
                y=y+dir2
                x=x+1/normal_slope*dir2
            else:
                x = x+dir
                y = y+normal_slope*dir
            x_i,y_i = int(round(x)),int(round(y))
            if y_i >= seg.shape[0] or x_i >= seg.shape[1]:
                inside_ep = False
            elif seg[y_i,x_i] == 0:
                inside_ep=False
            else:
                prevx, prevy = x_i, y_i
        point_to_add = np.array([prevx,prevy])
        result_points.append(point_to_add)
        distance = np.linalg.norm(point_to_add-prev_kpt)
        result_distances.append(distance)
        prev2_kpt = prev_kpt
        prev_kpt = kpt_lv
    return np.array(result_points),np.array(result_distances)

def remove_duplicates(points,axis=1): # remove duplicates and keep order
    rounded_points= np.round(points)
    _, idx = np.unique(rounded_points, axis=axis, return_index=True)
    if axis == 1:
        new_points = rounded_points[:,np.sort(idx)]
    elif axis == 0:
        new_points = rounded_points[np.sort(idx),:]
    else:
        raise ValueError('axis not supported')
    return new_points

def add_ep_landmarks(ep_points,distances,landmarks_EP,landmarks_LV):
    result = list(ep_points)
    result_distances = list(distances)
    result = [landmarks_EP['BaseSep']] + result
    distance_basesep = np.linalg.norm(landmarks_EP['BaseSep']-landmarks_LV['BaseSep'])
    result_distances = [distance_basesep] + result_distances
    distance_baselat = np.linalg.norm(landmarks_EP['BaseLat']-landmarks_LV['BaseLat'])
    result.append(landmarks_EP['BaseLat'])
    result_distances.append(distance_baselat)
    return np.array(result),result_distances


def get_point_in_dir(x0,y0,angle,distance):
    x = x0 +distance*math.cos(angle)
    y = y0 +distance*math.sin(angle)
    return [x,y]

def distances_to_kpts(contour_LV,distances_ep,as_np_array=True,transpose=False):
    if transpose:
        contour_LV=np.transpose(contour_LV)
    result = []
    x0,y0 = contour_LV[0]
    x1,y1 = contour_LV[-1]
    angle_base = math.atan((y1-y0)/(x1-x0))
    ep_point = get_point_in_dir(x0,y0,angle_base+math.pi,distances_ep[0])
    result.append(ep_point)
    prev2_lv_point = contour_LV[0]
    prev_lv_point = contour_LV[1]
    for lv_point,idx in zip(contour_LV[2:],range(len(distances_ep[1:-1]))):
        r, dir = get_normal(lv_point, prev2_lv_point)
        angle = math.atan(r)
        if dir==-1:
            angle+=math.pi
        ep_point = get_point_in_dir(prev_lv_point[0], prev_lv_point[1], angle, distances_ep[idx])
        result.append(ep_point)
        prev2_lv_point = prev_lv_point
        prev_lv_point = lv_point
    ep_point = get_point_in_dir(x1,y1,angle_base,distances_ep[-1])
    result.append(ep_point)
    if as_np_array:
        result = np.array(result)
    return result

def near_border(lv_point,img,vic=5):
    x,y = lv_point
    norm_factorx = img.shape[0]
    norm_factory = img.shape[0]
    for x_offset,y_offset in zip([-vic,-vic,-vic,0,vic,vic,vic],[-vic,0,vic,vic,vic,0,-vic,-vic]):
        x1 = int(np.round(norm_factorx * x)) + x_offset
        y1 = int(np.round(norm_factory * y)) + y_offset
        if img[y1, x1]==0:
            return True
    return False

def extent_to_border(p1,p2,img):
    x0, y0 = p1
    x1, y1 = p2
    norm_factorx = img.shape[0]
    norm_factory = img.shape[0]
    x0 = int(np.round(x0*norm_factorx))
    x1 = int(np.round(x1*norm_factorx))
    y0 = int(np.round(y0*norm_factory))
    y1 = int(np.round(y1*norm_factory))
    #r = (y1 - y0) / (x1 - x0)
    steep = abs(y1 - y0) > abs(x1 - x0)  # abs(r) > 0
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if y0 < y1:
        ystep = 1
    else:
        ystep = -1
    deltax = abs(x1 - x0)
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y1
    x= x1
    if x0<x1:
        dir=1
    else:
        dir=-1
    while x<img.shape[0]:
        if steep:
            if img[x,y] == 0:
                return (y/norm_factory,x/norm_factorx)
        else:
            if img[y, x] == 0:
                return (x/norm_factorx, y/norm_factory)
        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
        x+=dir
    raise ValueError('Error: Failed to find border')






