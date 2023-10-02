import numpy as np
from skimage.segmentation import find_boundaries
from skimage.measure import find_contours


def find_apex(segmentation, label, baseMid):
    asd = segmentation == label
    if label == 1 or label == 2:
        asd = np.flipud(asd)
    contours = find_contours(asd, level=0.5) # Flip to avoid issue with contour starting at bottom
    contour = []
    for i in range(len(contours)):
        contour.extend(contours[i])

    new_contour = []
    for point in contour:
        x = int(round(point[1]))
        if label == 1 or label == 2:
            y = segmentation.shape[0] - int(round(point[0])) - 1 # Flip due to flipping on find_contours
        else:
            y = int(round(point[0]))
        new_contour.append(np.array([x, y]))

    # Find apex, as point longest away from baseMid
    apex = new_contour[0]
    for point in new_contour:
        if np.linalg.norm(baseMid - point) > np.linalg.norm(baseMid - apex):
            apex = point

    return apex

def find_apexv1(base_boundary,LV_boundary,baseMid):

    # Find apex point on the endocardium, as point longest away from baseMid
    apex = base_boundary[0]
    distances = []
    points = []
    for point in LV_boundary:
        distances.append(np.linalg.norm(baseMid - point['pos']))
        points.append(point['pos'])
        # Old method
        #if np.linalg.norm(baseMid - point['pos']) > np.linalg.norm(baseMid - apex):
        #    apex = point['pos']

    apex_threshold = 0.9
    while True:
        # Get all the points on the endocardium which can be good candidates for apex (far enough)
        distances_to_apex = np.array(distances)
        apex_candidate_indexes = np.where(distances >= (apex_threshold * np.max(distances_to_apex)), 1, 0)
        possible_apex = np.where(distances >= (apex_threshold * np.max(distances_to_apex)), distances, 0)
        # Find the transitions between points that are apex candidates and not
        apex_candidate_index_limits = np.argwhere(np.abs(np.diff(apex_candidate_indexes)) == 1)
        if len(apex_candidate_index_limits) == 2:
            apex_candidate_index_start = np.argwhere(np.diff(apex_candidate_indexes) == 1)[0][0]
            apex_candidate_index_end = np.argwhere(np.diff(apex_candidate_indexes) == -1)[0][0] + 1 # Add 1 to retablish dissymetry caused by np.diff

            # Place true apex
            if apex_candidate_index_start > apex_candidate_index_end:
                apex_index = int(round(((apex_candidate_index_end + len(distances) + apex_candidate_index_start) / 2) % len(distances), 0))
            else:
                apex_index = int(round((apex_candidate_index_end + apex_candidate_index_start) / 2, 0))
            apex = points[apex_index]
            break
        else:
            # Two segments on the LV contour are detected to be candidated to be apex
            # Reduce the threshold until getting a single segment
            apex_threshold -= 0.02

    return apex


def find_apexv2(LV_boundary,baseLat,baseSep):

    # Find apex point on the endocardium, as point with highest metric
    # we want points with
    # 1) high distance from base points and midpoint
    # 2) low difference in distances from two base points
    # so the metric is (sum of distances from base points) + (difference between distances from base points)
    distances_Lat = []
    distances_Sep = []
    #distances_Mid = []
    points = []
    for point in LV_boundary:
        distances_Lat.append((np.linalg.norm(baseLat - point['pos'])))
        distances_Sep.append((np.linalg.norm(baseSep - point['pos'])))
        #distances_Mid.append((np.linalg.norm(baseMid - point['pos'])))
        points.append(point['pos'])

    #summed_distances = np.add(np.add(distances_Lat,distances_Sep),distances_Mid)
    summed_distances = np.add(distances_Lat,distances_Sep)

    diff_distances =  np.abs(np.subtract(distances_Lat, distances_Sep))

    metric = np.subtract(summed_distances,diff_distances)

    # Get point with highest metric
    apex_index = np.argmax(metric)
    apex = points[apex_index]

    return apex


def _landmarks(segmentation, label, strict_base=True, return_apex_epi=False):
    if segmentation.dtype != np.uint8:
        raise ValueError('Input segmentation has to be in categorial format')

    asd = segmentation == label
    dir = -1
    if label == 1 or label == 2:
        dir = 1
        asd = np.flipud(asd)
    contours = find_contours(asd, level=0.5) # Flip to avoid issue with contour starting at bottom
    lens = [len(cont) for cont in contours]
    contour = contours[np.argmax(lens)]

    # Classify each contour point
    LV_boundary = []
    base_boundary = []
    for point in contour:
        x = int(round(point[1]))
        if label == 1 or label == 2:
            y = segmentation.shape[0] - int(round(point[0])) - 1 # Flip due to flipping on find_contours
        else:
            y = int(round(point[0]))
        type = 'other'
        # Check if atrium is below
        for offset in range(dir*1, dir*5, dir*1):
            if y + offset < segmentation.shape[0]:
                if not strict_base:
                    if segmentation[y + offset, x] == 0:
                        if label == 1:
                            type = 'base'
                            break
                if segmentation[y + offset, x] == 1:
                    # LV
                    if label == 3:
                        type = 'base'
                        break
                elif segmentation[y + offset, x] == 2:
                    # Myocardium
                    pass
                elif segmentation[y + offset, x] == 3:
                    # Atrium
                    if label == 1:
                        type = 'base'
                        break
            else:
                type = 'other'
        if type == 'base':
            base_boundary.append(np.array([x, y]))
        LV_boundary.append(
            {
                'pos': np.array([x, y]),
                'type': type,
            }
        )

    if len(base_boundary) == 0:
        raise Exception('Error finding base boundary')


    base_boundary = sorted(base_boundary , key=lambda k: [k[0], k[1]])

    baseSep = base_boundary[0]
    baseMid = base_boundary[int(len(base_boundary)/2)]
    baseLat = base_boundary[-1]


    apex = find_apexv2(LV_boundary,baseLat,baseSep)
    #apex = find_apexv1(base_boundary, LV_boundary,baseMid)

    """import matplotlib.pyplot as plt
    plt.plot(distances, '*')
    plt.plot(possible_apex, '*')
    plt.show()"""

    if return_apex_epi:
        # Find apex point on the epicardium, as intersection of the long axis (Apex-BaseMid) and the epicardial border
        apexEpi = apex.copy()
        LV_dir = np.array([baseMid[0]-apex[0], baseMid[1]-apex[1]])
        LV_dir = LV_dir / LV_dir[1]
        for depth in range(5, apex[1]):
            point = (apex - LV_dir * depth).astype(int)
            if segmentation[point[1], point[0]] == 2:
                apexEpi = point
            else:
                break

        return {'BaseSep': baseSep, 'BaseLat': baseLat, 'BaseMid': baseMid, 'Apex': apex, 'ApexEpi': apexEpi}
    else:
        return {'BaseSep': baseSep, 'BaseLat': baseLat, 'BaseMid': baseMid, 'Apex': apex}

def left_ventricular_landmarks(segmentation, strict_base=True, return_apex_epi=True):
    """
    Calculates the three basal landmarks and the apex as x, y coordinates, from a segmentation image
    :return: a dictionary of numpy arrays {'BaseSep': (x,y), 'BaseMid': (x,y) 'BaseLat': (x, y), 'Apex': (x, y)}
    """
    return _landmarks(segmentation, 1, strict_base, return_apex_epi= return_apex_epi)


def left_atrium_landmarks(segmentation, strict_base=True):
    """
    Calculates the three basal landmarks and the apex as x, y coordinates, from a segmentation image
    :return: a dictionary of numpy arrays {'BaseSep': (x,y), 'BaseMid': (x,y) 'BaseLat': (x, y), 'Apex': (x, y)}
    """
    return _landmarks(segmentation, 3, strict_base)
