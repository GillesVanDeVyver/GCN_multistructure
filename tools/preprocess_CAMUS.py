"""
This script shows how to extract contours from recordings
The script will iterate over the subjects in the input directory
and create a .txt log file in the current directory logging errors and successes.

The input dir variable should point to the directory containing the CAMUS dataset.
It can be downloaded here:
https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8

The output dir variable should point to the directory where the extracted contours should be saved.

The log file is meant to be used for debugging purposes and for other datasets that are less 'clean' than CAMUS.
The log file will contain the following information for each frame:
- The subject ID
- The frame number
- The number of points extracted for each contour
- The error message if an error occurred
- The path to the saved contour if the extraction was successful
"""
import os
import numpy as np
from landmarks import left_ventricular_landmarks, left_atrium_landmarks
import matplotlib.pyplot as plt
from utils.utils_tools import write_log_entry, markersize, merge_base_points, get_epicardium_landmarks, \
    extract_contour, get_epicardium_landmarks_baseline_def,excract_normal_points_ep,add_ep_landmarks
import nibabel as nib
import gzip
import shutil
from tqdm import tqdm

input_dir = 'data/local_data/database_nifti'
output_dir = 'data/local_data/CAMUS_keypoint_annotations'
plot_dir = 'data/local_data/keypoint_extraction_log'
log_file_loc = 'data/local_data/keypoint_extraction_log/CAMUS_preprocess_log.txt'

if not os.path.exists('data/local_data/keypoint_extraction_log'):
    os.mkdir('data/local_data/keypoint_extraction_log')
if not os.path.exists('data/local_data/CAMUS_keypoint_annotations'):
    os.mkdir('data/local_data/CAMUS_keypoint_annotations')


NUMB_POINTS_LV = 20 # 20 each side + apex + base points = 43
NUMB_POINTS_EP = 20
NUMB_POINTS_LA = 10 # 20 each side + apex = 21
nb_points = [NUMB_POINTS_LV,NUMB_POINTS_EP,NUMB_POINTS_LA]

#show extraction of contours (for debugging)
show_plot=False

# split the data into ED and ES
ed_es_split = False

logfile_object = open(log_file_loc,'w+')

output_dir_ED = output_dir+'_ED'
output_dir_ES = output_dir+'_ES'


def pad_correction(contour):
    return contour-1

def unzip(file_path,file_path_unzipped):
    with gzip.open(file_path, 'rb') as f_in:
        with open(file_path_unzipped, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

landmark_error_dir = os.path.join(plot_dir, 'landmark_errors')
if not os.path.exists(landmark_error_dir):
    os.mkdir(landmark_error_dir)
for curr_subject in tqdm(os.listdir(input_dir)):
    subject_dir = os.path.join(input_dir,curr_subject)
    if os.path.isdir(subject_dir):
        if ed_es_split:
            output_subject_dir_ED = os.path.join(output_dir_ED,curr_subject)
            output_subject_dir_ES = os.path.join(output_dir_ES,curr_subject)
        else:
            output_subject_dir_ED = os.path.join(output_dir,curr_subject)
            output_subject_dir_ES = output_subject_dir_ED

        if not os.path.exists(output_subject_dir_ED):
            os.mkdir(output_subject_dir_ED)
        if not os.path.exists(output_subject_dir_ES):
            os.mkdir(output_subject_dir_ES)
        for file_name in os.listdir(subject_dir):
            file_path = os.path.join(subject_dir, file_name)
            if 'ED' in file_name:
                output_subject_dir = output_subject_dir_ED
            elif 'ES' in file_name:
                output_subject_dir = output_subject_dir_ES
            else:
                continue
            if file_path.endswith('ED.nii.gz') or file_path.endswith('ES.nii.gz') and not 'gt' in file_name:



                file_name=file_name[:-7]
                file_name_gt = file_name+'_gt'
                log_entry = ';'.join((curr_subject, file_name, ''))
                file_path_unzipped=file_path[:-3]
                file_path_unzipped_gt=file_path[:-7]+'_gt'+'.nii'
                file_path_gt=file_path_unzipped_gt+'.gz'

                unzip(file_path, file_path_unzipped)
                unzip(file_path_gt, file_path_unzipped_gt)

                us_image =  np.array(nib.load(file_path_unzipped).get_fdata())
                us_image = np.transpose(us_image)
                seg=nib.load(file_path_unzipped_gt).get_fdata()
                seg=np.transpose(seg)
                x_size = us_image.shape[1]
                y_size = us_image.shape[0]
                seg_padded = np.pad(seg, 1)
                seg_padded=seg_padded.astype('uint8')
                us_image = us_image.astype('uint8')
                f = plt.figure(figsize=(10, 10))
                plt.imshow(us_image, cmap='gray')
                plt.imshow(np.ma.masked_where(seg == 0, seg), alpha=0.25, vmin=1)
                plot_name=curr_subject+'_'+file_name
                lanmark_error = False
                landmarks_LV = None
                landmarks_LA = None
                landmarks_EP = None
                try:
                    landmarks_LV = left_ventricular_landmarks(seg_padded)
                    landmarks_LA = left_atrium_landmarks(seg_padded)
                    landmarks_EP = get_epicardium_landmarks_baseline_def(seg_padded,landmarks_LV,x_size=x_size,
                                                                        y_size=y_size,ver_vicinity=2)
                except:
                    log_entry += 'LANDMARK_ERROR1;'
                    lanmark_error=True
                if not lanmark_error:
                    try:
                        contour_LV = pad_correction(extract_contour(seg_padded, landmarks_LV, label=1,nb_points=nb_points))
                        contour_LA = pad_correction(extract_contour(seg_padded, landmarks_LA, label=3,nb_points=nb_points))
                        contour_EP = pad_correction(extract_contour(seg_padded, landmarks_EP, label=2, nb_points=nb_points))
                        if any(elem is None for elem in [contour_LV, contour_LA, contour_EP]):
                            log_entry += 'LANDMARK_ERROR2;'
                            for elem,name in zip([contour_LV, contour_LA, contour_EP],['LV','LA','EP']):
                                if elem is None:
                                    log_entry +=name
                            log_entry +=';'
                            lanmark_error=True
                    except:
                        log_entry += 'LANDMARK_ERROR3;'
                        lanmark_error = True
                if lanmark_error:
                    plt.savefig(os.path.join(landmark_error_dir,file_name+'_nb'+str(count)))
                else:
                    contour_LV, contour_LA = \
                        merge_base_points(contour_LV, contour_LA)
                    plt.scatter(contour_LV[:, 0], contour_LV[:, 1],s=markersize(50.0,contour_LV[:, 0]))
                    plt.scatter(contour_LA[:, 0], contour_LA[:, 1],s=markersize(50.0,contour_LA[:, 0]))
                    plt.scatter(contour_EP[:, 0], contour_EP[:, 1],s=markersize(50.0,contour_EP[:, 0]))
                    log_entry += 'SUCCESS;'
                    plt.savefig(os.path.join(plot_dir, plot_name))
                    plt.savefig(os.path.join(plot_dir, plot_name))

                    contour_LA=np.flip(contour_LA, axis=0)
                    contour_EP_left = np.flip(contour_EP[1:21], axis=0)
                    contour_EP_right = np.flip(contour_EP[22:-1], axis=0)
                    contour_EP = np.concatenate((np.array([contour_EP[0]]),contour_EP_left,
                                                 np.array([contour_EP[21]]),
                                                 contour_EP_right,np.array([contour_EP[-1]])),axis=0)
                    data_to_save = np.array([us_image, contour_LV, contour_LA, contour_EP, file_path], dtype=object)
                    np.save(os.path.join(output_subject_dir,file_name),data_to_save)
                if show_plot:
                    plt.show()


                os.remove(file_path_unzipped)

                plt.close('all')
                write_log_entry(log_entry,logfile_object)


logfile_object.close()


