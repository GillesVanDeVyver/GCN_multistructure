import torch
from torch.utils import data
import torchvision.transforms as torch_transforms
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict
import albumentations as A
import glob

# internal:
from utils.utils_plot import draw_kpts, plot_kpts_pred_and_gt
from utils.utils_files import ultrasound_img_load
from skimage.transform import resize

########################################################################
########################################################################
#     Base Dataset class for Ultrasound Keypoints detection models
########################################################################
########################################################################
class USKpts_CAMUS_displacement(data.Dataset):
    def __init__(self, dataset_config: Dict, filenames_list: str = None, transform: A.core.composition.Compose = None):

        self.img_folder = dataset_config["img_folder"]
        self.transform = transform
        self.input_size = dataset_config["input_size"]

        # get list of files in dataset:
        self.create_img_list(filenames_list=filenames_list)

        # kpts info:
        self.kpts_info = dataset_config["kpts_info"]
        self.num_kpts = dataset_config["num_kpts"]
        self.nb_classes = dataset_config["nb_classes"]


        self.basic_transform = torch_transforms.ToTensor()

        # fig for item plots:
        self.fig = plt.figure(figsize=(16, 10))

    def create_img_list(self, filenames_list: str) -> None:
        """ Creates a list containing paths to frames in the dataset."""
        self.filenames_list = filenames_list

        subject_list_from_file = []
        if filenames_list is not None:
            if not type(filenames_list)==list:  # (if single element then put in list)
                filenames_list = [filenames_list]

            for filenames_sublist in filenames_list:
                if os.path.exists(filenames_sublist):  # (filenames_sublist is file)
                    with open(filenames_sublist) as f:
                        subject_list_from_file.extend(f.read().splitlines())
                else:   # (filenames_sublist is a case_name)
                    subject_list_from_file.append(filenames_sublist)

        self.img_list = []
        if len(subject_list_from_file) > 0:
            for f in subject_list_from_file:
                fullpath = os.path.join(self.img_folder, f)
                if os.path.exists(fullpath):
                    if os.path.isdir(fullpath): # f is a subject
                        to_add = glob.glob(os.path.join(fullpath, "*.npy"))
                        to_add_subdir= glob.glob(os.path.join(fullpath, "*/*.npy"))
                        to_add=to_add+to_add_subdir
                        to_add = [f.replace(self.img_folder, "") for f in to_add]
                        self.img_list.extend(to_add)
                    else:   # fullpath is a single frame
                        self.img_list.append(f)
        else:
            img_list_from_folder = glob.glob(os.path.join(self.img_folder, "**/*.npy"))
            self.img_list = [os.path.basename(f) for f in img_list_from_folder]

    def add_kpts(self,kpts,new_kpts,ep=False):
        if kpts is None:
            return new_kpts
        else:
            if ep:
                return np.concatenate((kpts, np.array(new_kpts)))
            else:
                return np.concatenate((kpts,np.array(new_kpts[1:-1]))) # la also contains the base points => remove duplicates here

    def get_img_and_kpts(self, index: int):
        """
        Load and parse a single data point.
        Args:
            index (int): Index
        Returns:
            img (ndarray): RGB frame in required input_size
            kpts (ndarray): Denormalized, namely in img coordinates
            img_path (string): full path to frame file in image format (PNG or equivalent)
        """
        # ge paths:
        data_path = os.path.join(self.img_folder, self.img_list[index])
        data = np.load(data_path,allow_pickle=True)
        img,kpts_LV,kpts_LA,kpts_EP,distances_ep,seg,file_path = data
        kpts = None

        kpts = self.add_kpts(kpts,np.array(kpts_LV))
        kpts = self.add_kpts(kpts,np.array(kpts_LA))
        kpts = self.add_kpts(kpts,np.array(kpts_EP),ep=True)

        # resize to DNN input size:
        ratio = [self.input_size / float(img.shape[1]), self.input_size / float(img.shape[0])]
        if img.shape[0] != self.input_size or img.shape[1] != self.input_size:
            img =resize(img, output_shape=(self.input_size, self.input_size), preserve_range=True, anti_aliasing=False, order=0)
        # resizing keypoints:
        img = img.astype(np.uint8)
        kpts = np.round(kpts * ratio)   # also cast int to float

        data = {"img": img,
                "kpts": kpts,
                "img_path": data_path,
                "ignore_margin": None,
                "ratio": ratio
                }
        return data

    def img_to_torch(self, img: np.ndarray) -> torch.Tensor:
        """ Convert original image format to torch.Tensor """
        # resize:
        if img.shape[0] != self.input_size:
            img =resize(img, output_shape=(self.input_size, self.input_size), preserve_range=True, anti_aliasing=False, order=0)
        # transform:
        img = Image.fromarray(np.uint8(img),mode='L')
        img = self.basic_transform(img)

        return img

    def normalize_pose(self, pose: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """ Normalizing a set of frame keypoint to [0, 1] """
        for p in pose:
            p[0] = p[0] / frame.shape[1]
            p[1] = p[1] / frame.shape[0]
        return pose

    def denormalize_pose(self, pose: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """ DeNormalizing a set of frame keypoint back to image coordinates """
        for p in pose:
            p[0] = p[0] * frame.shape[1]
            p[1] = p[1] * frame.shape[0]
        return pose

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = self.get_img_and_kpts(index)
        if self.transform is not None:
            transformed = self.transform(image=data["img"], keypoints=data["kpts"], teacher_kpts=[])
            img, all_kpts = transformed["image"], np.asarray(transformed["keypoints"])
        else:
            img = data["img"]
            all_kpts = data["kpts"]
        all_kpts = self.normalize_pose(pose=all_kpts, frame=img)
        all_kpts = torch.tensor(all_kpts).float()

        img_rgb = np.repeat(img[:, :, np.newaxis], 3, axis=2) # to rgb for image encoder

        # transform:
        img_rgb = Image.fromarray(np.uint8(img_rgb))
        img_rgb = self.basic_transform(img_rgb)

        la_offset = self.num_kpts[0]
        ep_offset = self.num_kpts[0]+self.num_kpts[1]

        distances = []

        for kpt_lv,kpt_ep in zip(all_kpts[:la_offset],all_kpts[ep_offset:]):
            distances.append(np.linalg.norm(kpt_lv-kpt_ep))

        distances = np.array(distances)



        kpts,ep_kpts = all_kpts[0:ep_offset],all_kpts[ep_offset:]

        return img_rgb, kpts,distances, data["img_path"]

    def plot_item(self, index: int, do_augmentation: bool = True, print_folder: str = './visu/') -> None:
        """ Plot frame and gt annotations for a single data point """
        data = self.get_img_and_kpts(index)

        basename = os.path.splitext(data["img_path"].replace(self.img_folder, ""))[0].replace("/", "_")
        print_fname = "{}_INDX{}_gt".format(basename, index)

        # data aug:
        if do_augmentation and self.transform is not None:
            transformed = self.transform(image=data["img"], keypoints=data["kpts"])
            img, kpts = transformed['image'], transformed['keypoints']
            print_fname = "{}_aug".format(print_fname, index)
        else:
            img = data["img"]
            kpts = data["kpts"]

        # plot:
        img_and_gt = draw_kpts(img, kpts, kpts_connections=self.kpts_info["connections"], colors_pts=self.kpts_info["colors"])
        plt.clf()
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(img.astype(np.int))
        ax2 = plt.subplot(1, 2, 2)
        plt.imshow(img_and_gt.astype(np.int))
        plt.axis('off')

        nnm = os.path.join(print_folder, print_fname)
        plt.savefig(nnm)
        print(nnm)

    def plot_prediction(self, fig: plt.Figure, img_path: str, predicted_pose: np.ndarray, gt_pose: np.ndarray,
                        is_normalized: bool = True, print_output_filename: str = None) -> plt.Figure:
        """
        Plot keypoints prediction on input frame.
        Args:
            fig: plt.Figure
            img_path: str
            predicted_pose: (numpy array, size num_kpts x 2)
            gt_pose: (numpy array, size num_kpts x 2)
            is_normalized: bool
            print_output_filename: str
        """

        img, _ = ultrasound_img_load(img_path=img_path)
        #img = cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_AREA)

        #frame_pose = to_numpy(predicted_pose).reshape(self.num_kpts, 2)

        if not is_normalized:
            predicted_pose = self.denormalize_pose(predicted_pose, img)
            gt_pose = self.denormalize_pose(gt_pose, img)

        plot_kpts_pred_and_gt(fig, img, gt_kpts=gt_pose, pred_kpts=predicted_pose,
                              kpts_info=self.kpts_info, closed_contour=self.kpts_info['closed_contour'])

        if print_output_filename is not None:
            #cv2.imwrite(print_output_filename, image)
            fig.savefig(print_output_filename)
            print("Cardiac contour is shown in {}".format(print_output_filename))

        return fig

