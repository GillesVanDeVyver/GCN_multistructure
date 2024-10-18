import os
import sys

import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import scipy.interpolate as interpolate
from PIL import Image, ImageDraw
from typing import List
from utils.utils_tools import remove_duplicates

import colorsys
############################
############################
# Plot utils
############################
############################

def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])


def plot_spline(kpts,unew):
    k=3
    if len(kpts)<4:
        k = len(kpts)-1
    tck, _ = interpolate.splprep([np.array(kpts)[:, 0], np.array(kpts)[:, 1]], s=0,k=k)
    interpolation = interpolate.splev(unew, tck)

    return interpolation


def draw_kpts(img, kpts,kpts_info, kpts_connections=[], colors_pts=None, color_connection=[255, 255, 255]):
    splines = None
    spline_colors = None
    im = img.copy() # workaround a bug in Python OpenCV wrapper: https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
    # draw points
    ii = 0
    for k in kpts:
        x = int(k[0])
        y = int(k[1])
        if colors_pts is None:
            c = (255, 255, 0)
        else:
            c = colors_pts[ii]
        cv2.circle(im, (x, y), radius=2, thickness=-1, color=c)
        ii += 1
    # draw lines

    for i in range(len(kpts_connections)):
        cur_im = im.copy()
        limb = kpts_connections[i]
        [Y0, X0] = kpts[limb[0]]
        [Y1, X1] = kpts[limb[1]]
        mX = np.mean([X0, X1])
        mY = np.mean([Y0, Y1])
        length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        if type(color_connection[0]) == int:
            cv2.fillConvexPoly(cur_im, polygon, color_connection)
        else:
            cv2.fillConvexPoly(cur_im, polygon, color_connection[i])
        im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)
    return im,splines,spline_colors

def color_connections(kpts_info):
    colors_pts = []
    color_connection = []
    for i in range(len(kpts_info['num_kpts_multistructure'])):
        for j in range(kpts_info['num_kpts_multistructure'][i]):
            colors_pts.append(kpts_info['colors'][i])
            if j != kpts_info['num_kpts_multistructure'][i] - 1:
                color_connection.append(kpts_info['colors'][i])
    return color_connection,colors_pts


def plot_splines(splines,spline_colors,ax):
    if splines is not None:
        for spline,color in zip(splines,spline_colors):
            if not isinstance(color, str):
                c = [np.array(color)/255 for el in range(len(spline[0]))]
            else:
                c= color
            ax.scatter(spline[0], spline[1], marker='.', c=c, s=2)

def sort_ep_kpts(kpts,start_ep,img):
    inner_kpts = kpts[:start_ep]
    ep_kpts = kpts[start_ep:]
    sorted_ep_kpts = []
    kpt=inner_kpts[0]
    distances = []
    for kpt2 in ep_kpts:
        dist = np.linalg.norm(kpt - kpt2)
        distances.append(dist)
    kpt_idx = np.argmin(np.array(distances))
    kpt = ep_kpts[kpt_idx]
    sorted_ep_kpts.append(kpt)
    ep_kpts=np.delete(ep_kpts,kpt_idx,0)
    while len(ep_kpts)>0:
        distances = []
        for kpt2 in ep_kpts:
            dist = np.linalg.norm(kpt-kpt2)
            distances.append(dist)
        kpt_idx = np.argmin(np.array(distances))
        kpt = ep_kpts[kpt_idx]
        sorted_ep_kpts.append(kpt)
        ep_kpts=np.delete(ep_kpts, kpt_idx, 0)

    return np.concatenate((inner_kpts,np.array(sorted_ep_kpts)))

def plot_kpts_pred_and_gt(fig, img, cfg,gt_kpts=None, pred_kpts=None, kpts_info=[], closed_contour=False,
                          text=None, std_vals = None, #pred_only=False,
                          img_ktps=None, sort_ep = False):

    #fig.clf()
    clean_img = img
    if len(clean_img.shape) == 2: # grayscale to rgb
        clean_img = np.repeat(clean_img[:, :, np.newaxis], 3, axis=2)
    if img_ktps is None:
        img_ktps=clean_img
    interpolation_step_size = 0.001  # 0.01
    unew = np.arange(0, 1.00, interpolation_step_size)



    if gt_kpts is not None:
        if kpts_info['multistructure'] == True:
            color_connection,colors_pts = color_connections(kpts_info)
        else:
            colors_pts = [[0, 255, 0] for _ in pred_kpts]
            color_connection = [0, 255, 0]
        img_ktps,splines_pred,spline_colors_pred = draw_kpts(img=img_ktps, kpts=gt_kpts,
                        kpts_info=kpts_info,
                        kpts_connections=kpts_info["connections"],
                        colors_pts=colors_pts,
                        color_connection=color_connection)
        gt_kpts = remove_duplicates(gt_kpts,axis=0)
        if sort_ep:
            num_kpts = cfg.TRAIN.NUM_KPTS_MULTISTRUCTURE
            start_ep = num_kpts[0] + num_kpts[1]
            gt_kpts = sort_ep_kpts(gt_kpts,start_ep,img)
        if closed_contour:
            gt_kpts = np.concatenate((gt_kpts, gt_kpts[:1, :]), axis=0)
        if len(kpts_info['names']) > 3:
            try:
                gt_tck, _ = interpolate.splprep([gt_kpts[:, 0], gt_kpts[:, 1]], s=0)
                gt_interpolate = interpolate.splev(unew, gt_tck)
            except:
                print('ERROR plotting interpolation ')
                print(sys.exc_info()[0], "occurred.")
                gt_interpolate = None

    if pred_kpts is not None:
        if std_vals is None:
            if kpts_info['multistructure'] == True:
                color_connection, colors_pts_pred = color_connections(kpts_info)

            else:
                colors_pts_pred = [[255,255,0] for _ in pred_kpts]
                color_connection = [255,255,0]
        else:
            cmap = plt.cm.rainbow
            norm = matplotlib.colors.Normalize(vmin=np.min(std_vals), vmax=np.max(std_vals))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(std_vals)
            # for std_vals in std_vals:
            #    colors_pts_gt.append([0,,])
            # for var in std_vals:
            colors_hue = sm.get_array()
            colors_rgb = sm.to_rgba(colors_hue)
            colors_pts_pred = []
            for color in colors_rgb:
                colors_pts_pred.append(255 * color[:3])


            color_connection=colors_pts_pred
        img_ktps_pred,splines_gt,spline_colors_gt = draw_kpts(img=clean_img, kpts=pred_kpts,
                        kpts_info=kpts_info,
                        kpts_connections=kpts_info["connections"],
                        colors_pts=colors_pts_pred,
                        color_connection=color_connection)
        pred_kpts = remove_duplicates(pred_kpts,axis=0) # avoid interpolation error due to duplicate keypoints
        if sort_ep:
            num_kpts = cfg.TRAIN.NUM_KPTS_MULTISTRUCTURE
            start_ep = num_kpts[0] + num_kpts[1]
            pred_kpts = sort_ep_kpts(pred_kpts, start_ep,img)
        if closed_contour:
            pred_kpts = np.concatenate((pred_kpts, pred_kpts[:1,:]), axis=0)
        if len(kpts_info['names']) > 3:
            try:
                pred_tck, _ = interpolate.splprep([pred_kpts[:, 0], pred_kpts[:, 1]], s=0)
                pred_interpolate = interpolate.splev(unew, pred_tck)
            except:
                print('ERROR plotting interpolation ')
                print(sys.exc_info()[0], "occurred.")
                pred_interpolate = None

    # option 1: clean img + kpts_img
    #if not pred_only:
    ax = fig.add_subplot(2, 2, 1)

    ax.imshow(clean_img)
    ax.set_axis_off()


    if text is not None:
        ax.set_title(text)
        #
    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(clean_img)


    if gt_kpts is not None and len(kpts_info['names']) > 3 and gt_interpolate is not None:
        ax.scatter(gt_interpolate[0], gt_interpolate[1], marker='.', c='green', s=2)
    if pred_kpts is not None and len(kpts_info['names']) > 3 and pred_interpolate is not None:
        ax.scatter(pred_interpolate[0], pred_interpolate[1], marker='.', c='yellow', s=2) # fixme: change color back to 'white'
    ax.set_axis_off()

        #
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title('Target')

    ax.imshow(img_ktps)
    plot_splines(splines_pred, spline_colors_pred, ax)

    ax = fig.add_subplot(2, 2, 4)
    ax.set_title('Prediction')


    ax.imshow(img_ktps_pred)
    plot_splines(splines_gt, spline_colors_gt, ax)


    if std_vals is not None:
        # fig.colorbar(sm)
        fig.colorbar(sm)
    #ax.set_axis_off()
    # option 2: kpts_img only
    #plt.imshow(img)


    return fig, img, img_ktps_pred

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def resize_image(image: np.array, image_size) -> np.array:
    img = cv2.resize(image.transpose(1, 2, 0), (image_size, image_size))
    return img.transpose(2,0,1)

def plot_inference_movie(test_movie, test_movie_pred_kpts, input_size, metric_name, value):     # Andy Gilbert
    dpi = 70.0
    c ,xpixels, ypixels = resize_image(test_movie[0],input_size).shape


    fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
    ax = plt.gca()
    ax.axis('off')
    ax.set_title("{}: {}".format(metric_name,value))
    interpolation_step_size = 0.001  # 0.01
    unew = np.arange(0, 1.00, interpolation_step_size)

    mv = plt.imshow(rgb2gray(resize_image(test_movie[0],input_size).transpose(1,2,0)), cmap='gray')

    kpts = test_movie_pred_kpts[0]
    kpts *= input_size

    pts, = ax.plot(kpts[:, 0], kpts[:, 1], marker='o', c='red')

    def animate(i):
        mv.set_array(rgb2gray(resize_image(test_movie[i],input_size).transpose(1,2,0)))

        kpts = test_movie_pred_kpts[i]
        kpts *= input_size
        pts.set_data(kpts[:, 0], kpts[:, 1])
        return (mv, pts)

    anim = animation.FuncAnimation(fig, animate, frames=test_movie.shape[0], blit=False)
    return anim



def plot_grid(frames: List, labels: List, thumbnail_size: int = 30) -> Image:
    """ Plot grid of images and labels """

    num_frames = len(frames)
    dim_length = int(num_frames ** 0.5) + 1     # default: 10
    grid_image_length = dim_length * (thumbnail_size + 10)

    new_im = Image.new('RGB', (grid_image_length + 10, grid_image_length + 10))

    index = 0
    idddxxx = range(min(100, num_frames)) #np.random.randint(0, num_frames, min(100, num_frames))
    for i in range(10, grid_image_length, thumbnail_size + 10):
        for j in range(10, grid_image_length, thumbnail_size + 10):
            if index < num_frames:
                #im = Image.open(files[idddxxx[index]])
                numpy_image = frames[idddxxx[index]]
                label = labels[idddxxx[index]]
                im = Image.fromarray(np.uint8(numpy_image)).convert('RGB')
                im.thumbnail((thumbnail_size, thumbnail_size))
                draw = ImageDraw.Draw(im)
                draw.text((0, 0), str(label), fill=128)
                #draw.text((0, 0), str(label), fill=0) #color=(255, 0, 255))#'magenta')
                new_im.paste(im, (i, j))
                index += 1
    #new_im.show()
    #input("Press Enter to continue...")

    return new_im

def boxplot(data,loc,labels,ylims=None,show=False,ylabel=None,yticks=None):
    plt.figure()
    plt.boxplot(data,labels=labels)
    if yticks is not None:
        plt.yticks(yticks)
        if ylims is not None:
            plt.ylim(ylims)
    elif ylims is not None:
        ticks = np.arange(ylims[0], ylims[1], (ylims[1] - ylims[0]) / 10)
        ticks = np.append(ticks, ylims[1])
        plt.yticks(ticks)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(loc)
    if show:
        plt.show()
    plt.clf()

def bar_plot(values,names,loc,title,ylims=None,yticks=None):
    fig, ax = plt.subplots()
    bars = ax.bar(names, values)
    ax.bar_label(bars)
    if ylims is not None:
        plt.ylim(ylims)
    if yticks is not None:
        plt.yticks(yticks)
    plt.title(title)
    plt.savefig(loc)
    plt.clf()


def seg_comparison_plot(input,gt,seg1,title1,seg2,title2,loc):
    fig, axes = plt.subplots(2, 2)
    fig.set_figheight(16)
    fig.set_figwidth(16)
    axes[0,0].imshow(input)
    axes[0,0].set_title('input')
    axes[0,1].imshow(gt)
    axes[0,1].set_title('ground truth')
    axes[1,0].imshow(seg1)
    axes[1,0].set_title(title1)
    axes[1,1].imshow(seg2)
    axes[1,1].set_title(title2)
    plt.tight_layout()
    plt.savefig(loc)
    plt.clf()
    plt.close()





def create_visualization(ultrasound, segmentation):
    result = np.zeros((ultrasound.shape[0], ultrasound.shape[1], 3))
    if ultrasound.shape[-1]==3:
        result=ultrasound.copy()
    else:
        for i in range(3):
            result[:, :, i] = ultrasound

    if len(segmentation.shape) == 3:
        # One hot encoded segmentation
        for i in range(3):
            result[segmentation[:, :, i+1] > 0.5, i] = np.clip(0.25 + result[segmentation[:, :, i+1] > 0.5, i], 0.0, 1.0)
    else:
        # Segmentation
        for i in range(3):
            result[segmentation == i+1, i] = np.clip(0.25 + result[segmentation == i+1, i], 0.0, 1.0)

    return (result*255).astype(np.uint8)












