import tensorflow as tf
import sys
sys.path.append('create_models')
import create_models as sm

import os
import numpy as np
from skimage import io

from sklearn.metrics import confusion_matrix
from helper_function import precision, recall, f1_score, iou_calculate

import glob
from natsort import natsorted

## Set the which GPU to run
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  

## Set the directory where the trained models are located
model_root_dir = './trained_models/'  	  
model_name = 'livedead-net-Unet-bone-efficientnetb3-pre-True-epoch-200-batch-14-lr-0.0005-banl-False-dim-512-train-900-bk-0.5-one-True' ## trained model name
model_folder = model_root_dir+model_name

## Load the testing images and ground truth label maps
data_dir = './data'		# dataset diretory
image_dir = os.path.join(data_dir, 'images')	# image directory: save each input image as a RGB image [HxWx3], where each channel 
												# contain the same phase contrast image; normalize the values into pixel values 
												# of range: [0,255], dtype: np.uint8
map_dir = os.path.join(data_dir, 'gt_maps')		# ground truth directory: save ground truth as an gray-scale image: [HxW]
												# ground truth pixel value: [0,1,2,3]; dtype: np.uint8
image_fns = os.listdir(image_dir)
images = []; gt_maps = []
for img_fn in image_fns:
	image = io.imread(image_dir+'/{}'.format(img_fn)); images.append(image)
	gt_map =  io.imread(map_dir+'/{}'.format(img_fn)); gt_maps.append(gt_map)
images = np.stack(images); gt_maps = np.stack(gt_maps) # an array of image and ground truth label maps

## Image preprocessing
backbone = 'efficientnetb3'
preprocess_input = sm.get_preprocessing(backbone) ## preprocessing function
images = preprocess_input(images); #gt_maps = preprocess_input(gt_maps)

## Load the trained model
model=tf.keras.models.load_model(model_folder+'/ready_model.h5')

## Label map prediction
pr_masks = model.predict(images, batch_size=1); ## probability maps [N(num of images) x H x W x C(class)] for 0: live, 1: intermediate, 2: dead, 3: background 
pr_masks_ = np.zeros(pr_masks.shape, dtype = np.float32)
pr_masks_[:,:,:,0] = pr_masks[:,:,:,3]; pr_masks_[:,:,:,1:]= pr_masks[:,:,:,:3] # 0:background, 1:live, 2:intermediate, 3:dead 
pr_maps = np.argmax(pr_masks_,axis=-1)   # predicted label map

## calculate the performance metrics on the testing images

iou_classes, mIoU, dice_classes, mDice = iou_calculate(gt_maps, pr_maps)
print('iou_classes: {:.4f},{:.4f},{:.4f},{:.4f}; mIoU: {:.4f}'.format(iou_classes[0],iou_classes[1],iou_classes[2], iou_classes[-1], mIoU))
print('dice_classes: {:.4f},{:.4f},{:.4f},{:.4f}; mDice: {:.4f}'.format(dice_classes[0],dice_classes[1],dice_classes[2], dice_classes[-1], mDice))

# confusion matrix, precision, recall, and F1 score
y_true=gt_maps.flatten(); y_pred = pr_maps.flatten()
cf_mat = confusion_matrix(y_true, y_pred)
print('Confusion matrix:')
print(cf_mat)
prec_scores = []; recall_scores = []; f1_scores = []; iou_scores=[]
for i in range(cf_mat.shape[0]):
    prec_scores.append(precision(i,cf_mat))
    recall_scores.append(recall(i,cf_mat))
    f1_scores.append(f1_score(i,cf_mat))

print('Precision:{}, mean {}'.format(np.round(prec_scores,4), round(np.mean(prec_scores), 4)))
print('Recall:{}, mean {}'.format(np.round(recall_scores,4), round(np.mean(recall_scores), 4)))
print('F1 score:{}, mean {}'.format(np.round(f1_scores,4), round(np.mean(f1_scores), 4)))