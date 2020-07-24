import tensorflow as tf
import sys
sys.path.append('segment')
import segment as sm

import os
from skimage import io
import numpy as np
from sklearn.metrics import confusion_matrix

from helper_function import precision, recall, f1_score, iou_calculate

import glob
from natsort import natsorted

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_root_dir = './trained_models/'  # model root folder

model_name = 'livedead-net-Unet-bone-efficientnetb3-pre-True-epoch-200-batch-14-lr-0.0005-banl-False-dim-512-train-900-bk-0.5-one-True' # model dir
model_folder = model_root_dir+model_name

## parse model name
backbone = 'efficientnetb3'
val_dim = 832   # input image dimension dim x dim x 3
preprocess_input = sm.get_preprocessing(backbone)
data_dir = './data'
image_dir = os.path.join(data_dir, 'images'); map_dir = os.path.join(data_dir, 'gt_maps')
image_fns = os.listdir(image_dir)
images = []; gt_maps = []
for img_fn in image_fns:
	image = io.imread(image_dir+'/{}'.format(img_fn)); images.append(image)
	gt_map =  io.imread(map_dir+'/{}'.format(img_fn)); gt_maps.append(gt_map)
images = np.stack(images); gt_maps = np.stack(gt_maps)
images = preprocess_input(images); #gt_maps = preprocess_input(gt_maps)

# load the trained model
model=tf.keras.models.load_model(model_folder+'/ready_model.h5')

# calculate the performance metrics on the test image set
pr_masks = model.predict(images); ## probability maps [N(num of images) x H x W x C(class)] for 0: live, 1: intermediate, 2: dead, 3: background 
pr_masks_ = np.zeros(pr_masks.shape, dtype = np.float32)
pr_masks_[:,:,:,0] = pr_masks[:,:,:,3]; pr_masks_[:,:,:,1:]= pr_masks[:,:,:,:3] # 0:background, 1:live, 2:intermediate, 3:dead 
pr_maps = np.argmax(pr_masks_,axis=-1)

## IoU and dice coefficient
iou_classes, mIoU, dice_classes, mDice = iou_calculate(gt_maps, pr_maps)
print('iou_classes: {:.4f},{:.4f},{:.4f},{:.4f}; mIoU: {:.4f}'.format(iou_classes[0],iou_classes[1],iou_classes[2], iou_classes[-1], mIoU))
print('dice_classes: {:.4f},{:.4f},{:.4f},{:.4f}; mDice: {:.4f}'.format(dice_classes[0],dice_classes[1],dice_classes[2], dice_classes[-1], mDice))

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