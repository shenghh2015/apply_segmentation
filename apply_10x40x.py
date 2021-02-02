import tensorflow as tf
import sys
sys.path.append('create_models')
import create_models as cm

import os
import numpy as np
from skimage import io
from keras_applications.imagenet_utils import preprocess_input

from sklearn.metrics import confusion_matrix
from helper_function import precision, recall, f1_score, iou_calculate

## Set the which GPU to run
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  

## Set the directory where the trained models are located
model_root_dir = './cycle_models/'
model_name = 'net-Unet-bone-efficientnetb5-cyc2_1488x1512' # model for x40
#model_name = 'net-Unet-bone-efficientnetb6-cycle_736x752' # model for x10
model_folder = model_root_dir+model_name

# roughly parse the model name
splits = model_name.split('-')
for v in range(len(splits)):
	if splits[v] == 'bone':
		backbone=splits[v+1]	

## Load the testing images and ground truth label maps
''' the folder strcutures under the dataset
---- x40 dataset
ls /data/datasets/cyc2_1488x1512
images  masks  test_list.txt  train_list.txt  valid_list.txt
--- x10 dataset
ls /data/datasets/cycle_736x752
images  masks  test_list.txt  train_list.txt  valid_list.txt
'''

dataset = 'cyc2_1488x1512' 					# x40 dataset
dataset = 'cycle_736x752'						# x10 dataset
subset = 'test' 										# dataset subset
data_dir = '/data/datasets/{}'.format(dataset)		# dataset diretory
image_dir = data_dir+'/images'
gt_dir = data_dir+'/masks'

# fetch file names
sample_name_file = data_dir+'/{}_list.txt'.format(subset)
file_names = []
with open(sample_name_file,'r+') as f:
	file_names = [line.strip() for line in f.readlines()]

def gray2rgb(image):
	image = np.uint8((image-image.min())*255/(image.max()-image.min()))
	return np.stack([image,image,image], axis=-1)

# load data and transform gray images to rgb images
images = np.stack([gray2rgb(io.imread(image_dir+'/'+fn)) for fn in file_names], axis=0)
gt_maps = np.stack([io.imread(gt_dir+'/'+fn) for fn in file_names], axis=0)

# Zero padding the input images
pad_widths = [(0,0),(8,8),(12,12),(0,0)] if dataset == 'cyc2_1488x1512' else [(0,0),(0,0),(8,8),(0,0)]
images_pad = np.pad(images, pad_widths, mode='reflect')

## Image preprocessing
print('Preprocessing ...')
preprocess_input = cm.get_preprocessing(backbone) ## preprocessing function
images_pad = preprocess_input(images_pad); # will scale pixels between 0 and 1 and then will normalize each channel with respect to the ImageNet dataset
print('Preprocessing done !')

## Load the trained model
model=tf.keras.models.load_model(model_folder+'/ready_model.h5')

## Label map prediction
pr_masks = model.predict(images_pad, batch_size=1); ## probability maps [N(num of images) x H x W x C(class)] for 0: G1, 1: S, 2: G2, 3: background/M phase
# crop images back
pr_masks = pr_masks[:,pad_widths[1][0]:-pad_widths[1][1],pad_widths[2][0]:-pad_widths[2][1],:] if dataset == 'cyc2_1488x1512' else pr_masks[:,:,pad_widths[2][0]:-pad_widths[2][1],:]
pr_masks_ = np.zeros(pr_masks.shape, dtype = np.float32)
pr_masks_[:,:,:,0] = pr_masks[:,:,:,3]; pr_masks_[:,:,:,1:]= pr_masks[:,:,:,:3] # 0:background/M, 1:G1, 2:S, 3:G2 
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
