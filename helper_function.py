import numpy as np

def generate_folder(folder):
	import os
	if not os.path.exists(folder):
		os.system('mkdir -p {}'.format(folder))

# calculate the IoU and dice scores
def iou_calculate(gt_maps, pr_maps, num_classes = 4):
	# one hot encoding of predictions and ground truth
	num_classes = num_classes
	y_true = np.array([gt_maps ==i for i in range(num_classes)]).transpose(1,2,3,0)
	y_pred = np.array([pr_maps ==i for i in range(num_classes)]).transpose(1,2,3,0)
	print(y_pred.shape)
	axes = (1,2) # W,H axes of each image
	intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
	union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
	mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

	smooth = .00001
	iou_per_image_class = (intersection + smooth) / (union + smooth)
	dice_per_image_class = (2 * intersection + smooth)/(mask_sum + smooth)

	mean_iou_over_images = np.mean(iou_per_image_class, axis = 0)
	mean_iou_over_images_class = np.mean(mean_iou_over_images)
	dice_class = np.mean(dice_per_image_class, axis = 0)
	mean_dice = np.mean(dice_per_image_class)

	return mean_iou_over_images, mean_iou_over_images_class, dice_class, mean_dice

def precision(label, confusion_matrix):
	col = confusion_matrix[:, label]
	return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
	row = confusion_matrix[label, :]
	return confusion_matrix[label, label] / row.sum()

def f1_score(label, confusion_matrix):
	row = confusion_matrix[label, :]; col = confusion_matrix[:, label]
	prec_score = confusion_matrix[label, label] / col.sum()
	recall_score = confusion_matrix[label, label] / row.sum()
	return 2*prec_score*recall_score/(prec_score + recall_score)