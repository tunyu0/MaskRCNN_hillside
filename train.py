# -*- coding: utf-8 -*-

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib, utils  # modellib是模型本體
from mrcnn import visualize  # 視覺化模組
import yaml
from mrcnn.model import log
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT_DIR = os.getcwd() # ROOT_DIR是這個檔案(train.py)所在的位置
# MODEL_DIR是這個檔案(train.py)的位置進去logs這個資料夾，等一下訓練完的h5放的位置
MODEL_DIR = os.path.join(ROOT_DIR, "logs1")
iter_num = 0

# (train.py)所在的位置的mask_rcnn_coco.h5這個檔案路徑(是一個預訓練權重)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
	
class ShapesConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2 # 每顆GPU處理幾張圖片
    # Number of classes (including background)
    NUM_CLASSES = 1 + 5
    # background + 1 shapes # 修改1:如果你有超過一個label, 記得改1 + <label總量>
        
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384

    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  
    TRAIN_ROIS_PER_IMAGE = 100
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
	
config = ShapesConfig()
#config.display()
# 看參數(有些是預設的 有些是上面自己設的) 可不用印出來 不影響


class DrugDataset(utils.Dataset):
    # 得到該圖中有多少實體(Instance)
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]

        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):                    
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 覆寫load_shapes把原本的count換成start, end來指定圖片(原本count用迴圈跑完一定要拿全部的圖片)
    def load_shapes(self, start, end, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # 修改2  Add classes
        self.add_class("shapes", 1, "collapse")
        self.add_class("shapes", 2, "embankment")
        self.add_class("shapes", 3, "water")
        self.add_class("shapes", 4, "green")
        self.add_class("shapes", 5, "sky")

        # 把count換成start, end
        for i in range(start, end):
            # 獲取圖片的寬和高            
            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
			#如果圖片名稱含有_,則上一行改為 mask_path = mask_floder + "/" + filestr + "_json.png"
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)
    # 覆寫load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
		
		#修改3 label 有變要改
        for i in range(len(labels)):
            if labels[i].find("collapse") != -1:
                #str.find('collapse') 等於-1 就是沒找到'collapse'，不等於-1 就是有找到'collapse'
                labels_form.append("collapse") # 照順序加入labels_form
            elif labels[i].find("embankment") != -1:
                labels_form.append("embankment")
            elif labels[i].find("water") != -1:
                labels_form.append("water")
            elif labels[i].find("green") != -1:
                labels_form.append("green")
            elif labels[i].find("sky") != -1:
                labels_form.append("sky")
				
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    fig1, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# 修改4: 餵進去的訓練資料位置
dataset_root_path = "samples/mydataset/"
img_floder = dataset_root_path + "pic"  # 原圖
mask_floder = dataset_root_path + "cv2_mask"  # mask圖
# yaml_floder = dataset_root_path
imglist = os.listdir(img_floder)
count = len(imglist)  # 這2行是記下全部有多少張圖

# 準備train和val
dataset_train = DrugDataset()
print("dataset_train-->")
dataset_train.load_shapes(16, 26, img_floder, mask_floder, imglist, dataset_root_path)
dataset_train.prepare()

dataset_val = DrugDataset()
print("dataset_val-->")
dataset_val.load_shapes(0, 6, img_floder, mask_floder, imglist, dataset_root_path)
dataset_val.prepare()


# 可以去mrcnn/config.py那個檔案修改DETECTION_MIN_CONFIDENCE(就是confidence的threshold)
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)  # MODEL_DIR設定等一下log檔會在哪邊生成

init_with = "coco"  #修改5 imagenet, 第一次訓練(重新訓練)用coco, or 接續訓練用last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
	# 修改6 如果是接續訓練的話 .load_weights第一個參數要給接續哪次結果的權重檔(XX.h5檔)
    # 直接給h5檔的路徑
    model.load_weights(model.find_last()[1], by_name=True)
    #model.load_weights('已訓練好的.h5', by_name=True)

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,  # heads要訓練多少epoch
            layers='heads')  # 只訓練heads層，就是backbone的(預訓練)參數先不動 先訓練其他層

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,  # heads+all 要訓練多少epoch
            layers="all")  # 訓練全部的層