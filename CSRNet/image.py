import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path, train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)

    # for name in gt_file:
    #     for dataset_name in gt_file[name]:
    #         # dataset = gt_file[name][dataset_name]
    #         print(name,dataset_name.shape)

    target = np.asarray(gt_file['density'])

    # 是否进行数据增强
    if False:
        crop_size = (img.size[0]/2,img.size[1]/2)
        # 计算裁剪起点
        if random.randint(0,9)<= -1:
            
            
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
        
        
        # 裁剪图像和GT图
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        
        
        # 20%的概率对图像和目标进行水平翻转
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    

    # target = cv2.resize(target,(target.shape[1]/8,target.shape[0]/8),interpolation = cv2.INTER_CUBIC)*64
    # 加入插值方法，计算新尺寸下每个像素点时会考虑周围16个像素点，从而生成较平滑的图像边缘
    target = cv2.resize(target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64
    
    
    return img,target