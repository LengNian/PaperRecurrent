import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
# %matplotlib inline


from torchvision import datasets, transforms

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# root = '/home/leeyh/Downloads/Shanghai/'
root = './dataset/'

#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')

# path_sets = [part_A_test]
path_sets = [part_B_test]

# glob是一个在文件系统中查找匹配特定模式文件名的python模块。使用glob模块，可以用一个特定的模式来查找文件路径。
# 这通常用于获取一个目录下所有匹配特定扩展名（如 .jpg、.png）的文件。
# glob模块最常用的函数是 glob.glob()，它接收一个路径模式，并返回一个列表，其中包含了所有匹配该模式的文件路径。

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


model = CSRNet()

model = model.cuda()

# checkpoint = torch.load('model_best.pth.tar')
checkpoint = torch.load('./PartBmodel_best.pth.tar')
# checkpoint = torch.load('./PartAmodel_best.pth.tar')
# 这是测试的
# checkpoint = torch.load('./0model_best.pth.tar')

model.load_state_dict(checkpoint['state_dict'])

mae = 0


for i in range(len(img_paths)):
    # 对图像进行标准化
    img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
    # 减去图像中每个通道的均值
    img[0,:,:]=img[0,:,:]-92.8207477031
    img[1,:,:]=img[1,:,:]-95.2757037428
    img[2,:,:]=img[2,:,:]-104.877445883
    img = img.cuda()

    #img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    # .h5文件是由.mat文件转换而来的
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))

    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    print(i,mae)
print(mae/len(img_paths))

