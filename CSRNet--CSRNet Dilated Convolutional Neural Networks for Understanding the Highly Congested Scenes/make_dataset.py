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
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
# %matplotlib inline


#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    # 初始化一个与GT形状相同的密度图，并将其所有值设为0
    density = np.zeros(gt.shape, dtype=np.float32)
    # 计算输入的GT中的非零点个数
    gt_count = np.count_nonzero(gt)
    # 非零点个数为0，说明全是零点，直接返回一张空的密度图
    if gt_count == 0:
        return density
    # 获取非零点的坐标，
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    # print("pts's type:", type(pts.copy()))
    # print(pts)
    # 设置KDTree的叶子大小，这个参数用于优化搜索性能，KDTree是一个用于快速空间搜索的数据机构
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # 对每个非零点进行KDTree查询，找到最近的4个点(包括自身)，返回距离和位置
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    # pts中已经获取到了非零点的坐标
    for i, pt in enumerate(pts):
        # 创建一个二维点阵pt2d, 只有对应非零点的位置是1
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        # 计算高斯滤波的标准差sigma, 如果非零点超过一个，则取第二/三和第四个最近邻点的距离的平均值
        # 如果只有一个非零点，则取图像形状平均值的四分之一
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point

        # 对每个点阵pt2d应用高斯滤波，将结果累加到密度图上。
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


#set the root to the Shanghai dataset you download

# root = '/home/leeyh/Downloads/Shanghai/'
# root = r'E:\PostGrauate\PaperRecurrent\目标计数\CSRNet-pytorch-master\dataset'
root = './dataset'


#now generate the ShanghaiA's ground truth
# part_A_train = os.path.join(root,'part_A_final/train_data','images')
# part_A_test = os.path.join(root,'part_A_final/test_data','images')
# part_B_train = os.path.join(root,'part_B_final/train_data','images')
# part_B_test = os.path.join(root,'part_B_final/test_data','images')

# 不这样修改，会生成part_A_final/train_data\images这样的路径
part_A_train = os.path.join(root, 'part_A_final', 'train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final', 'test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final', 'train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final', 'test_data', 'images')

# 生成part_A
path_sets = [part_A_train, part_A_test]

# 获取路径集合下的图片路径，并添加到img_paths列表中
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


for img_path in img_paths:
    # 加载.mat文件
    # mat.keys()  dict_keys(['__header__', '__version__', '__globals__', 'image_info'])
    # mat['image_info'].shape/type    (1, 1)/object
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))

    # 加载当前图像路径对应的图像文件
    img= plt.imread(img_path)
    # 初始化一个存储二值掩码的矩阵，初始值都为0
    k = np.zeros((img.shape[0],img.shape[1]))
    # 获取真实的地面数据
    gt = mat["image_info"][0, 0][0, 0][0]
    # 遍历GT图，如果点在图像尺寸范围内，就在对应的掩码矩阵上将位置设为1
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    # 应用高斯滤波，生成平滑的密度图
    k = gaussian_filter_density(k)
    # 创建一个.h5文件，将经过高斯滤波后的密度图保存起来，关键字设为'density', 如果文件不存在，将自动出啊u女鬼剑新文件
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k


# now see a sample from ShanghaiA
plt.imshow(Image.open(img_paths[0]))


gt_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images','ground_truth'),'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)


np.sum(groundtruth)# don't mind this slight variation



# 生成part_B
path_sets = [part_B_train, part_B_test]


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = gaussian_filter(k,15)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k