"""
Training Code for Learning To Count Everything, CVPR 2021
Authors: Viresh Ranjan,Udbhav, Thu Nguyen, Minh Hoai

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
"""
import torch.nn as nn
from model import  Resnet50FPN,CountRegressor,weights_normal_init
from utils import MAPS, Scales, Transform,TransformTrain,extract_features, visualize_output_and_save
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists,join
import random
import torch.optim as optim
import torch.nn.functional as F


parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default='./data/', help="Path to the FSC147 dataset")
# parser.add_argument("-dp", "--data_path", type=str, default='/home/hoai/DataSets/AgnosticCounting/FSC147_384_V2/', help="Path to the FSC147 dataset")
parser.add_argument("-o", "--output_dir", type=str,default="./logsSave", help="/Path/to/output/logs/")
parser.add_argument("-ts", "--test-split", type=str, default='val', choices=["train", "test", "val"], help="what data split to evaluate on on")
parser.add_argument("-ep", "--epochs", type=int,default=1500, help="number of training epochs")
parser.add_argument("-g", "--gpu", type=int,default=0, help="GPU id")
parser.add_argument("-lr", "--learning-rate", type=float,default=1e-5, help="learning rate")
args = parser.parse_args()


data_path = args.data_path
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
im_dir = data_path + 'images_384_VarV2'
gt_dir = data_path + 'gt_density_map_adaptive_384_VarV2'

if not exists(args.output_dir):
    os.mkdir(args.output_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

criterion = nn.MSELoss().cuda()

resnet50_conv = Resnet50FPN()
resnet50_conv.cuda()
resnet50_conv.eval()

regressor = CountRegressor(6, pool='mean')

weights_normal_init(regressor, dev=0.001)
regressor.train()
regressor.cuda()
optimizer = optim.Adam(regressor.parameters(), lr = args.learning_rate)

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)


def train():
    print("Training on FSC147 train set data")
    # 选择train中的数据
    im_ids = data_split['train']
    random.shuffle(im_ids)
    train_mae = 0
    train_rmse = 0
    train_loss = 0
    pbar = tqdm(im_ids)
    cnt = 0
    for im_id in pbar:
        cnt += 1
        anno = annotations[im_id]
        # len(bboxes) : 3 ,看起来很多，但是应该是给了四个点的坐标
        # bboxse.shape : (3, 4, 2)
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        rects = list()
        # 循环获取三次，最后的rects中包括三个边界框的坐标，每个边界框存了两个点
        # rects: [[93, 100, 129, 134], [122, 204, 161, 231], [200, 354, 247, 391]]
        for bbox in bboxes:
            # 只获取四个坐标中的两个
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        # 打开图像和密度图
        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        density_path = gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
        density = np.load(density_path).astype('float32')

        # 这里的sample就包括图像、边界框和密度图
        sample = {'image':image,'lines_boxes':rects,'gt_density':density}
        sample = TransformTrain(sample)
        # shape: image: (384, 576, 3) --> (3, 384, 576)  boxes (3, 4) --> (1, 3, 5)  gt_density: (384, 512) --> (1, 1, 384, 512)
        image, boxes,gt_density = sample['image'].cuda(), sample['boxes'].cuda(),sample['gt_density'].cuda()


        with torch.no_grad():
            # 这里传入的image和boxes的维度都是四维
            # feature shape : torch.Size([1, 3, 6, 48, 80])
            features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

        features.requires_grad = True
        optimizer.zero_grad()
        # output's shape: [1, 1, 384, 576]
        output = regressor(features)

        #if image size isn't divisible by 8, gt size is slightly different from output size
        if output.shape[2] != gt_density.shape[2] or output.shape[3] != gt_density.shape[3]:
            orig_count = gt_density.sum().detach().item()
            gt_density = F.interpolate(gt_density, size=(output.shape[2],output.shape[3]),mode='bilinear')
            new_count = gt_density.sum().detach().item()
            # 如果原始的密度图中有人，就要在调整密度图之后重新调整密度大小
            if new_count > 0: gt_density = gt_density * (orig_count / new_count)


        loss = criterion(output, gt_density)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # 预测人数和实际人数
        pred_cnt = torch.sum(output).item()
        gt_cnt = torch.sum(gt_density).item()

        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2

        pbar.set_description('actual-predicted: {:6.1f}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f} Best VAL MAE: {:5.2f}, RMSE: {:5.2f}'.format( gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), train_mae/cnt, (train_rmse/cnt)**0.5,best_mae,best_rmse))
        print("")
    train_loss = train_loss / len(im_ids)
    train_mae = (train_mae / len(im_ids))
    train_rmse = (train_rmse / len(im_ids))**0.5

    return train_loss,train_mae,train_rmse


def eval():

    cnt = 0
    SAE = 0 # sum of absolute errors
    SSE = 0 # sum of square errors
    # 评估默认在验证集上测试
    print("Evaluation on {} data".format(args.test_split))
    im_ids = data_split[args.test_split]
    pbar = tqdm(im_ids)
    for im_id in pbar:
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])


        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        sample = {'image':image,'lines_boxes':rects}
        sample = Transform(sample)
        image, boxes = sample['image'].cuda(), sample['boxes'].cuda()

        with torch.no_grad():
            output = regressor(extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales))

        # dots's shape: [点注释个数, 2]
        gt_cnt = dots.shape[0]
        pred_cnt = output.sum().item()
        cnt = cnt + 1
        # 点注释的个数可以代表测试图片上物体的总数量
        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err**2

        pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5))
        print("")

    print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE/cnt, (SSE/cnt)**0.5))
    return SAE/cnt, (SSE/cnt)**0.5


best_mae, best_rmse = 1e7, 1e7
stats = list()

for epoch in range(0,args.epochs):
    regressor.train()
    train_loss,train_mae,train_rmse = train()
    regressor.eval()
    val_mae,val_rmse = eval()
    stats.append((train_loss, train_mae, train_rmse, val_mae, val_rmse))
    stats_file = join(args.output_dir, "stats" +  ".txt")
    with open(stats_file, 'w') as f:
        for s in stats:
            f.write("%s\n" % ','.join([str(x) for x in s]))
    if best_mae >= val_mae:
        best_mae = val_mae
        best_rmse = val_rmse
        model_name = args.output_dir + '/' + "FamNet.pth"
        torch.save(regressor.state_dict(), model_name)

    print("Epoch {}, Avg. Epoch Loss: {} Train MAE: {} Train RMSE: {} Val MAE: {} Val RMSE: {} Best Val MAE: {} Best Val RMSE: {} ".format(
              epoch+1,  stats[-1][0], stats[-1][1], stats[-1][2], stats[-1][3], stats[-1][4], best_mae, best_rmse))