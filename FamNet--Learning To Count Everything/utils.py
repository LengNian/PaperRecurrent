import numpy as np
import torch.nn.functional as F
import math
from torchvision import transforms
import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('agg')


MAPS = ['map3','map4']
Scales = [0.9, 1.1]
MIN_HW = 384
MAX_HW = 1584
# 均值和标准差
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


# 运行demo自己画框时的程序
def select_exemplar_rois(image):
    all_rois = []

    print("Press 'q' or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar, 'space' to save.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('n') or key == '\r':
            # cv2.selectROI()返回四个值(x, y, width, height) x, y是ROI左上角的坐标 width, height是矩形的宽度和高度
            rect = cv2.selectROI("image", image, False, False)
            x1 = rect[0]
            y1 = rect[1]
            x2 = x1 + rect[2] - 1
            y2 = y1 + rect[3] - 1

            all_rois.append([y1, x1, y2, x2])
            for rect in all_rois:
                y1, x1, y2, x2 = rect
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            print("Press q or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar")

    return all_rois


# 创建了一个类似于matlab中fspecial函数生成的二维高斯滤波核，用于模糊图像和减少噪声
# shape表示生成高斯核的大小
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    # 计算高斯核的半宽和半高
    m,n = [(ss-1.)/2. for ss in shape]
    # x, y代表高斯核中每个点的横纵坐标
    y,x = np.ogrid[-m:m+1,-n:n+1]
    # 计算高斯核的值
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    # np.finfo().eps就是获取最小的正数，×h.max()相当于得到一个相对于高斯核最大值的阈值，然后将高斯核中小于这个阈值的值设为0
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    # 计算高斯核的总和
    sumh = h.sum()
    # 如果总和不为0，就进行一个归一化的过程，使所得核的所有值加起来等于1
    if sumh != 0:
        h /= sumh

    return h

# 扰动损失
# output: 模型的输出，boxes: 边界框
def PerturbationLoss(output, boxes, sigma=8, use_gpu=True):
    Loss = 0.
    # 判断有几个边界框 boxes这里有三个维度，可能是batch, 边界框数量, 特征信息
    if boxes.shape[1] > 1:
        # 去除维度为1的维数
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            # 获得单个边界框的坐标
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            # FamNet的输出是一张密度图(batch, channel, width, height)这里，截取出密度图中与边界框中所对应给的部分
            # 这里的out就是论文中的Z_b
            out = output[:,:,y1:y2,x1:x2]
            # 这里的GaussKernel应该就是理想化的高斯核，通过最小化预测的密度图与理想化的高斯分布之间的差异
            GaussKernel = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
            # 高斯核转换为Pytorch张量
            GaussKernel = torch.from_numpy(GaussKernel).float()
            if use_gpu: GaussKernel = GaussKernel.cuda()
            Loss += F.mse_loss(out.squeeze(),GaussKernel)
    else:
        print("else:", boxes.shape)
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        out = output[:,:,y1:y2,x1:x2]
        Gauss = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
        GaussKernel = torch.from_numpy(Gauss).float()
        if use_gpu: GaussKernel = GaussKernel.cuda()
        Loss += F.mse_loss(out.squeeze(),GaussKernel)

    return Loss


# 最小计数损失
def MincountLoss(output, boxes, use_gpu=True):
    ones = torch.ones(1)
    if use_gpu: ones = ones.cuda()
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            # 如果边界框中感兴趣实例数目小于1
            X = output[:,:,y1:y2,x1:x2].sum()
            if X.item() <= 1:
                Loss += F.mse_loss(X,ones)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        X = output[:,:,y1:y2,x1:x2].sum()
        if X.item() <= 1:
            Loss += F.mse_loss(X,ones)

    return Loss

# 对四维特征矩阵进行零填充，使其高度和宽度达到期望的尺寸
def pad_to_size(feat, desire_h, desire_w):
    # feat: 输入的特征矩阵，形状为N,C,H,W  desire_h, desire_w: 期望的高度, 宽度
    """ zero-padding a four dim feature matrix: N*C*H*W so that the new Height and Width are the desired ones
        desire_h and desire_w should be largers than the current height and weight
    """
    # 获得当前的高度和宽度
    cur_h = feat.shape[-2]
    cur_w = feat.shape[-1]

    # 计算上下左右需要填充的列数和行数
    left_pad = (desire_w - cur_w + 1) // 2
    right_pad = (desire_w - cur_w) - left_pad
    top_pad = (desire_h - cur_h + 1) // 2
    bottom_pad =(desire_h - cur_h) - top_pad

    # 对特征矩阵进行填充，返回填充后的特殊矩阵
    return F.pad(feat, (left_pad, right_pad, top_pad, bottom_pad))

# 提取图像特征，并获得示例特征和图像特征之间的相关图
def extract_features(feature_model, image, boxes,feat_map_keys=['map3','map4'], exemplar_scales=[0.9, 1.1]):
    # image (1, 3, 384, 576)  boxes (1, 1, 3, 5)]
    # 获取批次大小 边界框数量
    N, M = image.shape[0], boxes.shape[2]
    """
    Getting features for the image N * C * H * W
    """
    # 这里得到的Image_features是堆叠了resnet50三四层之后的内容
    Image_features = feature_model(image)
    """
    Getting features for the examples (N*M) * C * h * w
    """
    for ix in range(0,N):
        # boxes = boxes.squeeze(0)
        # boxes.shape  (3, 5)
        boxes = boxes[ix][0]
        # print(np.array(boxes.cpu()).shape)
        cnter = 0
        Cnter1 = 0

        for keys in feat_map_keys:
            # 这里的image_features是经过卷积神经网络处理的特征图 1×512×h×w 或 1×1024×h×w
            # image_features's shape: 1×512×h×w or 1×1024×h×w
            image_features = Image_features[keys][ix].unsqueeze(0)
            if keys == 'map1' or keys == 'map2':
                Scaling = 4.0
            elif keys == 'map3':
                Scaling = 8.0
            elif keys == 'map4':
                Scaling =  16.0
            else:
                Scaling = 32.0

            boxes_scaled = boxes / Scaling

            # 对边界框坐标进行调整使符合图像的索引范围
            # 对起始点向下取整
            boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
            # 对结束点向上取整
            boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5])
            boxes_scaled[:, 3:5] = boxes_scaled[:, 3:5] + 1 # make the end indices exclusive

            feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
            # make sure exemplars don't go out of bound
            boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
            boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
            boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)
            # 计算边界框的高度和宽度
            box_hs = boxes_scaled[:, 3] - boxes_scaled[:, 1]
            box_ws = boxes_scaled[:, 4] - boxes_scaled[:, 2]
            # 计算出这三个边界框中的最大的高和宽
            max_h = math.ceil(max(box_hs))
            max_w = math.ceil(max(box_ws))

            # 获取每个边界框的坐标
            for j in range(0,M):
                y1, x1 = int(boxes_scaled[j,1]), int(boxes_scaled[j,2])  
                y2, x2 = int(boxes_scaled[j,3]), int(boxes_scaled[j,4]) 
                #print(y1,y2,x1,x2,max_h,max_w)
                if j == 0:
                    # 获取特征图中对应边界框位置的特征图
                    examples_features = image_features[:,:,y1:y2, x1:x2]
                    # 检查提取出来的特征值是否和边界框的最大高或最大宽相等
                    if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                        #examples_features = pad_to_size(examples_features, max_h, max_w)
                        # 如果不满足，就使用双线性插值法将尺寸调整为max_h和max_w
                        # examples_features's shape: torch.Size([1, 512, 14, 13])
                        examples_features = F.interpolate(examples_features, size=(max_h,max_w),mode='bilinear')
                else:
                    # 同上，但是把特征图堆叠起来
                    feat = image_features[:,:,y1:y2, x1:x2]
                    if feat.shape[2] != max_h or feat.shape[3] != max_w:
                        feat = F.interpolate(feat, size=(max_h,max_w),mode='bilinear')
                        #feat = pad_to_size(feat, max_h, max_w)
                        # 这里会堆叠两个边界框
                        # examples_fetures's shape: torch.Size([2, 512, 14, 13]) 堆叠两次
                        # examples_fetures's shape: torch.Size([3, 512, 14, 13]) 堆叠三次
                        # 就是将三个边界框堆起来，作为示例特征
                    examples_features = torch.cat((examples_features,feat),dim=0)
            """
            Convolving example features over image features
            """
            # 这里的examples_features是堆叠了三个边界框所对应的部分特征图的一个特征
            # 这里得到的h, w应该和上面的max_h, max_w一致
            # examples_features是从resnet50中得到的特征图对应的边界框部分的特征图
            # 而image_features是从resnet50中得到的特征图
            h, w = examples_features.shape[2], examples_features.shape[3]

            # 先对image_features进行填充，填充的目的是确保可以在图像的边缘进行卷积操作
            # 再进行卷积操作, examples_features就是卷积核
            # 一定要注意上面得到的三个边界框对应特征图堆叠起来的内容是作为卷积核，对整张图像的特征图进行卷积操作
            # features's shape: 1×3×h×w
            features =    F.conv2d(
                    F.pad(image_features, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
                    examples_features
                )
            # 重新排列张量的维度，将channel和batch互换
            # combined's shape: 3×1×h×w
            combined = features.permute([1,0,2,3])

            # computing features for scales 0.9 and 1.1 
            for scale in exemplar_scales:
                h1 = math.ceil(h * scale)
                w1 = math.ceil(w * scale)
                if h1 < 1: # use original size if scaled size is too small
                    h1 = h
                if w1 < 1:
                    w1 = w
                examples_features_scaled = F.interpolate(examples_features, size=(h1,w1),mode='bilinear')
                features_scaled =    F.conv2d(F.pad(image_features, ((int(w1/2)), int((w1-1)/2), int(h1/2), int((h1-1)/2))),
                examples_features_scaled)
                features_scaled = features_scaled.permute([1,0,2,3])
                # 这时候combined就是原始大小的特征图和示例特征进行卷积得到的内容，和两个不同尺度进行相同操作以后的内容堆叠起来
                # 注意这里的堆叠维度是1，而上面不同边界框堆叠维度是dim=0，我认为是因为上面使用permute调换了维度
                # 这里对combined再堆叠两个维度，所以最后combined内容就是使用实例中边界框对应的特征图作为卷积核对特征图进行卷积操作得到的特征图
                # 其中对应的三维分别是图像在1, 0.9, 1,1三个不同维度下完成卷积操作后的内容
                # 全部堆叠完成后, combined's shape: 3×3×h×w
                combined = torch.cat((combined,features_scaled),dim=1)

            if cnter == 0:
                # 这里的Combined的形状和combined的形状相同
                Combined = 1.0 * combined
            else:
                if Combined.shape[2] != combined.shape[2] or Combined.shape[3] != combined.shape[3]:
                    combined = F.interpolate(combined, size=(Combined.shape[2],Combined.shape[3]),mode='bilinear')
                Combined = torch.cat((Combined,combined),dim=1)
            cnter += 1

            # Combined对应着堆叠resnet不同层下得到的combined
            # Combined's shape: torch.Size([3, 6, 48, 72])
            # 3对应每张图像都有三个边界框
            # 6对应 3 + 3，分别对应map3和map4所对应的三个不同scale所堆叠起来的内容

        # 这里对应的是不同批次，将不同bctah的图像堆叠起来，本文batch为1，所以else不需要考虑
        # All_feat's shape: torch.Size([1, 3, 6, 48, 72])
        if ix == 0:
            All_feat = 1.0 * Combined.unsqueeze(0)
        else:
            All_feat = torch.cat((All_feat,Combined.unsqueeze(0)),dim=0)

    return All_feat


# 作用同resizeImageWithGT
class resizeImage(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes = sample['image'], sample['lines_boxes']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1
            resized_image = image

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        sample = {'image':resized_image,'boxes':boxes}
        return sample


# 调整图像大小，同时保持图像的宽高比，调整后的高度和宽度不会超过指定的最大值，并且都是8的倍数，同时调整相关的边界框和密度图。
class resizeImageWithGT(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    Modified by: Viresh
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            # 获取一个缩放系数
            scale_factor = float(self.max_hw)/ max(H, W)
            # 计算新图像的长和宽
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            # 调整图像和密度图的大小
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)
            # 对密度图进行调整
            if new_count > 0: resized_density = resized_density * (orig_count / new_count)
            
        else:
            scale_factor = 1
            resized_image = image
            resized_density = density

        # 对边界框进行调整
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            # 这里将边界框原本的某一维的四维变成了五维 (3, 4) --> (3, 5)
            boxes.append([0, y1,x1,y2,x2])
        # (3, 5) --> (1, 3, 5)
        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        # (384, 512) -->  (1, 1, 384, 512)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density}

        return sample


Normalize = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])
# 不处理密度图
Transform = transforms.Compose([resizeImage( MAX_HW)])
# 处理密度图
TransformTrain = transforms.Compose([resizeImageWithGT(MAX_HW)])

# 将归一化的张量反归一化，使像素值恢复到原始范围
def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD):
    """Reverses the normalisation on a tensor.
    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.
    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean
    Args:
        tensor (torch.Tensor, dtype=torch.float32): Normalized image tensor
    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(N, C, H, W)` (same shape as input)
    Return:
        torch.Tensor (torch.float32): Demornalised image tensor with pixel
            values between [0, 1]
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """
    # 创建tensor的一个副本，在不改变原始张量的情况下进行操作
    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized, means, stds):
        # 对于每个通道，使用mul_方法将通道的值×标准差std，然后用add_方法将均值mean添加到通道
        # 这样的操作原地进行，会修改denormalized的值
        channel.mul_(std).add_(mean)

    return denormalized


def scale_and_clip(val, scale_factor, min_val, max_val):
    "Helper function to scale a value and clip it within range"

    new_val = int(round(val*scale_factor))
    new_val = max(new_val, min_val)
    new_val = min(new_val, max_val)
    return new_val


def visualize_output_and_save(input_, output, boxes, save_path, figsize=(20, 12), dots=None):
    """
        dots: Nx2 numpy array for the ground truth locations of the dot annotation
            if dots is None, this information is not available
    """

    # get the total count
    pred_cnt = output.sum().item()
    boxes = boxes.squeeze(0)

    boxes2 = []
    # 上面对boxes进行了squeeze，所以这里的boxes.shape[0]就是边界框的个数
    for i in range(0, boxes.shape[0]):
        y1, x1, y2, x2 = int(boxes[i, 1].item()), int(boxes[i, 2].item()), int(boxes[i, 3].item()), int(
            boxes[i, 4].item())
        # 表示边界框内密度总和
        roi_cnt = output[0,0,y1:y2, x1:x2].sum().item()
        boxes2.append([y1, x1, y2, x2, roi_cnt])

    # 得到输入图像和密度图适合绘图的形式
    img1 = format_for_plotting(denormalize(input_))
    output = format_for_plotting(output)

    fig = plt.figure(figsize=figsize)

    # display the input image
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axis_off()
    ax.imshow(img1)

    # 可视化边界框，添加两个矩形，外边框是黄色实线，内边框是黑色虚线
    for bbox in boxes2:
        y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)

    # 如果有点注释，就把点注释加到图像上
    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], c='red', edgecolors='blue')
        # ax.scatter(dots[:,0], dots[:,1], c='black', marker='+')
        ax.set_title("Input image, gt count: {}".format(dots.shape[0]))
    else:
        ax.set_title("Input image")


    ax = fig.add_subplot(2, 2, 2)
    ax.set_axis_off()
    ax.set_title("Overlaid result, predicted count: {:.2f}".format(pred_cnt))

    # 对图像的不同通道×不同系数，然后相加，将其从RGB空间转换为灰度图像
    img2 = 0.2989*img1[:,:,0] + 0.5870*img1[:,:,1] + 0.1140*img1[:,:,2]
    ax.imshow(img2, cmap='gray')
    ax.imshow(output, cmap=plt.cm.viridis, alpha=0.5)

    # 展示密度图
    # display the density map
    ax = fig.add_subplot(2, 2, 3)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ax.imshow(output)
    # plt.colorbar()

    # # 在密度图上叠加边界框以及边界框内的计数
    ax = fig.add_subplot(2, 2, 4)
    # 关闭坐标轴显示
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ret_fig = ax.imshow(output)

    for bbox in boxes2:
        y1, x1, y2, x2, roi_cnt = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--',
                                  facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.text(x1, y1, '{:.2f}'.format(roi_cnt), backgroundcolor='y')

    # 在fig中添加一个颜色条，关联到ret_fig图像对象，并且只与子图ax相关联
    fig.colorbar(ret_fig, ax=ax)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close()

# 确保输入的张量适合绘图
def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.
    Tensors typically have a shape of :math:`(N, C, H, W)` or :math:`(C, H, W)`
    which is not suitable for plotting as images. This function formats an
    input tensor :math:`(H, W, C)` for RGB and :math:`(H, W)` for mono-channel
    data.
    Args:
        tensor (torch.Tensor, torch.float32): Image tensor
    Shape:
        Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        Output: :math:`(H, W, C)` or :math:`(H, W)`, respectively
    Return:
        torch.Tensor (torch.float32): Formatted image tensor (detached)
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """
    # 判断张量是否有批次维度，即其形状是否为四维。
    has_batch_dimension = len(tensor.shape) == 4
    # 创建张量的一个副本，在不改变原始张量的情况下操作
    formatted = tensor.clone()

    # 如果有batch维，则去除
    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    # 单通道返回长和宽
    # 否则将通道拿到最后一维
    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()