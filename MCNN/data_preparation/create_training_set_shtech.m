%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File to create training and validation set       %
% for ShanghaiTech Dataset Part A and B. 10% of    %
% the training set is set aside for validation     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc; clear all;
seed = 95461354;
rng(seed)
N = 9;
dataset = 'A';
dataset_name = ['shanghaitech_part_' dataset '_patches_' num2str(N)];

path = ['../data/original/shanghaitech/part_' dataset '_final/train_data/images/'];

output_path = '../data/formatted_trainval/';

train_path_img = strcat(output_path, dataset_name,'/train/');
train_path_den = strcat(output_path, dataset_name,'/train_den/');

val_path_img = strcat(output_path, dataset_name,'/val/');
val_path_den = strcat(output_path, dataset_name,'/val_den/');

gt_path = ['../data/original/shanghaitech/part_' dataset '_final/train_data/ground_truth/'];

mkdir(output_path)
mkdir(train_path_img);
mkdir(train_path_den);
mkdir(val_path_img);
mkdir(val_path_den);

if (dataset == 'A')
    num_images = 300;
else
    num_images = 400;
end

% 取10%作为验证集
num_val = ceil(num_images*0.1);
% 随机打乱图像索引
indices = randperm(num_images);

for idx = 1:num_images
    i = indices(idx);
    % 每处理10张图像打印一次进度
    if (mod(idx,10)==0)
        fprintf(1,'Processing %3d/%d files\n', idx, num_images);
    end
    load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat')) ;
    input_img_name = strcat(path,'IMG_',num2str(i),'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end

    % 保证wn2和hn2是8的倍数
    wn2 = w/8; hn2 = h/8;
    wn2 =8 * floor(wn2/8);
    hn2 =8 * floor(hn2/8);

    % 获取标注点
    annPoints =  image_info{1}.location;

    % ??? w不应该是恒大于2*wn2嘛，不太懂这里在做什么
    if( w <= 2*wn2 )
        im = imresize(im,[ h,2*wn2+1]);
        annPoints(:,1) = annPoints(:,1)*2*wn2/w;
    end
    if( h <= 2*hn2)
        im = imresize(im,[2*hn2+1,w]);
        annPoints(:,2) = annPoints(:,2)*2*hn2/h;
    end

    [h, w, c] = size(im);
    a_w = wn2+1; b_w = w - wn2;
    a_h = hn2+1; b_h = h - hn2;
    
    im_density = get_density_map_gaussian(im,annPoints);
    for j = 1:N
        % b_w 先- a_w，再乘rand,确保之后+a_w后不会超出边界， b_h同理
        x = floor((b_w - a_w) * rand + a_w);
        y = floor((b_h - a_h) * rand + a_h);
        % 这里的长和宽才是原图像尺寸的四分之一吧 (我个人理解是这样的)
        x1 = x - wn2; y1 = y - hn2;
        x2 = x + wn2-1; y2 = y + hn2-1;
        
        
        im_sampled = im(y1:y2, x1:x2,:);
        im_density_sampled = im_density(y1:y2,x1:x2);

        % 筛选补丁区域内的标注点
        annPoints_sampled = annPoints(annPoints(:,1)>x1 & ...
            annPoints(:,1) < x2 & ...
            annPoints(:,2) > y1 & ...
            annPoints(:,2) < y2,:);

        % 调整标注点的坐标, 使其相对于补丁左上角
        % 在裁剪之前，annPoints 中的坐标是相对于原始图像的。一旦裁剪出补丁，这些坐标需要转换为相对于补丁的新坐标系
        annPoints_sampled(:,1) = annPoints_sampled(:,1) - x1;
        annPoints_sampled(:,2) = annPoints_sampled(:,2) - y1;
        img_idx = strcat(num2str(i), '_',num2str(j));        

        % 保存补丁
        if(idx < num_val)
            imwrite(im_sampled, [val_path_img num2str(img_idx) '.jpg']);
            csvwrite([val_path_den num2str(img_idx) '.csv'], im_density_sampled);
        else
            imwrite(im_sampled, [train_path_img num2str(img_idx) '.jpg']);
            csvwrite([train_path_den num2str(img_idx) '.csv'], im_density_sampled);
        end
        
    end
    
end

