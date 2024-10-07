%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File to create grount truth density map for test set%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 清楚命令窗口和所有变量
clc; clear all;

dataset = 'A';
dataset_name = ['shanghaitech_part_' dataset ];
% 定义图像文件的路径
path = ['../data/original/shanghaitech/part_' dataset '_final/test_data/images/'];
% 定义真实密度图的路径
gt_path = ['../data/original/shanghaitech/part_' dataset '_final/test_data/ground_truth/'];
% 定义要保存csv文件的路径
gt_path_csv = ['../data/original/shanghaitech/part_' dataset '_final/test_data/ground_truth_csv/'];

mkdir(gt_path_csv )
if (dataset == 'A')
    num_images = 182;
else
    num_images = 316;
end

% 每处理10张图片，打印当前进度
for i = 1:num_images    
    if (mod(i,10)==0)
        fprintf(1,'Processing %3d/%d files\n', i, num_images);
    end

    % 加载图像, load这一步会产生image_info
    load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat')) ;
    input_img_name = strcat(path,'IMG_',num2str(i),'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
    % 使用rgb2gray函数将图像转换为灰度图
    if (c == 3)
        im = rgb2gray(im);
    end

    % 这个标点应该是数据集中那个ground_truth中就已经有了
    annPoints =  image_info{1}.location;
    [h, w, c] = size(im);
    im_density = get_density_map_gaussian(im,annPoints);    
    csvwrite([gt_path_csv ,'IMG_',num2str(i) '.csv'], im_density);       
end

