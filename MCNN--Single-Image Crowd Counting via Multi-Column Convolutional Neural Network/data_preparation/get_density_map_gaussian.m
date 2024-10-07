% 输出参数 im_density, 生成的密度图   输入参数 im, points 输入图像, 标注点矩阵
function im_density = get_density_map_gaussian(im,points)


im_density = zeros(size(im));
% 获取密度的高度和宽度
[h,w] = size(im_density);

% 检查标注点是否为空，如果是空的，返回空的密度图
if(length(points)==0)
    return;
end

% 如果point数组只有一行，即只有一个标注点
%   计算其坐标，保证在图像边界内
%   在密度图上将其值设置为255
if(length(points(:,1))==1)
    x1 = max(1,min(w,round(points(1,1))));
    y1 = max(1,min(h,round(points(1,2))));
    im_density(y1,x1) = 255;
    return;
end

% 设置高斯核大小和标准差sigma
for j = 1:length(points) 	
    f_sz = 15;
    sigma = 4.0;
    % 使用fspecial函数生成一个高斯核
    H = fspecial('Gaussian',[f_sz, f_sz],sigma);
    x = min(w,max(1,abs(int32(floor(points(j,1)))))); 
    y = min(h,max(1,abs(int32(floor(points(j,2))))));
    % 计算当前点的坐标，确保其在图像边界内
    if(x > w || y > h)
        continue;
    end


    % 计算高斯核在图像上的边界框
    % x1和x2是高斯核的左右边界, y1和y2是高斯核的上下边界
    x1 = x - int32(floor(f_sz/2)); y1 = y - int32(floor(f_sz/2));
    x2 = x + int32(floor(f_sz/2)); y2 = y + int32(floor(f_sz/2));

    % dfx1 x2 y1 y2表示高斯核需要调整的量, change_H用来判断是否需要重新计算高斯核
    dfx1 = 0; dfy1 = 0; dfx2 = 0; dfy2 = 0;
    change_H = false;

    % 图像的坐标系通常以 (1,1) 为原点，即图像的左上角
    % 如果 x1 小于 1（即左边界超出图像左边界），则调整 x1 为 1，并设置 dfx1 为需要裁剪的列数。
    % 如果 y1 小于 1（即上边界超出图像上边界），则调整 y1 为 1，并设置 dfy1 为需要裁剪的行数。
    % 如果 x2 大于图像宽度 w（即右边界超出图像右边界），则调整 x2 为 w，并设置 dfx2 为需要裁剪的列数。
    % 如果 y2 大于图像高度 h（即下边界超出图像下边界），则调整 y2 为 h，并设置 dfy2 为需要裁剪的行数。
    % 如果需要调整，则设置 change_H 为 true，表示需要重新计算高斯核。
    if(x1 < 1)
        dfx1 = abs(x1)+1;
        x1 = 1;
        change_H = true;
    end
    if(y1 < 1)
        dfy1 = abs(y1)+1;
        y1 = 1;
        change_H = true;
    end
    if(x2 > w)
        dfx2 = x2 - w;
        x2 = w;
        change_H = true;
    end
    if(y2 > h)
        dfy2 = y2 - h;
        y2 = h;
        change_H = true;
    end

    % x1h 和 y1h 分别表示调整后高斯核在原始核中的左上角坐标。
    % x2h 和 y2h 分别表示调整后高斯核在原始核中的右下角坐标。
    x1h = 1+dfx1; y1h = 1+dfy1; x2h = f_sz - dfx2; y2h = f_sz - dfy2;
    % 如果需要调整高斯核（即 change_H 为 true），则重新计算高斯核的大小和位置。
    if (change_H == true)
        H =  fspecial('Gaussian',[double(y2h-y1h+1), double(x2h-x1h+1)],sigma);
    end
    % 将高斯核叠加到密度图上
    im_density(y1:y2,x1:x2) = im_density(y1:y2,x1:x2) +  H;
     
end

end