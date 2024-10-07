import h5py
import torch
import shutil

# 保存模型参数
def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            # HDF5文件需要Numpy数组格式的数据
            h5f.create_dataset(k, data=v.cpu().numpy())


# 保存模型参数
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            # 使用np.asarray转换为numpy数组，再用from_numpy转换为pytorch张量
            param = torch.from_numpy(np.asarray(h5f[k]))
            # 使用copy_方法将读取的参数param复制到模型的相应参数v中
            v.copy_(param)


# 保存Pytorch模型的检查点
def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    torch.save(state, task_id+filename)
    # 如果是当前最后的模型，则新保存一个文件副本
    if is_best:
        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')            