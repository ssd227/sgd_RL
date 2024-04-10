'''
定义适合围棋数据的 Dataset、dataload 和 transforms
方便pytorch 直接调用

现有的数据输入都是npy格式的文件，每个文件存放的样本数量为chunksize 1024
label数据与feature数据对应

KGS-2008-19-14002-train_features_0.npy
KGS-2008-19-14002-train_features_1.npy
KGS-2008-19-14002-train_labels_0.npy
KGS-2008-19-14002-train_labels_1.npy
'''
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import v2

go_transform = v2.Compose([torch.from_numpy, v2.ToDtype(torch.float32, scale=True)])
go_target_transform = v2.Compose([torch.tensor, v2.ToDtype(torch.long, scale=True)])

class GoDataset(Dataset):
    def __init__(self,
                preprocess_dir,
                transform=go_transform,
                target_transform=go_target_transform,
                datatype='train',
                device=None):
        
        self.data_dir = preprocess_dir
        self.data_type = datatype # 'train' or 'test'
        self.chunksize = 1024
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.features_files = self.init_features_files(self.data_dir)

        # 弄个缓存，每次读一个文件，就把所有的pair对存好。
        # 尽量减少为了读一行数据，反复加载整个文件
        # 缓存几个文件呢，（数据读过了直接删除缓存）
        self.cache = {}
        
        self.device = device
        
    def init_features_files(self, data_dir):
        files = []
        base = data_dir + '/' + '*'+ self.data_type + '_features_*.npy'
        for feature_file in glob.glob(base):
            files.append(feature_file)                    
        return files

    def __len__(self):
        return len(self.features_files) * self.chunksize # 总样本数 = 文件数 * chunksize

    def __getitem__(self, idx):
        
        f_id = idx // self.chunksize
        f_offset = idx % self.chunksize
        # 需要加载的文件
        ff_path = self.features_files[f_id]
        lf_path = ff_path.replace("features", "labels")
        
        # 命中缓存直接返回数据
        if ff_path not in self.cache:
            # 读文件, 添加进缓存
            features = np.load(ff_path)
            labels = np.load(lf_path)

            # 清理缓存
            if len(self.cache) >= 2:
                self.cache = {}
            
            self.cache.setdefault(ff_path, {})
            self.cache[ff_path] = { i:(features[i], labels[i].astype(int)) for i in range(len(features))}
            
        feature, label = self.cache[ff_path][f_offset] # 从缓存读取数据
        
        if self.transform:
            feature = self.transform(feature)
            if self.device:
                feature = feature.to(self.device)
        if self.target_transform:
            label = self.target_transform(label)
            if self.device:
                label = label.to(self.device)
            

        return feature, label
    
# todo 感觉直接用pytorch的数据加载
#   有时候不好针对自己的数据进行有效的管理
#   比如随机性和数据类型，以及加载效率之间的平衡关系
    

