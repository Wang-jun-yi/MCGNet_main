import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.segmentation import slic,mark_boundaries,felzenszwalb,quickshift,random_walker
from sklearn import preprocessing
import cv2
import math

def LSC_superpixel(I, nseg):
    superpixelNum = nseg
    ratio = 0.075
    size = int(math.sqrt(((I.shape[0] * I.shape[1]) / nseg)))
    superpixelLSC = cv2.ximgproc.createSuperpixelLSC(
        I,
        region_size=size,
        ratio=0.005)
    superpixelLSC.iterate()
    superpixelLSC.enforceLabelConnectivity(min_element_size=25)
    segments = superpixelLSC.getLabels()
    return np.array(segments,np.int64)

def SEEDS_superpixel(I, nseg):
    I=np.array(I[:, :, 0:3], np.float32).copy()
    I_new = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    # I_new =np.array( I[:,:,0:3],np.float32).copy()
    height, width, channels = I_new.shape
    
    superpixelNum = nseg
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, int(superpixelNum), num_levels=2,prior=1,histogram_bins=5)
    seeds.iterate(I_new,4)
    segments = seeds.getLabels()
    # segments=SegmentsLabelProcess(segments) # 排除labels中不连续的情况
    return segments

def SegmentsLabelProcess(labels):
    '''
    对labels做后处理，防止出现label不连续现象
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))
    
    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i
    
    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SLIC(object):
    def __init__(self, HSI,labels, n_segments=1000, compactness=20, max_iter=20, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        # 数据standardization标准化,即提前全局BN
        height, width, bands = HSI.shape  # 原始高光谱数据的三个维度
        data = np.reshape(HSI, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        self.labels=labels
        
    
    def get_Q_and_S_and_Segments(self):
        # 执行 SLIC 并计算 Q 矩阵（像素属于哪个超像素）、S 矩阵（每个超像素的平均特征）以及分割图 segments
        img = self.data  # 获取当前图像数据
        (h, w, d) = img.shape  # 获取图像的高、宽、波段数（维度）
        # 计算超像素S以及相关系数矩阵Q
        # 使用 skimage 的 slic 方法对图像进行超像素分割，得到 segments 标签图
        segments = slic(
            img,
            n_segments=self.n_segments,  # 超像素数量
            compactness=self.compactness,  # 控制空间距离和颜色距离的权重
            max_num_iter=self.max_iter,  # 最大迭代次数
            convert2lab=False,  # 是否将图像转换到 Lab 颜色空间（这里为 False，表示不转换）
            sigma=self.sigma,  # 高斯平滑的 sigma 值
            enforce_connectivity=True,  # 保证每个超像素区域是连接的
            min_size_factor=self.min_size_factor,  # 控制最小超像素大小
            max_size_factor=self.max_size_factor,  # 控制最大超像素大小
            slic_zero=False  # 是否使用 SLIC-zero 算法（False 表示标准 SLIC）
        )
        # segments = felzenszwalb(img, scale=1,sigma=0.5,min_size=25)
        
        # segments = quickshift(img,ratio=1,kernel_size=5,max_dist=4,sigma=0.8, convert2lab=False)
        
        # segments=LSC_superpixel(img,self.n_segments)
        
        # segments=SEEDS_superpixel(img,self.n_segments)
        
        # 检查 segments 是否是连续编号的，如果不是就进行校正（保证标签是连续的 0 ~ N-1）
        if segments.max()+1!=len(list(set(np.reshape(segments,[-1]).tolist()))): segments = SegmentsLabelProcess(segments)
        self.segments = segments  # 保存分割图
        superpixel_count = segments.max() + 1  # 计算超像素的个数
        self.superpixel_count = superpixel_count  # 存储超像素数量
        print("superpixel_count", superpixel_count)  # 打印超像素数量
        
        # ######################################显示超像素图片
        out = mark_boundaries(img[:,:,[0,1,2]], segments) # 可视化超像素边界（使用图像前三个通道）
        # out = (img[:, :, [0, 1, 2]]-np.min(img[:, :, [0, 1, 2]]))/(np.max(img[:, :, [0, 1, 2]])-np.min(img[:, :, [0, 1, 2]]))
        #plt.figure()
        #plt.imshow(out)
        #plt.show()
        
        segments = np.reshape(segments, [-1])  # 将 segments 展平为一维（每个像素对应一个超像素编号）
        S = np.zeros([superpixel_count, d], dtype=np.float32) # 初始化 S 矩阵：每个超像素的平均特征（维度为超像素数 × 波段数）
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)  # 初始化 Q 矩阵：每个像素属于哪个超像素的 one-hot 编码（像素数 × 超像素数）
        x = np.reshape(img, [-1, d]) # 将图像展平为二维：每一行为一个像素的波段特征

        # 遍历每个超像素
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]  # 找到属于第 i 个超像素的像素索引
            count = len(idx)  # 该超像素包含的像素数
            pixels = x[idx]  # 获取这些像素的特征
            superpixel = np.sum(pixels, 0) / count  # 计算这些像素特征的均值
            S[i] = superpixel  # 存入 S 矩阵中
            Q[idx, i] = 1  # 在 Q 矩阵中将这些像素在第 i 个超像素位置置为 1（表示归属）

        self.S = S  # 保存 S 矩阵
        self.Q = Q  # 保存 Q 矩阵

        return Q, S, self.segments  # 返回超像素归属矩阵 Q，超像素特征矩阵 S，以及分割图 segments
    
    def get_N(self, sigma: float):
        '''
    根据 segments 判定超像素之间的邻接矩阵 N
    N[i, j] 表示第 i 个和第 j 个超像素是否邻接，以及它们之间的相似度（基于特征差异）
    :param sigma: 控制相似度计算中高斯核的尺度参数
    :return: 超像素邻接矩阵 N（大小为 超像素数 × 超像素数）
    '''
        # 初始化邻接矩阵 N，所有值设为 0，表示初始没有邻接关系
        N = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        # 获取分割图的高和宽（segments 是二维标签图）
        (h, w) = self.segments.shape
        # 遍历图像中的每一个 2x2 小区域，滑动窗口判断局部是否存在不同超像素编号
        for i in range(h - 2):
            for j in range(w - 2):
                # 提取当前位置的 2x2 区块
                sub = self.segments[i:i + 2, j:j + 2]
                # 获取该 2x2 区域中的最大和最小超像素编号
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # 如果这两个值不同，说明这 2x2 区域中存在两个不同的超像素 => 它们是邻接的
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    # 如果已经计算过这两个超像素之间的相似度，则跳过
                    if N[idx1, idx2] != 0:
                        continue
                    # 获取这两个超像素的平均特征向量（来自 S 矩阵）
                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    # 计算它们之间的欧式距离并通过高斯函数转换为相似度
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    # 设置邻接矩阵中两个超像素之间的相似度（对称）
                    N[idx1, idx2] = N[idx2, idx1] = diss
        return N

class LDA_SLIC(object):
    def __init__(self,data,labels,n_component):
        # 构造函数，初始化数据、标签、目标降维维度
        self.data = data  # 输入的高光谱图像数据，形状为 (height, width, bands)
        self.init_labels = labels  # 初始标签，用于监督信息
        self.curr_data = data  # 当前处理的数据，初始化为原始数据
        self.n_component = n_component  # LDA 目标降维的维度数
        self.height, self.width, self.bands = data.shape  # 获取图像的高、宽、波段数
        self.x_flatt = np.reshape(data, [self.width * self.height, self.bands])  # 将3D图像数据展开为二维数组 (像素数, 波段数)
        self.y_flatt = np.reshape(labels, [self.height * self.width])  # 将标签也展平成一维
        self.labes = labels  # 保存初始标签（似乎这里应为 self.labels，可能是拼写错误）
        
    def LDA_Process(self,curr_labels):
        '''
        :param curr_labels: 当前使用的标签 (height * width)
        :return: LDA降维后的图像数据 (height, width, n_component)
        '''
        curr_labels = np.reshape(curr_labels, [-1])  # 将标签展平为一维
        idx = np.where(curr_labels != 0)[0]  # 找到非零标签的索引（去除未标记的像素）
        x = self.x_flatt[idx]  # 获取对应索引的像素特征
        y = curr_labels[idx]  # 获取对应的标签值
        lda = LinearDiscriminantAnalysis()  # 创建 LDA 对象（没有设置维度数，默认最大维度）
        lda.fit(x, y - 1)  # 使用标签训练 LDA 模型（标签减1以保证从0开始）
        X_new = lda.transform(self.x_flatt)  # 对所有像素进行LDA变换，降维
        return np.reshape(X_new, [self.height, self.width, -1])  # 将降维结果恢复为原图形状
       
    def SLIC_Process(self,img,scale=25):
        # 使用 SLIC 超像素方法对图像进行分割
        n_segments_init = self.height * self.width / scale  # 计算初始分割块数量
        print("n_segments_init", n_segments_init)  # 打印超像素初始数量
        # 使用自定义的 SLIC 类对图像进行超像素分割
        myslic=SLIC(img,n_segments=n_segments_init,labels=self.labes, compactness=0.01,sigma=1.5, min_size_factor=0.1, max_size_factor=2)
        Q, S, Segments = myslic.get_Q_and_S_and_Segments()  # 获取 SLIC 的 Q、S 矩阵和分割图
        N = myslic.get_N(sigma=10)  # 获取区域邻接矩阵或相似度矩阵
        return Q, S, N, Segments  # 返回结果


    def simple_superpixel(self,scale):
        # 执行包含 LDA 降维的超像素分割过程
        curr_labels = self.init_labels  # 使用初始标签
        X = self.LDA_Process(curr_labels)  # 使用 LDA 对图像进行降维
        Q, S, N, Seg = self.SLIC_Process(X, scale=scale)  # 对降维后的图像执行 SLIC 超像素分割
        return Q, S, N, Seg  # 返回超像素结果
    
    def simple_superpixel_no_LDA(self,scale):
        # 执行不包含 LDA 降维的超像素分割过程，直接对原始图像处理
        Q, S, N, Seg = self.SLIC_Process(self.data, scale=scale)  # 对原始图像执行 SLIC 超像素分割
        return Q, S, N, Seg  # 返回结果