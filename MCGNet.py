import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from ours_de import decode_GCN


def noramlize(A: torch.Tensor):
    D = A.sum(1)
    D_hat = torch.diag(torch.pow(D, -0.5))
    A = torch.mm(torch.mm(D_hat, A), D_hat)
    return A


def spareadjacency(adjacency):
    if not adjacency.is_sparse:
        adjacency = adjacency.to_sparse()

    adjacency = adjacency.coalesce()
    indices = adjacency.indices()
    values = adjacency.values()
    size = adjacency.size()
    adjacency = torch.sparse_coo_tensor(indices, values, size, dtype=torch.float32, device=device)
    return adjacency


class SGC(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor, dropout):
        super(SGC, self).__init__()
        self.Activition = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(input_dim)
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))

        nodes_count = A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        A = noramlize(A + self.I)

        ratio = 1 - (torch.sum(A != 0) / (nodes_count ** 2))
        print(f"Sparsity ratio: {ratio}")

        if ratio > 0.92:
            self.A = spareadjacency(A)
        else:
            self.A = A

        self.dropout = dropout

    def forward(self, H):
        H = F.normalize(H)
        H = self.bn(H)
        output = torch.sparse.mm(self.A, self.GCN_liner_out_1(H))
        output = self.Activition(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output


class PGC(nn.Module):
    def __init__(self, input: int, output: int, dropout, A, res=False, MPNN='GCN'):
        super(PGC, self).__init__()
        self.Activition = nn.LeakyReLU()
        self.input = input
        self.output = output
        self.dropout = dropout
        self.BN = nn.BatchNorm1d(input)

        if MPNN == 'GCN':
            from torch_geometric.utils import dense_to_sparse
            if A.is_sparse:
                A = A.coalesce()
                self.edge_index = A.indices()
                self.edge_weight = A.values()
            else:
                A = A.cpu()
                self.edge_index, self.edge_weight = dense_to_sparse(A)
            self.gnn = GCNConv(input, output)
        elif MPNN == 'SAGE':
            self.gnn = SAGEConv(input, output, normalize=False, aggr='sum')
        elif MPNN == 'GAT':
            self.gnn = GATConv(input, output // 2, heads=2, concat=True, add_self_loops=True)

        self.A = A
        self.res = res
        if self.res:
            self.rslin = nn.Linear(input, output)

    def forward(self, x):
        x = F.normalize(x)
        x = self.BN(x)
        output = self.gnn(x, self.edge_index, self.edge_weight)
        if self.res:
            output += self.rslin(x)
        output = self.Activition(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output


class LSE(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(LSE, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class MCGNet(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor,
                 AX):
        super(MCGNet, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.AX=AX
        self.model = 'normal'
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q

        layers_count = 2
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())


        self.LSE_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.LSE_Branch.add_module('LSE_Branch' + str(i), LSE(128, 128, kernel_size=5))
            else:
                self.LSE_Branch.add_module('LSE_Branch' + str(i), LSE(128, 64, kernel_size=5))

        # Superpixel-level Graph Sub-Network
        gnncout=2
        self.SGC_Branch = nn.Sequential()
        for i in range(gnncout):
            if i < gnncout - 1:
                self.SGC_Branch.add_module('SGC_Branch' + str(i), SGC(128, 128, self.A,0))
            else:
                self.SGC_Branch.add_module('SGC_Branch' + str(i), SGC(128, 64, self.A,0))


        PGClayers_count = 2  # ,2,
        self.PGC_Branch = nn.Sequential()
        for i in range(PGClayers_count):
            print('PGC')
            if i < PGClayers_count - 1:
                self.PGC_Branch.add_module('PGC_Branch' + str(i), PGC(input=128,output= 128,dropout=0.8,A=self.AX))
            else:
                self.PGC_Branch.add_module('PGC_Branch' + str(i), PGC(input=128,output= 64,dropout=0.8,A=self.AX))

        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))
        self.decode = decode_GCN(input_dim = 64, output_dim = 64, dropout = 0.8,alpla = 0.1,globel=True) #input_dim=64



    def forward(self, x: torch.Tensor):

        (h, w, c) = x.shape


        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise

        clean_x_flatten = clean_x.reshape([h * w, -1])

        PGCruslt=self.PGC_Branch(clean_x_flatten)

        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # 低频部分


        hx = clean_x


        LSE_result = self.LSE_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))  # spectral-spatial convolution
        LSE_result = torch.squeeze(LSE_result, 0).permute([1, 2, 0]).reshape([h * w, -1])


        H = superpixels_flatten
        if self.model == 'normal':
            for i in range(len(self.SGC_Branch)): H = self.SGC_Branch[i](H)
        else:
            for i in range(len(self.SGC_Branch)): H, _ = self.SGC_Branch[i](H, model='smoothed')

        #GCN_result = torch.matmul(self.Q, H)+PGCruslt
        SGC_result = self.decode(LSE_result, H, self.Q)+PGCruslt


        Y = torch.cat([SGC_result, LSE_result], dim=-1)
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y
