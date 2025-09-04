import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random
import matplotlib.pyplot as plt
import os

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(42)


@torch.no_grad()
def getN(net_input, device, num_subgraphs=0, top_k=0,cross_subgraph_k=0,k_cos=0,row=True):
    height, width, _ = net_input.shape

    if row:
        out = net_input.reshape([height * width, -1]).to(device)
    else:
        out = net_input.permute(1, 0, 2).reshape(width * height, -1)
    out = F.layer_norm(out, [out.size(-1)])
    N = height * width


    rows, cols = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )

    coordinates = torch.stack((rows.flatten(), cols.flatten()), dim=1)



    edge_src, edge_dst = [], []


    subgraph_size = N // num_subgraphs
    subgraph_ranges = [(i * subgraph_size, min((i + 1) * subgraph_size, N))
                       for i in range(num_subgraphs)]


    for i in range(num_subgraphs):
        curr_start, curr_end = subgraph_ranges[i]
        curr_indices = torch.arange(curr_start, curr_end, device=device)
        curr_out = out[curr_indices]
        curr_coords = coordinates[curr_indices]


        if curr_end - curr_start > 1:
            feat_sim = torch.mm(curr_out, curr_out.T)
            coord_dist = torch.cdist(curr_coords.float(), curr_coords.float(), p=2)
            spatial_sim = torch.exp(-coord_dist ** 2 / 2.0)
            sim_matrix = torch.relu(feat_sim) * spatial_sim

            k = min(top_k, curr_end - curr_start - 1)
            topk_val, topk_idx = torch.topk(sim_matrix, k=k, dim=1) # [subgraph_size, top_k]
            #print(topk_idx)
            #print(curr_indices)

            src = curr_indices.view(-1, 1).expand(-1, k)
            #print(src)
            dst = curr_indices[topk_idx]
            #print(dst)

            edge_src.append(src.flatten())
            edge_dst.append(dst.flatten())


        if i < num_subgraphs - 1:
            next_start, next_end = subgraph_ranges[i + 1]


            curr_half = curr_start + (k_cos * (curr_end - curr_start)) // 10
            curr_half = min(max(curr_half, curr_start), curr_end - 1)
            curr_boundary = torch.arange(curr_half, curr_end, device=device)
            #print(curr_boundary)

            # 下一个子图的前半部分
            next_mid = next_start + ((10 - k_cos) * (next_end - next_start)) // 10
            next_half = min(max(next_mid, next_start + 1), next_end)
            next_boundary = torch.arange(next_start, next_half, device=device)
            #print(next_boundary)


            curr_feat = out[curr_boundary]
            next_feat = out[next_boundary]
            cross_sim = torch.mm(curr_feat, next_feat.T)

            curr_c = coordinates[curr_boundary]
            next_c = coordinates[next_boundary]
            coord_dist = torch.cdist(curr_c.float(), next_c.float(), p=2)
            spatial_sim = torch.exp(-coord_dist ** 2 / 2.0)

            combined_sim = torch.relu(cross_sim) * spatial_sim

            k = min(cross_subgraph_k, len(next_boundary))
            topk_val, topk_idx = torch.topk(combined_sim, k=k, dim=1) # [subgraph_size, top_k]

            src = curr_boundary.view(-1, 1).expand(-1, k)
            dst = next_boundary[topk_idx]

            edge_src.append(src.flatten())
            edge_dst.append(dst.flatten())


    if edge_src:
        sparse_A = torch.sparse_coo_tensor(
            torch.stack([torch.cat(edge_src), torch.cat(edge_dst)]),
            torch.ones(sum(len(es) for es in edge_src), dtype=torch.float, device=device),
            size=(N, N),
            device=device
        ).coalesce().to_sparse_csr()
    else:
        sparse_A = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float, device=device),
            size=(N, N),
            device=device
        ).to_sparse_csr()

    return sparse_A


from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

def row_hyp(FLAG):
    if FLAG==1:
        num_subgraphs = 1
        top_k = 100#100,80,60,40,20
        cross_subgraph_k = 5
        k_cos = 9
    elif FLAG==2:
        num_subgraphs = 40
        top_k = 30#30,24,18,12,6
        cross_subgraph_k = 2
        k_cos = 5
    elif FLAG==3:
        num_subgraphs = 4
        top_k = 70#70,56,42,28,14
        cross_subgraph_k = 5
        k_cos = 5
    elif FLAG == 4:  # 👈 添加这个分支
        num_subgraphs = 16  # 最关键改动
        top_k = 15
        cross_subgraph_k = 2
        k_cos = 5
    else:
        num_subgraphs = 0
        top_k = 0
        cross_subgraph_k = 0
        k_cos =0
    return num_subgraphs, top_k, cross_subgraph_k,k_cos

def col_hyp(FLAG):
    if FLAG==1:
        num_subgraphs = 3
        top_k = 50#50,40,30,20,10
        cross_subgraph_k = 5
        k_cos = 9
    elif FLAG==2:
        num_subgraphs = 20
        top_k = 20#20,16,12,8,4
        cross_subgraph_k = 2
        k_cos = 5
    elif FLAG==3:
        num_subgraphs = 4
        top_k = 50#50,40,30,20,10
        cross_subgraph_k = 5
        k_cos = 5
    elif FLAG == 4:  # 👈 添加这个分支
        num_subgraphs = 16  # 最关键改动
        top_k = 15
        cross_subgraph_k = 2
        k_cos = 5
    else:
        num_subgraphs = 0
        top_k = 0
        cross_subgraph_k = 0
        k_cos =0
    return num_subgraphs, top_k, cross_subgraph_k,k_cos


def combine(row_AX, col_AX, device):

    row_indices = row_AX.indices()
    col_indices = col_AX.indices()

    merged_indices = torch.cat([row_indices, col_indices], dim=1)

    unique_indices = torch.unique(merged_indices, dim=1)

    merged_AX = torch.sparse_coo_tensor(
        unique_indices,
        torch.ones(unique_indices.shape[1], device=device),  # 1
        size=row_AX.size(),
        device=device
    )

    return merged_AX.coalesce()



def process_sparse_A(net_input, device,row=False,col=False,combin=True,FLAG=-1):
    row_num_subgraphs,row_top_k,row_cross_subgraph_k,row_k_cos=row_hyp(FLAG)
    col_num_subgraphs,col_top_k,col_cross_subgraph_k,col_k_cos=col_hyp(FLAG)

    row_AX= getN(net_input, device, num_subgraphs=row_num_subgraphs, top_k=row_top_k, cross_subgraph_k=row_cross_subgraph_k, k_cos=row_k_cos,row=True)
    col_AX = getN(net_input, device, num_subgraphs=col_num_subgraphs, top_k=col_top_k, cross_subgraph_k=col_cross_subgraph_k,k_cos=col_k_cos, row=False)
    if row_AX.layout == torch.sparse_csr:
        row_AX = row_AX.to_sparse_coo()
        col_AX = col_AX.to_sparse_coo()

    if combin:
        AX=combine(row_AX,col_AX,device)
    elif row:
        AX=row_AX
    elif col:
        AX=col_AX

    edge_index = AX.indices()
    edge_attr = AX.values()

    #edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=AX.size(0))
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    #edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=AX.size(0))
    AX = torch.sparse_coo_tensor(
        edge_index,
        edge_attr,
        size=AX.size(),
        device=AX.device
    ).coalesce()

    return AX




@torch.no_grad()
def oldgetA(net_input,height, width, device, num_subgraphs=4, top_k=15):

    # Get embeddings for all pixels\
    out = net_input.reshape([height * width, -1]).to(device)
    out = F.layer_norm(out, [out.size(-1)])
    N = height * width  # total number of pixels

    # Get pixel coordinates (row, col) for all pixels
    row_indices, col_indices = torch.meshgrid(torch.arange(height, device=device),
                                              torch.arange(width, device=device))

    coordinates = torch.stack((row_indices.flatten(), col_indices.flatten()), dim=1)  # [N, 2]

    # Initialize sparse adjacency matrix
    indices = []
    values = []

    # Calculate subgraph boundaries
    subgraph_size = N // num_subgraphs
    subgraph_boundaries = [i * subgraph_size for i in range(num_subgraphs)] + [N]

    for i in range(num_subgraphs):
        # Get pixel indices for this subgraph
        start_idx = subgraph_boundaries[i]
        end_idx = subgraph_boundaries[i + 1]
        subgraph_indices = torch.arange(start_idx, end_idx, device=device)

        # Get embeddings and coordinates for this subgraph
        subgraph_out = out[subgraph_indices]  # [subgraph_size, d]
        subgraph_coords = coordinates[subgraph_indices]  # [subgraph_size, 2]

        # Compute similarity matrix (combination of feature similarity and spatial proximity)
        # 1. Feature similarity
        feature_sim = torch.matmul(subgraph_out, subgraph_out.T)  # [subgraph_size, subgraph_size]

        # 2. Spatial proximity (Gaussian kernel)
        coord_dist = torch.cdist(subgraph_coords.float(), subgraph_coords.float(),
                                 p=2)  # [subgraph_size, subgraph_size]
        spatial_sim = torch.exp(-coord_dist ** 2 / (2 * (1.0 ** 2)))  # sigma = 1.0

        # Combined similarity
        combined_sim = torch.relu(feature_sim) * spatial_sim

        # For each pixel, keep only top-k connections
        topk_values, topk_indices = torch.topk(combined_sim, k=top_k, dim=1)  # [subgraph_size, top_k]

        # Create indices for sparse matrix (global indices)
        row_indices = subgraph_indices.view(-1, 1).expand(-1, top_k)  # [subgraph_size, top_k]
        col_indices = subgraph_indices[topk_indices]  # [subgraph_size, top_k] (global indices)

        # Add to our sparse matrix components
        indices.append(torch.stack([row_indices.flatten(), col_indices.flatten()]))
        values.append(torch.ones_like(topk_values.flatten()))  # Set value to 1 for selected edges

    # Combine all subgraphs
    if indices:  # Check if we have any edges
        all_indices = torch.cat(indices, dim=1)  # [2, total_edges]
        all_values = torch.cat(values)  # [total_edges]

        # Create sparse adjacency matrix
        sparse_A = torch.sparse_coo_tensor(
            all_indices,
            all_values,
            size=(N, N),
            device=device
        ).coalesce()  # Remove potential duplicates
    else:
        # If no edges found (shouldn't happen), return empty sparse matrix
        sparse_A = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float, device=device),
            size=(N, N),
            device=device
        )

    return sparse_A


from torch_geometric.utils import coalesce
from torch_geometric.data import Data
import itertools

def construct_neighbor_matrix_with_self_loops_pyg(Q):
    n, m = Q.shape


    edge_index = []


    for k in range(m):

        rows_with_one = np.where(Q[:, k] == 1)[0]


        for i, j in itertools.combinations(rows_with_one, 2):
            edge_index.append([i, j])
            edge_index.append([j, i])


    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()


    #edge_index = torch.cat([edge_index, torch.arange(n).view(1, -1).repeat(2, 1)], dim=1)

    edge_index, _ = coalesce(edge_index, None, n, n)

    # 将边索引转换为稀疏矩阵
    data = Data(edge_index=edge_index, edge_attr=None, num_nodes=n)

    return data


def dense_to_sparse_coo(dense_adj, num_nodes=None, edge_values=None, device='cuda:0'):

    if not isinstance(dense_adj, torch.Tensor):
        dense_adj = torch.tensor(dense_adj, device=device)
    else:
        dense_adj = dense_adj.to(device)


    if num_nodes is None:
        num_nodes = torch.max(dense_adj).item() + 1


    if edge_values is None:
        edge_values = torch.ones(dense_adj.shape[1], device=device)
    elif not isinstance(edge_values, torch.Tensor):
        edge_values = torch.tensor(edge_values, device=device)
    else:
        edge_values = edge_values.to(device)


    sparse_adj = torch.sparse_coo_tensor(
        indices=dense_adj,
        values=edge_values,
        size=(num_nodes, num_nodes),
        device=device
    )


    return sparse_adj.coalesce()





class decode_GCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout,alpla=0.1,globel=True):
        super(decode_GCN, self).__init__()
        self.alpla = alpla
        self.Q = nn.Linear(input_dim,output_dim)
        self.K = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(input_dim, 64)
        self.BN = nn.BatchNorm1d(64)

        self.dropout = dropout
        self.globel=globel

        self.reset_parameters()

    def reset_parameters(self):
        self.Q.reset_parameters()
        self.K.reset_parameters()
        self.V.reset_parameters()
        self.BN.reset_parameters()



    def forward(self,X, H, M): #X是cnn模块的输出，H=超像素GCN，M=Q
        K = self.K(H)
        K = F.layer_norm(K, [K.size(-1)])
        Q = self.Q(X)
        Q = F.layer_norm(Q, [Q.size(-1)])
        V = self.V(H)
        V = self.BN(V)
        V = F.dropout(V, p=self.dropout, training=self.training)

        #print(Q.shape,K.shape,V.shape,M.shape,X.shape,"*******************")
        A=(torch.mm(Q,K.T)) / 8#N*m
        A=F.softmax(M*A,dim=0) #A=M*A,更快
        if self.globel:
            output = self.alpla*(torch.mm(A, V)) + torch.mm(M, H)
        else:
            output = torch.mm(M, X)

        return output










