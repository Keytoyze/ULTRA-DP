from .conv import *
import torch

class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout = 0.2,
                 conv_name = 'hgt', prev_norm = False, last_norm = False, use_RTE = True, final_l2_norm=True):
        super(GNN, self).__init__()
        print(f"Init GNN: {in_dim} {n_hid} {num_types} {num_relations} {n_heads} {n_layers} {dropout}")
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm, use_RTE = use_RTE))
        self.final_l2_norm = final_l2_norm

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        # print(f"GNN forward: l2 norm = {self.final_l2_norm}")
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        if self.final_l2_norm:
            meta_xs = meta_xs / torch.linalg.norm(meta_xs, dim=-1, keepdim=True)
        return meta_xs   
