from model.conv import *
import numpy as np


class PromptGNN(nn.Module):
    def __init__(self, gnn, types, out_types, anchor_num, prompt_size=0):
        super(PromptGNN, self).__init__()
        self.types = types
        self.gnn = gnn
        self.params = nn.ModuleList()
        if prompt_size != 0:
            knn_task_embedding = nn.Parameter(torch.zeros((10, prompt_size, gnn.in_dim)))
            link_task_embedding = nn.Parameter(torch.zeros((prompt_size, gnn.in_dim)))
            cl_task_embedding = nn.Parameter(torch.zeros((prompt_size, gnn.in_dim)))
        else:
            knn_task_embedding = None
            link_task_embedding = None
            cl_task_embedding = None
        if anchor_num != 0:
            self.position_embedding = nn.Linear(anchor_num, gnn.in_dim)
        else:
            self.position_embedding = None

        self.link_task_embedding = link_task_embedding
        self.knn_task_embedding = knn_task_embedding
        self.cl_task_embedding = cl_task_embedding
        self.zero_task_embedding = nn.Parameter(torch.zeros((prompt_size, gnn.in_dim)))
        self.ce = nn.CrossEntropyLoss(reduction = 'none')
        self.prototypical_embedding = nn.Parameter(torch.randn(out_types, gnn.n_hid))
    
    def get_task_embedding_pytorch(self, type):
        if type.startswith('knn_'):
            if self.knn_task_embedding is None:
                emb = None
            else:
                step = int(type.replace('knn_', ''))
                emb = self.knn_task_embedding[step]
        elif type == 'link':
            emb = self.link_task_embedding
        elif type == 'cl':
            emb = self.cl_task_embedding
        elif type == 'zero':
            emb = self.zero_task_embedding
        else:
            raise ValueError(type)
        return emb

    def get_prompt_features_numpy(self, type):
        emb = self.get_task_embedding_pytorch(type)
        if emb is None:
            return []
        array = emb.detach().cpu().numpy()
        return [np.expand_dims(array[i], axis=0) for i in range(len(array))]

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        return self.gnn(node_feature, node_type, edge_time, edge_index, edge_type)

