from torch_geometric.datasets import CitationFull, Coauthor
from model.data import *
import tqdm
import pandas as pd
import dill

import argparse
import os

parser = argparse.ArgumentParser(description='Pre-process dataset')
parser.add_argument('--dataset', choices=['DBLP', 'Pubmed', 'CoraFull', 'Coauthor-CS'])
parser.add_argument('--data_root', type=str, default="datadrive/dataset")
args = parser.parse_args()

if args.dataset == 'DBLP':
    save_name = 'dblp'
    dataset = CitationFull(root=args.data_root, name='DBLP')
elif args.dataset == 'Pubmed':
    save_name = 'pubmed'
    dataset = CitationFull(root=args.data_root, name='PubMed')
elif args.dataset == 'CoraFull':
    save_name = 'cora'
    dataset = CitationFull(root=args.data_root, name='Cora')
elif args.dataset == 'Coauthor-CS':
    save_name = 'cs'
    dataset = Coauthor(root=args.data_root, name='CS')
else:
    raise ValueError("Unknown dataset. Please setup PyG dataset on your own.")

graph = Graph()
el = defaultdict(  #target_id
                    lambda: defaultdict( #source_id(
                        lambda: int # time
                    ))
for i, j in tqdm(dataset.data.edge_index.t()):
    el[i.item()][j.item()] = 1

target_type = 'def'
graph.edge_list['def']['def']['def'] = el
n = list(el.keys())
degree = np.zeros(np.max(n)+1)
for i in n:
    degree[i] = len(el[i])
x = np.concatenate((dataset.data.x.numpy(), np.log(degree).reshape(-1, 1)), axis=-1)
graph.node_feature['def'] = pd.DataFrame({'emb': list(x)})

idx = np.arange(len(graph.node_feature[target_type]))
np.random.seed(43)
np.random.shuffle(idx)

graph.pre_target_nodes   = idx[: int(len(idx) * 0.7)]
graph.train_target_nodes = idx[int(len(idx) * 0.7): int(len(idx) * 0.8)]
graph.valid_target_nodes = idx[int(len(idx) * 0.8): int(len(idx) * 0.9)]
graph.test_target_nodes  = idx[int(len(idx) * 0.9):]

graph.y = dataset.data.y
dill.dump(graph, open(os.path.join(args.data_root, f"graph_{save_name}.pk"), 'wb'))
