import bdb
from model.data import *
from model.model import *
from warnings import filterwarnings
from PromptGNN import *
import torch
import random
import io
from sklearn.metrics import f1_score
import signal
import os
import dill
import time


def print_log(*values):
    print(*values)

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='Pre-training and fine-tuning on a given graph')

'''
   Prompt arguments 
'''
parser.add_argument('--prompt_size', type=int, default=0, 
                    help='Amount of prompt nodes to attach')
parser.add_argument('--prompt_width', type=int, default=0, 
                    help='How many hop around the target node should the prompt node attach to. 0: only attach to the target node.')
parser.add_argument('--task_embedding_init', type=str, default='knn_6', 
                    help='Which pre-trained task embedding is used for fine-tuning initialization. Valid values: knn_[step], cl, link, zero. ')
parser.add_argument('--position_embedding_step_t', type=int, default=9, 
                    help='Random walk step of position embedding')
parser.add_argument('--position_embedding_weight', type=float, default=1.0, 
                    help='Weight for position embedding')
parser.add_argument('--position_anchor_num', type=float, default=0.0, 
                    help='The ratio of number of anchors and number of nodes')
'''
   Pre-training task arguments 
'''
parser.add_argument('--pre_training_task', type=str, default='hybrid-knn_6-link', 
                    help='Pre-training tasks. Valid value: "knn_[step]", "link", "cl", and the hybrid tasks formulated as "hybrid-[task1]-[task2]-..."')
parser.add_argument('--knn_margin', type=float, default=1.0, 
                    help='Margin for k-NN task')
parser.add_argument('--knn_center_loss_ratio', type=float, default=0.5,
                    help='Ratio of center loss against triplet loss, range: [0-1]')
parser.add_argument('--link_margin', type=float, default=0.5, 
                    help='Margin for edge prediction task')
'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='datadrive/dataset/graph_dblp.pk',
                    help='The address of preprocessed graph.')
parser.add_argument('--pretrain_model_dir', type=str, default='datadrive/models/ultra_dp_dblp',
                    help='The address for storing the pre-trained models.')
parser.add_argument('--cache_dir', type=str, default='datadrive/cache',
                    help='The address for storing the cache.')
parser.add_argument('--cuda', type=int, default=1,
                    help='Avaiable GPU ID')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many layers within a mini-batch subgraph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')
parser.add_argument('--few_shot', type=int, default=32, 
                    help='Few shot sample count for each class. 32 means 32-shot.')
parser.add_argument('--fine_tuning_repeat', type=int, default=5,
                    help='How many fine-tuning repeat for.')
'''
   GNN arguments 
'''
parser.add_argument('--conv_name', type=str, default='gat',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn', 'sage'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=64,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers',
                    action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers',
                    action='store_true')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
'''
    Optimization arguments
'''
parser.add_argument('--max_lr', type=float, default=1e-3,
                    help='Maximum learning rate.')
parser.add_argument('--ft_max_lr', type=float, default=1e-3,
                    help='Maximum learning rate for fine-tuning.')
parser.add_argument('--scheduler', type=str, default='cycle',
                    help='Name of learning rate scheduler.')
parser.add_argument('--ft_scheduler', type=str, default='null',
                    help='Name of learning rate scheduler for fine-tuning.')
parser.add_argument('--ft_epoch', type=int, default=500,
                    help='Number of epoch to fine-tune')
parser.add_argument('--pretrain_epoch', type=int, default=20,
                    help='Number of epoch to pre-train')
parser.add_argument('--n_pool', type=int, default=4,
                    help='Number of process to sample subgraph')
parser.add_argument('--n_batch', type=int, default=36,
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--n_valid', type=int, default=8,
                    help='Number of validation batch (sampled graphs)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')
parser.add_argument('--clip', type=float, default=0.5,
                    help='Gradient Norm Clipping')
parser.add_argument('--weight_decay', type=float, default=1e-4, 
                    help='Weight decay')
parser.add_argument('--serial', action='store_true', 
                    help='Whether to run graph sampling serial')

args = parser.parse_args()
args_print(args)

is_serial = args.serial

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

print_log('Start Loading Graph Data...')
graph: Graph = dill.load(open(args.data_dir, 'rb'))
if len(graph.y.shape) == 2 and graph.y.shape[1] == 1:
    graph.y = graph.y[:, 0]
reachability_path = args.data_dir.replace(".pk", "_reachability.pk")
if os.path.exists(reachability_path) and args.position_anchor_num != 0:
    cache_reachability_dict = dill.load(open(reachability_path, 'rb'))
else:
    cache_reachability_dict = None
print_log('Finish Loading Graph Data!')


target_type = 'def'
rel_stop_list = ['self', 'prompt']

pre_target_nodes = graph.pre_target_nodes
train_target_nodes = graph.train_target_nodes

pre_target_nodes = np.concatenate([pre_target_nodes, np.ones(len(pre_target_nodes))]).reshape(2,
                                                                                              -1).transpose()
train_target_nodes = np.concatenate([train_target_nodes, np.ones(len(train_target_nodes))]).reshape(
    2, -1).transpose()

repeat_num = int(len(pre_target_nodes) / args.batch_size // args.n_batch)

types = graph.get_types()

node_num = len(graph.node_feature['def'])
position_embedding_step_t = int(args.position_embedding_step_t)

anchors = []
if args.position_anchor_num != 0:
    array = np.zeros((node_num, ), dtype=np.float64)
    for k in range(len(cache_reachability_dict)):
        buff = np.zeros((node_num, ), dtype=np.float64)
        rea_idx, rea_prob = cache_reachability_dict[k][position_embedding_step_t]
        buff[rea_idx] = rea_prob
        array += buff
    anchors = (-array).argsort(kind='stable')[:int(args.position_anchor_num * node_num)]
    print_log("anchors: ", anchors[:20])

gnn = GNN(conv_name=args.conv_name,
          in_dim=len(graph.node_feature[target_type]['emb'].values[0]), n_hid=args.n_hid, \
          n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, num_types=len(types) + 1, \
          num_relations=len(graph.get_meta_graph()) + 1, prev_norm=args.prev_norm,
          last_norm=args.last_norm, use_RTE=False, final_l2_norm=True)

prompt_gnn = PromptGNN(gnn=gnn, types=types, out_types=graph.y.max().item() + 1, 
                    anchor_num=len(anchors), 
                    prompt_size=args.prompt_size)
prompt_gnn = prompt_gnn.to(device)

sample_methods = []
if args.pre_training_task.startswith("hybrid-"):
    x = args.pre_training_task.replace("hybrid-", "")
    sample_methods += x.split("-")
else:
    sample_methods.append(args.pre_training_task)
print_log(f'pre-training task: {sample_methods}')

def prompting(node_feature, prompt_data, task_embedding, node_type):
    if task_embedding is not None:

        prompt_idx, center_idx = prompt_data

        # task embedding part
        prompt_feature = task_embedding.repeat(prompt_idx, 1)

        # position embedding part
        if len(anchors) > 0:
            center_idx = center_idx[:, 0].astype(np.int)
            x = []
            for idx in center_idx:
                reachability_list = cache_reachability_dict[idx]
                modulation_features = []
                for step in range(1, len(reachability_list)):
                    if position_embedding_step_t >= 0 and step != position_embedding_step_t:
                        continue
                    buff = np.zeros((node_num, ), dtype=np.float64)
                    rea_idx, rea_prob = reachability_list[step]
                    buff[rea_idx] = rea_prob
                    hub_prob = buff[anchors]
                    hub_prob = hub_prob / (np.std(hub_prob) + 1e-7)
                    modulation_features.extend(list(hub_prob))
                x.append(modulation_features)
            x = torch.FloatTensor(np.asarray(x)).to(device)
            y = torch.tanh(prompt_gnn.position_embedding(x))
            prompt_feature = prompt_feature + y * args.position_embedding_weight

        node_feature[-len(prompt_feature):] = prompt_feature
        node_type[-len(prompt_feature):] = 1


def get_reachability_dict(n_step, start):
    if cache_reachability_dict is None:
        edge_list = graph.edge_list['def']['def']['def']
        reachability_dict = defaultdict(  # step
            lambda: defaultdict(  # end
                lambda: 0.0
            )
        )
        reachability_weight_dict = defaultdict(  # step
            lambda: defaultdict(  # node, prob
                lambda: [], []
            )
        )
        reachability_dict[0][start] = 1.0
        for step in range(1, n_step + 1):
            for front_node in reachability_dict[step - 1]:
                leaves = edge_list[front_node].keys()
                leaves_count = len(leaves)
                if leaves_count > 256:
                    leaves = np.random.choice(list(leaves), 256, replace=False)
                for leaf in leaves:
                    reachability_dict[step][leaf] += reachability_dict[step - 1][front_node] / len(
                        leaves)
            for k, v in reachability_dict[step].items():
                if k != start:
                    reachability_weight_dict[step][0].append(k)
                    reachability_weight_dict[step][1].append(v)
            reachability_weight_dict[step][1] = np.asarray(reachability_weight_dict[step][1])
        return reachability_weight_dict
    return cache_reachability_dict[int(start)]



def sample_with_reachability(step, base_node, batch_size, target_nodes):
    reachability_dict = get_reachability_dict(step, base_node)
    pos_step = random.randint(1, step)
    if len(reachability_dict[pos_step][0]) <= 1:
        return None, None
    positive_reachability_nodes = np.random.choice(
        reachability_dict[pos_step][0], size=batch_size // 2, replace=True,
        p=reachability_dict[pos_step][1] / np.sum(reachability_dict[pos_step][1])
    )
    neg_prob = 1 / (reachability_dict[step][1].astype(np.float64))
    neg_prob[np.isnan(neg_prob)] = 0
    if len(reachability_dict[step][0]) <= 1:
        return None, None
    neg_prob = neg_prob / np.sum(neg_prob)
    if np.isnan(neg_prob).any():
        print_log(list(1 / (reachability_dict[step][1].astype(np.float64))))
    negative_reachability_nodes = np.random.choice(
        reachability_dict[step][0], size=batch_size // 2, replace=True,
        p=neg_prob
    )
    return [np.asarray([x, 1.0]) for x in positive_reachability_nodes], \
           [np.asarray([x, 1.0]) for x in negative_reachability_nodes]


def sample_data(seed, target_nodes, time_range, batch_size, feature_extractor,
                prompt_node_feature, sample_method):
    
    def index_node(initial_table, indx):
        indx = list(map(int, indx['def']))
        re = []
        for x in initial_table:
            re.append(indx.index(x))
        return re

    def knn_sample():
        np.random.seed(seed)
        random.seed(seed)

        head_nodes = list(target_nodes[np.random.choice(len(target_nodes), 1)])
        if sample_method.startswith('knn'):
            step = int(sample_method.replace('knn_', ''))
            while True:
                positive_nodes, negative_nodes = sample_with_reachability(
                    step,
                    head_nodes[0][0],
                    batch_size, target_nodes
                )
                if positive_nodes is not None and negative_nodes is not None:
                    break
                head_nodes = list(target_nodes[np.random.choice(len(target_nodes), 1)])
        else:
            raise
        # print_log(positive_nodes, negative_nodes)
        positive_index_start = 1
        negative_index_start = positive_index_start + batch_size // 2
        negative_index_end = negative_index_start + batch_size // 2
        samp_target_nodes = np.asarray(
            head_nodes + positive_nodes + negative_nodes)
        feature, times, edge_list, indxs, _, prompt_index = \
            sample_subgraph(graph, time_range,
                            inp={
                                target_type: samp_target_nodes},
                            feature_extractor=feature_extractor,
                            sampled_depth=args.sample_depth,
                            sampled_number=args.sample_width,
                            prompt_node_feature=prompt_node_feature,
                            return_prompt_index=True,
                            prompt_width=args.prompt_width)

        samp_target_nodes_indx = list(samp_target_nodes[:, 0].astype(np.int32))
        return_indx = list(indxs['def'].astype(np.int32))
        pos_indx = []
        neg_indx = []
        for i, samp_target_nodes_indx_i in enumerate(samp_target_nodes_indx):
            is_pos = i < negative_index_start
            assert samp_target_nodes_indx_i in return_indx, f"{samp_target_nodes_indx_i}, {samp_target_nodes_indx}, {return_indx}"
            idx = return_indx.index(samp_target_nodes_indx_i)
            if is_pos:
                pos_indx.append(idx)
            else:
                neg_indx.append(idx)
        removed_edge = []
        neg_edge = []

        y = graph.y[return_indx]
        return 'knn_' + str(step), to_torch(feature, times, edge_list, graph, ['prompt']), np.asarray(removed_edge), \
               np.asarray(neg_edge), pos_indx, neg_indx, y, \
               samp_target_nodes[:, 0].astype(int), prompt_index

    def cl_sample():
        np.random.seed(seed)
        random.seed(seed)
        nodes = target_nodes[np.random.choice(len(target_nodes), batch_size // 2, replace=False)]
        feature, times, edge_list, indxs, _, prompt_index_1 = \
            sample_subgraph(graph, time_range,
                            inp={
                                target_type: np.asarray(nodes)},
                            feature_extractor=feature_extractor,
                            sampled_depth=args.sample_depth,
                            sampled_number=args.sample_width,
                            prompt_node_feature=prompt_node_feature,
                            return_prompt_index=True,
                            prompt_width=args.prompt_width, augment=True)
        data1 = to_torch(feature, times, edge_list, graph, ['prompt'])
        feature, times, edge_list, indxs, _, prompt_index_2 = \
            sample_subgraph(graph, time_range,
                            inp={
                                target_type: np.asarray(nodes)},
                            feature_extractor=feature_extractor,
                            sampled_depth=args.sample_depth,
                            sampled_number=args.sample_width,
                            prompt_node_feature=prompt_node_feature,
                            return_prompt_index=True,
                            prompt_width=args.prompt_width, augment=True)
        data2 = to_torch(feature, times, edge_list, graph, ['prompt'])

        return 'cl', data1, prompt_index_1, data2, prompt_index_2

    def link_sample():
        np.random.seed(seed)
        random.seed(seed)
        edge_list = graph.edge_list['def']['def']['def']

        p1, p2, n1, n2 = [], [], [], []
        nodes = []
        while len(nodes) < batch_size:
            node = target_nodes[np.random.choice(len(target_nodes), 1)][0]
            node_neighbors = list(map(int, edge_list[node[0]].keys()))
            nodes.append(node)
            index = (len(nodes) - 1) // 2
            pos = index % 2 == 0
            if pos and len(node_neighbors) >= 1:
                pos_node = random.choice(node_neighbors)
                nodes.append(np.asarray([float(pos_node), 1.0]))
                p1.append(int(node[0]))
                p2.append(pos_node)
            else:
                while True:
                    neg_node = list(target_nodes[np.random.choice(len(target_nodes), 1)])[0]
                    neg_node_int = int(neg_node[0])
                    if neg_node_int not in node_neighbors:
                        break
                nodes.append(np.asarray([float(neg_node_int), 1.0]))
                n1.append(int(node[0]))
                n2.append(neg_node_int)

        feature, times, edge_list, indxs, _, prompt_index = \
            sample_subgraph(graph, time_range,
                            inp={
                                target_type: np.asarray(nodes)},
                            feature_extractor=feature_extractor,
                            sampled_depth=args.sample_depth,
                            sampled_number=args.sample_width,
                            prompt_node_feature=prompt_node_feature,
                            return_prompt_index=True,
                            prompt_width=args.prompt_width)

        p1 = index_node(p1, indxs)
        p2 = index_node(p2, indxs)
        n1 = index_node(n1, indxs)
        n2 = index_node(n2, indxs)

        return 'link', to_torch(feature, times, edge_list, graph, ['prompt']), prompt_index, \
               p1, p2, n1, n2

    while True:
        try:
            if sample_method.startswith('knn'):
                result = knn_sample()
            elif sample_method.startswith('link'):
                result = link_sample()
            elif sample_method.startswith('cl'):
                result = cl_sample()
            else:
                raise
            break
        except bdb.BdbQuit as e1:
            raise e1
        except Exception as e:
            import traceback
            traceback.print_exc()
            seed = None
            print_log("Error! retry...")

    return result

def node_classification_sample(seed, nodes, time_range, mark):
    if mark is not None:
        cache_path = os.path.join(args.cache_dir,
                                  f"{mark}_{args.sample_depth}_{args.sample_width}_"
                                  f"{args.prompt_size}_{args.prompt_width}_"
                                  f"downstream.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return dill.load(f)
    np.random.seed(seed)
    random.seed(seed)
    sample_nodes = np.concatenate([nodes, np.ones(len(nodes))]).reshape(2, -1).transpose()
    feature, times, edge_list, _, texts, prompt_idx = sample_subgraph(graph, time_range,
                                                                      inp={target_type: sample_nodes},
                                                                      sampled_depth=args.sample_depth,
                                                                      sampled_number=args.sample_width,
                                                                      feature_extractor=feature_pyg,
                                                                      return_prompt_index=True,
                                                                      prompt_node_feature=prompt_gnn.get_prompt_features_numpy(type='link'),
                                                                      prompt_width=args.prompt_width)

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
        to_torch(feature, times, edge_list, graph)

    x_ids = np.arange(len(nodes))
    result = (
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, graph.y[nodes],
        nodes, prompt_idx)

    if mark is not None:
        if not os.path.isdir(args.cache_dir):
            os.makedirs(args.cache_dir)
        with open(cache_path, 'wb') as f:
            dill.dump(result, f)

    return result


def prepare_data(pool):
    jobs = []

    class DummyGet:
        def __init__(self, data):
            self.data = data

        def get(self):
            return self.data


    prompt_node_feature = prompt_gnn.get_prompt_features_numpy('link')
    if is_serial:
        for _ in np.arange(args.n_batch - args.n_valid):
            jobs.append(DummyGet(sample_data(None, pre_target_nodes, {1: True}, args.batch_size,
                                             feature_pyg, prompt_node_feature, random.choice(sample_methods))))
        for i in np.arange(args.n_valid):
            jobs.append(DummyGet(
                sample_data(i, train_target_nodes, {1: True}, args.batch_size, feature_pyg,
                            prompt_node_feature, sample_methods[i % len(sample_methods)])))
    else:
        for _ in np.arange(args.n_batch - args.n_valid):
            jobs.append(pool.apply_async(sample_data, args=(
                None, pre_target_nodes, {1: True}, args.batch_size, feature_pyg,
                prompt_node_feature, random.choice(sample_methods))))
        for i in np.arange(args.n_valid):
            jobs.append(pool.apply_async(sample_data, args=(
                i, train_target_nodes, {1: True}, args.batch_size, feature_pyg,
                prompt_node_feature, sample_methods[i % len(sample_methods)])))
    return jobs


if not is_serial:
    pool = mp.Pool(args.n_pool, init_worker)
    st = time.time()
    jobs = prepare_data(pool)
else:
    st = time.time()
    jobs = prepare_data(None)

best_val = 100000
best_val_str = ""
best_val_step = -1
train_step = 0
stats = []
optimizer = torch.optim.AdamW(prompt_gnn.parameters(), weight_decay=args.weight_decay, eps=1e-06,
                              lr=args.max_lr)
# optimizer = torch.optim.Adam(prompt_gnn.parameters(), eps=1e-06, lr=args.max_lr)

if args.scheduler == 'cycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.02,
                                                    anneal_strategy='linear', final_div_factor=100, \
                                                    max_lr=args.max_lr,
                                                    total_steps=repeat_num * args.n_batch * args.pretrain_epoch + 1)
elif args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, repeat_num * args.n_batch,
                                                           eta_min=1e-6)
else:
    scheduler = None

print_log('Start Pretraining...', args.pretrain_epoch, repeat_num, len(pre_target_nodes))


def distance(a, b):
    return -torch.cosine_similarity(a, b, dim=0) + 1


labels_group = None
init = False

best_f1_str = ""
best_f1 = [-1, -1, -1]

metrics = defaultdict(lambda: defaultdict(lambda: []))


cri = torch.nn.NLLLoss()


def get_knn_loss(params, is_test):
    task_name, data, removed_edge, neg_edge, pos_indx, neg_indx, y, raw_node, prompt_index = params
    if type(y) == np.ndarray:
        y_np = y
    else:
        y_np = y.numpy()
    pos_correct_prob = np.mean((y_np[0] == y_np[pos_indx]).astype(np.int32))
    neg_correct_prob = np.mean((y_np[0] == y_np[neg_indx]).astype(np.int32))
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data
    node_feature = node_feature.to(device)  # [N, F]
    node_type = node_type.to(device)

    prompting(node_feature, prompt_index, prompt_gnn.get_task_embedding_pytorch(task_name),
                                node_type)

    edge_time = edge_time.to(device)  # [E, ]
    edge_index = edge_index.to(device)  # [2, E]
    edge_type = edge_type.to(device)  # [E, ]
    node_emb_gnn = prompt_gnn.gnn(node_feature, node_type, edge_time, edge_index,
                                edge_type)  # [N, F2]
    node_emb = node_emb_gnn

    anchor = torch.mean(node_emb[pos_indx, :], dim=0, keepdim=False)
    pos_dis = torch.stack(
        [distance(anchor, node_emb[i]) for i in pos_indx])
    neg_dis = torch.stack(
        [distance(anchor, node_emb[i]) for i in neg_indx])
    center_loss = (pos_dis ** 2).mean()
    max_pos = torch.log(torch.exp(pos_dis).sum())
    min_neg = torch.log(torch.exp(args.knn_margin - neg_dis).sum())
    neg_loss = torch.relu(max_pos + min_neg) ** 2
    loss = center_loss * args.knn_center_loss_ratio + neg_loss * (1 - args.knn_center_loss_ratio)
    return loss, 0.0, pos_correct_prob, neg_correct_prob

def get_link_loss(params, is_test):

    task_name, data, prompt_index, p1, p2, n1, n2 = params

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data
    node_feature = node_feature.to(device)  # [N, F]
    node_type = node_type.to(device)

    prompting(node_feature, prompt_index, prompt_gnn.get_task_embedding_pytorch(task_name), node_type)

    edge_time = edge_time.to(device)  # [E, ]
    edge_index = edge_index.to(device)  # [2, E]
    edge_type = edge_type.to(device)  # [E, ]
    node_emb = prompt_gnn.gnn(node_feature, node_type, edge_time, edge_index,
                               edge_type)  # [N, F2]

    pos_embedding_1 = node_emb[p1] # [Np, E]
    pos_embedding_2 = node_emb[p2] # [Np, E]
    neg_embedding_1 = node_emb[n1] # [Np, E]
    neg_embedding_2 = node_emb[n2] # [Np, E]

    loss = torch.cosine_embedding_loss(
        pos_embedding_1,
        pos_embedding_2,
        margin=args.link_margin,
        target=torch.ones(len(pos_embedding_1)).to(device)
    ).mean() + torch.cosine_embedding_loss(
        neg_embedding_1,
        neg_embedding_2,
        margin=args.link_margin,
        target=-torch.ones(len(neg_embedding_1)).to(device)
    ).mean()
    
    tp = (torch.cosine_similarity(pos_embedding_1, pos_embedding_2) >= 0).int().sum().item()
    fp = len(p1) - tp
    tn = (torch.cosine_similarity(neg_embedding_1, neg_embedding_2) < 0).int().sum().item()
    fn = len(n1) - tn
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)

    return loss, acc, precision, recall


def get_cl_loss(params, is_test):
    task_name, data1, prompt_index_1, data2, prompt_index_2 = params

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data1
    node_feature = node_feature.to(device)  # [N, F]
    node_type = node_type.to(device)
    prompting(node_feature, prompt_index_1, prompt_gnn.get_task_embedding_pytorch(task_name), node_type)
    edge_time = edge_time.to(device)  # [E, ]
    edge_index = edge_index.to(device)  # [2, E]
    edge_type = edge_type.to(device)  # [E, ]
    node_emb_1 = prompt_gnn.gnn(node_feature, node_type, edge_time, edge_index,
                               edge_type)[:args.batch_size // 2, :]  # [N, F2]

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data2
    node_feature = node_feature.to(device)  # [N, F]
    node_type = node_type.to(device)
    prompting(node_feature, prompt_index_2, prompt_gnn.get_task_embedding_pytorch(task_name), node_type)
    edge_time = edge_time.to(device)  # [E, ]
    edge_index = edge_index.to(device)  # [2, E]
    edge_type = edge_type.to(device)  # [E, ]
    node_emb_2 = prompt_gnn.gnn(node_feature, node_type, edge_time, edge_index,
                               edge_type)[:args.batch_size // 2, :]  # [N, F2]
    
    node_emb_neg = torch.concat([node_emb_2[1:, :], node_emb_2[:1, ]], dim=0)

    loss = torch.cosine_embedding_loss(
        node_emb_1,
        node_emb_2,
        margin=args.link_margin,
        target=torch.ones(len(node_emb_1)).to(device)
    ).mean() + torch.cosine_embedding_loss(
        node_emb_1,
        node_emb_neg,
        margin=args.link_margin,
        target=-torch.ones(len(node_emb_1)).to(device)
    ).mean()

    tp = (torch.cosine_similarity(node_emb_1, node_emb_2) >= 0).int().sum().item()
    fp = len(node_emb_1) - tp
    tn = (torch.cosine_similarity(node_emb_1, node_emb_neg) < 0).int().sum().item()
    fn = len(node_emb_1) - tn
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)

    return loss, acc, precision, recall

def get_loss(params, is_test):
    if params[0].startswith('knn'):
        return get_knn_loss(params, is_test)
    elif params[0] == 'link':
        return get_link_loss(params, is_test)
    elif params[0] == 'cl':
        return get_cl_loss(params, is_test)
    else:
        raise ValueError(params[0])

try:
    for epoch in np.arange(args.pretrain_epoch) + 1:
        for batch in np.arange(repeat_num) + 1:

            # torch.cuda.empty_cache()

            train_data = [job.get() for job in jobs[:-args.n_valid]]
            valid_data = [job.get() for job in jobs[-args.n_valid:]]
            if not is_serial:
                pool.close()
                pool.join()
                pool = mp.Pool(args.n_pool, init_worker)
                jobs = prepare_data(pool)
            else:
                jobs = prepare_data(None)
            et = time.time()
            print_log('Data Preparation: %.1fs' % (et - st))

            prompt_gnn.train()
            train_losses = []
            train_extras = []

            for params in train_data:
                loss, extra, prob1, prob2 = get_loss(params, False)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prompt_gnn.parameters(), args.clip)
                optimizer.step()

                train_losses.append(loss.item())
                train_extras.append(extra)

                if scheduler is not None:
                    scheduler.step()
                prompt_emb = prompt_gnn.get_task_embedding_pytorch(params[0])
                if prompt_emb is None:
                    prompt_emb_mean = 0.0
                    prompt_emb_std = 0.0
                else:
                    detach = prompt_emb.detach().cpu().numpy()
                    prompt_emb_mean = np.mean(detach)
                    prompt_emb_std = np.std(detach)
                print_log(f"[{params[0]}] loss: {loss:.5f}, prob: {prob1:.5f} / {prob2:.5f}, extra: {extra:.5f}, task_emb: {prompt_emb_mean:.5f}±{prompt_emb_std:.5f}")

                # torch.cuda.empty_cache()
            '''
                Valid
            '''
            lr = optimizer.param_groups[0]['lr']
            prompt_gnn.eval()
            print_log('start eval')
            with torch.no_grad():
                valid_losses = []
                valid_extras = []
                for params in valid_data:
                    valid_loss, valid_extra, valid_prob1, valid_prob2 = get_loss(params, True)
                    print_log(f"valid: [{params[0]}] loss: {valid_loss:.5f}, prob: {valid_prob1:.5f} / {valid_prob2:.5f}, extra: {valid_extra:.5f}")
                    valid_losses.append(valid_loss.item())
                    valid_extras.append(valid_extra)

                st = time.time()
                valid_loss_str = f"{float(np.average(valid_losses)):.5f}"
                print_log(f"Epoch: {epoch} ({batch} / {repeat_num}) {round(st - et, 1)}s  "
                      f"LR: {lr:.5f} "
                      f"Loss: {float(np.average(train_losses)):.5f} / {float(np.average(valid_losses)):.5f}  "
                      f"Extra: {np.average(train_extras):.5f} / {np.average(valid_extras):.5f}   ")

                metrics['loss']['train'].append(np.average(train_losses))
                metrics['loss']['valid'].append(np.average(valid_losses))
                metrics['LR']['LR'].append(optimizer.param_groups[0]['lr'])


            if np.average(valid_losses) < best_val:
                best_val = np.average(valid_losses)
                best_val_str = f"{epoch} ({batch} / {repeat_num}) - {valid_loss_str} - {best_val}"
                best_val_step = epoch
                print_log('UPDATE!!!')
                if not os.path.isdir(os.path.dirname(args.pretrain_model_dir)):
                    os.mkdir(os.path.dirname(args.pretrain_model_dir))
                torch.save(prompt_gnn.state_dict(), args.pretrain_model_dir)
            else:
                print_log(f"Previous best: {best_val_str}")

        if epoch - best_val_step >= 50:
            break

    if not is_serial:
        pool.terminate()
        pool.join()
except KeyboardInterrupt as e:
    print_log("Caught KeyboardInterrupt, terminating workers")
except Exception as e1:
    pool.terminate()
    pool.join()
    raise e1

# fine-tune with few shot
print_log("Start fine-tune")
if not os.path.isdir(os.path.dirname(args.pretrain_model_dir)):
    os.mkdir(os.path.dirname(args.pretrain_model_dir))

if not os.path.isfile(args.pretrain_model_dir):
    torch.save(prompt_gnn.state_dict(), args.pretrain_model_dir)
    print_log("warning: fine-tune with random initialization!")

def sample_fewshot(total_nodes, seed):
    total_nodes = list(total_nodes)
    random.seed(seed)
    random.shuffle(total_nodes)
    random.seed(None)
    result = []
    remain_num = [args.few_shot for _ in range(graph.y.max().item() + 1)]
    for n in total_nodes:
        if remain_num[graph.y[n]] > 0:
            remain_num[graph.y[n]] -= 1
            result.append(n)
        if sum(remain_num) == 0:
            break
    return np.asarray(result)


final_result = {
    'pt_step': int(best_val_step),
    'pt_micro': best_f1[0],
    'pt_macro': best_f1[1],
    'pt_weight': best_f1[2]
}

few_shot_metrics = set()
def update_one_result(dit):
    global few_shot_metrics
    for k, v in dit.items():
        few_shot_metrics.add(k)
        if k not in final_result:
            final_result[k] = []
        final_result[k].append(v)

ce = torch.nn.CrossEntropyLoss()

def prototypical_finetune(nodes, is_test=False, mask=None):
    node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, raw_ids, prompt_idx = \
        node_classification_sample(randint(), nodes, {1: True}, mask)
    node_feature = node_feature.to(device)
    node_type = node_type.to(device)
    prompting(node_feature, prompt_idx, prompt_gnn.get_task_embedding_pytorch(args.task_embedding_init), node_type)
    embedding_gnn = prompt_gnn.gnn.forward(node_feature, node_type,
                                        edge_time.to(device), edge_index.to(device),
                                        edge_type.to(device))[x_ids] # (B, H), ylabel: (B, ), proto: (T, H)

    types_sim = []
    for type_id in range(len(prompt_gnn.prototypical_embedding)):
        a = embedding_gnn
        b = prompt_gnn.prototypical_embedding[type_id:type_id + 1]
        sim = torch.cosine_similarity(a, b, dim=1)
        types_sim.append(sim)
    types_sim = torch.stack(types_sim).transpose(0, 1) # [B, T]
    ylabel = ylabel.to(device)
    loss = ce(types_sim, ylabel)
    predicts = torch.argmax(types_sim, dim=1)

    ylabel = ylabel.detach().cpu().numpy()
    predicts = predicts.detach().cpu().numpy()
    f1_micro = f1_score(ylabel, predicts, average='micro')
    f1_macro = f1_score(ylabel, predicts, average='macro')
    f1_weighted = f1_score(ylabel, predicts, average='weighted')
    return loss, (f1_micro, f1_macro, f1_weighted)

for repeat_i in range(args.fine_tuning_repeat):
    repeat_flag = f"{repeat_i}"
    if args.few_shot != 8:
        repeat_flag += f"{args.few_shot}"
    few_shot_tune_nodes = sample_fewshot(graph.train_target_nodes, repeat_i)
    few_shot_valid_nodes = sample_fewshot(graph.valid_target_nodes, repeat_i)
    print_log(f"start seed: {repeat_i}\ntune: {few_shot_tune_nodes}\nvalid: {few_shot_valid_nodes}\ntotal: {len(graph.train_target_nodes)}")
    few_shot_tune_nodes_concat = np.concatenate(
        [few_shot_tune_nodes, np.ones(len(few_shot_tune_nodes))]).reshape(
        2, -1).transpose()
    few_shot_valid_nodes_concat = np.concatenate(
        [few_shot_valid_nodes, np.ones(len(few_shot_valid_nodes))]).reshape(
        2, -1).transpose()
    prompt_gnn.load_state_dict(torch.load(args.pretrain_model_dir, map_location=device), strict=False)
    prompt_gnn.to(device)
    prompt_gnn.train()

    t = Texttable()
    t.add_row(["Name", "Shape", "Param", "Trainable"])
    s = 0
    for name, param in prompt_gnn.named_parameters():
        cur_size = np.prod(param.shape)
        if param.requires_grad:
            s += cur_size
        t.add_row((name, param.shape, cur_size, param.requires_grad))
    print_log(t.draw())
    print_log(f"Total trainable params: {s}")

    trainable_params = filter(lambda p: p.requires_grad, prompt_gnn.parameters())
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=args.weight_decay, eps=1e-06,
                                  lr=args.ft_max_lr)

    if args.ft_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)
    elif args.ft_scheduler == 'null':
        scheduler = None
    else:
        raise ValueError(args.ft_scheduler)

    best_f1 = [-1, -1, -1]
    early_stop_counter = 0
    best_val_step = -1
    best_val_loss = 100000000

    for name, param in prompt_gnn.named_parameters():
        param.requires_grad = True
    trainable_params = filter(lambda p: p.requires_grad, prompt_gnn.parameters())
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=args.weight_decay, eps=1e-06,
                                  lr=args.ft_max_lr)
    best_model_io = None

    for epoch in np.arange(args.ft_epoch) + 1:

        prompt_gnn.eval()
        with torch.no_grad():
            valid_loss, valid_f1s = prototypical_finetune(few_shot_valid_nodes, mask=f'valid-fs-{repeat_flag}')
        valid_loss = valid_loss.item()
        if valid_loss < best_val_loss:
            print_log(f"Save model: {valid_loss} < {best_val_loss} (best), f1 = {valid_f1s}")
            best_val_step = epoch
            best_val_loss = valid_loss
            early_stop_counter = 0
            best_f1 = valid_f1s

            best_model_io = io.BytesIO()
            torch.save(prompt_gnn.state_dict(), best_model_io)
            best_model_io.seek(0)
            # torch.save(prompt_gnn.state_dict(), args.pretrain_model_dir + "_ft")
        else:
            print_log(f"Not save: {valid_loss} >= {best_val_loss} (best), f1 = {valid_f1s}")
        early_stop_counter += 1
        if early_stop_counter >= 100:
            break

        prompt_gnn.train()
        loss, f1s = prototypical_finetune(few_shot_tune_nodes, mask=f'train-fs-{repeat_flag}-{epoch % 10}')
        print_log(f"{repeat_i}-{epoch} [{early_stop_counter}] loss = {loss.item():.5f}, f1 = {f1s}, lr = {optimizer.param_groups[0]['lr']:.5f}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_gnn.parameters(), args.clip)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    update_one_result({
        'ft_micro': best_f1[0],
        'ft_macro': best_f1[1],
        'ft_weight': best_f1[2],
        'ft_step': int(best_val_step)
    })

    prompt_gnn.load_state_dict(torch.load(best_model_io, map_location=device), strict=True)
    prompt_gnn.to(device)
    prompt_gnn.eval()

    with torch.no_grad():
        test_loss, test_f1s = prototypical_finetune(graph.test_target_nodes, is_test=True, mask=f'test')
        print_log(f"test: {test_loss.item():.5f}, f1 = {test_f1s}")

        val_loss, val_f1s = prototypical_finetune(few_shot_valid_nodes, mask=f'valid-fs-{repeat_flag}')
        print_log(f"valid: {val_loss.item():.5f}, f1 = {val_f1s}")

        train_loss, train_f1s = prototypical_finetune(few_shot_tune_nodes)
        print_log(f"train: {train_loss.item():.5f}, f1 = {train_f1s}")
    update_one_result({
        'test_loss': test_loss.item(),
        'test_micro': test_f1s[0],
        'test_macro': test_f1s[1],
        'test_weight': test_f1s[2],
        'test_train_micro': train_f1s[0],
        'test_valid_micro': val_f1s[0]
    })

for e in few_shot_metrics:
    final_result[e + "_std"] = float(np.std(final_result[e]))
    final_result[e] = float(np.mean(final_result[e]))

print_log(final_result)

