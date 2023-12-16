import sys

from model.data import *
from model.model import *
from PromptGNN import *
from tqdm import tqdm
import numba
from numba.core import types
from numba.typed import Dict
import dill

import argparse

parser = argparse.ArgumentParser(description='Pre-process reachability metrices')
parser.add_argument('--data_dir', type=str, default='datadrive/dataset/graph_dblp.pk', help='graph data file. ')
parser.add_argument('--max_step', type=int, default=9, help='max step for random walk. ')
args = parser.parse_args()

max_step = args.max_step

idx_type = np.int32
idx_nb_type = types.int32

print('Start Loading Graph Data...')
graph: Graph = dill.load(open(args.data_dir, 'rb'))
print('Finish Loading Graph Data!')

inner_dict_type = types.DictType(idx_nb_type, types.float64)

@numba.njit
def get_reachability_dict(start, edge_list):
    reachability_dict = Dict.empty(
        key_type=idx_nb_type,
        value_type=inner_dict_type,
    )
    step_target_nodes = [np.asarray([start], dtype=idx_type)]
    step_prob_nodes = [np.asarray([1.0], dtype=np.float64)]
    for step in range(0, max_step + 1):
        reachability_dict[step] = Dict.empty(
            key_type=idx_nb_type,
            value_type=types.float64,
        )
    reachability_dict[0][start] = 1.0
    for step in range(1, max_step + 1):
        target_nodes = []
        reachability_prob = []
        for front_node in reachability_dict[step - 1]:
            if front_node not in edge_list:
                continue
            leaves = edge_list[front_node]
            leaves_count = len(leaves)
            if leaves_count > 256:
                leaves = np.random.choice(leaves, 256, replace=False)
            for leaf in leaves:
                if leaf not in reachability_dict[step]:
                    reachability_dict[step][leaf] = 0
                reachability_dict[step][leaf] += reachability_dict[step - 1][front_node] / len(
                    leaves)
        for k, v in reachability_dict[step].items():
            if k != start:
                target_nodes.append(k)
                reachability_prob.append(v)
        step_target_nodes.append(np.asarray(target_nodes, idx_type))
        step_prob_nodes.append(np.asarray(reachability_prob, np.float64))
    return step_target_nodes, step_prob_nodes


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, np.ndarray):
        size += obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def size(num):
    suffix = "B"
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def save_reachability():
    node_dicts = [] # start -> {step -> [end], [prob]}

    pbar = tqdm(total=len(graph.node_feature['def']))
    edge_list_raw = graph.edge_list['def']['def']['def']

    edge_list = Dict.empty(
        key_type=idx_nb_type,
        value_type=idx_nb_type[:],
    )
    for k, v in edge_list_raw.items():
        v = list(v.keys())
        assert k < 2 ** 31
        edge_list[k] = np.asarray(v, dtype=idx_type)

    count = 0
    for node in range(len(graph.node_feature['def'])):
        reachability_dict = get_reachability_dict(node, edge_list)
        node_dict = []
        for i in range(len(reachability_dict[0])):
            node_dict.append(
                (reachability_dict[0][i], reachability_dict[1][i])
            )
        node_dicts.append(node_dict)
        count += 1
        pbar.update()

        if count % 1000 == 0:
            pbar.set_description(size(get_size(node_dicts)))

    pbar.close()
    target_path = args.data_dir.replace('.pk', '_reachability.pk')
    ofile = open(target_path, 'wb')
    dill.dump(node_dicts, ofile)
    ofile.close()

if __name__ == "__main__":
    save_reachability()