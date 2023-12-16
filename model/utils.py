import numpy as np
from texttable import Texttable
from collections import *

def args_print(args):
    _dict = vars(args)
    t = Texttable() 
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())

def randint():
    return np.random.randint(2**16 - 1)

def feature_pyg(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]
        
        feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'emb']), dtype=np.float)
        times[_type]   = tims
        indxs[_type]   = idxs
        
        if _type == 'def':
            attr = feature[_type]
    return feature, times, indxs, attr

def load_gnn(_dict):
    out_dict = {}
    for key in _dict:
        if 'gnn' in key:
            out_dict[key[4:]] = _dict[key]
    return OrderedDict(out_dict)
