from config import tableModeLinePath
import h5py
import numpy as np
import torch
from models.model import table_net

data = h5py.File(tableModeLinePath)
Model = table_net(2)
state_dict = Model.state_dict()
layer_keys = list(state_dict.keys())

n = 0
m = 0
for layer_name in layer_keys:
    if 'CNN' in layer_name:
        name = 'conv2d'
        if n > 0:
            name += '_{}'.format(str(n))
        to_data = torch.from_numpy(np.array(data[name][name]['kernel:0'])).permute(3,2,0,1)
        print('{}|{}|{}|{}'.format(layer_name, state_dict[layer_name].shape, name, to_data.shape))
        state_dict[layer_name] = to_data
        n += 1
    elif 'BatchNormal' in layer_name:
        name = 'batch_normalization'
        if m // 5 > 0:
            name += '_{}'.format(str(m//5))
        if 'weight' in layer_name:
            state_dict[layer_name] = torch.from_numpy(np.array(data[name][name]['gamma:0']))
            print('{}|{}|{}|{}'.format(layer_name, state_dict[layer_name].shape, name, data[name][name]['gamma:0'].shape))
        elif 'bias' in layer_name:
            state_dict[layer_name] = torch.from_numpy(np.array(data[name][name]['beta:0']))
            print('{}|{}|{}|{}'.format(layer_name, state_dict[layer_name].shape, name, data[name][name]['beta:0'].shape))
        elif 'running_mean' in layer_name:
            state_dict[layer_name] = torch.from_numpy(np.array(data[name][name]['moving_mean:0']))
            print('{}|{}|{}|{}'.format(layer_name, state_dict[layer_name].shape, name, data[name][name]['moving_mean:0'].shape))
        elif 'running_var' in layer_name:
            state_dict[layer_name] = torch.from_numpy(np.array(data[name][name]['moving_variance:0']))
            print('{}|{}|{}|{}'.format(layer_name, state_dict[layer_name].shape, name, data[name][name]['moving_variance:0'].shape))
        m += 1
    elif 'classify' in layer_name:
        name = 'conv2d_{}'.format(str(n))
        if 'weight' in layer_name:
            to_data = torch.from_numpy(np.array(data[name][name]['kernel:0'])).permute(3,2,0,1)
            print('{}|{}|{}|{}'.format(layer_name, state_dict[layer_name].shape, name, to_data.shape))
            state_dict[layer_name] = to_data
        elif 'bias' in layer_name:
            to_data = torch.from_numpy(np.array(data[name][name]['bias:0']))
            print('{}|{}|{}|{}'.format(layer_name, state_dict[layer_name].shape, name, to_data.shape))
            state_dict[layer_name] = to_data

torch.save(state_dict, './table_line_pt.pt')
