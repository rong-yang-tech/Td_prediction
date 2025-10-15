# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:21:20 2024

@author: PC
"""

import os

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import  Subset, random_split
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import pandas as pd
from torch_geometric.loader import DataLoader

import sys

from sklearn.metrics import mean_absolute_error,r2_score, mean_absolute_percentage_error
from ml_gnn_graph import CustomGraphDataset
from ml_gnn_layer_nn import CCPGraph
#import math 
from sklearn.model_selection import KFold
from ml_gnn_utils import DualOutput, draw_heat_map

att_dtype = np.float32

#delete_folder_recursive('prediction')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
dataset_name = 'Td_pre_new88'
dataset = CustomGraphDataset(root = dataset_name, mode = 'prediction')
# 'HUBLAU.RFD.CIF_821.xyz','HEYZIX.RFD.cif_737.xyz',LAZPIO.RFD.cif_421.xyz','LAZPIO.RFD.cif_511.xyz',,'ACETAF01.RFD.cif_566.xyz'
define_draw_name = ['[N-]=[N+]=Nc1n[nH]nc1-c1nnn[nH]1', 'O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1','O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])C1','[N-]=[N+]=Nc1cc(N=[N+]=[N-])nc(N=[N+]=[N-])n1']

batch_size = 88
# soap_truancate = 271# dif_max
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# torch.serialization.add_safe_globals([dataset])
u_dim = [dataset[0].u.shape[0]]

model = CCPGraph(u_dim,
                 # 255, 17,  44, 1724,  1471,  1885,  1975, 322,  0.10074925921752349, 0.10068535608528603, 0.14394447821081802)

               144, 137, 217, 126, 665, 1526, 1317,586, 0.19013568030892725, 0.18306424528822834, 0.11958168442951482)
                 # 227, 256,  198,  481,  287,
     # 1771,  1705,  845,  0.1094705632310795,
     #  0.10793048512795315,  0.19210173291469798)

#
# 229,  222,  227,  1745,  1239,  1128,
#      1571, 326,  0.13371345351421732,  0.13042231358065917, 0.22026937064318997)

# model = CCPGraph(u_dim, 178, 141, 62,
#                  323,370, 1936, 1949, 1705,
#                  0.12076090605972635, 0.44709334556648733, 0.4167116754945725)# dif_max
# model = CCPGraph( u_dim, 157, 206, 67,
#                  2981, 2850, 923, 1203, 2949,
#                    0.15533512523766185,0.11030400231364725,0.23186860260479136)#remove_duplicates
# model.load_state_dict(torch.load(dataset_name+'/best_model_train664.pt',map_location=torch.device('cpu')))# dif_max
model.load_state_dict(torch.load(dataset_name+'/best_model_Td_onlygnn_predict.pt',weights_only=True,map_location=torch.device('cpu')))
# print(model)
model.eval()
model.to(device)
config_names = []
inference_results = []

with torch.no_grad():
    for batch_idx, data_batch in enumerate(loader):
        data = data_batch.to(device)
        print(data.name)
        config_names.extend(data.name)
        outputs, att = model(data)
        #print(data.batch)
        # print(outputs)
        unique_indices = torch.unique(data.batch)

        # 存储分割后的子张量
        grouped_tensors = []

        for idx in unique_indices:
            mask = data.batch == idx
            sub_tensor = att[mask]
            grouped_tensors.append(sub_tensor)
       # label = data.y.unsqueeze(1)
        #loss = criterion(outputs, label)
        #test_losses.append(loss.item())
        # print(outputs.detach().cpu().numpy().shape)
        inference_results.extend(outputs.detach().cpu().numpy())

        # attention图
        for draw_name in data.name:
            print(draw_name)
            for i in define_draw_name:
                # print(i)
                if draw_name == i:
                    # print(draw_name)
                    draw_num = data.name.index(draw_name)
                    print(data[draw_num].name)
                    draw_heat_map(data[draw_num].rdkit_mol[0],
                                  grouped_tensors[draw_num].cpu().numpy().reshape(-1),
                                  draw_name)
        break

        # test_targets.extend(label.numpy())
        # test_predictions.extend(outputs.numpy())
print(config_names, inference_results)
pred_data = pd.DataFrame(inference_results,columns=['Td'])
# pred_data = pd.DataFrame({'CONFIGURATIONS':config_names, 'total_energy':inference_results})
pred_data['smiles'] = config_names
print(pred_data)
pred_data.to_csv(dataset_name + '/Td_only_gnn_prediction_properties.csv')
# original_data = pd.read_csv('initial_data/results.csv')
# all_data = pd.merge(pred_data, original_data, on = 'CONFIGURATIONS')
# all_data.to_csv(dataset_name + '/all_properties.csv')
# print(mean_absolute_error(all_data.TOTAL, all_data.total_energy))
# print(r2_score(all_data.TOTAL, all_data.total_energy))