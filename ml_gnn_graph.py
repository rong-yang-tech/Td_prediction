
# import ase
# import ase.io
# from ase.build import molecule
# from dscribe.descriptors import SOAP,CoulombMatrix,ACSF
# from rdkit import Chem #导入rdkit模块用于生成分子构象
# import subprocess,os
import numpy as np
# import math
from ml_gnn_layer_nn import CCPGraph


# 生成rdkit所有描述符
# import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D,rdMolDescriptors,AllChem,RDConfig,FragmentCatalog,Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.EState import Fingerprinter


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem


"""Data and graphs."""
import os


import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem import rdmolops
from torch.utils.data import Dataset, SubsetRandomSampler
from torch_geometric.data import Dataset, Data, InMemoryDataset
import scipy.sparse as sp

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import os
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_add_pool, TopKPooling, GCNConv, GraphConv, GINConv

att_dtype = np.float32

PeriodicTable = Chem.GetPeriodicTable()
try:
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
except:
    fdefName = os.path.join('/RDKit file path**/RDKit/Data/',
                            'BaseFeatures.fdef')  # The 'RDKit file path**' is the installation path of RDKit.
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


class Graph():
    def __init__(self, rdkit_mol):
        self.atom_type = ['H', 'C', 'N', 'O','S', 'F', 'Cl', 'Br', 'I']
        self.hybridization = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        self.bond_type = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        # self.smiles = molecule_smiles
        self.atom_pair = self.get_atom_pair_type()
        # print(self.atom_pair)
        self.mol = rdkit_mol
        if self.mol is None:
            self.mol = None
            return

        # Add hydrogens to molecule

        # self.mol = Chem.AddHs(self.mol)
        if self.mol is not None:
            self.smiles_to_graph()

    def get_atom_pair_type(self):

        from itertools import combinations_with_replacement

        # elements = ["C", "H", "N", "O", "F", "Cl", "Br", "I"]
        pair_types = ["-".join(sorted(pair)) for pair in combinations_with_replacement(self.atom_type, 2)]
        # print(len(pair_types))
        # 3. 创建独热编码映射表
        pair_to_index = {pair: i for i, pair in enumerate(pair_types)}
        # num_pairs = len(pair_types)
        ##one_hot_vector = np.zeros(num_pairs)
        # index = pair_to_index.get("-".join(sorted(atom_pair)), None)
        # if index is not None:
        # one_hot_vector[index] = 1
        return list(pair_to_index.keys())

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            print(x, type(x))
            raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def donor_acceptor(self, rd_mol):
        is_donor = defaultdict(int)
        is_acceptor = defaultdict(int)
        feats = factory.GetFeaturesForMol(rd_mol)
        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                for u in feats[i].GetAtomIds():
                    is_donor[u] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                for u in feats[i].GetAtomIds():
                    is_acceptor[u] = 1
        return is_donor, is_acceptor

    def AtomAttributes(self, rd_atom, is_donor, is_acceptor, extra_attributes=[]):

        rd_idx = rd_atom.GetIdx()
        # Inititalize
        attributes = []
        # Add atimic number
        attributes += self.one_of_k_encoding(rd_atom.GetSymbol(), self.atom_type)
        # Add heavy neighbor count
        attributes += self.one_of_k_encoding(len(rd_atom.GetNeighbors()), [0, 1, 2, 3, 4, 5, 6])
        # Add neighbor hydrogen count
        attributes += self.one_of_k_encoding(rd_atom.GetTotalNumHs(includeNeighbors=True), [0, 1, 2, 3, 4])
        # Add hybridization type
        attributes += self.one_of_k_encoding(rd_atom.GetHybridization().__str__(), self.hybridization)
        # Add boolean if chiral
        attributes += self.one_of_k_encoding(int(rd_atom.GetChiralTag()), [0, 1, 2, 3])
        # Add boolean if in ring
        attributes.append(rd_atom.IsInRing())
        # Add boolean if aromatic atom
        attributes.append(rd_atom.GetIsAromatic())
        # Add boolean if donor
        attributes.append(is_donor[rd_idx])
        # Add boolean if acceptor
        attributes.append(is_acceptor[rd_idx])

        attributes += extra_attributes
        return np.array(attributes, dtype=att_dtype)

    def atom_featurizer(self, rd_mol):

        is_donor, is_acceptor = self.donor_acceptor(rd_mol)

        #### add atoms descriptors####
        V = []
        for k, atom in enumerate(rd_mol.GetAtoms()):
            all_atom_attr = self.AtomAttributes(atom, is_donor, is_acceptor)
            V.append(all_atom_attr)
        return np.array(V, dtype=att_dtype)

    def bond_features(self, bond):
        """
        提取键的特征。

        参数：
        - bond: RDKit 键对象。

        返回：
        - features: 键的特征列表。
        """
        bt = bond.GetBondType()
        # print(bt)
        is_conjugated = bond.GetIsConjugated()
        is_in_ring = bond.IsInRing()
        bond_dict = {
            Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0],
            Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0],
            Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0],
            Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1],
        }
        bond_feature = bond_dict.get(bt, [0, 0, 0, 0])
        # print(bond_feature)
        # 添加额外的特征
        features = bond_feature + [
            1 if is_conjugated else 0,
            1 if is_in_ring else 0
        ]

        return features

    def bond_featurizer(self, mol):
        """
        将 RDKit 分子对象转换为 edge_index。

        参数：
        - mol: RDKit 分子对象。

        返回：
        - edge_index: 形状为 [2, num_edges] 的 torch.LongTensor。
        - edge_attr: 边的特征，形状为 [num_edges, num_edge_features]（可选）。
        """
        # 获取分子中的键
        # num_atoms = mol.GetNumAtoms()
        # AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        # edges = []
        edge_attrs = []
        # weights = []
        # bond_mat = molDG.GetMoleculeBoundsMatrix(mol)
        # print(bond_mat)
        for a in mol.GetAtoms():  ##全部原子
            i = a.GetIdx()  ##原子索引
            # print(atom.GetSymbol())  ##
            for neighbor in a.GetNeighbors():
                m = neighbor.GetIdx()  ##原子索引
                # print(neighbor.GetSymbol())  ##具体对应的原子
                bond_atoms = self.one_of_k_encoding("-".join(sorted([a.GetSymbol(),
                                                                     neighbor.GetSymbol()])), self.atom_pair)
                bond = self.mol.GetBondBetweenAtoms(i, m)  ##原子之间键
                bond_mat = molDG.GetMoleculeBoundsMatrix(mol)
                bond_length = bond_mat[i][m]
                s = self.bond_features(bond)
                # print(s)
                edge_attrs.append(bond_atoms + s + [bond_length])

        # for bond in mol.GetBonds():
        #     start = bond.GetBeginAtomIdx()
        #     end = bond.GetEndAtomIdx()
        #     #bond_type = bond.GetBondType()

        #     # 将键表示为双向边
        #     edges.append((start, end))
        #     edges.append((end, start))

        #     # 获取键的特征（例如键类型）
        #     bond_atoms = self.get_one_hot_encoding([mol.GetAtomWithIdx(start).GetSymbol(),
        #                                mol.GetAtomWithIdx(end).GetSymbol()]).tolist()

        #     bond_length =  bond_mat[start][end]
        #     weights.append(bond_length)

        #     bond_feature = self.bond_features(bond)
        #     edge_attrs.append(bond_feature + bond_atoms + [bond_length])
        # 转换为 torch.Tensor
        # edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        # edge_weight = torch.tensor(weights, dtype=torch.float)

        return edge_attr  # edge_index, edge_attr, edge_weight

    def smiles_to_graph(self):
        """
        Converts smiles to a graph.
        """
        # 节点特征1--原子信息
        # atom_types= ['H','C','N','O','F','Cl','Br','I']
        atoms = self.mol.GetAtoms()

        # 获取原子
        atoms_list = []
        for i in atoms:
            atom = i.GetSymbol()  ##具体对应的原子
            atoms_list.append(atom)  # 原子列表
        # print(atoms_list)

        # 文献node
        node_mat = self.atom_featurizer(self.mol)
        # print(node_mat)

        edge_mat = self.bond_featurizer(self.mol)

        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol)
        # print(adj_mat.shape)
        self.std_adj_mat = np.copy(adj_mat)

        # Create distance matrix
        dist_mat = molDG.GetMoleculeBoundsMatrix(self.mol)
        dist_mat[dist_mat == 0.0] = 1

        # Get modified adjacency matrix
        adj_mat = adj_mat * (1 / dist_mat)

        # Pad the adjacency matrix
        dim_add = len(atoms_list) - adj_mat.shape[0]
        adj_mat = np.pad(
            adj_mat, pad_width=((0, dim_add), (0, dim_add)), mode="constant"
        )

        # Add an identity matrix to adjacency matrix
        # This will make an atom its own neighbor
        adj_mat = adj_mat + np.eye(len(atoms_list))
        # print(adj_mat)

        # # 稀疏矩阵
        # 去除自身的稀疏矩阵        adj_mat[~np.eye(adj_mat.shape[0], dtype=bool)].reshape(adj_mat.shape[0], -1)
        np.fill_diagonal(adj_mat, 0)
        edge_index = sp.coo_matrix(adj_mat)

        values = edge_index.data  # 边上对应权重值weight
        indices = np.vstack((edge_index.row, edge_index.col))  # 我们真正需要的coo形式
        edge_index = torch.tensor(indices)  # 我们真正需要的coo形式

        i = torch.tensor(indices)  # 转tensor
        v = torch.tensor(values)  # 转tensor
        edge_index = torch.sparse_coo_tensor(i, v, edge_index.shape)
        edge_index, edge_weight = edge_index._indices(), edge_index._values()
        # 带值的
        # edge_index = edge_index.to_dense()
        # edge_index=edge_index.add(edge_index).to_dense()
        # print(edge_index)
        # edge_no1, edge_mat, edge_no2 = self.molecule_to_edge_index_attr_weight(self.mol)
        # print(edge_attr.shape)
        # print(edge_weight)
        # Save both matrices
        self.node_mat = node_mat
        self.edge_mat = edge_mat
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        # print(self.adj_mat)
        # print(self.node_mat)
# if __name__ == '__main__':
#
#     mol = Chem.MolFromSmiles('CC')
#     mol = Chem.AddHs(mol)
#     Graph(mol)




class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, mode, transform=None, pre_transform=None,
                 pre_filter=None):
        self.mode = mode

        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root

        # print(self.processed_paths[0])
        self.load(self.processed_paths[0])
        # self.atom_pair = self.get_atom_pair_type()

    @property
    def raw_file_names(self):
        # 返回原始文件的文件名列表，如果没有原始文件则返回空列表
        return []

    @property
    def processed_file_names(self):
        # 返回预处理文件的文件名列表，如果没有预处理文件则返回空列表
        return ['data_bde.pt']

    # @property
    # def soap_truancate(self):
    #     # 返回预处理文件的文件名列表，如果没有预处理文件则返回空列表
    #     return self.soap_truancate

    def download(self):
        # 如果需要，可以在此处下载原始数据
        pass

    def GetRdkitDescriptors(self,smile):

        def get_bde(csv):
            data_bde=pd.read_csv(csv)
            for i, smi in enumerate(data_bde['can_smiles']):

                if smi==smile:
                    bde=data_bde['bde_gnn'][i]
                    return bde

        def Estate_calculator(mol):
            exestate1, exestate2 = Fingerprinter.FingerprintMol(mol)
            estate1 = np.squeeze(exestate1)
            estate2 = np.squeeze(exestate2)

            s1 = slice(6, 38, 1)
            s2 = slice(53, 54, 1)
            s3 = slice(69, 70, 1)
            s4 = slice(74, 75, 1)
            selected_estate_1 = np.append(np.append(np.append(estate1[s1], estate1[s2]), estate1[s3]),
                                          estate1[s4])
            selected_estate_2 = np.append(np.append(np.append(estate2[s1], estate2[s2]), estate2[s3]),
                                          estate2[s4])
            return np.append(selected_estate_1, selected_estate_2)

        def count_functional_groups(mol):

            smarts_patterns1 = {
                'N-NO2': 'N(N(=O)=O)',
                'O-NO2': 'O(N(=O)=O)',
                'C(NO2)3': 'C(N(=O)=O)(N(=O)=O)(N(=O)=O)',
                'C(NO2)2': 'C(N(=O)=O)(N(=O)=O)',
                'C(NO2)': 'C(N(=O)=O)',
            }

            # mol= Chem.MolFromSmiles(smiles)
            # mol = Chem.AddHs(mol)
            # if mol is None:
            #     return None  # 无效的SMILES

            counts = {}
            for name, pattern in smarts_patterns1.items():
                substruct = Chem.MolFromSmiles(pattern)
                if substruct is None:
                    counts[name] = 0
                    continue
                matches = mol.GetSubstructMatches(substruct)
                counts[name] = len(matches)

            C_2NO2 = counts['C(NO2)2'] - 3 * counts['C(NO2)3']
            counts['new_C(NO2)2'] = C_2NO2
            if (counts['C(NO2)3'] != 0) or (counts['C(NO2)2'] != 0):
                C_NO2 = counts['C(NO2)'] - 3 * counts['C(NO2)3'] - 2 * counts['new_C(NO2)2']
                counts['new_C1(NO2)'] = C_NO2
                return counts['N-NO2'], counts['O-NO2'], counts['C(NO2)3'], counts['new_C(NO2)2'], counts[
                    'new_C1(NO2)']
            else:
                C_NO2 = counts['C(NO2)']
                counts['new_C1(NO2)'] = C_NO2
                return counts['N-NO2'], counts['O-NO2'], counts['C(NO2)3'], counts['new_C(NO2)2'], counts[
                    'new_C1(NO2)']

        # smiles = 'O=[N+]([O-])C(=C1NCN(CN2CNC(=C([N+](=O)[O-])[N+](=O)[O-])NC2)CN1)[N+](=O)[O-]'
        # m =Chem.MolFromSmiles(smiles)
        # print(Estate_calculator(m))
        mol = Chem.MolFromSmiles(smile, sanitize=True)
        des_list = ['NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                    'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
                    'NumAromaticRings',
                    'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'RingCount', 'NumRotatableBonds',
                    'fr_aryl_methyl', 'fr_NH1', 'fr_NH2', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
                    'fr_nitro_arom_nonortho', 'MolWt', 'MaxPartialCharge', 'MinPartialCharge',
                    'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'TPSA',

                    ]
        desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
        descriptors0 = desc_calc.CalcDescriptors(mol)

        # #descriptors0.columns = des_list
        # #descriptors0.index = smile
        descriptors =list(descriptors0)

        e_state = Estate_calculator(mol)
        # e_state.index = smile

        descriptors.extend(list(e_state))
        # print(descriptors)

        # 统计官能团
        # smile = Chem.MolToSmiles(mol)
        no2 = count_functional_groups(mol)
        # print(no2)
        descriptors.append(no2[0])
        descriptors.append(no2[1])
        descriptors.append(no2[2])
        descriptors.append(no2[3])
        descriptors.append(no2[4])



        mol = Chem.AddHs(mol)
        # 优化3d坐标
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)

        npr1 = rdMolDescriptors.CalcNPR1(mol)
        npr2 = rdMolDescriptors.CalcNPR2(mol)
        pmi3 = rdMolDescriptors.CalcPMI3(mol)
        pbf = rdMolDescriptors.CalcPBF(mol)
        descriptors.append(npr1)
        descriptors.append(npr2)
        descriptors.append(pmi3)
        descriptors.append(pbf)

        mwg = Descriptors.MolWt(mol)
        nH = 0
        nC = 0
        nO = 0
        nN = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                nH = nH + 1
            elif atom.GetAtomicNum() == 6:
                nC = nC + 1
            elif atom.GetAtomicNum() == 7:
                nN = nN + 1
            elif atom.GetAtomicNum() == 8:
                nO = nO + 1

        # 氮含量、氧平衡
        per_n = nN * 14 / mwg
        ban_o = (nO - nC * 2 - nH * 0.5) * 16 / mwg
        descriptors.append(per_n)
        descriptors.append(ban_o)
        descriptors.append(nC)
        descriptors.append(nH)
        descriptors.append(nO)
        descriptors.append(nN)

        #

        bde=get_bde("./ml_data13.csv")
        # bde=get_bde("./smi_new_allrings3.csv")
        descriptors.append(bde)
        #
        # print(descriptors)
        return descriptors


    def process(self):
        # 在此处进行数据的预处理，并保存到self.processed_dir目录下

        # df = pd.read_csv('tesall_data_rdx.csv')

        if self.mode == 'train':


            df = pd.read_csv('./ml_data13.csv')

            # y = df["tot"].to_list()
            # x = []
            # edge_feats = []
            # adj_mat = []
            data_list = []
            out_list = []
            wrong_mol = []
            not_even_mol = []

            for index, row in df.iterrows():
                # 该行的索引
                # 获取该行的某个字段值
                smi = row['can_smiles']
                # print(smi)
                u = self.GetRdkitDescriptors(smi)
                u = torch.tensor(u, dtype=torch.float)

                mol = Chem.MolFromSmiles(smi)
                mol = Graph(mol)
                y = row["Td"]
                y = torch.tensor(y, dtype=torch.float)

                node_mat_tot = torch.tensor(mol.node_mat, dtype=torch.float)
                edge_mat_tot = torch.tensor(mol.edge_mat, dtype=torch.float)
                edge_index = torch.tensor(mol.edge_index, dtype=torch.int64)
                edge_weight = torch.tensor(mol.edge_weight, dtype=torch.int64)


                data = Data(x=node_mat_tot, edge_attr=edge_mat_tot,
                            edge_index=edge_index, edge_weight=edge_weight,
                            y=y, u=u, name=smi)  # 根据需要创建图数据

                data_list.append(data)

            self.save(data_list, self.processed_paths[0])

        elif self.mode == 'prediction':

            # df = pd.read_csv('D:/smi_xyz/smil_to_xyz/all_smi_temp.csv')
            df = pd.read_csv("./smi_new_allrings3.csv")

            # y = df["tot"].to_list()
            # x = []
            # edge_feats = []
            # adj_mat = []
            data_list = []
            out_list = []
            wrong_mol = []
            not_even_mol = []

            for index, row in df.iterrows():
                # 该行的索引
                # 获取该行的某个字段值
                smi = row['can_smiles']
                # print(smi)
                u = self.GetRdkitDescriptors(smi)
                u = torch.tensor(u, dtype=torch.float)

                mol = Chem.MolFromSmiles(smi)
                mol = Graph(mol)

                node_mat_tot = torch.tensor(mol.node_mat, dtype=torch.float)
                edge_mat_tot = torch.tensor(mol.edge_mat, dtype=torch.float)
                edge_index = torch.tensor(mol.edge_index, dtype=torch.int64)
                edge_weight = torch.tensor(mol.edge_weight, dtype=torch.int64)

                data = Data(x=node_mat_tot, edge_attr=edge_mat_tot,
                            edge_index=edge_index, edge_weight=edge_weight,
                            u=u, name=smi)  # 根据需要创建图数据

                data_list.append(data)

            self.save(data_list, self.processed_paths[0])

# if __name__ == "__main__":
#     # 假设molecules是分子列表，labels是标签
#     dataset = CustomGraphDataset(root='tm_gnn')
#     print(dataset[0])
#     prediction_dataset = CustomGraphDataset(root='bde_pre_gnn')
#     print(dataset[0])


# 定义GNN模型
# MPNN
# from torch_geometric.nn import MessagePassing, GraphNorm
# from torch_scatter import scatter
# import torch.nn.functional as F
# from torch import nn
# from torch_scatter import scatter_add
# from torch_geometric.utils import softmax

# class Conv(MessagePassing):
#     def __init__(self, in_channels, out_channels, edge_dim=52, aggr='sum'):
#         super().__init__(aggr=aggr)
#         self.aggr = aggr
#         self.lin_neg = nn.Linear(in_channels+ edge_dim, out_channels)
#         self.lin_root = nn.Linear(in_channels,out_channels)
#
#
#     def forward(self, x, edge_index, edge_attr):
#         #print('x shape:', edge_attr.shape)
#         x_adj = torch.cat([x[edge_index[0]], edge_attr], dim=1)
#        # print('x shape:',x.shape)
#         x_adj = F.tanh(self.lin_neg(x_adj))
#         #print('x_adj shape:',x_adj.shape)
#         #print('edge shape:',edge_index.shape)
#         #x_adj = torch.cat((x_adj, x_adj),0)
#         #edge_index_new = torch.cat((edge_index[0],edge_index[1]),0)
#         neg_sum = scatter(x_adj, edge_index[0], dim=0, reduce=self.aggr)
#         #unique_elements = np.unique(edge_index[1])
#         #neg_sum = x_adj.scatter_reduce_(dim=0, index=edge_index, src=x_adj, reduce='sum', include_self=False)
#         #print('neg_sum',neg_sum.shape)
#         x_out = F.tanh(self.lin_root(x))
#         #if x_out.shape[0] != neg_sum.shape[0] :
#         #print(x_adj.shape, edge_index.shape, len(unique_elements))
#         #print(x_out.shape, neg_sum.shape)
#         x_out = x_out + neg_sum
#         # x_out = self.bn1(x_out)
#         return x_out
#
#
# class GNN(torch.nn.Module):
#     def __init__(self,  hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4,hidden_dim5,dp_rate1,dp_rate2):
#         super(GNN, self).__init__()
#         self.conv1 = Conv(36, hidden_dim1)
#         self.conv2 = Conv(hidden_dim1, hidden_dim2)
#         self.conv3 = Conv(hidden_dim2, hidden_dim3)
#         self.fc = torch.nn.Sequential(
#             torch.nn.Linear(hidden_dim3+112, hidden_dim4),
#             torch.nn.Dropout(dp_rate1),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim4, hidden_dim5) ,
#             torch.nn.Dropout(dp_rate2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim5, 1),
#             # 输出层，回归任务
#         )
#
#     def forward(self, data):
#         x, edge_index,edge_attr, u, batch = data.x, data.edge_index, data.edge_attr,data.u,data.batch
#         x = self.conv1(x, edge_index,edge_attr)
#         x = self.conv2(x, edge_index,edge_attr)
#         x = self.conv3(x, edge_index,edge_attr)
#
#         x = global_mean_pool(x, batch)
#         # print(x.shape)
#         size = data.batch[-1].item() + 1
#         x = torch.cat([x,u.view(size,-1)],dim=1)
#
#   # 全局池化
#         return self.fc(x).squeeze(-1)


# 分子转图数据工具函数
# class EarlyStopping:
#     def __init__(self, patience=100, delta=0):
#         self.patience = patience
#         self.delta = delta
#         self.counter = 0
#         self.best_loss = np.inf
#
#     def __call__(self, val_loss):
#         if val_loss < self.best_loss - self.delta:
#             self.best_loss = val_loss
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False


# 早停类
# from torch.utils.data import Subset, random_split
# def run_model(model,hidden_dim1, hidden_dim2,hidden_dim3,hidden_dim4, hidden_dim5,dp_rate1, dp_rate2):
#
#
#
#     # 训练流程
#     model =CCPGraph(hidden_dim1, hidden_dim2,hidden_dim3,hidden_dim4, hidden_dim5,dp_rate1, dp_rate2)
#     dataset = CustomGraphDataset(root='bde_gnn')
#     train_size = int(0.8 * len(dataset))  # 80% 用于训练
#     test_size = len(dataset) - train_size  # 20% 用于测试
#     print(dataset[0])
#
#     generator1 = torch.Generator().manual_seed(2024)
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator1)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
#     val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
#     # train_size = int(0.8 * len(dataset))
#     # train_dataset = dataset[:train_size]
#     # val_dataset = dataset[train_size:]
#     # train_loader = DataLoader(train_dataset, batch_size=32)
#     # val_loader = DataLoader(val_dataset, batch_size=32)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.9)
#     # criterion = torch.nn.MSELoss()
#     criterion = torch.nn.L1Loss()
#
#     best_loss = np.inf
#     count_unchange = 0
#     early_stop = 50
#
#
#     model.train()
#
#     for epoch in range(1000):
#         model.train()
#         loss_all = 0
#         for data in train_loader:
#             # print(data)
#             optimizer.zero_grad()
#             out = model(data)
#             loss = criterion(out, data.y)
#             loss.backward()
#             optimizer.step()
#             loss_all += loss.item() * data.num_graphs
#         train_loss = loss_all / len(train_loader.dataset)
#
#         model.eval()
#         loss_all = 0
#         with torch.no_grad():
#             for data in val_loader:
#                 out = model(data)
#                 loss = criterion(out, data.y)
#                 loss_all += loss.item() * data.num_graphs
#         val_loss= loss_all / len(val_loader.dataset)
#
#         # 验证
#         scheduler.step()
#         print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
#
#         if val_loss < best_loss:
#             best_loss = val_loss
#             torch.save(model.state_dict(), 'best_model_bde.pt')
#             count_unchange = 0
#
#         else:
#
#             count_unchange += 1
#             if count_unchange > early_stop:
#                 return best_loss
#
#                 break
#
#
#
#     return best_loss
#
#
#     # 评估函数
#
#
#
#
# # 主函数
# if __name__ == "__main__":
#     # # 假设molecules是分子列表，labels是标签
#     # dataset = CustomGraphDataset(root='tm_gnn')
#     # # print(dataset[0])
#     #
#     # # 划分训练集和验证集（80-20）
#     # train_size = int(0.8 * len(dataset))
#     # train_dataset = dataset[:train_size]
#     # val_dataset = dataset[train_size:]
#     #
#     # train_loader = DataLoader(train_dataset, batch_size=1)
#     # val_loader = DataLoader(val_dataset, batch_size=1)
#     #
#     # model = GNN(num_features=36)  # 根据实际特征维度调整
#     # train(model, train_loader, val_loader)
#
#     import optuna
#
#     def objective(trial):
#         hidden_dim1 = trial.suggest_int('hidden_dim_1', 16, 128)
#         hidden_dim2 = trial.suggest_int('hidden_dim_2', 16, 128)
#         hidden_dim3 = trial.suggest_int('hidden_dim_3', 16, 128)
#         hidden_dim4 = trial.suggest_int('hidden_dim_4', 100, 1000)
#         hidden_dim5 = trial.suggest_int('hidden_dim_5', 100, 1000)
#         dp_rate1 = trial.suggest_float('dp_rate1', 0.1, 0.5)
#         dp_rate2 = trial.suggest_float('dp_rate2', 0.1, 0.5)
#
#         #learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
#         print(hidden_dim1, hidden_dim2,hidden_dim3,
#               hidden_dim4, hidden_dim5,
#                           dp_rate1, dp_rate2)
#
#         score = run_model(GNN, hidden_dim1,hidden_dim2,hidden_dim3, hidden_dim4, hidden_dim5,
#                           dp_rate1, dp_rate2)
#         return score
#
#     study = optuna.create_study(
#     study_name="bde_gnn_optimization",  # Name the study
#     storage="sqlite:///bde_gnn_study.db",  # SQLite database URL
#     direction="minimize",
#     load_if_exists=True  # Load the existing study if it exists
#     )
#     study.optimize(objective, n_trials=1000)
#     print(f"Best value: {study.best_value} (params: {study.best_params})")
#
#
#     # run_model(GNN, 123,50,119, 145, 200,
#     #                           0.1, 0.14862447718340935)
#
#
#     # 预测BDE
#     dataset = CustomGraphDataset(root='bde_pre_gnn',)
#     loader = DataLoader(dataset, batch_size=32, shuffle=False)
#     model = GNN(97,98, 120,650,320,0.1,0.3)  # all
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.load_state_dict(
#         torch.load('/best_model_bde.pt', weights_only=True, map_location=torch.device('cpu')))
#     # print(model)
#     # model.eval()
#     # model.to(device)
#     # config_names = []
#     # inference_results = []
#     #
#     # with torch.no_grad():
#     #     for batch_idx, data_batch in enumerate(loader):
#     #         data = data_batch.to(device)
#     #         print(data.name)
#     #         config_names.extend(data.name)
#     #         outputs, att = model(data)
#     #
#     #         inference_results.extend(outputs.detach().cpu().numpy())
#     #
#     #
#     # print(config_names, inference_results)
#     # pred_data = pd.DataFrame(inference_results, columns=['BDE'])
#     # # pred_data = pd.DataFrame({'CONFIGURATIONS':config_names, 'total_energy':inference_results})
#     # pred_data['CONFIGURATIONS'] = config_names
#     # print(pred_data)
#     # pred_data.to_csv('bde_prediction_properties.csv')