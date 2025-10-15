"""Data and graphs."""
import os
import pandas as pd
import numpy as np
import torch

from rdkit import Chem
from torch.utils.data import Subset, random_split
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

from torch_geometric.loader import DataLoader
from ml_gnn_utils import DualOutput
import sys

from sklearn.metrics import mean_absolute_error,r2_score
from ml_gnn_graph import CustomGraphDataset
from ml_gnn_layer_nn import CCPGraph
#import math 
from sklearn.model_selection import KFold
# from ml_gnn_utils import DualOutput
att_dtype = np.float32

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

PeriodicTable = Chem.GetPeriodicTable()
try:
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
except:
    fdefName = os.path.join('/RDKit file path**/RDKit/Data/',
                            'BaseFeatures.fdef')  # The 'RDKit file path**' is the installation path of RDKit.
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def run_model(root,  hidden_dim_1, hidden_dim_2, hidden_dim_3,
             hidden_dim_4, hidden_dim_5, hidden_dim_6, hidden_dim_7,
             hidden_dim_8, dp_rate_1, dp_rate_2, dp_rate_3):

    
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    

    
    
    dataset = CustomGraphDataset(root=root,mode='train')
    #print(dataset[0].u.shape[0])
    
    train_size = int(0.8 * len(dataset))  # 80% 用于训练
    test_size = len(dataset) - train_size  # 20% 用于测试
    
    generator1 = torch.Generator().manual_seed(2025)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator1)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    print('total data number %d'%(len(dataset)))
    print('train data number %d'%(len(train_dataset)))
    print('test data number %d'%(len(test_dataset)))
    #dataset = dataset.shuffle()
    # data_size = 9396
    # print(dataset.num_node_features)
    for batch in test_loader:
        # for i in batch
        print(batch.name)


    u_dim = [dataset[0].u.shape[0]]
        
    print(u_dim)
    num_epochs = 1000
    k_split = 5
    kfold = KFold(n_splits=k_split, shuffle=True, random_state=2024)
    
    model = CCPGraph(u_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3,
                 hidden_dim_4, hidden_dim_5, hidden_dim_6, hidden_dim_7,
                 hidden_dim_8, dp_rate_1, dp_rate_2, dp_rate_3)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.9)
    criterion = torch.nn.MSELoss()
    
    
    loss = []
    #mae = []
    epoch = []
    #vals_mae =[]
    best_loss = np.inf
    count_unchange = 0
    early_stop = 50
    torch.cuda.empty_cache()
    for epoch in range(num_epochs):
        print('\n')
        fold_val_losses = []
        fold_val_maes = []
        fold_val_r2s = []
        
        fold_train_losses = []
        fold_train_maes = []
        fold_train_r2s = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            print('*****epoch %d*****' % (epoch+1), f'Fold {fold + 1}/{k_split}')
            print('-' * 30)
            # print(len(train_idx))
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=False)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        
            
                
           # 模型训练
            model.train()
            train_losses = []
            train_targets = []
            train_predictions = []
            for batch_idx, data_batch in enumerate(train_loader):
                # 前向传播
                data = data_batch.to(device)
                outputs, embedding2 = model(data)
                label = data.y.view(outputs.shape)
                loss = criterion(outputs, label)
                train_losses.append(loss.item())
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_targets.extend(label.cpu().numpy())
                train_predictions.extend(outputs.detach().cpu().numpy())
            
            # 模型验证
            model.eval()
            val_losses = []
            val_targets = []
            val_predictions = []
            with torch.no_grad():
                for batch_idx, data_batch in enumerate(val_loader):
                    data = data_batch.to(device)
                    outputs,embedding2 = model(data)
                    label = data.y.view(outputs.shape)
                    loss = criterion(outputs, label)
                    val_losses.append(loss.item())
                    
                    val_targets.extend(label.cpu().numpy())
                    val_predictions.extend(outputs.detach().cpu().numpy())
            
            # 计算指标
            avg_train_loss = np.mean(train_losses)
            train_mae = mean_absolute_error(train_targets, train_predictions)
            train_r2 = r2_score(train_targets, train_predictions)
            
            avg_val_loss = np.mean(val_losses)
            val_mae = mean_absolute_error(val_targets, val_predictions)
            val_r2 = r2_score(val_targets, val_predictions)
            
            fold_train_losses.append(avg_train_loss)
            fold_train_maes.append(train_mae)
            fold_train_r2s.append(train_r2)
            
            fold_val_losses.append(avg_val_loss)
            fold_val_maes.append(val_mae)
            fold_val_r2s.append(val_r2)
            
            # print(f'Validation Loss for fold {fold + 1}: {avg_val_loss:.4f}')
            # print(f'Validation MAE for fold {fold + 1}: {mae:.4f}')
            # print(f'Validation R2 for fold {fold + 1}: {r2:.4f}')
            # print('-' * 20)
        
        # 计算当前 epoch 的平均指标
        epoch_avg_train_loss = np.mean(fold_train_losses)
        epoch_std_train_loss = np.std(fold_train_losses)
        epoch_avg_train_mae = np.mean(fold_train_maes)
        epoch_std_train_mae = np.std(fold_train_maes)
        epoch_avg_train_r2 = np.mean(fold_train_r2s)
        epoch_std_train_r2 = np.std(fold_train_r2s)
        
        epoch_avg_val_loss = np.mean(fold_val_losses)
        epoch_std_val_loss = np.std(fold_val_losses)
        epoch_avg_val_mae = np.mean(fold_val_maes)
        epoch_std_val_mae = np.std(fold_val_maes)
        epoch_avg_val_r2 = np.mean(fold_val_r2s)
        epoch_std_val_r2 = np.std(fold_val_r2s)
        
        
        #with open('print_results.txt', 'a+') as f:
        print(f'Average Train Loss for epoch {epoch + 1}: {epoch_avg_train_loss:.4f}±{epoch_std_train_loss:.4f}')
        print(f'Average Train MAE for epoch {epoch + 1}: {epoch_avg_train_mae:.4f}±{epoch_std_train_mae:.4f}')
        print(f'Average Train R2 for epoch {epoch + 1}: {epoch_avg_train_r2:.4f}±{epoch_std_train_r2:.4f}')
        print(f'Average Validation Loss for epoch {epoch + 1}: {epoch_avg_val_loss:.4f}±{epoch_std_val_loss:.4f}')
        print(f'Average Validation MAE for epoch {epoch + 1}: {epoch_avg_val_mae:.4f}±{epoch_std_val_mae:.4f}')
        print(f'Average Validation R2 for epoch {epoch + 1}: {epoch_avg_val_r2:.4f}±{epoch_std_val_r2:.4f}')
        print(f'Unchanged epoch: {count_unchange}')
        print('-' * 30)
        
        scheduler.step()
        
        if epoch_avg_val_loss < best_loss:
            best_loss = epoch_avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            test_losses = []
            test_targets = []
            test_predictions = []
            predict_label = []

            model.eval()
            with torch.no_grad():
                for batch_idx, data_batch in enumerate(test_loader):
                    data = data_batch.to(device)
                    outputs, embedding2 = model(data)
                    label = data.y.view(outputs.shape)
                    loss = criterion(outputs, label)
                    test_losses.append(loss.item())
                    test_targets.extend(label.cpu().numpy())
                    test_predictions.extend(outputs.detach().cpu().numpy())

                    # predict value and true value
                    array = torch.cat((outputs, label), dim=1).cpu().numpy()
                    predict_label.extend(array)
            # predict_label = np.stack(predict_label, axis=0)
            # pd.DataFrame(predict_label).to_csv("predict_label_bde_gnn.csv")
            # print(predict_label)

            avg_test_loss = np.mean(test_losses)
            test_mae = mean_absolute_error(test_targets, test_predictions)
            test_r2 = r2_score(test_targets, test_predictions)

            print(f'Average Test Loss for epoch {epoch + 1}: {avg_test_loss:.4f}')
            print(f'Average Test MAE for epoch {epoch + 1}: {test_mae:.4f}')
            print(f'Average Test R2 for epoch {epoch + 1}: {test_r2:.4f}')
            print(f'Unchanged epoch: {count_unchange}')
            print('-' * 30)
            

            
            count_unchange = 0
            
        else:
            
            count_unchange += 1
            if count_unchange > early_stop:
                return best_loss


                break
            
    return best_loss 
   
if __name__ == '__main__':


    
    # if (os.path.exists('log_model_Td_pre_gnnnobde.txt')) :
    # 	os.remove('log_model_Td_pre_gnnnobde.txt')
    dual_output = DualOutput('log_model_train_test.txt')
    sys.stdout = dual_output
    # run_model('Td_gnn_bde_MLP', 144, 137, 217, 126,  665,  1526,
    # 1317, 586, 0.19013568030892725, 0.18306424528822834,  0.11958168442951482229)
    run_model('Td_gnn4',
              # 255, 17,  44, 1724,  1471,  1885,  1975, 322,  0.10074925921752349, 0.10068535608528603, 0.14394447821081802)
    144,  137,  217,  126, 665,  1526,  1317,586, 0.19013568030892725,  0.18306424528822834,  0.11958168442951482)
    # 227, 256,  198,  481,  287,
    #  1771,  1705,  845,  0.1094705632310795,
    #   0.10793048512795315,  0.19210173291469798)
    sys.stdout = dual_output.console
    dual_output.close()
    # import optuna
    # def objective(trial):
    #     hidden_dim_1 = trial.suggest_int('hidden_dim_1', 16, 256)
    #     hidden_dim_2 = trial.suggest_int('hidden_dim_2', 16, 256)
    #     hidden_dim_3 = trial.suggest_int('hidden_dim_3', 16, 256)
    #     hidden_dim_4 = trial.suggest_int('hidden_dim_4', 100, 2000)
    #     hidden_dim_5 = trial.suggest_int('hidden_dim_5', 100, 2000)
    #     hidden_dim_6 = trial.suggest_int('hidden_dim_6', 100, 2000)
    #     hidden_dim_7 = trial.suggest_int('hidden_dim_7', 100, 2000)
    #     hidden_dim_8 = trial.suggest_int('hidden_dim_8', 100, 2000)
    #     dp_rate_1 = trial.suggest_float('dp_rate_1', 0.1, 0.5)
    #     dp_rate_2 = trial.suggest_float('dp_rate_2', 0.1, 0.5)
    #     dp_rate_3 = trial.suggest_float('dp_rate_3', 0.1, 0.5)
    #
    #     #learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    #     print(hidden_dim_1, hidden_dim_2,
    #                       hidden_dim_3, hidden_dim_4, hidden_dim_5,
    #                       hidden_dim_6, hidden_dim_7, hidden_dim_8,
    #                       dp_rate_1, dp_rate_2, dp_rate_3)
    #
    #     score = run_model('Td_gnn',  hidden_dim_1,
    #                       hidden_dim_2,hidden_dim_3, hidden_dim_4, hidden_dim_5,
    #                       hidden_dim_6, hidden_dim_7, hidden_dim_8,
    #                       dp_rate_1, dp_rate_2, dp_rate_3)
    #     return score
    #
    # study = optuna.create_study(
    # study_name="ml_gnn_Td_predict",  # Name the study
    # storage="sqlite:///ml_gnn_Td_predict.db",  # SQLite database URL
    # direction="minimize",
    # load_if_exists=True  # Load the existing study if it exists
    # )
    # study.optimize(objective, n_trials=1000)
    # print(f"Best value: {study.best_value} (params: {study.best_params})")
    # # sys.stdout = dual_output.console
    # # dual_output.close()


