import numpy as np  # numpy库
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法

from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor,StackingRegressor  # 集成算法
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score,train_test_split, GridSearchCV # 交叉检验，
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score,make_scorer ,mean_absolute_percentage_error # 批量导入指标算法
# from sklearn.manifold import TSNE
import pandas as pd  # 导入pandas
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt  # 导入图形展示库
# import seaborn as sns
# data = pd.read_csv("C:\\Users\\Yangxiurong\\Desktop\\0209d.csv")


# 数据准备
# rdk = pd.read_csv("E:/data_supp/exist/rdk_descriptor_ss.csv")
# dis = pd.read_csv("E:/data_supp/exist/distance_newhh.csv")
# dis = dis.drop(['Unnamed: 0'],axis=1)
# # # 部分能量数据还未获得
# energy = pd.read_csv("E:/data_supp/exist/energy_num.csv")
# # estate = pd.read_csv("E:/data_supp/exist/estate_desc.csv")
# # name = energy.iloc[:,1].str.split('.',expand=True)
# # name = name.drop([1,2],axis=1)
# # energy = energy.drop(['name'],axis=1)
# # energy['name'] = name
# # energy = energy.drop( energy.iloc[:,[0]],axis=1)
# # print(energy)# energy = energy.drop( energy.iloc[:,[0]],axis=1)
# # data = pd.merge(energy,data0, how='outer', on='name')
# # data = data.drop(['Unnamed: 0'],axis=1)
# # data = data.dropna(axis=0)
#
# # 拼加CM特征
# # h = pd.read_csv("E:\data_supp\exist\hcal.csv")
# soap = pd.read_csv("E:\data_supp\exist\soap0_all_mu.csv")
# data = pd.merge(energy,dis, how='inner', on='name')
# # print(data.shape)
# # data = pd.concat([energy,cm],axis = 1)
# # data = pd.concat([data,rdk],axis = 1)
# data = pd.concat([data,soap],axis = 1)
# # data = pd.concat([data,h],axis = 1)
# data = data.drop(['Unnamed: 0'],axis=1)
# data.to_csv('a_allhh.csv')
# print('finish')
# data = data.dropna(axis=0,how='any')

# # data1 = pd.read_csv("E:/Downloads/tes_data/8.0/results/each_configs_8.0.csv")
# # data2 = pd.read_csv("E:/Downloads/tes_data/7.0/results/each_configs_7.0.csv")
# # data3 = pd.read_csv("E:/Downloads/tes_data/6.0_NEW/results/each_configs_6.0.csv")
# # data1 = pd.concat([data1, data2], axis=0)
# # data2 = pd.concat([data1, data3], axis=0)
# # data2.to_csv('tesall_data.csv')
# data=pd.read_csv("E:/Downloads/tes_data/6.0_NEW/soap_all_mu.csv")



# 删除相互作用能弱的值-10，
# data = data.drop(data[data['tot']>0].index)
# print(data.shape)
# data = pd.read_csv('a_all_del.csv')

# 定义rmse函数
def RMSE(y,y_pred):
    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(y,y_pred,squared=False)
    return rmse
# 定义r2_score
def r2_score(y, y_pred):
    from sklearn.metrics import r2_score
    r2_score = r2_score(y, y_pred)
    return r2_score
def MAE(y, y_pred):
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y, y_pred)
    return mae
# 定义随机交叉验证
# scoring RMSE:neg_root_mean_squared_error或,'neg_mean_absolute_error'
    # std = StandardScaler()
    # pipeline = Pipeline([('transformer', std), ('estimator', model)])

def cross_val(model, x, y):
    from sklearn.model_selection import cross_val_score, ShuffleSplit,StratifiedKFold,LeaveOneOut
    scores_model = cross_val_score(estimator= model,
                                   X=x,
                                   y=y,
                                   cv=ShuffleSplit(n_splits=100, test_size=0.2,random_state=42),#LeaveOneOut(),  10,
                                   scoring='neg_mean_absolute_error',# 'r2',
                                   n_jobs=-1) # ShuffleSplit打乱   数据标准化

    return np.mean(scores_model)

import shap

def ml(x,y,random,new_csv,corr):
# 特征工程
# 特征相关性分析
    dat = pd.concat([x,y],axis=1)
    # dat.to_csv('a1.csv')
    # print(dat)
    x_cor = dat.corr(method='pearson')
    # x_cor.to_csv('a1.csv')
    x_cor = x_cor.abs()
    print(x_cor[y.name])
    drop_col=x_cor[x_cor[y.name]<corr].index
    # drop_col2=x_cor[x_cor[y.name]>0.1].index
#
# #
# #
    print(drop_col)
    x = x.drop(drop_col, axis=1)
    print(x.shape)
# #     # x.to_csv('u.csv')
#     print('与y相关',x.shape)
# #     # print(x)


    corr_matrix = x.corr(method='pearson')
    highly_corr = {}
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                col = corr_matrix.columns[i]
                related_col = corr_matrix.columns[j]
                if col not in highly_corr:
                    highly_corr[col] = set()
                highly_corr[col].add(related_col)
    x = x.drop(highly_corr, axis=1)
    print('相关性', x.shape)
    print(highly_corr)

# 方差选择法
#     from sklearn.feature_selection import VarianceThreshold
#     vt = VarianceThreshold(threshold=0)
#     x=vt.fit_transform(x)
#     print('方差',x.shape)



    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=random)
    # data_pre=pd.read_csv('D:/smi_xyz/smil_to_xyz/rdk_des.csv')
    # x_pre=data_pre.drop(['can_smiles'],axis=1)
# 添加标准化
#     std = StandardScaler()
#     x_train = std.fit_transform(x_train)
#     x_test = std.transform(x_test)
# 归一化
    scaler = MinMaxScaler()


    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # x_pre= scaler.transform(x_pre)

    # print(x_train)


    scoring = {'r2': make_scorer(r2_score), 'variance': make_scorer(explained_variance_score),
               'mae': make_scorer(mean_absolute_error), 'mse': make_scorer(mean_squared_error),
               'mape': make_scorer(mean_absolute_percentage_error)}

# 异常值检
#箱线图
# 特征工程降维处理
# PCA降维
#     pca = PCA(n_components=15)
#     x_train = pca.fit_transform(x_train)
#     x_test = pca.transform(x_test)

    # x_all = scaler.fit_transform(x)
    # x_all = pca.fit_transform(x)


# #梯度提升会回归0.52
    from sklearn.ensemble import GradientBoostingRegressor

    param = {'learning_rate': [1.0,1e-1,1e-2,1e-3],
      'n_estimators': [300, 400,500,600, 700,800],
      'max_depth': [2,4,6,8,10],
       "random_state": [42]
             }
    # param = {'learning_rate': [0.08],
    #   'n_estimators': [750],
    #   'max_depth': [6],
    #    "random_state": [42]
    #          }


    gbr=GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), param_grid=param, cv=3
                     , n_jobs=-1)
    gbr.fit(x_train,y_train)
    print("params_gpr:", gbr.best_params_)
    print("scores_gbr:", gbr.best_score_)
    gbr = gbr.best_estimator_
    gbr.fit(x_train,y_train)
    y_pre_train_gbr=gbr.predict(x_train)
    y_pre_test_gbr=gbr.predict(x_test)
    print('r2(train)gbr:',r2_score(y_train,y_pre_train_gbr),MAE(y_train,y_pre_train_gbr))
    print('r2(test)gbr:',r2_score(y_test,y_pre_test_gbr),MAE(y_test,y_pre_test_gbr))
    # print('100 times r2_score_gbr:',cross_val(gbr,x_all,y))






# refit = 'r2',SCORING=scoring
# # SVR回归0.74

    # svr = GridSearchCV(SVR(kernel='sigmoid', gamma=0.05), scoring=None, cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3,1e4,1e5,],
    #                                                                        "gamma": np.logspace(-2, 2, 5),
    #                                                                    })
    # # # # 训练
    # svr.fit(x_train, y_train)
    # # print("params:", svr.best_params_)
    # svr = svr.best_estimator_
    # svr.fit(x_train, y_train)
    # y_pre_train_svr=svr.predict(x_train)
    # y_pre_test_svr = svr.predict(x_test)
    #
    # print('RMSE(test):',RMSE(y_test,y_pre_test_svr))
    # print('r2(train)svr:',r2_score(y_train,y_pre_train_svr),MAE(y_train,y_pre_train_svr))
    # print('r2(test)svr:',r2_score(y_test,y_pre_test_svr),MAE(y_test,y_pre_test_svr))
    # print('100 times r2_svr:',cross_val(svr,x,y))


# # KRR回归
    from sklearn.kernel_ridge import KernelRidge
    # feature_std = np.std(x_train, axis=0)
    # gamma_ref = 1 / (feature_std.max( ) ** 2)
    krr = GridSearchCV(KernelRidge(kernel='rbf'),scoring=None,
                      param_grid={'alpha': np.logspace(-3, 0, 10),  # 15个对数间隔值
                                    'gamma': np.logspace(-2,0,10)} ,  # 基于数据尺度的gamma,
                    #    param_grid={'alpha': [0.01],  # 15个对数间隔值
                    #
                    #                  'gamma': [0.03162]},  # 基于数据尺度的gamma
                    cv=3)
    krr.fit(x_train, y_train)
    print("params:", krr.best_params_)
    print("scores_krr:", krr.best_score_)
    krr = krr.best_estimator_
    krr.fit(x_train, y_train)
    y_pre_train_krr=krr.predict(x_train)
    y_pre_test_krr = krr.predict(x_test)
    print('r2(train)krr:',r2_score(y_train,y_pre_train_krr),MAE(y_train,y_pre_train_krr))
    print('r2(test)krr:',r2_score(y_test,y_pre_test_krr),MAE(y_test,y_pre_test_krr))


    # plt.figure(figsize=(12, 16))
    #
    # features1=data.columns.drop(['can_smiles','T'])
    # # print(features1)
    # explainer = shap.KernelExplainer(krr.predict, x_train)
    # shap_values = explainer(x_test)
    # # # 2.1 SHAP摘要图（特征重要性）
    # plt.subplot(1, 2, 1)
    # shap.summary_plot(shap_values, x_test, feature_names=features1, plot_type="bar", max_display=15,show=False)
    # plt.title("Feature importance (global)", fontsize=8)
    # # # 2.2 SHAP摘要图（特征影响分布）
    # plt.subplot(1, 2, 2)
    # shap.summary_plot(shap_values, x_test, feature_names=features1, max_display=15,show=False)
    # plt.title("feature impact", fontsize=12)
    # plt.show()





#最小平方回归linear regression
# from sklearn.linear_model import LinearRegression
# lr=LinearRegression()
# lr.fit(x_train,y_train)
# y_pre_train_lr=lr.predict(x_train)
# y_pre_test_lr=lr.predict(x_test)
# print('r2(train)lr:',r2_score(y_train,y_pre_train_lr))
# print('r2(test)lr:',r2_score(y_test,y_pre_test_lr))
# print('100 times r2_lr:',cross_val(lr,x,y))


# # ElasticNet回归0.5
# from sklearn.linear_model import ElasticNet
# etc = GridSearchCV(estimator=ElasticNet(),
#                param_grid={"alpha":[0.1,0.5,1],"l1_ratio":[0.1, 0.3, 0.5,0.7,1]},
#                cv=3)
# etc.fit(x_train, y_train)
# etc= etc.best_estimator_
# y_pre_train_etc=etc.predict(x_train)
# y_pre_test_etc=etc.predict(x_test)
# print('r2(train)etc:',r2_score(y_train,y_pre_train_etc))
# print('r2(test)etc:',r2_score(y_test,y_pre_test_etc))
# print('100 times r2_etc:',cross_val(etc,x,y))

    # Lasso回归
    # from sklearn.linear_model import LassoCV
    # lasso = LassoCV(cv =10)
    # lasso.fit(x_train, y_train)
    # y_pre_train_lasso=lasso.predict(x_train)
    # y_pre_test_lasso=lasso.predict(x_test)
    # print('r2(train)lasso:',r2_score(y_train,y_pre_train_lasso))
    # print('r2(test)lasso:',r2_score(y_test,y_pre_test_lasso))
    # print('100 times r2_lasso:',cross_val(lasso,x,y))

# 贝叶斯回归0.51
# from sklearn.linear_model import BayesianRidge
# br = BayesianRidge()
# br.fit(x_train, y_train)
# y_pre_train_br=br.predict(x_train)
# y_pre_test_br=br.predict(x_test)
# print('r2(train)br:',r2_score(y_train,y_pre_train_br))
# print('r2(test)br:',r2_score(y_test,y_pre_test_br))
# print('100 times r2_br:',cross_val(br,x,y))



#
# # 高斯过程回归0.42
#     import sys
#     import warnings
#     import os
#     if not sys.warnoptions:
#         warnings.simplefilter("ignore")
#         os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
#
#
#     # # 高斯过程回归0.42
#     from sklearn.gaussian_process import GaussianProcessRegressor
#     from sklearn.gaussian_process.kernels import ConstantKernel, RBF,Matern,RationalQuadratic,ExpSineSquared,DotProduct,WhiteKernel
#     #ConstantKernel(constant_value=1, constant_value_bounds=(1e-5, 1e5)) * RBF(length_scale=1, length_scale_bounds=(1e-5, 1e5)
#     # kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-05, 1e5),nu=1.5)
#     # *RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-05, 1e5), alpha_bounds=(1e-05, 1e5))
#     # kernel = ExpSineSquared(length_scale=1.0, periodicity=1.0, length_scale_bounds=(1e-05, 1e5), periodicity_bounds=(1e-05, 1e5))
#     # DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-05, 100000.0))+ WhiteKernel()
#
#     kernel = ConstantKernel(constant_value=2, constant_value_bounds=(1e-100, 1e5))* Matern(length_scale=1.0, length_scale_bounds=(1e-05, 1e30),nu=1.5)
#     param={'alpha':[10,0,1.0,1e-1,1e-2,1e-3,1e-4,1e-5]}
#     gs=GridSearchCV(estimator=GaussianProcessRegressor(kernel=kernel,
#                                                        normalize_y=True),
#                    param_grid=param,
#                    cv=3)
#     gs.fit(x_train,y_train)
#     #
#     GPR=gs.best_estimator_
#     GPR.fit(x_train,y_train)
#     y_pre_train_gpr=GPR.predict(x_train)
#     y_pre_test_gpr=GPR.predict(x_test)
#     print('r2(train)gpr:',r2_score(y_train,y_pre_train_gpr))
#     print('r2(test)gpr:',r2_score(y_test,y_pre_test_gpr))
#     print('100 times R2_gpr:',cross_val(GPR,x_all,y))

    # # 特征重要性PI
    # plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # plt.rcParams['font.family'] = ['serif']
    # import eli5
    # from eli5.sklearn import PermutationImportance
    # feature_names = ['Atomic Number','Valence Electron', 'Group', 'Atomic Radius', 'Electron Negativity',
    #                  'Ionization Energy', 'Work Function','d band', 'Surface Energy',
    #                  'HOMO','LUMO','Dipole Moment', 'C', 'H', 'N']
    # perm = PermutationImportance(GPR, random_state=42).fit(x_test,y_test)
    # eli5.show_weights(perm, feature_names = feature_names)




 # # # # 随机森林回归0.59
    from sklearn.ensemble import RandomForestRegressor
    rfr=GridSearchCV(estimator=RandomForestRegressor(random_state=42 ),
               # param_grid={'n_estimators': [300],
               #             'max_depth': [4,  6],  # 深度严格限制
               #             'min_samples_split': [8,],
               #             'min_samples_leaf': [4, ],
               #             'max_features': [0.5, ],
               #             'max_samples': [0.7]},
                     param_grid={'n_estimators': [300,400,450,500,600],
                                 'min_samples_leaf': [2,4,6,8],},
               cv=3,scoring='neg_mean_absolute_error')
    rfr.fit(x_train,y_train)
    print("params_rfr:", rfr.best_params_)
    print("scores_rfr:", rfr.best_score_)

    rfr = rfr.best_estimator_
    rfr.fit(x_train,y_train)
    y_pre_train_rfr=rfr.predict(x_train)
    y_pre_test_rfr=rfr.predict(x_test)
    # 预测
    # y_PRE_BDE=rfr.predict(x_pre)
    # bde_prediction=pd.DataFrame(y_PRE_BDE)
    # bde_prediction['smi']=data_pre['can_smiles']
    # bde_prediction.to_csv('bde_prediction.csv')

    print('r2(train)rfr:',r2_score(y_train,y_pre_train_rfr),MAE(y_train,y_pre_train_rfr))
    print('r2(test)rfr:',r2_score(y_test,y_pre_test_rfr),MAE(y_test,y_pre_test_rfr))
    # print('100 times R2_rfr:',cross_val(rfr,x,y))



    # 神经网络多层感知机回归
    # from sklearn.neural_network import MLPRegressor
    # param ={'hidden_layer_sizes':[(16),(32),(16,32),(24,12),(32,40),(50,40),(50,60),(60,40)],'alpha':[0.0001,0.001,0.01], 'activation': ['relu'],'solver': ['adam'],'learning_rate_init':[0.001,0.005,0.01], }
    # mlp = GridSearchCV(MLPRegressor( batch_size=32,max_iter=1000,early_stopping=True,validation_fraction=0.2, random_state=42), param_grid=param, scoring='neg_mean_absolute_error',cv=5 , n_jobs=-1)
    # mlp.fit(x_train,y_train)
    # print("params_mlp:", mlp.best_params_)
    # print("scores_mlp:", mlp.best_score_)
    # mlp = mlp.best_estimator_
    # y_pre_train_mlp = mlp.predict(x_train)
    # y_pre_test_mlp = mlp.predict(x_test)
    # # print('label',y_train)
    # # print('pre',y_pre_train_mlp)
    # print('r2(train)mlp:',r2_score(y_train,y_pre_train_mlp),MAE(y_train,y_pre_train_mlp))
    # print('r2(test)mlp:',r2_score(y_test,y_pre_test_mlp),MAE(y_test,y_pre_test_mlp))
    # print('100 times R2_mlp:',cross_val(mlp,x_all,y))
    # stacking_new_smi = mlp.predict(x_new)
    #
    # print('r2(new_smi)stacking:', r2_score(new_pre_smi["Td"], stacking_new_smi), MAE(new_pre_smi["Td"], stacking_new_smi))


    # stacking 模型
    # 定义基模型
    base_models = [
        ('gbr', gbr),
        ('ridge', krr),
        ('rf', rfr),
    ]

    # 定义元模型
    final_estimator = LinearRegression()

    # 实例化StackingRegressor
    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=final_estimator,
        cv=5)

    # 训练基模型和Stacking集成模型
    for name, model in base_models:
        model.fit(x_train, y_train)

    stacking_regressor.fit(x_train, y_train)

    # 预测
    # base_predictions = {name: model.predict(x_test) for name, model in base_models}
    y_pre_train_stacking = stacking_regressor.predict(x_train)
    stacking_prediction = stacking_regressor.predict(x_test)
    # 评估Stacking集成的性能（可选）
    print('r2(train)stacking:',r2_score(y_train, y_pre_train_stacking),MAE(y_train, y_pre_train_stacking))
    print('r2(test)stacking:',r2_score(y_test, stacking_prediction),MAE(y_test, stacking_prediction))

    # explainer = shap.KernelExplainer(stacking_regressor.predict, x_train)
    # # explainer = shap.Explainer(model, x_train)
    # shap_values = explainer(x_test)
    #
    # # 2. 创建SHAP可视化图表
    # plt.figure(figsize=(12, 16))
    # features1=data.columns.drop(['NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
    #    'NumAliphaticRings', 'NumHAcceptors', 'fr_NH1', 'fr_nitrile', 'MolWt',
    #    '1', '4', '8', '16', '19', '31', '32', '33', '34', '36', '39', '43',
    #    '46', '51', '54', '66', '67', '68', '69', 'PMI3', 'N', 'six_N4',
    #    'five_N1', 'five_N4',])
    # # 2.1 SHAP摘要图（特征重要性）
    # plt.subplot(1, 2, 1)
    # shap.summary_plot(shap_values, x_test, feature_names=features1, plot_type="bar", show=False)
    # plt.title("Feature importance (global)", fontsize=8)
    # # # 2.2 SHAP摘要图（特征影响分布）
    # plt.subplot(1, 2, 2)
    # shap.summary_plot(shap_values, x_test, feature_names=features1, show=False)
    # plt.title("feature impact", fontsize=12)



# 预测新数据
#     new_pre_smi = pd.read_csv(new_csv)
    x_new = new_csv.drop(['can_smiles', 'T'], axis=1)
    x_new = x_new.drop(drop_col, axis=1)
    x_new = x_new.drop(highly_corr, axis=1)
    print(x_new.shape)

    x_new = scaler.transform(x_new)
    # x_new = pca.transform(x_new)

    gbr_new_smi = gbr.predict(x_new)
    # print(gbr_new_smi)
    # print('r2(new_smi)gbr:', r2_score(new_pre_smi["Td"], gbr_new_smi), MAE(new_pre_smi["Td"], gbr_new_smi))

    krr_new_smi = krr.predict(x_new)
    # print(krr_new_smi)
    # print('r2(new_smi)krr:', r2_score(new_pre_smi["Td"], krr_new_smi), MAE(new_pre_smi["Td"], krr_new_smi))

    rfr_new_smi = rfr.predict(x_new)
    # print('预测值:', rfr_new_smi)
    # print('r2(new_smi)rfr:', r2_score(new_pre_smi["Td"], rfr_new_smi), MAE(new_pre_smi["Td"], rfr_new_smi))

    stacking_new_smi = stacking_regressor.predict(x_new)
    # print('预测值:',stacking_new_smi)
    # print('r2(new_smi)stacking:', r2_score(new_pre_smi["Td"], stacking_new_smi), MAE(new_pre_smi["Td"], stacking_new_smi))

    # mae为基准
    mae_list = [(MAE(y_test, y_pre_test_gbr),gbr_new_smi), (MAE(y_test, y_pre_test_krr),krr_new_smi ),(MAE(y_test, y_pre_test_rfr),rfr_new_smi),]
    mae_list = sorted(mae_list, key=lambda x: x[0])
    best_new_mae = mae_list[0]
    predict_new_value = best_new_mae[1]
    print('new_value',predict_new_value)

    # R2为基准
    # r2_list = [(r2_score(y_test, y_pre_test_gbr),gbr_new_smi), (r2_score(y_test, y_pre_test_krr),krr_new_smi ),(r2_score(y_test, y_pre_test_rfr),rfr_new_smi),(r2_score(y_test,  stacking_prediction),stacking_new_smi)]
    # r2_list = sorted(r2_list, key=lambda x: x[0],reverse=True)
    # best_new_r2 = r2_list[0]
    # predict_new_value = best_new_r2[1]
    # print('new_value',predict_new_value)





    return r2_score(y_test,y_pre_test_gbr),MAE(y_test,y_pre_test_gbr), r2_score(y_test,y_pre_test_krr),MAE(y_test,y_pre_test_krr), r2_score(y_test,y_pre_test_rfr),MAE(y_test,y_pre_test_rfr),r2_score(y_test, stacking_prediction),MAE(y_test, stacking_prediction),predict_new_value

# x = data.drop(["name",'Dispersion',"Electrostatics",'Exchange','Induction','tot'], axis=1)
# x = data.drop(['ELECTRO'	,'EXCHANGE',	'INDUCTION'	,'DISPERISION'	,'TOTAL'], axis=1)

# print(x.shape)
# print('dis')
# ml(x,data["Dispersion"])
# print('ele')
# ml(x,data["Electrostatics"])
# ml(x,data["Exchange"])
# ml(x,data["Induction"])
# print('tot')

# data=pd.read_csv("D:/smi_xyz/mol_opt/energy/rdk_energy_all.csv")
# data0=pd.read_csv("D:/smi_xyz/all_smi_temp.csv")
# data = pd.merge(data,data0,how='inner',on='can_smiles')
# data = data.drop(['Unnamed: 0_x','Unnamed: 0_y','name','n'],axis=1)
# data = data.dropna(axis = 0)
# data.to_csv('ml_data.csv')
# print(data)

if __name__ == '__main__':
    # data = pd.read_csv("C:/Users/888/Desktop/ml_data13.csv")
    # data=pd.read_csv("D:/smi_xyz/smil_to_xyz/ml_data13_rdk000.csv")
    data = pd.read_csv("D:\pythonProject\ml_data13_rdk_147.csv")
    # data=pd.read_csv("D:/smi_xyz/smil_to_xyz/similarity_smi_new_rdk_des.csv")
    # data = pd.read_excel("D:/smi_xyz/smil_to_xyz/dft_properties/new_useful_data/new_useful_data/model decom/clean000_bde.xlsx")
    # data =pd.read_csv("D:/smi_xyz/smil_to_xyz/dft_properties/new_useful_data/new_useful_data/model decom/clean_del0_bde.csv")
    # data =pd.read_csv("D:/smi_xyz/smil_to_xyz/dft_properties/new_useful_data/new_useful_data/model decom/clean_bde.csv")
    # data=pd.read_csv('D:/smi_xyz/smil_to_xyz/clean_del.csv')
    # data = pd.read_csv('D:/smi_xyz/smil_to_xyz/BDE_des.csv')
    # data=pd.read_csv("C:/Users/888/Desktop/ml_data13.csv")
    # data=pd.read_csv('D:/smi_xyz/smil_to_xyz/rdk_des.csv')
    # x= data.drop(['Td','smiles'],axis=1)
    x = data.drop(['can_smiles','T'], axis=1)
    # x = data.drop(['smiles', 'BDE(kJ/mol)'], axis=1)
    print(x)
    test_lists = pd.DataFrame()
    for i in range(1,2):
        # test_list = ml(x,data["BDE(kJ/mol)"],i)
        csv=pd.read_csv("C:/Users/888/Desktop/Td_predict/smi_new_rdk_des3.csv")
        test_list = ml(x, data["T"], i,csv,0.1)
        test_lists = pd.concat([test_lists,pd.DataFrame(test_list).T],axis=0)
    print('list',test_lists)
    print('gbr_performance: ', test_lists.iloc[:,0].mean(),test_lists.iloc[:,1].mean())
    print('krr_performance: ', test_lists.iloc[:,2].mean(), test_lists.iloc[:,3].mean())
    print('rfr_performance: ', test_lists.iloc[:,4].mean(), test_lists.iloc[:,5].mean())
    print('mlp_performance: ', test_lists.iloc[:,6].mean(), test_lists.iloc[:,7].mean())
    print('stacking_performance: ', test_lists.iloc[:,8].mean(), test_lists.iloc[:,9].mean())

    # # DFT descriptors
    # data=pd.read_excel("D:/smi_xyz/smil_to_xyz/dft_properties/new_useful_data/new_useful_data/model decom/all0_all.xlsx")
    # x= data.drop(['DECOMPOSITION[℃]','code_name'],axis=1)
    # x.columns = x.columns.astype(str)
    # print(x)
    # test_lists = pd.DataFrame()
    # for i in range(1,20):
    #     test_list = ml(x,data['DECOMPOSITION[℃]'],i)
    #     test_lists = pd.concat([test_lists,pd.DataFrame(test_list).T],axis=0)
    # print('list',test_lists)
    # print('gbr_performance: ', test_lists.iloc[:,0].mean(),test_lists.iloc[:,1].mean())
    # print('krr_performance: ', test_lists.iloc[:,2].mean(), test_lists.iloc[:,3].mean())
    # print('rfr_performance: ', test_lists.iloc[:,4].mean(), test_lists.iloc[:,5].mean())
    # print('mlp_performance: ', test_lists.iloc[:,6].mean(), test_lists.iloc[:,7].mean())
    # print('stacking_performance: ', test_lists.iloc[:,8].mean(), test_lists.iloc[:,9].mean())





# # ETR
# # from sklearn.tree import ExtraTreeRegressor
# # #
# # # param = {'n_estimators': [100,250,500],
# # #           "min_samples_leaf": [1,3,5],
# # #           "random_state": [42]}
# #
# # etr = GridSearchCV(estimator=ExtraTreeRegressor(),
# #                  param_grid={"min_samples_leaf": [1,3,5],
# #             'max_depth': [1,2,3,4,6,8,10],
# #        "random_state": [42]},
# #                cv=3)
# # etr.fit(x_train,y_train)
# # etr = etr.best_estimator_
# # etr.fit(x_train,y_train)
# # y_pre_train_etr=etr.predict(x_train)
# # y_pre_test_etr=etr.predict(x_test)
# # print('r2(train)etr:',r2_score(y_train,y_pre_train_etr))
# # print('r2(test)etr:',r2_score(y_test,y_pre_test_etr))
# # print('100 times R2_etr:',cross_val(etr,x,y))
#
#
# # # 绘图
# # import seaborn as sns
# # train_pred = [ y_pre_train_krr, y_pre_train_svr,y_pre_train_rfr, y_pre_train_gbr]
# # test_pred = [ y_pre_test_krr, y_pre_test_svr,y_pre_test_rfr, y_pre_test_gbr]
# #
# # for i in range(1, 5):
# #     plt.rc('font', family='Times New Roman')
# #     plt.subplot(2,2, i)
# #
# #     sns.scatterplot(y_train, train_pred[i - 1].ravel())
# #     sns.scatterplot(y_test, test_pred[i - 1].ravel())
# #     plt.plot(np.arange(-2,50,5),np.arange(-2,50,5),'grey')
# #     plt.subplots_adjust(wspace=0.2, hspace=0.2)
# #
# #     # 坐标轴
# #     plt.xlim([0, 11])
# #     plt.ylim([0, 11])
# #     plt.yticks([0,2,4,6,8,10])
# #
# #     # x轴和y轴标题
# #     plt.xlabel('DFT calculated/eV')
# #     plt.ylabel('Predicted/eV')
# # plt.show()
#
#
# # def plot_learning_curve(estimator, x_train, x_test, y_train, y_test):
# #     """绘制学习曲线：只需要传入算法(或实例对象)、X_train、X_test、y_train、y_test"""
# #     """当使用该函数时传入算法，该算法的变量要进行实例化，如：PolynomialRegression(degree=2)，变量 degree 要进行实例化"""
# #     train_score = []
# #     test_score = []
# #     plt.rc('font', family='Times New Roman')
# #     for i in range(1, len(x_train) + 1):
# #         estimator.fit(x_train[:i], y_train[:i])
# #
# #         y_train_predict = estimator.predict(x_train[:i])
# #         train_score.append(mean_squared_error(y_train[:i], y_train_predict))
# #
# #         y_test_predict = estimator.predict(x_test)
# #         test_score.append(mean_squared_error(y_test, y_test_predict))
# #
# #     plt.plot([i for i in range(1, len(x_train) + 1)],
# #              np.sqrt(train_score), linewidth=2, label="train")
# #     # plt.hlines(0.8, xmin=0, xmax=62, ls='--', lw=1, color='grey')
# #
# #     plt.plot([i for i in range(1, len(x_train) + 1)],
# #              np.sqrt(test_score), label="test")
# #
# #     plt.xlabel('Training set size',fontsize = 15)
# #     plt.ylabel('Error',fontsize = 15)
# #     plt.legend(loc=1,fontsize = 15)
# #     plt.axis([0, len(x_train) + 1, 0, 2.5])
# #     plt.tick_params(labelsize=12)
# #
# #
# #
# # plot_learning_curve(krr, x_train, x_test, y_train, y_test)
# # plt.savefig('./learning/krr.png', bbox_inches='tight', transparent=False)
# # plt.show()
# #
# # plot_learning_curve(svr, x_train, x_test, y_train, y_test)
# # plt.savefig('./learning/svr.png', bbox_inches='tight', transparent=False)
# # plt.show()
# #
# #
# # plot_learning_curve(rfr, x_train, x_test, y_train, y_test)
# # plt.savefig('./learning/rfr.png', bbox_inches='tight', transparent=False)
# # plt.show()
# #
# # plot_learning_curve(gbr, x_train, x_test, y_train, y_test)
# # plt.savefig('./learning/gbr.png', bbox_inches='tight', transparent=False)
# # plt.show()