import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from math import sqrt
from matplotlib import pyplot
from numpy import concatenate
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, LeavePOut, cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, ensemble
import xgboost as xgb
from sklearn.model_selection import learning_curve
import shap
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import shap
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor,StackingRegressor
def ml(x,y,df_pre_new,corr):
    def model_fit(model):

        model.fit(x_train,y_train)
        y_pre_train=model.predict(x_train)
        y_pre_test=model.predict(x_test)
        print('r2(train):',r2_score(y_train,y_pre_train),mean_absolute_error(y_train,y_pre_train))
        print('r2(test):',r2_score(y_test,y_pre_test),mean_absolute_error(y_test,y_pre_test))
        y_prediction = model.predict(x_pre_new)
        return mean_absolute_error(y_test,y_pre_test),y_prediction
        # print(list(y_prediction))
        # print('r2(test_new):',r2_score(y_pre_new,y_prediction),mean_absolute_error(y_pre_new,y_prediction))

    # 步骤一：构造黑盒目标函数
    def rfr_cv(n_estimators, min_samples_split, max_features, max_depth):
        val = cross_val_score(
            RandomForestRegressor(n_estimators=int(n_estimators),
                                  min_samples_split=int(min_samples_split),
                                  max_features=min(max_features, 0.999),  # float
                                  max_depth=int(max_depth),
                                  random_state=2),
            x_train, y_train, scoring='neg_mean_absolute_error', cv=3, error_score='raise'
        ).mean()
        return val


    def krr_cv(alpha, gamma):
        val = cross_val_score(
            KernelRidge(
                alpha=alpha,
                gamma=gamma, kernel='rbf'),
            x_train, y_train, scoring='neg_mean_absolute_error', cv=3, error_score='raise'
        ).mean()
        return val


    def gbr_cv(n_estimators, max_depth, min_samples_split,min_samples_leaf, learning_rate,subsample,alpha):
        val = cross_val_score(
            ensemble.GradientBoostingRegressor(n_estimators=int(n_estimators),
                                               min_samples_split=int(min_samples_split),
                                               min_samples_leaf=int(min_samples_leaf),
                                               learning_rate=min(learning_rate, 0.999),  # float
                                               max_depth=int(max_depth),subsample=subsample,alpha=alpha,
                                               loss='huber',
                                               random_state=2),
            x_train, y_train, scoring='neg_mean_absolute_error', cv=3, error_score='raise'
        ).mean()
        return val


    # def xgb_cv(learning_rate, reg_alpha, reg_lambda, subsample, colsample_bytree, gamma, n_estimators, min_child_weight,
    #            max_depth):
    #     val = cross_val_score(
    #         xgb.XGBRegressor(n_estimators=int(n_estimators),
    #                          min_child_weight=int(min_child_weight),  # float
    #                          max_depth=int(max_depth), gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
    #                          reg_alpha= reg_alpha,
    #                          reg_lambda=reg_lambda,
    #                         learning_rate=learning_rate,
    #                          objective='reg:squarederror',
    #                          random_state=2),
    #         x_train, y_train, scoring='neg_mean_absolute_error', cv=10, error_score='raise'
    #     ).mean()
    #     return val


    def mlp_cv(hidden_layer_sizes_layer1,hidden_layer_sizes_layer2,alpha,learning_rate_init):
        # 创建MLP模型实例
        val = cross_val_score(
            MLPRegressor(hidden_layer_sizes=(int(hidden_layer_sizes_layer1),int(hidden_layer_sizes_layer2)),
                            max_iter=5000,
                            alpha=float(alpha),
                         learning_rate_init=float(learning_rate_init),

                            solver='adam',
                         activation='relu',batch_size='auto',random_state=2),
            x_train, y_train, scoring='neg_mean_absolute_error', cv=3, error_score='raise'
        ).mean()
        # 计算交叉验证得分

        return val
    # 定义搜索空间


    # 步骤二：确定取值空间
    pbounds_rfr = {'n_estimators': (50, 1000), 'min_samples_split': (2, 20),'max_features': (0.3, 0.9),
                  'max_depth': (1, 10)}
    pbounds_krr = {'alpha': (0.001, 1), 'gamma': (0.001, 1)}
    pbounds_gbr = {"n_estimators": (50, 1000), "max_depth": (1, 10), "min_samples_split": (2, 20),'min_samples_leaf': [1, 20],
                   "learning_rate": (0.001, 0.1),'subsample':(0.5,1),'alpha':(0.01,0.1)}
    # pbounds_xgb = {"n_estimators": (50, 1000), "max_depth": (1, 10), "min_child_weight": (1, 10),
    #                "learning_rate": (0.01, 0.1), "gamma": (0.1, 1),
    #                'subsample': (0.5, 1.0), 'colsample_bytree': (0.5, 1), 'reg_alpha': (0, 10),'reg_lambda':(0,10)}


    # pbounds_mlp = {'hidden_layer_sizes_layer1': (10, 100),'hidden_layer_sizes_layer2': (10, 100),
    #                'alpha': (0.001, 0.1), 'learning_rate_init': (0.001, 0.02),}

    # 步骤三：构造贝叶斯优化器
    def opt(pbounds, model_cv):

        optimizer = BayesianOptimization(
            f=model_cv,  # 黑盒目标函数
            pbounds=pbounds,  # 取值空间
            verbose=1,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
            random_state=1,allow_duplicate_points=True
        )
        optimizer.maximize(  # 运行
            init_points=5,  # 随机搜索的步数
            n_iter=500,  # 执行贝叶斯优化迭代次数
        )
        params_best = optimizer.max["params"]
        score_best = optimizer.max["target"]
        # print(optimizer.res)  # 打印所有优化的结果
        print(optimizer.max)  # 最好的结果与对应的参数
        # 打印最佳参数与最佳分数
        print("\n", "\n", "best params: ", params_best,
              "\n", "\n", "best cvscore: ", score_best)

        # 返回最佳参数与最佳分数
        return params_best, score_best


    # 特征相关性分析
    dat = pd.concat([x,y],axis=1)
    x_cor = dat.corr(method='pearson')
    # x_cor.to_csv('a1.csv')
    x_cor = x_cor.abs()
    print(x_cor[y.name])
    drop_col=x_cor[x_cor[y.name]<corr].index
    print(drop_col)
    # drop_col2=x_cor[x_cor[y.name]>0.1].index
    x = x.drop(drop_col, axis=1)

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
    print(highly_corr)
    x = x.drop(highly_corr, axis=1)

    # print('x.shape',x.shape)
    # 获得features name for
    # data_all=pd.concat([x,y],axis=1)
    # data_all.columns = data_all.columns.astype(str)
    # headers1 = data_all.columns[0:]
    #
    # features1 = list(headers1)
    # print(features1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)


    # df_pre_new=pd.read_csv("C:/Users/888/Desktop/Td_predict/pre_smi_new_rdk_des3.csv")
    x_pre_new= df_pre_new.drop(['T', 'can_smiles'], axis=1)
    x_pre_new = x_pre_new.drop(drop_col, axis=1)
    x_pre_new = x_pre_new.drop(highly_corr, axis=1)
    x_pre_new =scaler.transform(x_pre_new)
    # 特征工程降维处理
    # PCA降维
    # pca = PCA(n_components=20)
    # x_train = pca.fit_transform(x_train)
    # x_test = pca.transform(x_test)

    MAE_list=[]
    params_best_rfr, score_best_rfr =opt(pbounds_rfr,rfr_cv)
    rfr=RandomForestRegressor(max_depth=int(params_best_rfr['max_depth']),max_features=float (params_best_rfr['max_features']),min_samples_split=int(params_best_rfr['min_samples_split']),n_estimators=int(params_best_rfr['n_estimators']), random_state=2)
    MAE_list.append(model_fit(rfr))

    params_best_krr, score_best_krr=opt(pbounds_krr,krr_cv)
    krr=KernelRidge(kernel='rbf',alpha=params_best_krr['alpha'],gamma=params_best_krr['gamma'])
    MAE_list.append(model_fit(krr))

    params_best_gbr, score_best_gbr=opt(pbounds_gbr,gbr_cv)
    gbr=ensemble.GradientBoostingRegressor(n_estimators=int(params_best_gbr['n_estimators']),
                                       min_samples_split=int(params_best_gbr['min_samples_split']),
                                       min_samples_leaf=int(params_best_gbr['min_samples_leaf']),
                                       learning_rate=  params_best_gbr['learning_rate'], # float
                                       max_depth=int(params_best_gbr['max_depth']), subsample= params_best_gbr['subsample'], alpha=  params_best_gbr['alpha'],
                                       loss='huber',
                                       random_state=2)
    MAE_list.append(model_fit(gbr))

    # params_best_mlp, score_best_mlp=opt(pbounds_mlp,mlp_cv)
    # mlp=MLPRegressor(hidden_layer_sizes=(int(params_best_mlp['hidden_layer_sizes_layer1']),int(params_best_mlp['hidden_layer_sizes_layer2'])),
    #              max_iter=1000,
    #              alpha=params_best_mlp['alpha'],
    #              learning_rate_init=params_best_mlp['learning_rate_init'],
    #              solver='adam',
    #              activation='relu', batch_size='auto', random_state=2)
    # MAE_list.append(model_fit(mlp))

    # 定义基模型
    base_models = [
        ('gbr', gbr),
        ('ridge', krr),
        ('rf', rfr),
    ]

    # 定义元模型
    final_estimator = LinearRegression()

    # 实例化StackingRegressor
    stack = StackingRegressor(
        estimators=base_models,
        final_estimator=final_estimator,
        cv=3)

    MAE_list.append(model_fit(stack))
    print('mae for model/ pre_value',MAE_list)
    mae_list = sorted(MAE_list, key=lambda x: x[0])
    best_new_mae = mae_list[0]
    new_pre = best_new_mae[1]
    print('new_mae',new_pre)
    return new_pre


