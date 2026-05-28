import pandas as pd
# from ml_all_predict_22 import ml
# from ml_all_predict_nostack import ml
from ml_all_predict_bayies import ml
# 搜集重要官能团
df_pre = pd.read_csv("./pre_smi_new_allrings3.csv")
df_pre_group =df_pre.loc[ :,'benzene_0no2_0nh2':'five_N2O1_4no2_4nh2']
# df_pre = pd.read_csv("C:/Users/888/Desktop/Td/pre_smi_new_allrings00_no2nh2.csv")
# df_pre_group =df_pre.loc[ :,'benzene_no_sub':'five_N3O1_NH2_NO2']
df_pre_group2=df_pre[['N_NO2','c_3NO2','count_3n']]
df_pre_group3=pd.concat([df_pre_group,df_pre_group2],axis=1)

print(df_pre_group3)

# # 统计预测分子含有的官能团，用于后续挑选具有相同结构的分子
results=df_pre_group3.apply(lambda x: x.index[(x != 0.0) ].to_list(), axis=1).to_list()
print(results)
pre_value=[]
for i, list in enumerate(results):
    print('**********',list,df_pre['con_rings'][i])
    cols_to_check = list
    df = pd.read_csv('./ml_data13_rdk_allrings.csv')
    # df = pd.read_csv('C:/Users/888/Desktop/Td/ml_data13_rdk_allrings00_no2nh2.csv')

    # # 额外统计预测分子中含有3个环以上的结构
    # if int(df_pre['con_rings'][i]) >2:
    #     filtered_df = df[((df[cols_to_check] != 0).sum(axis=1) >=1) | (df['con_rings']>2)]


    filtered_df= df[(df[cols_to_check] != 0).sum(axis=1) >=1]


    drop_df= filtered_df.loc[:, 'benzene_0no2_0nh2':'five_N2O1_4no2_4nh2',]
    # drop_df= filtered_df.loc[:,'benzene_no_sub': 'five_N3O1_NH2_NO2']
    drop_columns=drop_df.columns
    # print(drop_columns)
    filtered_df=filtered_df.drop(columns=drop_columns)
    if len(filtered_df)<30:
        print('sample is not enough')
        pre_value.append('nan')
    else:
    # filtered_df.to_csv('ml_data13_rdk_147.csv')
    # print(filtered_df)
        x = filtered_df.drop(['can_smiles','T'], axis=1)
        # print(x)
        pre_smi_df = df_pre.drop(columns=drop_columns)

        # # 超参数搜索G rid search CV
        # test_list = ml(x, filtered_df["T"], 1,pre_smi_df,corr=0.1)
        # pre_value.append(test_list[8][i])
        # print('corr',0.1,'pre_value_new_smi',test_list[8][i])
        # print(pre_value)

        # 超参数搜索 beyies search
        test_list = ml(x, filtered_df["T"],  pre_smi_df, corr=0.1)
        pre_value.append(test_list[i])
        print('pre_value_new_smi', test_list[i])
        print('num',i, pre_value)


