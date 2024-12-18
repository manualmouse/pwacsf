import pandas as pd
from nd_fsc import ja_stacking_fenliu
from nd_tools import sl2ml
import os
import re
import random
# default parameter
ap_dict = {
    # wacsf1
    'en': 'ele_negativity', 'n': 'n', 'm': 'mass', 'z': 'Z', 
    'ie': 'ionization_energy', 'ea': 'ele_affinity', 'p': 'polarizability', 'r': 'radui',

    # for bin
        'pn':'period_num', 'gn':'group_num',
    }
"""
[DIR_NAME]
in: rc, eta, y, mode_cal_ca, ndw2mode:[fo_list(fodir, ndw2name, wap), bap], fscmode:{else_dict_ori}
fo: fodir, ndw2name, y, rc eta ap
sn: rc eta mode_cal_ca, wap, y, ndw2mode, bap, fscmode 
"""


dict_ndw2mode_fodir={
    'l4':[
        './feature_out_',
        [
            [['0322'],['l4'],[],[],['ea','ie','r']],
        ],
        [['_'],'__dis_wacsf_0'],

    ],
    'foap':[
        './feature_out_',
        [

            [['0322'],['20'],[],[],['ea','ie','r']],
        ],
        [['_','__cap_'],'_0'],
    ],
}



if '__main__'==__name__:
    fsc_list =[
            ['5'],['2'],['kp'],
            [


                '230413_0-04__e->230223__230223',
                '230413_0-08__e->230223__230223',
            ],
            [
                ['s15',{
                    'std':0.15,'corr_list':[300,200,100,60,30,15],
                    'imb':['smote','adasyns','rando']}],
                ['s15_cif',{
                    'std':0.15,'corr_list':[300,200,100,60,30,15],
                    'imb':['smote','adasyns','rando'],
                    'else_feature':'./feature_out_cif/T_wacsf_feature_20230310999999.csv'}],
                ['s15_for',{
                    'std':0.15,'corr_list':[300,200,100,60,30,15],
                    'imb':['smote','adasyns','rando'],
                    'else_feature':'./feature_out_for/T_wacsf_feature_20230310999999.csv'}],
                ['s15_for-cif',{
                    'std':0.15,'corr_list':[300,200,100,60,30,15],
                    'imb':['smote','adasyns','rando'],
                    'else_feature':'./feature_out_for-cif/T_wacsf_feature_20230310999999.csv'}],
                
            ],

            [
                {'foap':[['ea'],['ea']],'l4':[['ea']]},

                ],
        
    ]
    
    fsc_para = [
        'nlond',
        [
            'des_fscorr',
            ]]
    
    
    
    
    
    
    random.seed(24)
    rand_list = [24]
    
    fsc_ml = sl2ml(fsc_list)
    for h in rand_list:
        for i in fsc_ml:
            # print(i)
            # for j in i:
            #     print(j,i[j])
            # print()      
            
            # {0: '5', 1: '2', 2: 'kp', 3: '230310', 4: ['s15', {'std': 0.15}], 5: {'foap': [['p', 'en', 'm'], ['p', 'en', 'm', 'gn',
            # 'pn']], 'l4': ['p', 'en', 'm']}}
            # 0 5
            # 1 2
            # 2 kp
            # 3 230310
            # 4 ['s15', {'std': 0.15}]
            # 5 {'foap': [['p', 'en', 'm'], ['p', 'en', 'm', 'gn', 'pn']], 'l4': ['p', 'en', 'm']}
            
            curr_name = [
                ''.join([str(h)+'-'+i[0],i[1],i[2]]),         # 0  rc eta mode_cal_ca     <- i[0] i[1] i[2]
                '',                                # 1  wap                    <- i[5]
                '',                                # 2  y                      <- i[3] 
                '',                                # 3  ndw2mode               <- i[5]
                '',                                # 4  bap                    <- i[5]
                '',                                # 5  fsmode                 <- i[4]
            ]
                

            # processing i[4]
            curr_name[5]=i[4][0]
            else_dict_ori=i[4][1]
            
            # processing i[3]  1/2
            ori_i3_split = i[3].split('->')
            curr_i3_split = ori_i3_split[0].split('__')
            fo_i3_split   = ori_i3_split[1].split('__')
            if len(curr_i3_split)==2 and curr_i3_split[1][0]=='e':
                else_dict_ori['extra_y_list']=[['./nlo_ced/348_15_56/crystal-in-train'+''.join(curr_i3_split)+'.csv',fo_i3_split[1]]]
            curr_name[2]=''.join(curr_i3_split)
            y_csv_ori=pd.read_csv('./nlo_ced/348_15_56/crystal-in-train'+curr_i3_split[0]+'.csv',index_col = [0],header=None,low_memory=False)
            y_csv_ori.index = [str(j) for j in y_csv_ori.index.tolist()]

            # processing i[5]
            curr_name[1]='--'.join([''.join(i[5][j][0]) for j in i[5]])
            curr_name[3]='--'.join([j for j in i[5]])
            curr_name[4]='--'.join(['-'.join([''.join(k) for k in i[5][j][1:]]) for j in i[5]])
            # print(curr_name)
            # print('\n\n')
            # continue





            fo_tmp_name = '-'.join('_'.join(curr_name).split('-')[1:])


            feature_to_dir = './nlo_ced/348_15_56/feature_out_'+fo_tmp_name
            if not(os.path.exists(feature_to_dir)):
                os.mkdir(feature_to_dir)
            fs_to_dir = './nlo_ced/348_15_56/snresult_nlondc_'+'_'.join(curr_name)
            if not(os.path.exists(fs_to_dir)):
                os.mkdir(fs_to_dir) 

            def feature_df_gen(i_5,dict_ndw2mode_fodir,fo_i3_split_0,y_csv_ori_index_tolist):
                fodir_wbkl_dict = {}
                for j in i_5:
                    curr_fodir_wbkl_d = {}
                    for k in dict_ndw2mode_fodir[j][1]:
                        for l in i_5[j][0]:
                            if not(l in k[-1]):
                                continue
                            fo_dir = dict_ndw2mode_fodir[j][0]+\
                                '_'.join([''.join([n for n in m]) for m in k[:-3]+[fo_i3_split_0,[i[0],i[1],l]]])
                            if not(fo_dir in curr_fodir_wbkl_d):
                                curr_fodir_wbkl_d[fo_dir]=[[]]+i_5[j][1:]
                            curr_fodir_wbkl_d[fo_dir][0].append(l)
                    for k in curr_fodir_wbkl_d:
                        if not(k in fodir_wbkl_dict):
                            fodir_wbkl_dict[k]=[]
                        curr_sl = []
                        for l,ll in enumerate(dict_ndw2mode_fodir[j][2][0]):
                            curr_sl+=[[ll],[]]
                            for mm in curr_fodir_wbkl_d[k][l]:
                                curr_sl[-1].append(ap_dict[mm])
                        curr_sl+=[[dict_ndw2mode_fodir[j][2][1]]]
                        curr_ml = sl2ml(curr_sl)
                        for l in curr_ml:
                            tmp_key = ''.join([l[m] for m in sorted(list(l.keys()))])
                            if not(tmp_key in fodir_wbkl_dict[k]):
                                fodir_wbkl_dict[k].append(tmp_key)
                            else:
                                print("ERROR    fodir_wbkl_dict[k]    tmp_key")
                                print(i_5)
                                print(j)
                                print(k)
                                print(curr_fodir_wbkl_d)
                                print(fodir_wbkl_dict[k])
                                exit()
                # for j in fodir_wbkl_dict:
                #     print(j)
                #     print(fodir_wbkl_dict[j])
                #     print()
                fodir_wbkl_count_d = {}
                feature_df = pd.DataFrame()
                for j in fodir_wbkl_dict:
                    for k in [k for k in os.walk(j)][0][-1]:
                        if not('T_wacsf_feature' in k):
                            continue
                        curr_csv = pd.read_csv(j+'/'+k,index_col=[0],low_memory=False)
                        curr_csv.columns = [str(l) for l in curr_csv.columns.tolist()]
                        curr_csv = curr_csv.loc[:,y_csv_ori_index_tolist]
                        # 0.01_0.0_3.0_1.0_1.0_8.0_1.0_(0.0, 1.0, 0.0)_x_boilingpoint__cap_boilingpoint_0_0__1_+
                        curr_index = []
                        for l in curr_csv.index.tolist():
                            for m in fodir_wbkl_dict[j]:
                                if m in l:
                                    if not(m in fodir_wbkl_count_d):
                                        fodir_wbkl_count_d[m]=0
                                    if not(l in curr_index):
                                        curr_index.append(l)
                                        fodir_wbkl_count_d[m]+=1
                                    else:
                                        print("ERROR feature is in")
                                        print(l)
                                        exit()
                                    break
                        curr_csv = curr_csv.loc[curr_index,:]
                        feature_df = pd.concat([feature_df,curr_csv],axis=0,sort=False)
                # print(feature_df)
                for j in fodir_wbkl_count_d:
                    print(j,fodir_wbkl_count_d[j])
                print()
                return feature_df


            feature_df = feature_df_gen(i[5],dict_ndw2mode_fodir,fo_i3_split[0],y_csv_ori.index.tolist())
            feature_df.to_csv(feature_to_dir+'/T_wacsf_feature_20999999999999.csv')


            else_dict = {}
            for j in [j for j in else_dict_ori.keys() if j!='extra_y_list']:
                else_dict[j]=else_dict_ori[j]



            # processing i[3]  2/2
            if 'extra_y_list' in else_dict_ori:
                extra_df_x = pd.DataFrame()
                extra_df_y = pd.DataFrame()
                for j in else_dict_ori['extra_y_list']:
                    curr_y = pd.read_csv(j[0],index_col = [0],header=None,low_memory=False)
                    curr_y.index = [str(k) for k in curr_y.index.tolist()]
                    curr_x = feature_df_gen(i[5],dict_ndw2mode_fodir,j[1],curr_y.index.tolist())

                    extra_df_x = pd.concat([extra_df_x,curr_x],axis=0,sort=False)
                    extra_df_y = pd.concat([extra_df_y,curr_y],axis=0,sort=False)
                    extra_df_x.to_csv(feature_to_dir+'/'+'/T_wacsf_extra_x_20999999999999.csv')
                    extra_df_y.to_csv(feature_to_dir+'/'+'/T_wacsf_extra_y_20999999999999.csv')
                    else_dict['extra_test_x']=feature_to_dir+'/'+'/T_wacsf_extra_x_20999999999999.csv'
                    else_dict['extra_test_y']=feature_to_dir+'/'+'/T_wacsf_extra_y_20999999999999.csv'
                    # print(extra_df_x)
                    # print(extra_df_y)
                
            y_csv = pd.DataFrame(data =y_csv_ori.iloc[:,5].values.tolist(),index = y_csv_ori.index.tolist(),columns=['delta_n'])
            dir_path = [
                feature_to_dir,
                ]
            dir_ptitle_list =[fsc_para[0],fs_to_dir]
            for j in fsc_para[1]:
                ja_stacking_fenliu(dir_path,dir_ptitle_list,y_csv,fsc_para[0],j,else_dict=else_dict,jd_name='delta_n',random_state=h)


