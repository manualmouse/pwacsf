import numpy as np
import pandas as pd
import re
import time
import os
import copy
import math
import random
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from nd_wacsf1p import PMGLOAD
from nd_wacsf1p import BASICINFO

from nd_tools import sl2oao
from nd_tools import sl2f

def npzloader(npz_dir,pcl_list,ano_dict,oao_dict):
    """
    npz_dir: str do not involve different ca_mode - cal !!!
    pcl_list: generated from nd_wacsf1u
    """
    work_pcl_dict = {}
    for i in pcl_list:
        curr_key = tuple([float(j) for j in i[:7]]+[tuple([float(j) for j in i[7]]),str(i[8]),str(i[9])])
        if not(curr_key in work_pcl_dict):
            work_pcl_dict[curr_key]={}
            work_pcl_dict[curr_key]['count']=[[],[]]
        for j in i[14]:
            work_pcl_dict[curr_key][j]=len(list(ano_dict[j].keys()))
    dir_key_dict = {}
    ret_dict = {}
    for i in [i for i in os.walk(npz_dir)][0][-1]:
        fn_split1 = i[:-4].split('__')[1:]
        fn_split2o = fn_split1[0].split('_')
        fn_split2 = fn_split2o[:9]+['_'.join(fn_split2o[9:])]
        fn_para = [float(j) for j in fn_split2[:7]]+\
            [tuple([float(j) for j in re.split(r'[\[\],]',fn_split2[7]) if j])]+\
            [str(fn_split2[8]),str(fn_split2[9])]
        curr_key = tuple([tuple(fn_para),int(fn_split1[1]),fn_split1[2],fn_split1[3]]
        )
        curr_fn = list(curr_key[0])+list(curr_key)[1:]
        if not(sl2oao(curr_fn,oao_dict)):
            continue
        print(curr_key)
        if curr_key in dir_key_dict:
            print("ERROR npzloader repeat",i,curr_key,dir_key_dict[curr_key])
        dir_key_dict[curr_key]=npz_dir+'/'+i
    
        # p c a w -> p w c a
        if not curr_key[0] in ret_dict:
            ret_dict[curr_key[0]]={}
        if not curr_key[3] in ret_dict[curr_key[0]]:
            ret_dict[curr_key[0]][curr_key[3]]={}
        if not curr_key[2]=='all':
            if not int(curr_key[1]) in ret_dict[curr_key[0]][curr_key[3]]:
                ret_dict[curr_key[0]][curr_key[3]][int(curr_key[1])]={}
            if not int(curr_key[2]) in ret_dict[curr_key[0]][curr_key[3]][int(curr_key[1])]:
                pass
            else:
                print("ERROR npzloader repeat    p-c-a     ",curr_key[0],curr_key[1],curr_key[2])
            ret_dict[curr_key[0]][curr_key[3]][int(curr_key[1])][int(curr_key[2])]=np.load(npz_dir+'/'+i,allow_pickle=True)['arr_0']
        else:
            if not int(curr_key[1]) in ret_dict[curr_key[0]][curr_key[3]]:
                pass
            else:
                print("ERROR npzloader repeat    p-c    ",curr_key[0],curr_key[1])
            ret_dict[curr_key[0]][curr_key[3]][int(curr_key[1])]={}


            curr_list = np.load(npz_dir+'/'+i,allow_pickle=True)['arr_0']
            for j in range(len(curr_list)):
                ret_dict[curr_key[0]][curr_key[3]][int(curr_key[1])][j]=curr_list[j]
        work_pcl_dict[curr_key[0]]['count'][0].append(copy.deepcopy(curr_key[1]))
        if int(curr_key[1]) in work_pcl_dict[curr_key[0]] and len(ret_dict[curr_key[0]][curr_key[3]][int(curr_key[1])])!=work_pcl_dict[curr_key[0]][int(curr_key[1])]:
            work_pcl_dict[curr_key[0]]['count'][1].append(copy.deepcopy(curr_key[1]))
    print("len pcl    ",len(pcl_list))
    print("len p    ",len(list(ret_dict.keys())))
    work_pcl = [tuple([float(j) for j in i[:7]]+[tuple([float(j) for j in i[7]]),str(i[8]),str(i[9])]) for i in pcl_list]
    work_p = [i for i in list(ret_dict.keys())]
    print("-"*20)
    print(len(set(work_pcl)-set(work_p)),set(work_pcl)-set(work_p))
    print(len(set(work_p)-set(work_pcl)),set(work_p)-set(work_pcl))
    print("-"*20)
    for i in work_pcl_dict:
        c_l = [j for j in work_pcl_dict[i] if not j=='count']
        if work_pcl_dict[i]['count'][0] and set(work_pcl_dict[i]['count'][0])!=set(c_l):
            d1 = set(work_pcl_dict[i]['count'][0])-set(c_l)
            d2 = set(c_l)-set(work_pcl_dict[i]['count'][0])
            print("p-c ERROR",i)
            print(len(d1),d1)
            print(len(d2),d2)
        if work_pcl_dict[i]['count'][1]:
            print('p-c-a ERROR',i)
            print(work_pcl_dict[i]['count'][1])
    print("-"*20+' end of init '+"-"*20)
    return ret_dict


class HGGEN(object):
    def __init__(self,input_dict):
        """
        input_dict:
            'pwca_dict': 
                pwca_dict
            'pmg_dict':
                for PMGLOAD
            'bi_dict':
                for BASICINFO
            'hg_dict'
                'cap_prop':[['global' 'local',int]], 'prop': atom_prop_name 
                
                'dis_wacsf':[['global' 'local',int]]
                
                'clu_?'[[
                    'global',
                    [int,,,],
                    [
                        [
                            {'kms':{para:,metric:},'dbs':{}},
                            {'pca':{},'t-sne':{}}],
                        ['cap_prop','dis_wacsf',,]]
                    ],
                ]

                'psc_cap_prop':[['global' 'local',int,[int,,]]],
                'psc_dis_wacsf':[['global' 'local',int,[int,,]]]
                'psc_psc_sum':[['local',1,[int,,]]],
                'psc_psc_mean':[['local',1,[int,,]]],
                'psc_psc_clu'[[
                    'global',
                    [int,,,],
                    [
                        [
                            {'kms':{para:,metric:},'dbs':{}},
                            {'pca':{},'t-sne':{}}],
                        ['cap_prop','dis_wacsf',,]]
                    ],
                ]

                'rand24':[['global' 'local',int,]]
                



            'cal_para':['+','max','min','mean','maxd','mind','meand']
            'save_path':''
            'atom_mean':'1' 'an' 'cn'
        """
        self.input_dict = input_dict
        pmg_obj = PMGLOAD(load_dict=input_dict['pmg_dict'])
        self.cas_dict = pmg_obj.d_neighbor(neighbor_dict={
            'cut_rad':input_dict['pmg_dict']['cut_rad'],
            'return_mode':'ca_species'
        })
        work_pwca_dict = {}
        for i in self.input_dict['pwca_dict']:
            work_pwca_dict[i]={}
            for j in self.input_dict['pwca_dict'][i]:
                work_pwca_dict[i][j]={}
                for k in self.input_dict['pwca_dict'][i][j]:
                    if k in self.cas_dict:
                        work_pwca_dict[i][j][k]=self.input_dict['pwca_dict'][i][j][k]
        self.input_dict['pwca_dict'] = work_pwca_dict
        print('finish self.cas_dict')
        bi_obj = BASICINFO()
        bi_obj.load_element_in(csv_dict=input_dict['bi_dict'])
        self.element_df = bi_obj.element_df
        print('finish self.element_df')

        self.feature_dict = {}
        self.atom_column_neighbor_dict= {}

        self.psc_tmp_dict = {}
        self.psc_old_feature_dict={}

        self.ret_dict = {}


 
        
        # p_eta = h[0]
        # p_miu = h[1]
        # p_rc = h[2]
        # p_zeta1 = h[3]
        # p_lambda1 = h[4]
        # p_zeta2 = h[5]
        # p_lambda2 = h[6]
        # p_axis = h[7]
        # p_h = h[8]
        # p_ap = h[9]


        
        random_seed = [i for i in self.input_dict['hg_dict'].keys() if i[:4]=='rand']
        if random_seed:
            random_seed = int(random_seed[0][4:])
            random.seed(random_seed)
        for i_index,i in enumerate(self.input_dict['pwca_dict']):
            """
            每组参数下应抽取与合并的信息
            以下共 个阶段
                拆分 j 信息类    
                循环 g 信息名
                    
            """
            # if i_index>int(len(self.input_dict['pwca_dict'])/10) and i_index%int(len(self.input_dict['pwca_dict'])/10)<1e-3:
            #     print(i_index,i_index/int(len(self.input_dict['pwca_dict'])/10))
            print(i_index,i)
            for j in self.input_dict['hg_dict']:
                print(j)
                cat_j = j[:4]
                work_j = j[4:]
                if cat_j=='cap_':
                    if work_j not in self.element_df.columns.tolist():
                        print('cap_? ERROR    ',j)
                        return
                elif cat_j=='dis_':
                    if work_j not in self.input_dict['pwca_dict'][i].keys():
                        print('dis_?    ',j)
                        return
                elif cat_j=='psc_':
                    cat_j = work_j[:4]
                    work_j = work_j[4:]
                elif cat_j=='rand':
                    pass
                elif cat_j=='clu_':
                    pass
                else:
                    print("?_?    ERROR    ",j)
                    return

                for g_index,g in enumerate(self.input_dict['hg_dict'][j]):
                    print('g_index',g_index)
                    atom2bar_dict = {}
                    bar2atom_list = []
                    if g[0]=='global':
                        glo_prop_v = []
                        glo_prop_i = []

                        if cat_j=='clu_' or (cat_j=='psc_' and work_j=='clu'):
                            print('start global clu_?')
                            if cat_j=='clu_':
                                # g[0] glo loc, g[1] [int], g[2][0][0] alg para met, g[2][0][1] pca t-sne, g[2][1] prop
                                # 'clu_?'[[
                                #     'global',
                                #     [int,,,],
                                #     [
                                #         [
                                #             {'kms':{'random_state':} 'spe':{'random_state':}},
                                #             {'pca':{'random_state':},'t-sne':{'random_state':}}],
                                #         ['cap_prop','dis_wacsf',,]]
                                #     ],
                                # ]
                                work_g = g
                            elif (cat_j=='psc_' and work_j=='clu'):
                                # g[0] glo loc, g[1] [int], g[2][0][0] alg para met, g[2][0][1] pca t-sne, g[2][1] prop
                                work_g = copy.deepcopy(g)


                            clu_dict = {}
                            is_dis = 0
                            for k in self.input_dict['pwca_dict'][i]['wacsf']:
                                for l in self.input_dict['pwca_dict'][i]['wacsf'][k]:
                                    curr_key = tuple([k,l])
                                    if not(curr_key in clu_dict):
                                        clu_dict[curr_key]={}
                                    for m in work_g[2][1]:
                                        cat_m = m[:4]
                                        work_m = m[4:]
                                        if cat_m=='dis_':
                                            clu_dict[curr_key][m]=self.input_dict['pwca_dict'][i][work_m][k][l]
                                            is_dis=1
                                        elif cat_m=='cap_':
                                            clu_dict[curr_key][m]=float(self.element_df.loc[self.cas_dict[k][l]['self_species'],work_m])
                            
                            
                            clu_dfo = pd.DataFrame().from_dict(clu_dict).T
                            print('finish global clu_dfo')
                            # print("clu_dfo global index    ",clu_dfo.index.tolist())
                            # print("clu_dfo global    ",clu_dfo)
                            if cat_j=='clu_':
                                if len(list(work_g[2][0][0].keys()))>1:
                                    print("ERROR   work_g[2][0][0]>1")
                                    return
                                clu_result = []
                                clu_pre = []
                                clu_index = copy.deepcopy(clu_dfo.index.to_list())
                                clu_df = StandardScaler().fit_transform(clu_dfo)
                                print("clu_df    ",clu_df)
                                print("clu_df shape ",np.shape(clu_df))
                                for k in work_g[1]:
                                    print('clu_k    ',k)
                                    if 'kms' in work_g[2][0][0]:
                                        clu_model = KMeans(n_clusters=k,random_state=work_g[2][0][0]['kms']['random_state']).fit(clu_df)
                                    elif 'spe' in work_g[2][0][0]:
                                        clu_model = SpectralClustering(n_clusters=k,random_state=work_g[2][0][0]['spe']['random_state'],n_jobs=16).fit(clu_df)
                                    clu_result.append(silhouette_score(clu_df,clu_model.labels_))
                                    clu_pre.append(list(clu_model.labels_))
                                    print(silhouette_score(clu_df,clu_model.labels_))
                                best_pre = clu_pre[clu_result.index(max(clu_result))]
                                print(clu_result.index(max(clu_result)))
                                atom2bar_dict = {}
                                for k in range(len(clu_index)):
                                    atom2bar_dict[clu_index[k]]=best_pre[k]
                                    clu_dfo.loc[clu_index[k],'pre_result'] = best_pre[k]
                                bar2atom_list = [[] for _ in range(work_g[1][clu_result.index(max(clu_result))])]
                                for k in range(len(best_pre)):
                                    bar2atom_list[best_pre[k]].append(clu_index[k])
                            elif (cat_j=='psc_' and work_j=='clu'):
                                atom2bar_dict = {}
                                bar2atom_list = []

                            if (
                                ((cat_j=='clu_') and (is_dis or (not(is_dis) and i_index==0))) 
                                # ((cat_j=='psc_' and work_j=='clu') and i_index==len(self.input_dict['pwca_dict'])-1)
                            ):
                                if (
                                    ((cat_j=='clu_') and (not(is_dis) and i_index==0)) or 
                                    ((cat_j=='clu_') and (is_dis)) 
                                    # ((cat_j=='psc_' and work_j=='clu') and i_index==len(self.input_dict['pwca_dict'])-1)
                                ):
                                    clu_dfo.to_csv(self.input_dict['save_path']+'/T_wacsf_clu_'+str(i_index)+'_'+time.strftime("%Y%m%d%H%M%S", time.localtime())+'.csv',index=True,header=True)
                                color_list = ['black','aquamarine', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'hotpink', 'indianred', 'indigo', 'lawngreen', 'lime', 'limegreen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegreen', 'palevioletred', 'peachpuff', 'peru', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat',]
                                if (cat_j=='clu_'):
                                    for k in work_g[2][0][1]:
                                        if k=='pca':
                                            pca_transform = list(PCA(n_components=2,random_state=work_g[2][0][1]['pca']['random_state']).fit_transform(clu_df))
                                        elif k=='t-sne':
                                            pca_transform = list(TSNE(n_components=2,random_state=work_g[2][0][1]['t-sne']['random_state']).fit_transform(clu_df))
                                        print('finish pca/tsne')
                                        plt.figure()
                                        for l in range(len(pca_transform)):
                                            plt.scatter(pca_transform[l][0],pca_transform[l][1],color=color_list[atom2bar_dict[clu_index[l]]])
                                        if (
                                            ((cat_j=='clu_') and (not(is_dis) and i_index==0)) or 
                                            ((cat_j=='clu_') and (is_dis))
                                        ):
                                            plt.savefig(self.input_dict['save_path']+'/T_wacsf_clufig_'+\
                                                str(i_index)+'_'+str(list(work_g[2][0][0].keys())[0])+'_'+str(k)+'_'+\
                                                time.strftime("%Y%m%d%H%M%S", time.localtime())+'.png',bbox_inches='tight')
                                        plt.close()

                        else:
                            for k in self.input_dict['pwca_dict'][i]['wacsf']:
                                for l in self.input_dict['pwca_dict'][i]['wacsf'][k]:
                                    glo_prop_i.append(tuple([k,l]))
                                    if cat_j=='dis_':
                                        glo_prop_v.append(self.input_dict['pwca_dict'][i]['wacsf'][k][l])
                                    elif cat_j=='cap_':
                                        glo_prop_v.append(float(self.element_df.loc[self.cas_dict[k][l]['self_species'],work_j]))
                                    elif cat_j=='rand':
                                        glo_prop_v.append(random.randint(0,g[1]))                                        
                            atom2bar_dict,bar2atom_list = self.d_vi2dl(glo_prop_v,glo_prop_i,g[1])


                    for k in self.input_dict['pwca_dict'][i]['wacsf']:
                        if g[0]=='local':
                            loc_prop_v = []
                            loc_prop_i = []
                            for l in self.input_dict['pwca_dict'][i]['wacsf'][k]:
                                loc_prop_i.append(tuple([k,l]))
                                if cat_j in ['dis_','psc_']:
                                    loc_prop_v.append(self.input_dict['pwca_dict'][i]['wacsf'][k][l])
                                elif cat_j=='cap_':
                                    loc_prop_v.append(float(self.element_df.loc[self.cas_dict[k][l]['self_species'],work_j]))
                                elif cat_j=='rand':
                                    loc_prop_v.append(random.randint(0,g[1]))
                            atom2bar_dict,bar2atom_list = self.d_vi2dl(loc_prop_v,loc_prop_i,g[1])

                        if not(k in self.atom_column_neighbor_dict):
                            self.atom_column_neighbor_dict[k]={}
                            self.feature_dict[k]={}
                        if set(self.cas_dict[k].keys())!=set(self.input_dict['pwca_dict'][i]['wacsf'][k].keys()):
                            print("len(atom2bar_dict[self.cas_dict[k])!=len(self.input_dict['pwca_dict'][i]['wacsf'][k])")
                            print(k)
                            print(set(self.cas_dict[k].keys()).difference(set(self.input_dict['pwca_dict'][i]['wacsf'][k].keys())))
                            print("\n\n\nERROR\n\n\n")
                            # return
                        l_list= list(self.input_dict['pwca_dict'][i]['wacsf'][k].keys())
                        for l in range(len(l_list)):
                            curr_key = tuple([k,l_list[l]])
                            if (cat_j=='psc_' and work_j=='clu'):
                                # for m in [i,j,g_index,k,l_list[l],'\n']:
                                #     print(m)
                                for m in clu_dict[curr_key]:
                                    cat_m = m[:4]
                                    work_m = m[4:]
                                    if cat_m=='cap_':
                                        curr_bar_column = 'p_s_c_c_l_u_s_t_e_r__'+str(j)+'_'+str(m)+'_'+str(g_index)+'_0'+'__'
                                    elif cat_m=='dis_':
                                        curr_bar_column = '_'.join([str(m) for m in i])+'__'+str(j)+'_'+str(m)+'_'+str(g_index)+'_0'+'__'
                                    if not(curr_bar_column in self.atom_column_neighbor_dict[k]):
                                        self.atom_column_neighbor_dict[k][curr_bar_column]=['' for _ in range(len(l_list))]
                                        self.feature_dict[k][curr_bar_column]=['' for _ in range(len(l_list))]
                                    self.atom_column_neighbor_dict[k][curr_bar_column][l_list[l]]=str(l_list[l])+\
                                        '_'+str(self.cas_dict[k][l_list[l]]['self_species'])+'_'+\
                                        str(round(self.input_dict['pwca_dict'][i]['wacsf'][k][l_list[l]],3))
                                    self.feature_dict[k][curr_bar_column][l_list[l]]=clu_dict[curr_key][m]


                            else:
                                curr_bar_column = '_'.join([str(m) for m in i])+'__'+str(j)+'_'+str(g_index)+'_'+str(atom2bar_dict[curr_key])+'__'
                                if not(curr_bar_column in self.atom_column_neighbor_dict[k]):
                                    self.atom_column_neighbor_dict[k][curr_bar_column]=[]
                                    self.feature_dict[k][curr_bar_column]=[]
                                self.atom_column_neighbor_dict[k][curr_bar_column].append(
                                    str(l_list[l])+'_'+str(self.cas_dict[k][l_list[l]]['self_species'])+'_'+\
                                    str(round(self.input_dict['pwca_dict'][i]['wacsf'][k][l_list[l]],3)))
                                self.feature_dict[k][curr_bar_column].append(self.input_dict['pwca_dict'][i]['wacsf'][k][l_list[l]])

     
        is_psc = [i for i in self.input_dict['hg_dict'].keys() if i[:4]=='psc_']
        if is_psc:
            if 'psc_psc_clu' in is_psc:
                psc_clu_dict = {}
            curr_fd_df = pd.DataFrame().from_dict(self.feature_dict)
            curr_fd_len = {}
            for i in self.feature_dict:
                if not(i in curr_fd_len):
                    curr_fd_len[i]={}
                for j in self.feature_dict[i]:
                    if not(j in curr_fd_len):
                        curr_fd_len[i][j]=len(self.feature_dict[i][j])
            curr_fd_len_df =pd.DataFrame().from_dict(curr_fd_len)
            print("curr_fd_df\n",curr_fd_df)
            print("curr_fd_len_df",curr_fd_len_df)
            # print(curr_fd_len_df.index.tolist())

            for i in self.feature_dict:
                print('2df    ',i)
                for j in self.feature_dict[i]:
                    if '__psc_' in j:
                        j_re = [k.split('_') for k in j.split('__') if k]
                        j_re[0] = j_re[0][:9]+['_'.join(j_re[0][9:])]
                        j_re[1] = ['_'.join(j_re[1][:-2])]+j_re[1][-2:]
                        # print(j_re[1][0])
                        if 'psc_psc_clu' in j_re[1][0]:
                            curr_key0 = j_re[1][1]
                            if not(curr_key0 in psc_clu_dict):
                                psc_clu_dict[curr_key0]={}
                            for k in range(len(self.feature_dict[i][j])):
                                curr_key1 = str(i)+'_'+str(k)
                                # print(curr_key0,'\n',curr_key1,'\n',j)
                                if not(curr_key1 in psc_clu_dict[curr_key0]):
                                    psc_clu_dict[curr_key0][curr_key1]={}
                                if not(j in psc_clu_dict[curr_key0][curr_key1]):
                                    psc_clu_dict[curr_key0][curr_key1][j]=self.feature_dict[i][j][k]
                        else:
                            curr_key0='_'.join([
                                [j_re[0][k] if not(k in self.input_dict['hg_dict'][j_re[1][0]][int(j_re[1][1])][2]) else str(k)+'&&'][0] for k in range(len(j_re[0]))
                            ])
                            curr_key = curr_key0+'__'+'_'.join([j_re[1][0]]+j_re[1][1:])+'__'

                            if not(i in self.psc_tmp_dict):
                                self.psc_tmp_dict[i]={}
                            if not(curr_key in self.psc_tmp_dict[i]):
                                self.psc_tmp_dict[i][curr_key]=[]
                            if j_re[1][0]=='psc_psc_mean':
                                self.psc_tmp_dict[i][curr_key].append(np.sum(np.array(self.feature_dict[i][j]))/len(np.array(self.feature_dict[i][j])))
                            else: 
                                # j_re[1][0]=='psc_psc_sum' 或其他 i 内已分桶的情形，新版本将提出更多归并方案
                                self.psc_tmp_dict[i][curr_key].append(np.sum(np.array(self.feature_dict[i][j])))
                    # print()
            if 'psc_psc_clu' in is_psc:
                for g_index,g in enumerate(self.input_dict['hg_dict']['psc_psc_clu']):
                    psc_tmp = psc_clu_dict[str(g_index)]
                    clu_dfo = pd.DataFrame().from_dict(psc_tmp).T
                    print('finish psc clu_dfo')
                    print("clu_dfo psc    ",clu_dfo)

                    if len(list(g[2][0][0].keys()))>1:
                        print("ERROR   g[2][0][0]>1")
                        return
                    clu_result = []
                    clu_pre = []
                    clu_index = copy.deepcopy(clu_dfo.index.to_list())
                    clu_df = StandardScaler().fit_transform(clu_dfo)
                    print("clu_df    ",clu_df)
                    print("clu_df shape ",np.shape(clu_df))
                    for k in g[1]:
                        print('clu_k    ',k)
                        if 'kms' in g[2][0][0]:
                            clu_model = KMeans(n_clusters=k,random_state=g[2][0][0]['kms']['random_state']).fit(clu_df)
                        elif 'spe' in g[2][0][0]:
                            clu_model = SpectralClustering(n_clusters=k,random_state=g[2][0][0]['spe']['random_state'],n_jobs=16).fit(clu_df)
                        clu_result.append(silhouette_score(clu_df,clu_model.labels_))
                        clu_pre.append(list(clu_model.labels_))
                        print(silhouette_score(clu_df,clu_model.labels_))
                    best_pre = clu_pre[clu_result.index(max(clu_result))]
                    print(g[1][clu_result.index(max(clu_result))])
                    atom2bar_dict = {}
                    for k in range(len(clu_index)):
                        atom2bar_dict[clu_index[k]]=best_pre[k]
                        clu_dfo.loc[clu_index[k],'pre_result'] = best_pre[k]
                    bar2atom_list = [[] for _ in range(g[1][clu_result.index(max(clu_result))])]
                    for k in range(len(best_pre)):
                        bar2atom_list[best_pre[k]].append(clu_index[k])
                    color_list = ['black','aquamarine', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'hotpink', 'indianred', 'indigo', 'lawngreen', 'lime', 'limegreen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegreen', 'palevioletred', 'peachpuff', 'peru', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat',]
                    if len(bar2atom_list)>len(color_list):
                        print("ERROR color_list")
                        return

                    clu_dfo.to_csv(self.input_dict['save_path']+'/T_wacsf_clu_-1_'+time.strftime("%Y%m%d%H%M%S", time.localtime())+'.csv',index=True,header=True)
                    for k in g[2][0][1]:
                        if k=='pca':
                            pca_transform = list(PCA(n_components=2,random_state=g[2][0][1]['pca']['random_state']).fit_transform(clu_df))
                        elif k=='t-sne':
                            pca_transform = list(TSNE(n_components=2,random_state=g[2][0][1]['t-sne']['random_state']).fit_transform(clu_df))
                        print('finish pca/tsne')
                        plt.figure()
                        for l in range(len(pca_transform)):
                            plt.scatter(pca_transform[l][0],pca_transform[l][1],color=color_list[atom2bar_dict[clu_index[l]]],marker='.')
                        plt.savefig(self.input_dict['save_path']+'/T_wacsf_clufig_-1_'+\
                            '_'+str(list(g[2][0][0].keys())[0])+'_'+str(k)+'_'+str(round(clu_result[clu_result.index(max(clu_result))],3))+\
                            time.strftime("%Y%m%d%H%M%S", time.localtime())+'.png',bbox_inches='tight')
                        plt.close()


                    
                    
                    for k in atom2bar_dict:
                        k_key = [int(l) for l in k.split('_')]
                        curr_bar_column = 'p_s_c_c_l_u_s_t_e_r__psc_psc_clu_result_'+str(g_index)+'_'+str(atom2bar_dict[k])+'__'

                        if not(k_key[0] in self.atom_column_neighbor_dict):
                            self.atom_column_neighbor_dict[k_key[0]]={}
                        if not(curr_bar_column in self.atom_column_neighbor_dict[k_key[0]]):
                            self.atom_column_neighbor_dict[k_key[0]][curr_bar_column]=[]

                        if not(k_key[0] in self.psc_tmp_dict):
                            self.psc_tmp_dict[k_key[0]]={}
                        if not(curr_bar_column in self.psc_tmp_dict[k_key[0]]):
                            self.psc_tmp_dict[k_key[0]][curr_bar_column]=[]

                        self.atom_column_neighbor_dict[k_key[0]][curr_bar_column].append(str(k_key[1])+'_'+str(self.cas_dict[k_key[0]][k_key[1]]['self_species']))
                        for l in self.feature_dict[k_key[0]]:
                            self.psc_tmp_dict[k_key[0]][curr_bar_column].append(copy.deepcopy(self.feature_dict[k_key[0]][l][k_key[1]]))



            self.psc_old_feature_dict = copy.deepcopy(self.feature_dict)
            self.feature_dict=self.psc_tmp_dict



        for i in self.feature_dict:
            print('2df    ',i)
            if not i in self.ret_dict:
                self.ret_dict[i]={}
            for j in self.feature_dict[i]:
                for k in self.input_dict['cal_para']:
                    for l in  self.input_dict['atom_mean']:
                        if l=='1':
                            nj = j+'1_'
                        elif l=='an':
                            nj = j+'an_'
                            self.feature_dict[i][j] = np.array(
                                self.feature_dict[i][j])/np.array([len(self.cas_dict[i][m]['neighbor_species']) for m in [int(n.split('_')[1]) for n in self.atom_column_neighbor_dict[i][j]]])
                        elif l=='cn':
                            self.feature_dict[i][j] = np.array(self.feature_dict[i][j])/len(self.feature_dict[i].keys())
                        self.ret_dict[i][nj+k]=sl2f(np.array(self.feature_dict[i][j]),k)
                        # if k=='+':
                        #     self.ret_dict[i][nj+'+']=np.sum(np.array(self.feature_dict[i][j]))
                        # elif k=='max':
                        #     self.ret_dict[i][nj+'max']=np.max(np.array(self.feature_dict[i][j]))
                        # elif k=='min':
                        #     self.ret_dict[i][nj+'min']=np.min(np.array(self.feature_dict[i][j]))
                        # elif k=='mean':
                        #     self.ret_dict[i][nj+'mean']=np.mean(np.array(self.feature_dict[i][j]))
                        # elif k=='maxd':
                        #     curr_d = [abs(m-n) for m in np.array(self.feature_dict[i][j]) for n in np.array(self.feature_dict[i][j])]
                        #     self.ret_dict[i][nj+'maxd'] = max(curr_d)
                        # elif k=='mind':
                        #     curr_d = [abs(m-n) for m in np.array(self.feature_dict[i][j]) for n in np.array(self.feature_dict[i][j])]
                        #     self.ret_dict[i][nj+'mind'] = min(curr_d)
                        # elif k=='meand':
                        #     curr_d = [abs(m-n) for m in np.array(self.feature_dict[i][j]) for n in np.array(self.feature_dict[i][j])]
                        #     self.ret_dict[i][nj+'meand'] = np.mean(curr_d)/2





        if self.psc_old_feature_dict:
            pd.DataFrame().from_dict(self.psc_old_feature_dict).to_csv(self.input_dict['save_path']+'/T_wacsf_pscori_'+time.strftime("%Y%m%d%H%M%S", time.localtime())+'.csv',index=True,header=True)
        pd.DataFrame().from_dict(self.ret_dict).to_csv(self.input_dict['save_path']+'/T_wacsf_feature_'+time.strftime("%Y%m%d%H%M%S", time.localtime())+'.csv',index=True,header=True)
        pd.DataFrame().from_dict(self.atom_column_neighbor_dict).to_csv(self.input_dict['save_path']+'/T_wacsf_acn_'+time.strftime("%Y%m%d%H%M%S", time.localtime())+'.csv',index=True,header=True)



    def d_vi2dl(self,vl,il,n_bar):
        atom2bar_dict = {}
        bar2atom_list = []
        if type(n_bar)==type([]):
            bar2atom_list = copy.deepcopy(n_bar)
            for k in range(len(bar2atom_list)):
                for l in bar2atom_list[k]:
                    atom2bar_dict[l]=k
        elif type(n_bar)==type(1):
            bar2atom_list = [[] for _ in range(n_bar)]
            if abs(max(vl)-min(vl))>1e-7 and n_bar>1:
                if (max(vl)-min(vl))%(n_bar)<1e-10:
                    buchang = (max(vl)-min(vl))/(n_bar)
                else:
                    buchang = (max(vl)-min(vl))/(n_bar-1)
                for j in range(len(il)):
                    bar_int = math.ceil((vl[j]-min(vl)-0.5*buchang)/buchang)
                    atom2bar_dict[il[j]]=copy.deepcopy(bar_int)
                    bar2atom_list[bar_int].append(il[j])
            else:
                for j in range(len(il)):
                    bar_int = 0
                    atom2bar_dict[il[j]]=copy.deepcopy(bar_int)
                    bar2atom_list[bar_int].append(il[j])
        else:
            print("n_bar ERROR    ",n_bar)
        return atom2bar_dict,bar2atom_list


