import numpy as np
import pandas as pd
from multiprocessing import Pool
import re
import time
import os
import copy
import sys
from scipy.spatial import distance_matrix
import gc
import math
import matplotlib.pyplot as plt

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from nd_tools import sl2bar

class PMGLOAD(object):
    def __init__(self,load_dict = {
        }):
        """
        laod_dict: 'csv_path','cif_path'，'csv_from_to'
        """
        self.crystal_dict = {}
        crystal_df = pd.read_csv(load_dict['csv_path'],index_col = [0],header=None)
        self.crystal_df = crystal_df
        self.load_dict = load_dict


        if load_dict['csv_from_to']=='all':
            work_list = crystal_df.index.tolist()
        else:
            from_to_re =load_dict['csv_from_to'].split(':')
            if from_to_re[0] and from_to_re[1]:
                work_list = crystal_df.index.tolist()[int(from_to_re[0]):int(from_to_re[1])]
            elif from_to_re[0] and not(from_to_re[1]):
                work_list = crystal_df.index.tolist()[int(from_to_re[0]):]
            elif not(from_to_re[0]) and from_to_re[1]:
                work_list = crystal_df.index.tolist()[:int(from_to_re[1])]
        # for i_cp in ['173806']:
        #    # specie ERROR
        for i_cp in work_list:



            curr_cif_name = [
                str(i_cp)+'.cif' if (set(str(i_cp)).difference('0123456789')) or int(i_cp)>27 else '0'*(6-len(str(i_cp)))+str(i_cp)+'.cif'][0]
            # print(curr_cif_name)
            i_s = Structure.from_file(
                load_dict['cif_path']+'/'+curr_cif_name,primitive=True,
                )
            self.crystal_dict[i_cp]={
                'cif_id_str':curr_cif_name.split('.')[0],
                'pmg_obj':copy.deepcopy(i_s),
            }

    def d_neighbor(self,neighbor_dict={}):
        """
        neighbor_dict: 'cur_rad'
                       'return_mode'
        """
        if not('cut_rad' in neighbor_dict):
            print('ERROR    cut_rad')
            return
        else:
            cut_rad = neighbor_dict['cut_rad']
        for i_key in self.crystal_dict:
            i_obj = self.crystal_dict[i_key]['pmg_obj']
            self.crystal_dict[i_key]['neighbor_ret_list']=[]
            for j,j_site in enumerate(i_obj):
                j_neighbor = i_obj.get_all_neighbors(r=cut_rad,sites=[j_site])
                self.crystal_dict[i_key]['neighbor_ret_list'].append(copy.deepcopy(j_neighbor))


        ret_dict = {}
        if  neighbor_dict['return_mode']=='atom_coords':
            for i_key in self.crystal_dict:
                i_obj = self.crystal_dict[i_key]['pmg_obj']
                for j,j_site in enumerate(i_obj):
                    ret_key = tuple([i_key,j])
                    if not(ret_key in ret_dict):
                        ret_dict[ret_key]={}
                    ret_dict[i_key][j]['self_coords']=[[re.split(r'[^a-z^A-Z]',j_site.species_string)[0],list(j_site.coords)]]
                    ret_dict[ret_key]['neighbor_coords'] =[]
                    for k in self.crystal_dict[i_key]['neighbor_ret_list'][j]:
                        for l in k:
                            ret_dict[i_key][j]['neighbor_coords'].append([re.split(r'[^a-z^A-Z]',l.species_string)[0],list(l.coords)])
                # for j in ret_dict:
                #     print(j)
                #     for k in ret_dict[j]:
                #         print(k)
                #         print(ret_dict[j][k])
                #         print()
                #     print()
        elif  neighbor_dict['return_mode']=='ca_coords':
            for i_key in self.crystal_dict:
                i_obj = self.crystal_dict[i_key]['pmg_obj']
                ret_dict[i_key]={}
                for j,j_site in enumerate(i_obj):
                    ret_dict[i_key][j]={}
                    # ret_dict[i_key][j]['self_coords']=[[j_site.specie.name,list(j_site.coords)]]
                    ret_dict[i_key][j]['self_coords']=[re.split(r'[^a-z^A-Z]',j_site.species_string)[0],list(j_site.coords)]
                    ret_dict[i_key][j]['neighbor_coords'] =[]
                    for k in self.crystal_dict[i_key]['neighbor_ret_list'][j]:
                        for l in k:
                            # ret_dict[i_key][j]['neighbor_coords'].append([l.specie.name,list(l.coords)])
                            ret_dict[i_key][j]['neighbor_coords'].append([re.split(r'[^a-z^A-Z]',l.species_string)[0],list(l.coords)])
        elif  neighbor_dict['return_mode']=='ca_species':
            for i_key in self.crystal_dict:
                i_obj = self.crystal_dict[i_key]['pmg_obj']
                ret_dict[i_key]={}
                for j,j_site in enumerate(i_obj):
                    ret_dict[i_key][j]={}
                    ret_dict[i_key][j]['self_species']=re.split(r'[^a-z^A-Z]',j_site.species_string)[0]
                    ret_dict[i_key][j]['neighbor_species'] =[]
                    for k in self.crystal_dict[i_key]['neighbor_ret_list'][j]:
                        for l in k:
                            ret_dict[i_key][j]['neighbor_species'].append(re.split(r'[^a-z^A-Z]',l.species_string)[0])



        return ret_dict

    def d_crystal_system(self,para_dict):
        """
           
        "orthorhombic","tetragonal",
        "triclinic","hexagonal",
        "monoclinic",
        "trigonal",
        "cubic"


        'combination_filter':[[str,str],]
        'save_path':str
        'f_name':str_
        """
        cs_dict = {}
        for i_key in self.crystal_dict:
            print(i_key)
            i_obj = self.crystal_dict[i_key]['pmg_obj']
            sga =SpacegroupAnalyzer(i_obj)
            ics = sga.get_crystal_system()
            if not(ics in cs_dict):
                cs_dict[ics]=pd.DataFrame()
            cs_dict[ics]=pd.concat([cs_dict[ics],self.crystal_df.loc[i_key,:]],axis=1,sort=False)
        # for i in cs_dict:
        #     print(i)
        #     print(cs_dict[i])
        ret_dict={}
        combined_list=[]
        else_list = ["cubic"]
        for i in para_dict['combination_filter']:
            curr_df=pd.DataFrame()
            curr_name = ''
            for j in i:
                curr_name+=j+'_'
                combined_list.append(j)
                curr_df=pd.concat([curr_df,cs_dict[j].T],axis=0,sort=False)
            ret_dict[curr_name]=copy.deepcopy(curr_df)
        for i in [i for i in cs_dict.keys() if not(i in combined_list) and not(i in else_list)]:
            ret_dict[i]=cs_dict[i].T
        # for i in ret_dict:
        #     print(i)
        #     print(ret_dict[i])
        #     print(ret_dict[i].columns.tolist())
        for i in ret_dict:
            ret_dict[i].to_csv(para_dict['save_path']+'/'+para_dict['f_name']+i+'_.csv',header=None)
            print(ret_dict[i])
            curr_delta_n = ret_dict[i].loc[:,6].values.tolist()
            # d_ml2sl_bar(curr_delta_n,bar_para={
            #     'n_bar':20,
            #     'fig_show':para_dict['save_path'],
            #     'fig_name':para_dict['f_name']+i,
            # })
        return ret_dict

class BASICINFO(object):
    def __init__(self,bi_dict={}):
        pass

    def load_element_in(self,csv_dict):
        """
        'ele_path_str'
        """
        element_col = [i for i in [
            'index',
            # 'name',
            'mass','period_num','group_num','radui',
            'ele_negativity',
            's_valence_ele_num','p_valence_ele_num','d_svalence_ele_num','f_valence_ele_num',
            's_unfilled_state','p_unfilled_state','d_unfilled_state','f_unfilled_state',
            'ionization_energy','ele_affinity','meltingpoint','boilingpoint','density','polarizability'
            ]]
        self.element_df = pd.read_csv(csv_dict['ele_path_str'],index_col = [1],names=element_col)
        self.element_df['1'] = [1 for _ in self.element_df.index.tolist()]
        self.element_df['n'] = [i+1 for i in range(len(self.element_df.index.tolist()))]
        self.element_df['Z'] = [i for i in self.element_df.loc[:,'index']]

def peratom(input_list):
    for h in input_list:

        h_key = tuple([str(i) for i in h[:10]])

        p_eta = h[0]
        p_miu = h[1]
        p_rc = h[2]
        
        p_zeta1 = h[3]
        p_lambda1 = h[4]

        p_zeta2 = h[5]
        p_lambda2 = h[6]
        p_axis = h[7]

        p_h = h[8]
        p_ap = h[9]
        m_ap_jk = h[10]
        

        save_path = h[11]
        work_end = h[12]
        ca_mode = h[13]

        ano_ori_dict = h[14]
        
        for g in ano_ori_dict:
            print('peratom g    ',g,os.getpid())
            if ca_mode['ca']=='c':
                curr_name =  'of__'+'_'.join(list(h_key))+'__'+\
                        str(g)+'__all__wacsf.npz'
                if (curr_name in [j for j in os.walk(save_path)][0][-1]):
                    print("In!    ",curr_name)
                    continue
                ret_dict = {}
                if not(h_key in ret_dict):
                    ret_dict[h_key]={}
                if not(g in ret_dict[h_key]):
                    ret_dict[h_key][g]={}

            for f in ano_ori_dict[g]:

                if ca_mode['ca']=='ca':
                    curr_name =  'of__'+'_'.join(list(h_key))+'__'+\
                            str(g)+'__'+str(f)+'__wacsf.npz'
                    if (curr_name in [j for j in os.walk(save_path)][0][-1]):
                        print("In!    ",curr_name)
                        continue
                    ret_dict = {}
                    if not(h_key in ret_dict):
                        ret_dict[h_key]={}
                    if not(g in ret_dict[h_key]):
                        ret_dict[h_key][g]={}




                if not(f in ret_dict[h_key][g]):
                    ret_dict[h_key][g][f]={}
                n1_coords = np.array([ano_ori_dict[g][f]['self_coords'][1]]+[j[1] for j in ano_ori_dict[g][f]['neighbor_coords']])
                n1_atom = [ano_ori_dict[g][f]['self_coords'][0]]+[j[0] for j in ano_ori_dict[g][f]['neighbor_coords']]
                dm = distance_matrix(n1_coords,n1_coords)
                ret_dict[h_key][g][f]['dm']=copy.deepcopy(dm)
                if 'dm' in work_end:
                    continue


                for j in range(1):
                    angm =[]
                    
                    for k in range(len(n1_coords)):
                        if ca_mode['cal']=='ijka':
                            for l in range(len(n1_coords)):
                                if j==k or j==l or k==l:
                                    continue
                                
                                ap_k = m_ap_jk[n1_atom[k]]
                                ap_l = m_ap_jk[n1_atom[l]]
                                if p_h=='+':
                                    wm=ap_l+ap_k
                                elif p_h=='-':
                                    wm=abs(ap_l-ap_k)
                                elif p_h=='x':
                                    wm=ap_l*ap_k
                                elif p_h=='-1':
                                    wm=ap_l+ap_k/(ap_l*ap_k)
                                else:
                                    return
                                jk = n1_coords[k]-n1_coords[j]
                                jl = n1_coords[l]-n1_coords[j]
                                kl = n1_coords[l]-n1_coords[k]
                                expeq = 1
                                feq = 1
                                for m in [jk,jl,kl]:
                                    expeq=expeq*np.exp(-p_eta*(np.linalg.norm(m)-p_miu)**2)
                                    feq=feq*0.5*(np.cos((np.pi*np.linalg.norm(m))/p_rc)+1)
                                jk = [jk/np.linalg.norm(jk) if float(np.linalg.norm(jk)) else jk][0]
                                jl = [jl/np.linalg.norm(jl) if float(np.linalg.norm(jl)) else jl][0]
                                kl = [kl/np.linalg.norm(kl) if float(np.linalg.norm(kl)) else kl][0]
                                cos_ijk = np.dot(jk,jl)
                                if p_axis:
                                    ab_nn = np.linalg.norm(p_axis)*np.linalg.norm(jk+jl)
                                    cos_ab=[np.dot(p_axis,jk+jl)/ab_nn if float(ab_nn) else 0][0]
                                zeta1eq = (1+p_lambda1*cos_ijk)**p_zeta1
                                if p_axis:
                                    zeta2eq = (1+p_lambda2*cos_ab)**p_zeta2
                                    angm.append(wm*expeq*feq*zeta1eq*zeta2eq)
                                else:
                                    angm.append(wm*expeq*feq*zeta1eq)
                        if ca_mode['cal']=='ija':
                            if j==k:
                                continue
                            ap_k = m_ap_jk[n1_atom[k]]
                            wm = ap_k
                            jk = n1_coords[k]-n1_coords[j]
                            expeq=np.exp(-p_eta*(np.linalg.norm(jk)-p_miu)**2)
                            feq=0.5*(np.cos((np.pi*np.linalg.norm(jk))/p_rc)+1)
                            jk = [jk/np.linalg.norm(jk) if float(np.linalg.norm(jk)) else jk][0]
                            if p_axis:
                                ab_nn = np.linalg.norm(p_axis)*np.linalg.norm(jk)
                                cos_ab=[np.dot(p_axis,jk)/ab_nn if float(ab_nn) else 0][0]
                            if p_axis:
                                zeta1eq = (1+p_lambda1*cos_ab)**p_zeta1
                                angm.append(wm*expeq*feq*zeta1eq)
                            else:
                                angm.append(wm*expeq*feq)
                    ret_dict[h_key][g][f]['wacsf']=copy.deepcopy((2**(1-p_zeta1))*sum(angm))

                if ca_mode['ca']=='ca':
                    for i in [j[4:] for j in work_end if j[:4]=='npz_']:
                        np.savez(save_path+'/'+curr_name,ret_dict[h_key][g][f][i])
                    sys.stdout.flush()
                    del ret_dict
                    gc.collect()

            if ca_mode['ca']=='c':
                for i in [j[4:] for j in work_end if j[:4]=='npz_']:
                    np.savez(save_path+'/'+curr_name,[ret_dict[h_key][g][f][i] for f in ano_ori_dict[g]])
                sys.stdout.flush()
                del ret_dict
                gc.collect()
    return 1

def acsf_job(job_dict,ano_dict,element_df):
    """
        'p_eta'         [float]
        'p_miu'         [float]
        'p_rc'          [float] DON'T more than 1 

        'p_zeta1'       [float,,,]
        'p_lambda1'     [1,-1]

        'p_zeta2'       [float],['p_zeta1']
        'p_lambda2'     [1,-1],['p_lambda1']
        'p_axis'        ['' [float,float,float] 'crystal_%f_%f_%f',]

        'p_h'           ['+','-','x','^-1'] --- > sl2f()
        'p_ap'          [BASICINFO.element_df.columns.tolist()[?]]
                        --- > 'm_ap_jk':{}
        'save_path'     [str],
        'work_output'   ['npz_dm','npz_wacsf']
        'ca_mode'       [{'cal':'ijka' 'ija','ca':'c' 'ca'}]
        'ano_dict'      'all' list(ano_dict.keys())[:],

        'pcl_para'      calculation mode of per job
                        name  para cry  center
                        'pca' 1    1    1
                        'pc'  1    1    all
                        'p'   1    all  all
                        'c'   all  1    all
        'job_int'       'ori',int, pcl_list output groups
        'job_path'      output mode of pcl_list
                        'ret_list'
                        'path_%str'
        'job_name_start'int, start int of job files
        'job_content'   {'y_path':str,,,} special para for file generation
    """
    new_ano_list = []
    if len(job_dict['p_rc'])>1:
        print("WARNING rc")
        # return
    if job_dict['ano_dict']!=['all']:
        curr_ano_dict={}
        for i in job_dict['ano_dict']:
            curr_ano_dict[i]=ano_dict[i]
    else:
        curr_ano_dict = ano_dict
    
    if job_dict['pcl_para']=='p':
        new_ano_list.append('all')
    else:
        for i in curr_ano_dict:
            if job_dict['pcl_para'] in ['pca']:
                for j in curr_ano_dict[i]:
                    new_ano_list.append([i,[j]])
            elif job_dict['pcl_para'] in ['pc','c']:
                new_ano_list.append([i,list(curr_ano_dict[i].keys())])
    job_dict['ano_dict'] = new_ano_list

    pcl_list = []
    para_order = [
        'p_eta','p_miu','p_rc',          
        'p_zeta1','p_lambda1',  
        'p_zeta2','p_lambda2','p_axis', 
        'p_h','p_ap',  
        'save_path','work_output','ca_mode','ano_dict',     
    ]
    for i in para_order:
        ori_rl = [copy.deepcopy(pcl_list) if pcl_list else [[]]][0]
        pcl_list = []
        for j in job_dict[i]:
            for k in ori_rl:
                curr_k = copy.deepcopy(k)
                if j in ['p_zeta1','p_lambda1']:
                    curr_k+=[curr_k[-2]]
                elif i=='p_ap':
                    curr_k += [j,element_df.to_dict()[j]]
                elif i=='ano_dict':
                    if j=='all':
                        curr_k+=[curr_ano_dict]
                    else:
                        curr_dict={}
                        curr_dict[j[0]]={}
                        for l in j[1]:
                            curr_dict[j[0]][l]=curr_ano_dict[j[0]][l]
                        curr_k += [curr_dict]
                else:
                    curr_k += [j]
                pcl_list.append(curr_k)
    print('ori len(pcl_list)    ',len(pcl_list))

    ret_list = []
    if job_dict['job_int']=='ori':
        ret_list = pcl_list
    else:
        b2i_list = sl2bar([i for i in range(len(pcl_list))],{'n_bar':job_dict['job_int'],'bar_mode':['int_equ_split'],'return_mode':['b2i_list']})[0]
        for i in b2i_list:
            ret_list.append(pcl_list[min(i):max(i)+1])
    
    if job_dict['job_path']=='ret_list':
        return ret_list
    elif job_dict['job_path'][:5]=='path_':
        job_path = job_dict['job_path'][5:]
        for i,ii in enumerate(b2i_list):
            curr_path = job_path+'/sj__'+str(i+job_dict['job_name_start'])+'.py'
            print(curr_path)
            curr_file = open(curr_path,'w')
            curr_file.write("""from nd_wacsf1p import *
global_rc = """+str(job_dict['p_rc'][0])+"""
pmg_obj = PMGLOAD(load_dict={
    'csv_path':'"""+str(job_dict['job_content']['csv_path'])+"""',
    'cif_path':'"""+str(job_dict['job_content']['cif_path'])+"""',
    'csv_from_to':'"""+str(job_dict['job_content']['csv_from_to'])+"""',
})
ano_dict = pmg_obj.d_neighbor(neighbor_dict={
    'cut_rad':global_rc,
    'return_mode':'ca_coords'
})
print(len(ano_dict))
bi_obj = BASICINFO()
bi_obj.load_element_in(csv_dict={
    'ele_path_str':'./nlo_delta_n/element-in.csv'
})
element_df = bi_obj.element_df
para_combination_list = acsf_job({
    'p_eta':"""+str(job_dict['p_eta'])+""",
    'p_miu':"""+str(job_dict['p_miu'])+""",
    'p_rc':[global_rc],
    'p_zeta1':"""+str(job_dict['p_zeta1'])+""",
    'p_lambda1':"""+str(job_dict['p_lambda1'])+""",
    'p_zeta2':"""+str(job_dict['p_zeta2'])+""",
    'p_lambda2':"""+str(job_dict['p_lambda2'])+""",
    'p_axis':"""+str(job_dict['p_axis'])+""",
    'p_h':"""+str(job_dict['p_h'])+""",
    'p_ap':"""+str(job_dict['p_ap'])+""",

    'save_path':"""+str(job_dict['save_path'])+""",
    'work_output':"""+str(job_dict['work_output'])+""",
    'ca_mode':"""+str(job_dict['ca_mode'])+""",
    'ano_dict':"""+str(job_dict['ano_dict'])+""",
    'pcl_para':'"""+str(job_dict['pcl_para'])+"""',
    'job_int':'ori',
    'job_path':'ret_list',
    'job_name_start':0,
    'job_content':{},
},ano_dict,element_df)
peratom(para_combination_list["""+str(min(ii))+""":"""+str(int(max(ii)+1))+"""])
            """)

        return pcl_list
    else:
        print("ERROR job_dict['job_path']")
        return
    

