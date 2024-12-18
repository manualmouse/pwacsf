import pandas as pd
from nd_wacsf1p import *
from nd_wacsf2p import npzloader
from nd_wacsf2p import HGGEN


if __name__=='__main__':
    y_path = './348/crystal-in-train230223.csv'
    global_rc = 5.0
    y_csv_ori=pd.read_csv(y_path,index_col = [0],header=None)
    y_csv = pd.DataFrame(data =y_csv_ori.iloc[:,5].values.tolist(),index = y_csv_ori.index.tolist(),columns=['delta_n'])
    
    load_dict={
        'csv_path':y_path,
        'cif_path':'./348/cif',
        'csv_from_to':'0:',
    }
    pmg_obj = PMGLOAD(load_dict=load_dict)
    ano_dict = pmg_obj.d_neighbor(neighbor_dict={
        'cut_rad':global_rc,
        'return_mode':'ca_coords'
    })
    cas_dict = pmg_obj.d_neighbor(neighbor_dict={
            'cut_rad':global_rc,
            'return_mode':'ca_species'
        })

    bi_obj = BASICINFO()
    bi_obj.load_element_in(csv_dict={
        'ele_path_str':'./element-in.csv'
    })
    element_df = bi_obj.element_df


    pcl_list = acsf_job({
        'p_eta':[0.01,0.1],
        'p_miu':[0],
        'p_rc':[global_rc],
        'p_zeta1':[1,8,16],
        'p_lambda1':[-1,1],
        'p_zeta2':[8,16],
        'p_lambda2':[-1,1],
        'p_axis':[[0,0,1],[1,0,0],[0,1,0],[]],
        'p_h':['x'],
        'p_ap':['ionization_energy','radui','ele_affinity'],


        'save_path':['./feature_230322'],
        'work_output':[['npz_wacsf']],
        'ca_mode':[{'cal':'ijka','ca':'c'}],
        'ano_dict':['all'],

        'pcl_para':'p',
        'job_int':'ori',
        'job_path':'ret_list',
        'job_name_start':0,
        'job_content':{},
    },ano_dict,element_df)
    oao_dict = [
        {7:['tuple',[(0,1,0),(1,0,0),(0,0,1)]],12:['str',['wacsf']],0:['float_1e-3',['0.01','0.1']],9:['str',['ele_affinity']]},
                    {5:['int',[8]],6:['int',[1]],7:['tuple',[()]],12:['str',['wacsf']],0:['float_1e-3',['0.01','0.1']],9:['str',['ele_affinity']]}
    ]
    pwca_dict = npzloader('./feature_230322',pcl_list,ano_dict,oao_dict)




    hg_dict = {}

    hg_key_list = ['ele_affinity']
    
    for i in hg_key_list:
            hg_dict['cap_'+i]=[['global',20]]



    print(hg_dict)
    hg_obj = HGGEN({
        'pwca_dict':pwca_dict,
        'pmg_dict':{
            'csv_path':y_path,
            'cif_path':'./348/cif',
            'csv_from_to':'0:',
            'cut_rad':global_rc,
        },
        'bi_dict':{
            'ele_path_str':'./element-in.csv'
        },
        'hg_dict':hg_dict,
        'cal_para':['+','max','min','mean','std'],
        'save_path':'./feature_out_0322_20_230223_52ea',
        'atom_mean':['1']
    })




