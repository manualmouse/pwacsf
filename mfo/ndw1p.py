from nd_wacsf1p import *

global_rc =5.0
# global_rc =3.0

load_dict={
    'csv_path':'./348/crystal-in-train230223.csv',
    'cif_path':'./348/cif',
    'csv_from_to':'0:',
}

pmg_obj = PMGLOAD(load_dict=load_dict)
ano_dict = pmg_obj.d_neighbor(neighbor_dict={
    'cut_rad':global_rc,
    'return_mode':'ca_coords'
})
bi_obj = BASICINFO()
bi_obj.load_element_in(csv_dict={
    'ele_path_str':'./element-in.csv'
})
element_df = bi_obj.element_df

para_combination_list = acsf_job({
    'p_eta':[0.01,0.1],
    'p_miu':[0],
    'p_rc':[global_rc],
    'p_zeta1':[1,8,16],
    'p_lambda1':[-1,1],
    'p_zeta2':[8,16],
    'p_lambda2':[-1,1],
    'p_axis':[[0,0,1],[1,0,0],[0,1,0],[]],
    'p_h':['x'],
    'p_ap':['ele_affinity',],

    'save_path':['./feature_ang-ea'],
    'work_output':[['npz_wacsf']],
    'ca_mode':[{'cal':'ijka','ca':'c'}],
    'ano_dict':['all'],

    'pcl_para':'p',
    'job_int':96,
    'job_path':'path_./job_ang-ea',
    'job_name_start':0,
    'job_content':load_dict,
},ano_dict,element_df)