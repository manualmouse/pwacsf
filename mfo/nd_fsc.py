from numpy.core.fromnumeric import sort
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import pandas as pd
import numpy as np
import copy
import re
import os
import time
import sys







def ja_stacking_csv(dir_path,dir_ptitle_list,y_csv,file_filter_str,des_def_name,else_dict={},jd_name = 'delta_n',random_state=24):
    for i in dir_path:
        dir_ptitle_name = dir_ptitle_list[0]
        dir_ptitle_path = dir_ptitle_list[1]
        return_list = []
        jd_tot = ''
        for j in os.walk(i):
            for k in j[-1]:

                des_name = re.findall(r'(\S+)_\d+\.\S+',k)[0]
                




                if d_file_filter(file_filter_str,des_def_name,des_name):
                    continue
                print('\n',jd_name)
                print(j[0]+'/'+k)
                return_list.append(j[0]+'/'+k)
                return_list.append(jd_name)
                y_data = copy.deepcopy(y_csv[jd_name])
                y_data.index = [str(l) for l in y_data.index.tolist()]
                
                x_data = pd.read_csv(j[0]+'/'+k,index_col=[0]).T
                x_data = x_data.fillna(0)
                x_data = x_data.loc[y_data.index.tolist(),:]
               

                data_layer1 = []
                data_l1name = []

                # data_layer1 plan divide 2 kind : ShuffleSplit[0], StratifiedShuffleSplit[1]

                # [][][][][][][][][][][]    data_layer1 ShuffleSplit
                x_train_ori,x_test_ori,y_train_ori,y_test_ori = train_test_split(x_data,y_data,test_size=0.2,random_state =random_state)
                if 'extra_test_x' in else_dict and 'extra_test_y' in else_dict:
                    # print(x_train_ori,'\n',x_test_ori,'\n',y_train_ori,'\n',y_test_ori)
                    x_test_extra = pd.read_csv(else_dict['extra_test_x'],index_col=[0]).T
                    x_test_selected = []
                    for l in x_train_ori.columns.tolist():
                        if l in x_test_extra:
                            x_test_selected.append(l)
                    x_test_extra=x_test_extra.loc[:,x_test_selected]                    
                    x_test_ori = pd.concat([x_test_ori,x_test_extra],sort=False,axis=0)
                    x_test_ori = x_test_ori.fillna(0)
                    # print(x_test_ori)
                    y_test_extra = pd.read_csv(else_dict['extra_test_y'],index_col = [0])
                    y_test_extra = copy.deepcopy(y_test_extra[jd_name])
                    y_test_extra.index = [str(l) for l in y_test_extra.index.tolist()]
                    y_test_ori = pd.concat([y_test_ori,y_test_extra],sort=False,axis=0)
                    if set(x_test_ori.index.tolist())!=set(y_test_ori.index.tolist()):
                        print(set(x_test_ori.index.tolist())-set(y_test_ori.index.tolist()))
                        print(set(y_test_ori.index.tolist())-set(x_test_ori.index.tolist()))
                        print('set(x_test_ori.index.tolist())!=set(y_test_ori.index.tolist())')
                        return
                    x_test_ori = x_test_ori.loc[y_test_ori.index.tolist(),:]
                
                if 'else_feature' in else_dict:
                    else_df = pd.read_csv(else_dict['else_feature'],index_col=[0]).T
                    if set(x_train_ori.index.tolist()).difference(set(else_df.index.tolist())) or set(x_test_ori.index.tolist()).difference(set(else_df.index.tolist())):
                        print("index ERROR")
                        return
                    x_train_ori = pd.concat([x_train_ori,else_df.loc[x_train_ori.index.tolist(),:]],sort=False,axis=1)
                    x_test_ori = pd.concat([x_test_ori,else_df.loc[x_test_ori.index.tolist(),:]],sort=False,axis=1)                
                else:
                    else_df=pd.DataFrame()
                
                x_train_index_ori = copy.deepcopy(x_train_ori.index.tolist())
                x_test_index_ori = copy.deepcopy(x_test_ori.index.tolist())
                x_columns_ori = [re.sub(r'[\(\)\[\]\,]','*',l) for l in x_train_ori.columns.tolist()]

                curr_train_ori = copy.deepcopy(x_train_ori)
                curr_test_ori = copy.deepcopy(x_test_ori)
                curr_train_ori.columns = [re.sub(r'[\(\)\[\]\,]','*',l) for l in curr_train_ori.columns.tolist()]
                curr_test_ori.columns = [re.sub(r'[\(\)\[\]\,]','*',l) for l in curr_test_ori.columns.tolist()]




                mms = MinMaxScaler()
                mms.fit(x_train_ori)
                x_train_ori = mms.transform(x_train_ori)
                x_test_ori = mms.transform(x_test_ori)
                x_train_ori = pd.DataFrame(data = x_train_ori,columns = x_columns_ori,index = x_train_index_ori)
                x_test_ori = pd.DataFrame(data = x_test_ori,columns = x_columns_ori,index = x_test_index_ori)

                
                x_train_std = np.std(x_train_ori,ddof=1)
                if 'std' in else_dict:
                    curr_std= else_dict['std']
                else:
                    curr_std = 0.15
                x_train_ori = x_train_ori.loc[:,[l for l in x_train_std.index.tolist() if x_train_std[l]>curr_std or l in else_df.columns.tolist()]]
                x_test_ori = x_test_ori.loc[:,[l for l in x_train_ori.columns.tolist()]]
                if 'corrn85' in else_dict:
                    print("corrn85")
                    work_y = pd.DataFrame(data = y_train_ori,index=x_train_ori.index.tolist(),columns=[jd_name])
                    work_xy = pd.concat([x_train_ori,work_y],axis=1,sort=False)
                    work_corr = work_xy.corr(method='pearson')
                    print('finish corrn85 matrix')
                    work_c_index= [l for l in work_corr.index.tolist() if not(l==jd_name)]
                    work_c_corr = {}
                    for l in work_c_index:
                        work_c_corr[l] = abs(work_corr.loc[l,jd_name])
                    # print(work_c_corr)
                    work_c_sorted = sorted(work_c_corr.items(),key = lambda kv:(kv[1],kv[0]),reverse=True)
                    work_c_rank = [l[0] for l in work_c_sorted]
                    work_c_del = []
                    print('finish corrn85 sorting')
                    for l in range(len(work_c_rank)):
                        if len(work_c_rank)>11 and int(l%int(len(work_c_rank)/10))==0.0 and int(l/int(len(work_c_rank)/10)):
                            print("corrn85 l ",l,round(l/len(work_c_rank),2))
                        curr_l = work_c_rank[l]
                        if not(curr_l in work_c_del):
                            curr_del = [m for m in work_corr.loc[(work_corr[curr_l]>=0.85)|(work_corr[curr_l]<=-0.85)].index.tolist() if (
                                (m!=curr_l) and
                                (m in work_c_rank[l+1:]) and 
                                not(m in work_c_del) and 
                                not(m==jd_name) and 
                                not(m in else_df.columns.tolist()))]
                            work_c_del = list(set(work_c_del).union(set(curr_del)))
                    x_train_ori = x_train_ori.drop(work_c_del,axis=1)
                    x_test_ori = x_test_ori.drop(work_c_del,axis=1)




                if 'imb' in else_dict:
                    y_train_dict = dict(y_train_ori.value_counts())
                    if set(y_train_dict.keys())=={0,1} and abs(y_train_dict[0]-y_train_dict[1])/abs(y_train_dict[0]+y_train_dict[1])>0.1:
                        for l in else_dict['imb']:
                            if l=='smote':
                                sampler = SMOTE(random_state=10)
                            elif l=='adasyns':
                                sampler=ADASYN(random_state=10)
                            elif l=='rando':
                                sampler=RandomOverSampler(random_state=10)
                            x_train_imb,y_train_imb = sampler.fit_resample(x_train_ori,y_train_ori)
                            x_train_imb.index=[l+'_'+str(m) for m in x_train_imb.index.tolist()]
                            y_train_imb.index=[l+'_'+str(m) for m in y_train_imb.index.tolist()]
                            data_l1name.append('dl1ss'+l)
                            data_layer1.append([x_train_imb,x_test_ori,y_train_imb,y_test_ori])
                    else:
                        print("WARNING imb NOT WORK")
                        x_train_imb= x_train_ori
                        y_train_imb= y_train_ori
                        data_l1name.append('dl1ss')
                        data_layer1.append([x_train_imb,x_test_ori,y_train_imb,y_test_ori])
                    
                else:
                    data_layer1.append([x_train_ori,x_test_ori,y_train_ori,y_test_ori])
                    data_l1name.append('dl1ss')

                # [][][][][][][][][][][]    data_layer1 StratifiedShuffleSplit
                # dataset_split_sss = sss_split(x_data,y_data,0.2,24,jd_name)
                # data_layer1.append(dataset_split_sss)

                for l in range(len(data_layer1)):
                    return_list.append(data_l1name[l])
                    des_operator_dict_ori={
                        'des_corr':42,
                        'des_random':1379,

                        'des_fscorr':42,
                        'des_nnrandom':1379,


                        }
                    des_operator_dict = {}
                    if des_def_name in des_operator_dict_ori.keys():
                        des_operator_dict[des_def_name] = des_operator_dict_ori[des_def_name]

                    des_operatot_dk_list = list(des_operator_dict.keys())

                    # data_layer2 plan divide 3 kind : ShuffleSplit , Bagging , StratifiedShuffleSplit
                    #     based on every random seed len(des_operator_dict) divide train_re <-> test_re 



                    for n in range(len(des_operatot_dk_list)):
                        return_list.append(des_operatot_dk_list[n])
                        xy_ori_namel1 =dir_ptitle_name+"_"+data_l1name[l]


                        if 'ONLY OUTPUT: x_train_ori' in else_dict:
                            curr_train_ori.to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_"+des_name+"_x_train_ori_.csv",index=True,header=True)
                            curr_train_ori.T.to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_"+des_name+"_x_train_oriT_.csv",index=True,header=True)
                            curr_test_ori.to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_"+des_name+"_x_test_ori_.csv",index=True,header=True)
                            curr_test_ori.T.to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_"+des_name+"_x_test_oriT_.csv",index=True,header=True)
                        else:
                            curr_train_ori.to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_"+des_name+"_x_train_ori_.csv",index=True,header=True)
                            curr_train_ori.T.to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_"+des_name+"_x_train_oriT_.csv",index=True,header=True)
                            curr_test_ori.to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_"+des_name+"_x_test_ori_.csv",index=True,header=True)
                            curr_test_ori.T.to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_"+des_name+"_x_test_oriT_.csv",index=True,header=True)

                            data_layer1[l][0].to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_"+des_name+"_x_train_re_.csv",index=True,header=True)
                            data_layer1[l][1].to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_"+des_name+"_x_test_re_.csv",index=True,header=True)
                            data_layer1[l][0].T.to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_"+des_name+"_x_train_reT_.csv",index=True,header=True)
                            data_layer1[l][1].T.to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_"+des_name+"_x_test_reT_.csv",index=True,header=True)
                            data_layer1[l][2].to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_y_train_rew_.csv",index=True,header=True)
                            data_layer1[l][3].to_csv(dir_ptitle_path+"/"+xy_ori_namel1+"_y_test_re_.csv",index=True,header=True)
                            sys.stdout.flush()
                            sys.stderr.flush()

                            des_operator_obj = DesOperator(
                                    data_layer1[l],
                                    {
                                        "dir_ptitle_name":dir_ptitle_name,
                                        "dir_ptitle_path":dir_ptitle_path,
                                        "jd_name":jd_name,
                                        "jd_tot":jd_tot,
                                        "des_name":des_name,
                                        "data_l1name[l]":data_l1name[l],
                                        "des_operatot_dk_list[n]":des_operatot_dk_list[n],
                                        "else_dict":else_dict
                                    },
                                    )
                            des_operator_obj.save_csv(['proc','result'])
                            return_log_n = des_operator_obj.save_csv(['log'])
                            return_list+=return_log_n

                    return_list.append("\n")
                    print('\n')
                return_list.append("\n")
                print('\n')
            return_list.append("\n")
            print('\n')
        # print("---------------------------------------------------------------")
        # for j in return_list:
        #     print(j)
        log_o = open(dir_ptitle_path+'/sflog_'+time.strftime("%Y%m%d%H%M%S", time.localtime())+'.txt','w',encoding='utf-8')
        for j in return_list:
            log_o.write(j+"\n")

def sincol_corr(x_df,y_df,jd_name):
    ret_df = pd.DataFrame()
    for i in x_df.columns.tolist():
        curr_df = pd.concat([x_df.loc[:,i],y_df.loc[:,jd_name]],axis=1,sort=False)
        curr_corr = curr_df.corr(method='pearson')
        if not(curr_corr.empty):
            ret_df.loc[i,jd_name] = curr_corr.loc[i,jd_name]
        else:
            print("WARNING curr_corr is empty")
            print(curr_corr)
            ret_df.loc[i,jd_name] = 0.0
    return ret_df
class DesOperator(object):
    def __init__(self,input_data,jindu_para):
        
        self.corr_des_num_list = [2000,1500,1000,800,500,200,150,100,80,70,60,55,50,45,40,35,30,25,20,15,10]
        self.ret_df_dict = {}
        self.ret_log_list = []
        self.ret_proc_dict = {}

        self.x_train_re = input_data[0]
        # self.x_test_re = input_data[1]
        self.y_train_re = input_data[2]
        # self.y_test_re = input_data[3]
        self.jindu_para = jindu_para


        if 'else_feature' in self.jindu_para['else_dict']:
            else_df = pd.read_csv(self.jindu_para['else_dict']['else_feature'],index_col=[0]).T
            self.x_train_else = copy.deepcopy(self.x_train_re.loc[:,[i for i in self.x_train_re.columns.tolist() if (i in else_df.columns.tolist())]])
            self.x_train_re = self.x_train_re.loc[:,[i for i in self.x_train_re.columns.tolist() if not(i in else_df.columns.tolist())]]
        else:
            self.x_train_else = pd.DataFrame()



        print("----[DesOperator(object) : __init__()]----")
        self.ret_log_list.append("----[DesOperator(object) : __init__()]----")
        def_para = self.jindu_para['des_operatot_dk_list[n]']
        self.ret_proc_dname_str = self.jindu_para["dir_ptitle_name"]+'_'+self.jindu_para["jd_name"]+'_'+self.jindu_para["des_name"]+'_'+\
            self.jindu_para["data_l1name[l]"]+'_'+self.jindu_para["des_operatot_dk_list[n]"]
        # print(self.ret_proc_dname_str,"    self.ret_proc_dname_str")



        if def_para == 'des_corr':
            self.des_corr()
        elif def_para == 'des_random':
            self.des_random()
        elif def_para == 'des_fscorr':
            self.des_fscorr(self.jindu_para['else_dict']['corr_list'])
        elif def_para == 'des_nnrandom':
            self.des_nnrandom(self.jindu_para['else_dict']['n_list'],self.jindu_para['else_dict']['seed_list'])
        else:
            pass
            # print("class DesOperator(object):"+"\n"+"def __init__(self,input_data):")
            # return
    

    def des_fscorr(self,corr_list):
        print("----[DesOperator(object) : des_fscorr()]----")
        self.ret_log_list.append("----[DesOperator(object) : des_fscorr()]----")
           
        work_x_train_re = copy.deepcopy(self.x_train_re)
        work_y_train_re = copy.deepcopy(self.y_train_re)

        # print(type(work_y_train_re))
        work_y_train_re = pd.DataFrame(work_y_train_re,columns=[self.jindu_para["jd_name"]])

        # work_x_train_r0 = pd.concat([work_x_train_re,work_y_train_re],axis=1,sort=False)
        # print()
        # work_x_train_re_cor0 = work_x_train_r0.corr(method='pearson')
        # print(len(work_x_train_re_cor0.index.tolist()))

        work_x_train_re_cor0=sincol_corr(work_x_train_re,work_y_train_re,self.jindu_para["jd_name"])
        work_x_train_re_cas = abs(work_x_train_re_cor0).sort_values(by=self.jindu_para["jd_name"],inplace=False,ascending=False)
        # print(work_x_train_re_cas)
        # print(work_x_train_re)
        self.ret_proc_dict[self.ret_proc_dname_str] = work_x_train_re_cas

        for i in corr_list:
            x_re_selected =work_x_train_re_cas.index.tolist()[:i]
            if not(self.x_train_else.empty):
                self.ret_df_dict['fscorr-'+str(i)]=pd.concat([work_x_train_re.loc[:,x_re_selected],self.x_train_else],axis=1,sort=False)
            else:
                self.ret_df_dict['fscorr-'+str(i)]=work_x_train_re.loc[:,x_re_selected]
            curr_ps=min(work_x_train_re_cas.loc[x_re_selected,:])
            self.ret_log_list.append(str(curr_ps)+"    curr_ps")
            x_re_sstr = ''
            for j in x_re_selected:
                x_re_sstr+=j+"    "
            self.ret_log_list.append(str(len(x_re_selected))+"    "+x_re_sstr)
            print(curr_ps,"    curr_ps")
            print(len(x_re_selected),"    len(x_re_selected)")

    def des_nnrandom(self,n_list,seed_list):
        rd_work_x_train_re_col = self.x_train_re.columns.tolist()
        rd_x_train_re_index = [i for i in range(len(rd_work_x_train_re_col))]
        for i in n_list:
            if i>len(rd_work_x_train_re_col):
                continue
            for j in seed_list:
                curr_index_list = shuffle(rd_x_train_re_index,random_state=j,n_samples=i)
                rd_curr_index_str = ''
                for k in curr_index_list:
                    rd_curr_index_str+=(str(k)+"    ")
                self.ret_log_list.append(rd_curr_index_str+"    rd_curr_index_str")
                if not(self.x_train_else.empty):
                    self.ret_df_dict[str('nnrandom-'+str(j)+'-'+str(i))]=pd.concat([copy.deepcopy(self.x_train_re.iloc[:,curr_index_list]),self.x_train_else],axis=1,sort=False)
                else:
                    self.ret_df_dict[str('nnrandom-'+str(j)+'-'+str(i))]=copy.deepcopy(self.x_train_re.iloc[:,curr_index_list])

    def save_csv(self,save_para):
        if 'log' in save_para:
            return self.ret_log_list
        if 'proc' in save_para:
            for i in self.ret_proc_dict:
                self.ret_proc_dict[i].to_csv(self.jindu_para["dir_ptitle_path"]+'/'+i+"_proc_.csv",index=True,header=True)
        if 'result' in save_para:
            for i in self.ret_df_dict:
                self.ret_df_dict[i].to_csv(self.jindu_para["dir_ptitle_path"]+'/'+self.ret_proc_dname_str+"_"+str(i)+"_x_train_rew_.csv",index=True,header=True)



def ja_stacking_fenliu(dir_path,dir_ptitle_list,y_csv,file_filter_str,des_def_name,else_dict={},jd_name = '',random_state=24):

    ja_stacking_csv(dir_path,dir_ptitle_list,y_csv,file_filter_str,des_def_name,else_dict,jd_name,random_state=random_state)



def d_file_filter(file_filter_str,des_def_name,des_name):
    """
    Matching:   'nlond'
                    'T_wacsf_feature'
    """    
    is_ret_true = 0


    if file_filter_str=='nlond':
        if not 'T_wacsf_feature' in des_name:
            is_ret_true = 1




    if is_ret_true:
        return True
    else:
        return False

if '__main__'==__name__:
    pass


