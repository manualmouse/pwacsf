"""
WARNING: To avoid coding problems, only English is allowed here!
"""
import copy
import numpy as np
import re
import math
import pandas as pd


def sl2ml(a):
    """
    l1,l2,l3,l4 -> l1*l2*l3*l4
    [[a],[b,c],[[c],d]] --- > [a,b,[c]]
                              [a,c,[c]]
                              [a,b,d]
                              [a,c,d]
    """
    b = []
    for i,ii in enumerate(a):
        ori_b = [copy.deepcopy(b) if b else [dict()]][0]
        b=[]
        for j in ii:
            for k in ori_b:
                curr_k = copy.deepcopy(k)
                curr_k[i]=j
                b.append(curr_k)
    return b

def ml2sl(input_list):
    """
    unfolding a list to a single dimonsion list
    [a,[b,[[[c]],d],e,f],g] --- > [a,b,c,d,e,f,g]
    """
    sl = []
    def dd(l):
        for i in l:
            if type(i) in [type([]),type(np.ndarray([]))]:
                dd(i)
            else:
                sl.append(copy.deepcopy(i))
    dd(list(input_list))
    return sl

def sl2bar(vl,para_dict):
    """
    vl: single dimonsion list, int or float
    para_dict:
        'n_bar': int
        'bar_mode': [
            # 'percentile',
            'min-0.5' 'max-min' 'int_equ_split']
        'il': []     optional matching with vl
        'return_mode': ['ret_name']
    --------
    return: [i2b_dict, b2i_list, v2b_dict, b2v_list, r2n_dict]
    """
    if not('n_bar' in para_dict) or not(para_dict['n_bar']) or para_dict['n_bar']<1:
        print("ERROR sl2bar para_dict['n_bar']")
        return
    else:
        n_bar = para_dict['n_bar']
    if not('il' in para_dict):
        il = [i for i in range(len(vl))]
    else:
        if len(para_dict['il'])!=len(vl):
            print("ERROR sl2bar len(para_dict['il'])!=len(vl)")
            return 
        else:
            il = para_dict['il']
    
    # if 'percentile' in para_dict['bar_mode']:
    #     vl = [i for i in sl2si(vl,{'sort_para':'min2max','same_index':1e-7})]
    ret_tot_dict = {
        'i2b_dict':{},
        'v2b_dict':{},

        'b2i_list':[],
        'b2v_list':[],

        'b2r_list':[],
        'r2n_dict':{}
    }
    if abs(max(vl)-min(vl))<1e-7:
        n_bar=1
    for i in ret_tot_dict:
        if '_list' in i:
            ret_tot_dict[i]=[[] for _ in range(n_bar)]
    


    if 'min-0.5' in para_dict['bar_mode']:
        buchang = abs(max(vl)-min(vl))/(n_bar-1)
    elif 'max-min' in para_dict['bar_mode']:
        buchang = abs(max(vl)-min(vl))/(n_bar)
    elif 'int_equ_split' in para_dict['bar_mode']:
        buchang = int(abs(max(vl)-min(vl))/n_bar)





    for i,ii in enumerate(vl):
        # 'max-min' 'int_equ_split'
        if n_bar==1:
            bar_int = 0
            curr_range = str(round(min(vl),3))+'_'+str(round(max(vl),3))
        elif 'min-0.5' in para_dict['bar_mode']:
            bar_int = math.ceil((ii-min(vl)-0.5*buchang)/buchang)
            curr_range=''
        elif 'max-min' in para_dict['bar_mode']:
            bar_int = [int((ii-min(vl))/buchang) if ii!=max(vl) else int((ii-min(vl))/buchang-1)][0]
            curr_range=''
        elif 'int_equ_split' in para_dict['bar_mode']:
            bc1=buchang+1
            bc1_len = len(vl)-buchang*n_bar
            if ii<bc1_len*bc1:
                bar_int=int(ii/bc1)
                curr_range = str(bc1*bar_int)+'_'+str((bar_int+1)*bc1)
            else:
                bar_int=int((ii-bc1_len*bc1)/buchang+bc1_len)
                curr_range = str(buchang*(bar_int-bc1_len)+bc1_len*bc1)+'_'+str(buchang*(bar_int+1-bc1_len)+bc1_len*bc1)
        else:
            print("ERROR '?' mode hasn't finished yet")
            return
        
        ret_tot_dict['i2b_dict'][il[i]]=bar_int
        if not(vl[i] in ret_tot_dict['v2b_dict']):
            ret_tot_dict['v2b_dict'][vl[i]]=bar_int

        ret_tot_dict['b2i_list'][bar_int].append(il[i])
        ret_tot_dict['b2v_list'][bar_int].append(vl[i])

        ret_tot_dict['b2r_list'][bar_int].append(curr_range)
    for i,ii in enumerate(ret_tot_dict['b2i_list']):
        if 'min-0.5' in para_dict['bar_mode']:
            curr_range = str(round((min(vl)+(i-0.5)*buchang),3))+'_'+str(round((min(vl)+(i+0.5)*buchang),3))
        elif 'max-min' in para_dict['bar_mode']:
            curr_range = str(round((min(vl)+(i)*buchang),3))+'_'+str(round((min(vl)+(i+1)*buchang),3))
        ret_tot_dict['b2r_list'][bar_int][0]=curr_range
        ret_tot_dict['r2n_dict'][curr_range]=len(ii)

    ret_list = []
    for i in para_dict['return_mode']:
        ret_list.append(ret_tot_dict[i])
    return ret_list

def sl2f(l,cal):
    """
    calaulation of a list of float or int
    '+', '-' only 2 [0]-[1], 'x', '^-1',
    'max', 'min', 'mean',
    'maxd', 'mind' 'meand'
    """
    ret = ''
    if cal=='+':
        ret=np.sum(l)
    elif cal=='-':
        if len(l)!=2:
            print('ERROR sl2f \'-\' ',l)
            return
        ret=l[0]-l[1]
    elif cal=='x':
        ret=np.prod(l)
    elif cal=='^-1':
        ret=np.sum([m**(-1) for m in l])
    elif cal=='max':
        ret=np.max(l)
    elif cal=='min':
        ret=np.min(l)
    elif cal=='mean':
        ret=np.mean(l)
    
    elif cal=='maxd':
        curr_d = [abs(m-n) for m in l for n in l]
        ret = max(curr_d)
    elif cal=='mind':
        curr_d = [abs(mm-nn) for m,mm in enumerate(l) for n,nn in enumerate(l) if m>n]
        ret = min(curr_d)
    elif cal=='meand':
        curr_d = [abs(m-n) for m in l for n in l]
        # matrix is sensitive to m =? n
        ret= np.mean(curr_d)/2
    elif cal=='std':
        ret = np.std(l,ddof=1)
    else:
        print('ERROR sl2f')
        return
    return ret

def sl2oao(sl,oao_dict):
    """
    return a boolean result of the list matching with oao_dict
    sl: one dimonsion list, the type of each content is str
    oao_dict: {[int:['type',[or_condition]],],}
    available type: 
        str            : retransformation
        int            : through 
        float_tol      : str                        tol and range filter
        list tuple dict: no extra transformation
    """
    is_o = 0
    for i in oao_dict:
        is_a=0
        for j in i:
            if i[j][0]=='str':
                if str(sl[j]) in [str(k) for k in i[j][1]]:
                    is_a+=1
            elif i[j][0]=='int':
                if int(sl[j]) in [int(k) for k in i[j][1]]:
                    is_a+=1
            elif i[j][0] in ['list','tuple','dict']:
                if sl[j] in i[j][1]:
                    is_a+=1
            elif i[j][0][:5]=='float':
                tmp_a = 0
                for k in i[j][1]:
                    if '<' in k and (float(sl[j])<float(k[1:])):
                        tmp_a=1
                    elif '>' in k and (float(sl[j])>float(k[1:])):
                        tmp_a=1
                    elif abs(float(sl[j])-float(k))<=float(i[j][0][6:]):
                        tmp_a=1
                    if tmp_a:
                        break
                if tmp_a:
                    is_a+=1
        if is_a==len(list(i.keys())):
            is_o+=1
    if is_o:
        return True
    else:
        return False

def slaon(sl,aon,mode='or_1'):
    is_a = 1
    is_o = 0
    is_n = 0
    for i in aon[0]:
        if not(i in sl):
            is_a=0
            break
    if aon[1]:
        
        for i in aon[1]:
            if i in sl:
                is_o=1
                break
        if mode=='or_1':
            pass
        elif mode=='or_n':
            if (set(aon[1])-sl):
                is_o=0
    else:
        is_o=1
    for i in aon[2]:
        if i in sl:
            is_n=1
            break
    if is_a and is_o and not(is_n):
        return True
    else:
        return False


        





def sl2si(in_list,para_dict):
    """
    value_list   = [5, 7, 4, 3, 0, 8, 9, 7, 4, 3]
    sorted_index = [4, 3, 6, 8, 9, 1, 0, 2, 5, 7] without 'same_index'

                   [4, 2, 5, 7, 9, 1, 0, 2, 5, 7] with 'same_index':1e-2
    para_dict:
        'sort_para': 'max2min' 'min2max'
        'same_index': float      
    """
    sort_dict = {}
    for i in range(len(in_list)):
        sort_dict[i]=in_list[i]
    sorted_list = sorted(sort_dict.items(),key=lambda kv:(kv[1],kv[0]),reverse=[True if para_dict['sort_para']=='max2min' else False][0])
    ret_list = [-1 for _ in in_list]
    # print("sorted_list    ",sorted_list)
    curr_same = 0
    for i in range(len(sorted_list)):
        if 'same_index' in para_dict and i>0 and abs(sorted_list[i][1]-sorted_list[curr_same][1])<para_dict['same_index']:
            ret_list[sorted_list[i][0]]=copy.deepcopy(curr_same)
        else:
            curr_same = copy.deepcopy(i)
            ret_list[sorted_list[i][0]]=i
    if -1 in ret_list:
        print("ERROR sl2si -1")
        return
    else:
        return ret_list

def iscontain(short_str, long_str):
    """
    only matching strings connected by '_'
    eg: 'adfa_fa' < ---- > 'adff_adfa_adfa_fa'
    """
    short_re = re.split(r'_+', short_str)
    long_re = re.split(r'_+', long_str)
    # print(short_re,long_re)
    if not(short_re[0] in long_re):
        # print("?")
        return []
    long_index = 0
    ret_list_tmp = []
    while long_index <= len(long_re)-1:
        if long_re[long_index] == short_re[0]:
            ret_list_tmp.append([])
            short_index = 0
            while short_index <= len(short_re)-1:
                if long_index+short_index <= len(long_re)-1:
                    # print(long_index+short_index,short_re[short_index],long_re[long_index+short_index])
                    if short_re[short_index] == long_re[long_index+short_index]:
                        ret_list_tmp[-1].append(long_index+short_index)
                    else:
                        ret_list_tmp[-1].append("False")
                short_index += 1
            # long_index += short_index   210917  'mlr_ff_ori'  >>  'mlr_ff_mlr_ff_ori'
            long_index += 1
        else:
            long_index += 1
    ret_list_ret = []
    # print(ret_list_tmp)
    for i in ret_list_tmp:
        if not("False" in i):
            ret_list_ret.append(i)
    return ret_list_ret

def sl2md(input_key_list,deepest_content):
    """
    sl2md([a,b,c,d],e) < --- > {a:{b:{c:{d:e}}}}
    """
    curr_dict = {input_key_list[-1]:deepest_content}
    curr_int = len(input_key_list)-2
    while curr_int>=0:
        curr_dict = {input_key_list[curr_int]:curr_dict}
        curr_int = curr_int-1
    return curr_dict




def d2pl(dict,extraspace=2):
    ret_list = []
    if not(dict):
        return ret_list
    maxspace = max([len(i) for i in dict])
    for i in dict:
        ret_list.append(i+' '*(maxspace-len(i))+' '*extraspace+str(dict[i]))
    return ret_list

def xycorr(x_df,y_df,jd_name):
    ret_df = pd.DataFrame()
    for i in x_df.columns.tolist():
        curr_df = pd.concat([x_df.loc[:,i],y_df.loc[:,jd_name]],axis=1,sort=False)
        curr_corr = curr_df.corr(method='pearson')
        ret_df.loc[i,jd_name] = curr_corr.loc[i,jd_name]
    return ret_df

