import os
if '__main__'==__name__:

    for h in [24]:
        h = str(h)
        cla_jobs_dir = './ndml2_jobs_52kp'+h
        if not(os.path.exists(cla_jobs_dir)):
            os.makedirs(cla_jobs_dir)
        f_index = 0
        for i in [
            './snresult_nlondc_24-52kp_ea--ea_230413_0-04e_foap--l4_ea--_s15',
            './snresult_nlondc_24-52kp_ea--ea_230413_0-04e_foap--l4_ea--_s15_cif',
            './snresult_nlondc_24-52kp_ea--ea_230413_0-04e_foap--l4_ea--_s15_for',
            './snresult_nlondc_24-52kp_ea--ea_230413_0-04e_foap--l4_ea--_s15_for-cif',
            './snresult_nlondc_24-52kp_ea--ea_230413_0-08e_foap--l4_ea--_s15',
            './snresult_nlondc_24-52kp_ea--ea_230413_0-08e_foap--l4_ea--_s15_cif',
            './snresult_nlondc_24-52kp_ea--ea_230413_0-08e_foap--l4_ea--_s15_for',
            './snresult_nlondc_24-52kp_ea--ea_230413_0-08e_foap--l4_ea--_s15_for-cif',



            ]:
                i = i.replace('_24-','_'+h+'-')
                curr_f = open(cla_jobs_dir+'/ndml2_jobs_'+str(f_index)+'.py','w')
                curr_f.write("""from nd_ml2 import LEARNEROBJ
if '__main__'==__name__:
        l_o = LEARNEROBJ()
        l_o.d_learner_pipeline(
                {
                    'snresult_path_str':'"""+i+"""',
                    'learner_dict':{
                        'mode_str':'cla',
                        'learner_list':['lgbm'],
                        'cv_list':['shu'],
                        'scoring_ll':[['acc','accuracy'],['pre','precision'],['rec','recall'],['f2','make_scorer']],
                        'n_jobs_int':64
                    },
                    'learner_op_dict':{
                        'op_save_path_str':'"""+i+"""',
                        'op_pred_scoring_dll':{
                            tuple(['x_train_rew','curr']):[['acc','make_scorer'],['pre','make_scorer'],['rec','make_scorer'],['f1','make_scorer'],['f2','make_scorer'],['auc','make_scorer']],
                            tuple(['x_test_re','every']):[['acc','make_scorer'],['pre','make_scorer'],['rec','make_scorer'],['f1','make_scorer'],['f2','make_scorer'],['auc','make_scorer']]
                            }
                    }
                }
            )
    """)
                f_index+=1