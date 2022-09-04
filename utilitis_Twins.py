import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,pairwise_distances
import tqdm
import seaborn as sns
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,average_precision_score,confusion_matrix,f1_score,\
    roc_auc_score,roc_curve,precision_recall_curve
from collections import Counter


#######################################################################################
# Auxillary functions
#######################################################################################

# Get data
def defineNLI(X_add_toMatch,X_MS_toMatch,X_LTS_toMatch,X_PPS_toMatch,sc_TMS_toMatch,dermatomes_toMatch,ms_fields_toMatch):
    # get new NLI
    all_indMS_nli = []
    nli_score = []
    for p in tqdm.tqdm(X_add_toMatch.index):
        inds = np.where(np.logical_or((X_LTS_toMatch.loc[p,:]<2),(X_PPS_toMatch.loc[p,:]<2)))[0]
        if len(inds)>0:
            derm_notInt = X_LTS_toMatch.columns[inds[0]]
            this_ind_nli = dermatomes_toMatch.index(derm_notInt[:-2])
            if this_ind_nli>24:
                this_ind_nliMS = len(sc_TMS_toMatch)+1
            else:
                ms_blnli = [value for value in dermatomes_toMatch[this_ind_nli:] if value in sc_TMS_toMatch]
                this_ind_nliMS = 2 * sc_TMS_toMatch.index(ms_blnli[0])
        else:
            this_ind_nliMS = len(sc_TMS_toMatch)+1
            this_ind_nli = len(dermatomes_toMatch)+1

        inds_MS = np.where(X_MS_toMatch.loc[p,ms_fields_toMatch]<5)[0]
        if len(inds_MS)>0:
            myonot_notInt_ind = inds_MS[0]
        else:
            myonot_notInt_ind = len(sc_TMS_toMatch)+1
        ind_nli_notInt = np.min([this_ind_nliMS,myonot_notInt_ind])

        derm_ind_newnli = dermatomes_toMatch.index(sc_TMS_toMatch[int(ind_nli_notInt/2)])
        derm_nli = dermatomes_toMatch[np.min([derm_ind_newnli,this_ind_nli])]
        nli_score.append(derm_nli)

        # convert derm to myotom
        all_indMS_nli.append(ind_nli_notInt)

    X_add_toMatch['nli_ind_MS'] = all_indMS_nli
    X_add_toMatch['nli']        = nli_score
    return X_add_toMatch
def getaddData(X_addfile_ref,aisa_grade_field,plegia_field,age_field, sex_field,cause_field,
               useToSubset, ms_fields,X,X_LTS,X_PPS,pats,dermatomes_toMatch,sc_TMS_toMatch,Vac_field = [], DAP_field = []):



    X_add_load = pd.read_csv(X_addfile_ref, index_col=0)
    if type(pats) == str:
        pats_use =  X_add_load.index
    else:
        pats_use = pats.copy()
    X_add = pd.DataFrame(columns = useToSubset, index = pats_use)

    # Get date of SCI
    import calendar
    from datetime import datetime
    month_dict = {month: index for index, month in enumerate(calendar.month_name) if month}
    dates_raw = X_add_load.loc[pats_use, 'DOI']
    all_DOIs = []
    for d in dates_raw:
        month = month_dict[d.split(',')[1].split(' ')[1]]
        day   = int(d.split(',')[1].split(' ')[2])
        year  = int(d.split(',')[2].split(' ')[1])
        all_DOIs.append(datetime(year=year, month=month, day=day))
    X_add.loc[pats_use, 'DOI'] = all_DOIs


    if len(plegia_field)>0:
        X_add.loc[pats_use,'plegia'] = X_add_load.loc[pats_use, plegia_field]
    else:
        raise('plegia_field needs to be given!')

    if len(aisa_grade_field)>0:
        X_add.loc[pats_use,'aisa'] = X_add_load.loc[pats_use, aisa_grade_field]
        #X_add.loc[pats_use, 'aisa_rec'] = X_addrec.loc[pats_use, aisa_grade_field]
    else:
        raise('aisa_grade_field needs to be given!')

    if len(age_field)>0:
        X_add.loc[pats_use,'age'] = X_add_load.loc[pats_use, age_field]
    if len(sex_field)>0:
        X_add.loc[pats_use,'sex'] = X_add_load.loc[pats_use, sex_field]
    if len(cause_field)>0:
        X_add.loc[pats_use, 'cause'] = X_add_load.loc[pats_use, cause_field]
    if len(Vac_field)>0:
        X_add.loc[pats_use, 'VAC'] = X_add_load.loc[pats_use, Vac_field].map(dict(Yes=1, No=0))
    if len(DAP_field)>0:
        X_add.loc[pats_use, 'DAP'] = X_add_load.loc[pats_use, DAP_field].map(dict(Yes=1, No=0))



    # Get nli indices for MS (in order as anticipated!)
    X_add = defineNLI(X_add,X,X_LTS,X_PPS,sc_TMS_toMatch,dermatomes_toMatch,ms_fields)




    for sub in useToSubset:
        # Summary scores
        if sub == 'LEMS':
            X_add.loc[pats_use, 'LEMS'] = X.loc[pats_use, ms_fields[10:]].sum(axis = 1)

        if sub == 'MeanScBlNLI':
            for p in pats_use:
                X_add.loc[p, 'MeanScBlNLI'] = get_MeanScBlNLI(p,X_add, ms_fields,X)

        if sub == 'PropSc':
            X_add.loc[pats_use, 'PropSc'] = X.loc[pats_use, ms_fields[10:]].sum(axis = 1)


    return X_add
def getSygen(path_group,sc_TMS_toMatch,dermatomes_toMatch ):
    # Load Sygen data
    X_addfile = os.path.join(path_group, 'SCI/Sygen/JohnKramersProject_DATA_2019-10-07_0111.csv')
    df_data = pd.read_csv(X_addfile)
    df_data.set_index('ptid', drop=True, inplace=True)
    sygenSc = ['elbfl', 'wrext', 'elbex', 'finfl', 'finab', 'hipfl', 'kneex', 'ankdo', 'greto', 'ankpl']
    sygenScr = ['elbfl', 'wrext', 'elbex', 'finfl', 'finab', 'hipfl', 'kneet', 'ankdo', 'greto', 'ankpl']
    dermatomes = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'T1', 'T2', 'T3',
                  'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                  'L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'S2', 'S3', 'S45']
    derm_fields_lr = ['C2_l', 'C3_l', 'C4_l', 'C5_l', 'C6_l', 'C7_l', 'C8_l', 'T1_l', 'T2_l',
                      'T3_l', 'T4_l', 'T5_l', 'T6_l', 'T7_l', 'T8_l', 'T9_l', 'T10_l',
                      'T11_l', 'T12_l', 'L1_l', 'L2_l', 'L3_l', 'L4_l', 'L5_l', 'S1_l',
                      'S2_l', 'S3_l', 'S45_l', 'C2_r', 'C3_r', 'C4_r', 'C5_r', 'C6_r', 'C7_r',
                      'C8_r', 'T1_r', 'T2_r', 'T3_r', 'T4_r', 'T5_r', 'T6_r', 'T7_r', 'T8_r',
                      'T9_r', 'T10_r', 'T11_r', 'T12_r', 'L1_r', 'L2_r', 'L3_r', 'L4_r',
                      'L5_r', 'S1_r', 'S2_r', 'S3_r', 'S45_r']
    ms_fields_lr = ['C5_l', 'C6_l', 'C7_l', 'C8_l', 'T1_l', 'L2_l', 'L3_l', 'L4_l', 'L5_l', 'S1_l',
                    'C5_r', 'C6_r', 'C7_r', 'C8_r', 'T1_r', 'L2_r', 'L3_r', 'L4_r', 'L5_r', 'S1_r']
    ms_fields = ['C5_l', 'C5_r', 'C6_l', 'C6_r', 'C7_l', 'C7_r', 'C8_l', 'C8_r', 'T1_l', 'T1_r', 'L2_l', 'L2_r', 'L3_l',
                 'L3_r',
                 'L4_l', 'L4_r', 'L5_l', 'L5_r', 'S1_l', 'S1_r']

    ltl_sygen = []
    ltr_sygen = []
    ppl_sygen = []
    ppr_sygen = []
    for sc in dermatomes:
        if sc[0] == 'S':
            sc = 's' + sc[1:]
        elif sc[0] == 'C':
            sc = 'c' + sc[1:]
        elif sc[0] == 'T':
            sc = 't' + sc[1:]
        elif sc[0] == 'L':
            sc = 'l' + sc[1:]

        ltl_sygen.append(sc + 'ltl')
        ltr_sygen.append(sc + 'ltr')
        ppl_sygen.append(sc + 'ppl')
        ppr_sygen.append(sc + 'ppr')

    msl_sygen = [s + 'l' for s in sygenSc]
    msr_sygen = [s + 'r' for s in sygenScr]

    all_scSygen = msl_sygen + msr_sygen + ltl_sygen + ltr_sygen + ppl_sygen + ppr_sygen
    all_scSygen_tp1 = [s + '01' for s in all_scSygen]
    all_scSygen_tp2 = [s + '26' for s in all_scSygen]

    # Load the scores
    sygenX = df_data[all_scSygen_tp1]
    sygenY = df_data[all_scSygen_tp2]
    sygenX.dropna(inplace=True)
    sygenX.columns = all_scSygen
    sygenY.dropna(inplace=True)
    sygenY.columns = all_scSygen
    sygenpats = list(set(sygenX.index) & set(sygenY.index))

    # Additional data
    sygenX_add_lr = df_data[['age', 'sexcd', 'tx1_r', 'ais1', 'splvl', 'lower01','vaccd01','anyana01']]
    sygenX_add_lr = sygenX_add_lr.loc[sygenpats]
    sygenX_add_lr.columns = ['age', 'sex', 'treat', 'aisa', 'nli', 'LEMS','VAC','DAP']

    plegia = []
    TMS = []
    aisa = []
    nli_ind = []
    nli = []
    nli_coarse = []
    ind_nli_MS = []
    sex = []
    for p in tqdm.tqdm(sygenpats):

        this_nli = sygenX_add_lr.loc[p, 'nli'][0] + str(int(sygenX_add_lr.loc[p, 'nli'][-2:]))
        nli.append(this_nli)
        if this_nli[0] == 'C':
            nli_coarse.append('cervical')
        elif this_nli[0] == 'T':
            nli_coarse.append('thoracic')
        elif this_nli[0] == 'L':
            nli_coarse.append('lumbar')
        elif this_nli[0] == 'S':
            nli_coarse.append('sacral')

        this_nli = sygenX_add_lr.loc[p, 'nli'][0] + str(int(sygenX_add_lr.loc[p, 'nli'][-2:]))
        if this_nli == 'C1':
            this_nliind = 0
        else:
            this_nliind = dermatomes.index(this_nli)
        nli_ind.append(this_nliind)

        ms_blnli = [value for value in dermatomes_toMatch[this_nliind:] if value in sc_TMS_toMatch]
        this_ind_nliMS = 2 * sc_TMS_toMatch.index(ms_blnli[0])
        ind_nli_MS.append(this_ind_nliMS)

        if this_nli[0] in ['T','L','S']:
            plegia.append('para')
        else:
            plegia.append('tetra')

        this_aislong = sygenX_add_lr.loc[p, 'aisa']
        try:
            aisa.append(this_aislong[-1])
        except:
            aisa.append(np.nan)

        TMS.append(sygenX.loc[p, msl_sygen + msr_sygen].sum())

        if sygenX_add_lr.loc[p, 'sex'] == 1:
            sex.append('F')
        else:
            sex.append('M')

    sygenX_add_lr['plegia'] = plegia
    sygenX_add_lr['TMS'] = TMS
    sygenX_add_lr['nli_ind'] = nli_ind
    sygenX_add_lr['aisa'] = aisa
    sygenX_add_lr['nli'] = nli
    sygenX_add_lr['sex'] = sex
    sygenX_add_lr['nli_coarse'] = nli_coarse
    sygenX_add_lr['nli_ind_MS'] = ind_nli_MS

    sygenX_MS = pd.DataFrame(columns=ms_fields_lr)
    sygenX_LTS = pd.DataFrame(columns=derm_fields_lr)
    sygenX_PPS = pd.DataFrame(columns=derm_fields_lr)
    sygenY_MStrue = pd.DataFrame(columns=ms_fields_lr)
    sygenX_add = pd.DataFrame(columns=sygenX_add_lr.columns)

    ms_use = msl_sygen + msr_sygen
    lts_use = ltl_sygen + ltr_sygen
    pps_use = ppl_sygen + ppr_sygen
    meanSc = []
    for p in tqdm.tqdm(sygenpats):
        sygenX_MS.loc[p, :] = list(sygenX.loc[p, ms_use])
        sygenX_LTS.loc[p, :] = list(sygenX.loc[p, lts_use])
        sygenX_PPS.loc[p, :] = list(sygenX.loc[p, pps_use])

        sygenY_MStrue.loc[p, :] = list(sygenY.loc[p, ms_use])
        sygenX_add.loc[p, :] = sygenX_add_lr.loc[p, :]

        this_ind_nliMS = sygenX_add_lr.loc[p, 'nli_ind_MS']
        this_meanSc = sygenX_MS.loc[p, ms_fields[this_ind_nliMS:]].mean()
        meanSc.append(this_meanSc)
    sygenX_add['MeanScBlNLI'] = meanSc

    # drop patients missing an AISA grade
    pats_drop = list(sygenX_add[sygenX_add['aisa'].isnull()].index)
    sygenX_add.drop(pats_drop, inplace=True)
    sygenX_MS.drop(pats_drop, inplace=True)
    sygenX_LTS.drop(pats_drop, inplace=True)
    sygenX_PPS.drop(pats_drop, inplace=True)
    sygenY_MStrue.drop(pats_drop, inplace=True)

    return sygenX_MS, sygenX_LTS, sygenX_PPS, sygenY_MStrue, sygenX_add
def getEMSCI(pats_include_file, outcome_file_toMatch, inputMS_file_toMatch, inputLTS_file_toMatch,
             inputPPS_file_toMatch,
             X_addfile_toMatch, aisa_grade_field_toMatch, plegia_field_toMatch, age_field_toMatch,
             sex_field_toMatch, cause_field_toMatch, ms_fields_toMatch,
             dermatomes_toMatch, sc_TMS_toMatch, vac_field, dap_field):
    # Load baseline and recovery MS for the patients to be matched (true results)
    Y_toMatch     = pd.read_csv(outcome_file_toMatch, index_col=0)
    X_MS_toMatch = pd.read_csv(inputMS_file_toMatch, index_col=0)
    X_LTS_toMatch = pd.read_csv(inputLTS_file_toMatch, index_col=0)
    X_PPS_toMatch = pd.read_csv(inputPPS_file_toMatch, index_col=0)
    X_add_toMatch = getaddData(X_addfile_toMatch, aisa_grade_field_toMatch, plegia_field_toMatch,
                               age_field_toMatch, sex_field_toMatch, cause_field_toMatch,
                               ['aisa', 'nli', 'age', 'plegia', 'LEMS', 'MeanScBlNLI'], ms_fields_toMatch, X_MS_toMatch,
                               Y_toMatch.index, dermatomes_toMatch, sc_TMS_toMatch, Vac_field=vac_field,
                               DAP_field=dap_field)

    # Optional load patient subset
    if len(pats_include_file) > 0:
        pats_use = list(pd.read_csv(pats_include_file, index_col=0).index)
        Y_toMatch = Y_toMatch.loc[pats_use]
        X_MS_toMatch = X_MS_toMatch.loc[pats_use]
        X_LTS_toMatch = X_LTS_toMatch.loc[pats_use]
        X_PPS_toMatch = X_PPS_toMatch.loc[pats_use]
        X_add_toMatch = X_add_toMatch.loc[pats_use]

    return X_MS_toMatch, X_LTS_toMatch, X_PPS_toMatch, Y_toMatch, X_add_toMatch
def get_MeanScBlNLI(p, X_add, ms_fields, X):
    this_ind_nliMS = X_add.loc[p, 'nli_ind_MS']
    meanScBlNLI = X.loc[p, ms_fields[int(this_ind_nliMS):]].mean()

    return meanScBlNLI
def getName(useToSubset,sc_tol, useKNN, n_NN, useClosestTwin,useMean):
    name = 'Twin_'

    # all subsets
    for sub in useToSubset:
        name += sub+'_'

    # score tolerance
    if np.any(['LEMS' in useToSubset,'MeanScBlNLI' in useToSubset,'RMSE' in useToSubset]):
        name += 'SCtol'+ str(sc_tol) + '_'

    # KNN
    if ('NN_MS_SS' in useToSubset) or ('NN_MS' in useToSubset):
        if useKNN:
            name += 'KNN_' + str(n_NN)+ '_'
            if useClosestTwin:
                name += 'ClosestTwin_'

    if useMean:
        name += '_mean'
    else:
        name += '_median'
    return name
def getFlowDiagram():


    X_addAll = pd.read_csv('/Users/sbrueningk/PycharmProjects/SCI/SCI/EMSCI/emsci_data_2020.csv', index_col=0)
    cols_check2 = ['AIS']+list(X_addAll.columns[42:42+132])
    cols_check26 = list(X_addAll.columns[42:42 + 132])
    ms_cols = X_addAll.columns[42:62]
    X_2W = X_addAll[X_addAll['ExamStage_weeks'] == 2]
    pats2W = list(X_2W[~X_2W.loc[:,cols_check2].isnull().any(axis = 1)].index)
    pats2W_drop = []
    pats2W_keep = [] #2495
    for p in pats2W:
        if ('NT' in X_2W.loc[p,cols_check2].values) | ('5*' in X_2W.loc[p,cols_check2].values):
            pats2W_drop.append(p)
        else:
            pats2W_keep.append(p)


    X_26W = X_addAll[X_addAll['ExamStage_weeks'] == 26]
    pats26W = list(X_26W[~X_26W.loc[:, cols_check26].isnull().any(axis=1)].index)
    pats26W_drop = []
    pats26W_keep = []# 2710
    for p in pats26W:
        if ('NT' in X_26W.loc[p,cols_check26].values)| ('5*' in X_2W.loc[p,cols_check26].values):
            pats26W_drop.append(p)
        else:
            #if  ('5*' in X_26W.loc[p,cols_check].values):
            pats26W_keep.append(p)

    pats_keep = list(set(pats2W_keep)&set(pats26W_keep)) # 1333


    # Check for deterioration
    pats_nodet_EMSCI = []
    pats_det = []
    pats5 = []
    for p in pats_keep:
        try:
            diff = X_26W.loc[p, ms_cols].astype(float) - X_2W.loc[p, ms_cols].astype(float)
            if np.any(diff.values < -1):
                pats_det.append(p)
            else:
                pats_nodet_EMSCI.append(p)
        except:
            pats5.append(p)
            diff = X_26W.replace('5*','5').loc[p,ms_cols].astype(float)-X_2W.replace('5*','5').loc[p,ms_cols].astype(float)

    # Exclude sacral patients
    nlis = X_addAll.loc[pats_nodet_EMSCI,'NLI'].unique()
    patsS1 = list(X_2W.loc[pats_nodet_EMSCI][ X_2W.loc[pats_nodet_EMSCI,'NLI'] == 'S1'].index)
    pats_nodet_EMSCI = [ele for ele in pats_nodet_EMSCI if ele not in patsS1]
    pats_EMSCI = pd.DataFrame(index =pats_nodet_EMSCI )
    pats_EMSCI.to_csv('/Volumes/borgwardt/Data/SCI/Models/SharedPatients/usedPatsEMSCI.csv')



    # SYGEN
    # Load Sygen data
    path_group = '/Volumes/borgwardt/Data'
    X_addfile = os.path.join(path_group, 'SCI/Sygen/JohnKramersProject_DATA_2019-10-07_0111.csv')
    df_data = pd.read_csv(X_addfile)
    df_data.set_index('ptid', drop=True, inplace=True)
    sygenSc = ['elbfl', 'wrext', 'elbex', 'finfl', 'finab', 'hipfl', 'kneex', 'ankdo', 'greto', 'ankpl']
    sygenScr = ['elbfl', 'wrext', 'elbex', 'finfl', 'finab', 'hipfl', 'kneet', 'ankdo', 'greto', 'ankpl']
    dermatomes = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'T1', 'T2', 'T3',
                  'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                  'L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'S2', 'S3', 'S45']
    derm_fields_lr = ['C2_l', 'C3_l', 'C4_l', 'C5_l', 'C6_l', 'C7_l', 'C8_l', 'T1_l', 'T2_l',
                      'T3_l', 'T4_l', 'T5_l', 'T6_l', 'T7_l', 'T8_l', 'T9_l', 'T10_l',
                      'T11_l', 'T12_l', 'L1_l', 'L2_l', 'L3_l', 'L4_l', 'L5_l', 'S1_l',
                      'S2_l', 'S3_l', 'S45_l', 'C2_r', 'C3_r', 'C4_r', 'C5_r', 'C6_r', 'C7_r',
                      'C8_r', 'T1_r', 'T2_r', 'T3_r', 'T4_r', 'T5_r', 'T6_r', 'T7_r', 'T8_r',
                      'T9_r', 'T10_r', 'T11_r', 'T12_r', 'L1_r', 'L2_r', 'L3_r', 'L4_r',
                      'L5_r', 'S1_r', 'S2_r', 'S3_r', 'S45_r']
    ms_fields_lr = ['C5_l', 'C6_l', 'C7_l', 'C8_l', 'T1_l', 'L2_l', 'L3_l', 'L4_l', 'L5_l', 'S1_l',
                    'C5_r', 'C6_r', 'C7_r', 'C8_r', 'T1_r', 'L2_r', 'L3_r', 'L4_r', 'L5_r', 'S1_r']
    ms_fields = ['C5_l', 'C5_r', 'C6_l', 'C6_r', 'C7_l', 'C7_r', 'C8_l', 'C8_r', 'T1_l', 'T1_r', 'L2_l', 'L2_r', 'L3_l',
                 'L3_r',
                 'L4_l', 'L4_r', 'L5_l', 'L5_r', 'S1_l', 'S1_r']

    ltl_sygen = []
    ltr_sygen = []
    ppl_sygen = []
    ppr_sygen = []
    for sc in dermatomes:
        if sc[0] == 'S':
            sc = 's' + sc[1:]
        elif sc[0] == 'C':
            sc = 'c' + sc[1:]
        elif sc[0] == 'T':
            sc = 't' + sc[1:]
        elif sc[0] == 'L':
            sc = 'l' + sc[1:]

        ltl_sygen.append(sc + 'ltl')
        ltr_sygen.append(sc + 'ltr')
        ppl_sygen.append(sc + 'ppl')
        ppr_sygen.append(sc + 'ppr')

    msl_sygen = [s + 'l' for s in sygenSc]
    msr_sygen = [s + 'r' for s in sygenScr]

    all_scSygen = msl_sygen + msr_sygen + ltl_sygen + ltr_sygen + ppl_sygen + ppr_sygen
    all_scSygen_tp1 = [s + '01' for s in all_scSygen]
    all_scSygen_tp2 = [s + '26' for s in all_scSygen]

    # Load the scores
    sygenX = df_data[all_scSygen_tp1]
    sygenY = df_data[all_scSygen_tp2]
    sygenX.dropna(inplace=True)
    sygenX.columns = all_scSygen
    pats1W = sygenX.index
    pats_keep = df_data.loc[pats1W][~df_data.loc[pats1W, 'ais1'].isnull().values].index
    sygenX = sygenX.loc[pats_keep]


    sygenY.dropna(inplace=True)
    sygenY.columns = all_scSygen
    sygenpats = list(set(sygenX.index) & set(sygenY.index))

    # Additional data
    sygenX_MS = pd.DataFrame(columns=ms_fields_lr)
    sygenY_MStrue = pd.DataFrame(columns=ms_fields_lr)
    ms_use = msl_sygen + msr_sygen
    for p in tqdm.tqdm(sygenpats):
        sygenX_MS.loc[p, :] = list(sygenX.loc[p, ms_use])
        sygenY_MStrue.loc[p, :] = list(sygenY.loc[p, ms_use])


    # Check for deterioration
    pats_nodet_sygen = []
    pats_det = []
    for p in sygenpats:
        diff =  sygenY_MStrue.loc[p].astype(float) - sygenX_MS.loc[p].astype(float)
        if np.any(diff.values < -1):
            pats_det.append(p)
        else:
            pats_nodet_sygen.append(p)

    pats_sygen = pd.DataFrame(index=pats_nodet_sygen)
    pats_sygen.to_csv('/Volumes/borgwardt/Data/SCI/Models/SharedPatients/usedPatsSYGEN.csv')

    return pats_nodet_EMSCI,pats_nodet_sygen
def getAISgrade(X_MS, X_LTS, X_PPS,X_add, dermatomes_toMatch,ms_fields_toMatch,emsci_data = [], ais_assigned = []):

    # Emsci TP 2W: 201 changed!
    ais_grades = []
    changedPats = [] # 201
    ais_grades_newDef = []
    changedPats_newDef = [] # 701
    for p in tqdm.tqdm(X_MS.index):

        try:
            this_vac = emsci_data.loc[:,'VAC'].map({'No':0,'Yes':1}).loc[p]
            this_dap = emsci_data.loc[:,'DAP'].map({'No':0,'Yes':1}).loc[p]
        except:
            this_vac = X_add.loc[p, 'VAC']
            this_dap = X_add.loc[p, 'DAP']

        ind_NLI_MS = X_add.loc[p,'nli_ind_MS']

        derms_blNLI = [d+'_l' for d in dermatomes_toMatch[dermatomes_toMatch.index(X_add.loc[p,'nli']):]] +\
        [d + '_r' for d in dermatomes_toMatch[dermatomes_toMatch.index(X_add.loc[p, 'nli']):]]

        this_ms = X_MS.loc[p,ms_fields_toMatch].values
        ms_bl_nli = this_ms[ind_NLI_MS:].astype(float)
        ss_bl_nli = list(X_LTS.loc[p, derms_blNLI].astype(float))+list(X_PPS.loc[p, derms_blNLI].astype(float))


        #ss_bl_nli_vac = ss_bl_nli+[this_vac,this_dap]
        ss_S45_bl_nli = list(X_LTS.loc[p, ['S45_r','S45_l']].astype(float)) + list(X_PPS.loc[p, ['S45_r','S45_l']].astype(float))

        # My definition
        # Ais A: No sensory and no motor function below the NLI
        this_ais = np.nan
        if np.sum([this_vac,this_dap])== 0:
            if np.sum(ss_S45_bl_nli)>0:
                this_ais = 'B'
            else:
                this_ais = 'A'
        else:

            # Ais B: sensory but no motor function below the NLI
            if (np.sum(ms_bl_nli) == 0):
                this_ais = 'B'

            # AIS C: in more than half of the MS below the NLI the MS is <3
            elif (np.sum(ms_bl_nli >= 3) < 0.5 * len(ms_bl_nli)):
                this_ais = 'C'

            # AIS D: in more than half of the MS below the NLI the MS is >=3
            elif (np.sum(ms_bl_nli >= 3) >= 0.5 * len(ms_bl_nli)):


                # AIS E: normal motor and sensory function
                if (np.sum(ss_bl_nli) == 2 * len(ss_bl_nli)) & (np.sum(ms_bl_nli) == 5 * len(ms_bl_nli)):
                    this_ais = 'E'
                else:
                    this_ais = 'D'


        if ais_assigned.loc[p] != this_ais:
            changedPats_newDef.append(p)
        ais_grades_newDef.append(this_ais)



        # Try to implement ISNCSCI
        this_ais = np.nan
        if np.isnan(this_vac+this_dap) or (this_vac+this_dap)>0:

            # Ais B: sensory but no motor function below the NLI
            if (np.sum(ms_bl_nli) == 0):
                this_ais = 'B'

            # AIS C: in more than half of the MS below the NLI the MS is <3
            elif (np.sum(ms_bl_nli >= 3) <= 0.5 * len(ms_bl_nli)):
                this_ais = 'C'

            # AIS D: in more than half of the MS below the NLI the MS is >=3
            elif (np.sum(ms_bl_nli >= 3) > 0.5 * len(ms_bl_nli)):


                # AIS E: normal motor and sensory function
                if (np.sum(ss_bl_nli) == 2 * len(ss_bl_nli)) & (np.sum(ms_bl_nli) == 5 * len(ms_bl_nli)):
                    this_ais = 'E'
                else:
                    this_ais = 'D'

        else:
            # no DAP/VAC
            # Check S4/5
            if (np.sum(ss_S45_bl_nli)>0):
                this_ais = 'B'
            else:
                this_ais = 'A'


        if ((np.sum(ss_bl_nli)==0) & (np.sum(ms_bl_nli)==0)) | ((this_dap==0) & (this_vac==0)) :
            this_ais = 'A'

        else:

            # Ais B: sensory but no motor function below the NLI
            if (np.sum(ss_bl_nli) > 0) & (np.sum(ms_bl_nli)==0):
                this_ais = 'B'

            # AIS C: in more than half of the MS below the NLI the MS is <3
            elif (np.sum(ss_bl_nli) > 0) & (np.sum(ms_bl_nli>=3)<=0.5*len(ms_bl_nli)):
                this_ais = 'C'

            # AIS D: in more than half of the MS below the NLI the MS is >=3
            elif (np.sum(ss_bl_nli) > 0) & (np.sum(ms_bl_nli>=3)>0.5*len(ms_bl_nli)):
                this_ais = 'D'

            # AIS E: normal motor and sensory function
            elif (np.sum(ss_bl_nli) == 2*len(ss_bl_nli)) & (np.sum(ms_bl_nli)==5*len(ms_bl_nli)):
                this_ais = 'E'
            else:
                print('AIS grade failed for '+str(p))
                this_ais = np.nan


        # Check with assigned AIS grade
        if ais_assigned.loc[p] != this_ais:
            changedPats.append(p)
            #print('Missmatch for pat '+str(p))
        ais_grades.append(this_ais)

    X_add['aisa_corr'] = ais_grades
    X_add['aisa_corrSarah'] = ais_grades_newDef

    print(len(changedPats))
    print(len(changedPats_newDef))
    return X_add, changedPats,changedPats_newDef
def getOutcomeData_EMSCI(emsci_data_file,time_point,dermatomes_toMatch,pats):

    emsci_data = pd.read_csv(emsci_data_file, index_col = 0)
    emsci_data = emsci_data[emsci_data['ExamStage_weeks']==time_point]

    lts_cols = ['LLT_'+s for s in dermatomes_toMatch]+['RLT_'+s for s in dermatomes_toMatch]
    pps_cols = ['LPP_' + s for s in dermatomes_toMatch] + ['RPP_' + s for s in dermatomes_toMatch]
    Y_LTS = emsci_data[lts_cols]
    Y_PPS = emsci_data[pps_cols]
    Y_LTS.columns = [s+'_l' for s in dermatomes_toMatch]+[s+'_r' for s in dermatomes_toMatch]
    Y_PPS.columns = [s + '_l' for s in dermatomes_toMatch] + [s + '_r' for s in dermatomes_toMatch]

    emsci_data = emsci_data.loc[:,['AIS','SCIM3_TotalScore','SCIM2_TotalScore','SCIM23_TotalScore','WISCI',
                                   '6min','10m','TUG', 'SCIM2_12','SCIM3_12','VAC','DAP','SCIM23_SubScore1']]
    emsci_data = emsci_data.loc[pats]
    emsci_data['SCIM_12'] = emsci_data.loc[:,['SCIM2_12','SCIM3_12']].mean(axis =1)
    emsci_data['Walker'] = emsci_data['SCIM_12']>3
    emsci_data.loc[emsci_data['SCIM_12'].isna(),'Walker'] = np.nan

    emsci_data['Eater'] = emsci_data['SCIM23_SubScore1']>15
    emsci_data.loc[emsci_data['SCIM23_SubScore1'].isna(),'Eater'] = np.nan

    return emsci_data, Y_LTS, Y_PPS



# For matching
def generate_pat_uncert(df_uncert, this_X_use, n_bootstrap, nli_ind):

    ms_true = this_X_use.values

    # initialize dataframe
    df_new = pd.DataFrame()

    # draw samples for each of the scores below the nli
    for i, this_ms in enumerate(ms_true):

        if i < nli_ind:
            sc_boot = [this_ms for this_n_rand in range(0, n_bootstrap)]

        else:

            # Draw random numbers
            n_rands = np.random.uniform(0, 1, n_bootstrap)

            # get score pdf
            sc_prop = df_uncert.loc[this_ms, :].cumsum()

            # Assign score based on random numbers
            sc_boot = [int(sc_prop.index[(sc_prop > this_n_rand).values][0]) for this_n_rand in n_rands]

        # add to dataframe
        df_new.loc[:, this_X_use.index[i]] = sc_boot


    return df_new
def getPatCar(useToSubset, this_X_add):
    this_char = []
    for sub in useToSubset:
        try:
            this_char.append(this_X_add.loc[sub])
        except:
            this_char.append(np.nan)

    return this_char
def getTwin(useToSubset, data_Knn_use, this_X_use,X_add_ref_use, this_char,useKNN,n_NN,useClosestTwin,useMean,
            Y_MS_ref, X_MS_ref,sc_tol, this_ind_nliMS,addMatch_tol):


    # First generate patient subset
    addMatchPos = 1
    data_sub = X_add_ref_use.copy()
    for i_sc,sub in enumerate(useToSubset):
        this_sc = this_char[i_sc]

        if sub in ['RMSEblNLI']:
            this_X_MS = this_X_use[:20]
            this_rmse = [mean_squared_error(X_MS_ref.loc[i,:][int(this_ind_nliMS):].values,
                                            this_X_MS[int(this_ind_nliMS):].values*5, squared=False) for i in list(data_sub.index)]
            inds_keep =np.where(np.array(this_rmse)<=sc_tol)[0]
            data_sub_sc = data_sub.iloc[inds_keep]
            if data_sub_sc.shape[0] >= 1:
                data_sub = data_sub_sc
            else:
                print(sub + ' match not possible')
                return [], None, None, [],addMatchPos

        if sub in ['LEMS', 'MeanScBlNLI']:

            data_sub_sc = data_sub[data_sub[sub] >= this_sc - sc_tol]
            data_sub_sc = data_sub_sc[data_sub_sc[sub] <= this_sc + sc_tol]

            if data_sub_sc.shape[0] >= 1:
                data_sub = data_sub_sc
            else:
                print(sub + ' match not possible')
                return [], None, None, [],addMatchPos

        elif sub == 'age':
            data_sub_age = data_sub[data_sub['age'] >= this_sc - 10]
            data_sub_age = data_sub_age[data_sub_age['age'] <= this_sc + 10]

            if data_sub_age.shape[0]>=1:
                data_sub = data_sub_age.copy()
            else:
                print('Age match not possible but I continue')

        elif sub == 'sex':
            data_sub_sex = data_sub[data_sub['sex'] == this_sc ]

            if data_sub_sex.shape[0]>=1:
                data_sub = data_sub_sex.copy()
            else:
                print('Sex match not possible but I continue')

        elif sub in ['aisa','nli_coarse', 'cause', 'nli']:

            if addMatch_tol >0:

                if sub == 'nli':
                    nli_pos = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'T1', 'T2', 'T3',
                              'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                              'L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'S2', 'S3', 'S45']
                    ind_start = np.max([0,nli_pos.index(this_sc)- addMatch_tol])
                    ind_end = np.min([len(nli_pos),nli_pos.index(this_sc)+ addMatch_tol+1])
                    nlis_use = nli_pos[ ind_start: ind_end]

                    inds = [(n in nlis_use) for n in data_sub['nli']]
                    data_sub_ais = data_sub[inds]


                if sub in ['cause','aisa','nli_coarse']:
                    data_sub_ais = data_sub
            else:
                data_sub_ais = data_sub[data_sub[sub] == this_sc]



            if data_sub_ais.shape[0] > 1:
                data_sub = data_sub_ais
            else:
                print(sub + ' match not possible!')
                addMatchPos = 0
                return [], None, None, [],addMatchPos
        else:
            continue

    # Check if any possible matches remain
    if data_sub.shape[0] > 0:

        thisdist = (pairwise_distances(data_Knn_use.loc[data_sub.index],
                                       np.expand_dims(np.array(this_X_use), axis=0), metric='hamming') *
                    data_Knn_use.loc[data_sub.index].shape[1])#.astype(int)


        # Find all 'neighbors'
        if useKNN:

            n_inds_all = sorted(range(len(thisdist)), key=lambda k: thisdist[k])[:n_NN]


            if useClosestTwin:
                inds_minDist = list(np.where(thisdist == np.min(thisdist))[0])
                min_dist = np.mean([thisdist[j][0] for j in inds_minDist])
                pats_neigh = list(data_Knn_use.loc[data_sub.index].index[inds_minDist].values)
            else:

                # use k nearest neighbours
                if n_NN>len(n_inds_all):
                    this_inds_use = n_inds_all[:n_NN]
                else:
                    this_inds_use = n_inds_all

                min_dist = np.mean([thisdist[j][0] for j in this_inds_use])
                pats_neigh = list(data_Knn_use.loc[data_sub.index].index[this_inds_use].values)
        else:
            pats_neigh = data_sub.index
            min_dist   = np.mean(thisdist)


        # Calcualte the mean from them
        y_neigh   = Y_MS_ref.loc[pats_neigh, :]
        x_neigh   = X_MS_ref.loc[pats_neigh, :]

        if useMean:
            twin_mean  = y_neigh.mean().values
            twin_mean_X= x_neigh.mean().values
        else:
            twin_mean  = y_neigh.median().values
            twin_mean_X= x_neigh.median().values


        return twin_mean, min_dist, pats_neigh, twin_mean_X, addMatchPos

    else:
        addMatchPos = 1
        return [], None, None, [],addMatchPos

# Evaluation
def getClusterPrediciton(X_add_Sygen,all_dfsY_Sygen,all_dfsX_Sygen,df_cluster,df_median_low_Sygen,df_median_top_Sygen):
    df_pred_clusterX = pd.DataFrame(columns = all_dfsX_Sygen[0].columns)
    df_pred_clusterY = pd.DataFrame(columns=all_dfsY_Sygen[0].columns)
    df_pred_clusterY_low = pd.DataFrame(columns=all_dfsY_Sygen[0].columns)
    df_pred_clusterY_top = pd.DataFrame(columns=all_dfsY_Sygen[0].columns)
    for p in X_add_Sygen.index:
        this_cl =X_add_Sygen.loc[p,'cluster']
        m = df_cluster.loc[this_cl,'model']
        ind_model = all_names_short.index(m)
        df_pred_clusterY.loc[p,:] = all_dfsY_Sygen[ind_model].loc[p,:]
        df_pred_clusterY_low.loc[p, :] = df_median_low_Sygen[ind_model].loc[p, :]
        df_pred_clusterY_top.loc[p, :] = df_median_top_Sygen[ind_model].loc[p, :]
        df_pred_clusterX.loc[p, :] = all_dfsX_Sygen[ind_model].loc[p,:]


    return df_pred_clusterY,df_pred_clusterY_low,df_pred_clusterY_top, df_pred_clusterX
def getpredictionSmallestRMSEX(all_names_short,all_dfsY_Sygen,all_dfsX_Sygen,
                            df_allRMSEY_Sygen, df_allRMSEX_Sygen,df_allDelLEMSY_Sygen,df_allDelLEMSX_Sygen,
                            X_add_Sygen,Y_Sygen):


    #df_allRMSEX.drop('best_models', inplace=True, axis=1)
    if not 'best_models_mean' in all_names_short:
        all_names_short.append('best_models_mean')
    cols = all_dfsY_Sygen[0].columns
    df_X = pd.DataFrame(columns =all_dfsY_Sygen[0].columns )
    df_Y = pd.DataFrame(columns =all_dfsY_Sygen[0].columns )
    for p in tqdm.tqdm(X_add_Sygen.index):

        if 'best_models_mean' in df_allRMSEX_Sygen.columns:
            df_allRMSEX_Sygen_use = df_allRMSEX_Sygen.drop('best_models_mean', axis=1)
        else:
            df_allRMSEX_Sygen_use = df_allRMSEX_Sygen.copy()
        this_min = np.min(df_allRMSEX_Sygen_use.loc[p])
        best_models = list(df_allRMSEX_Sygen_use.loc[p][df_allRMSEX_Sygen_use.loc[p]==this_min].index)

        if len(best_models)>1:
            df_pat = pd.DataFrame()
            df_patX = pd.DataFrame()
            for m in best_models:
                if m == 'best_models_mean':
                    continue
                else:
                    ind = all_names_short.index(m)
                    df_pat[m] = all_dfsY_Sygen[ind].loc[p]
                    df_patX[m] = all_dfsX_Sygen[ind].loc[p]

            df_Y.loc[p,:]  = df_pat.mean(axis = 1)[cols]
            df_X.loc[p, :] = df_patX.mean(axis=1)[cols]

            this_ind_nliMS = X_add_Sygen.loc[p,'nli_ind_MS']

            this_rmseY, this_deltaLEMSY, this_nonlinY = get_eval_scores(Y_Sygen.loc[p,cols].values,
                                                                        df_Y.loc[p,cols].values.astype(float),
                                                                        this_ind_nliMS)

            df_allRMSEY_Sygen.loc[p, 'best_models_mean'] = this_rmseY
            df_allRMSEX_Sygen.loc[p, 'best_models_mean'] = df_allRMSEX_Sygen.loc[p,best_models].mean()
            df_allDelLEMSY_Sygen.loc[p, 'best_models_mean'] = this_deltaLEMSY
            df_allDelLEMSX_Sygen.loc[p, 'best_models_mean'] = df_allDelLEMSX_Sygen.loc[p,best_models].mean()
        else:
            ind = all_names_short.index(best_models[0])
            df_Y.loc[p, :] = all_dfsY_Sygen[ind].loc[p,:]
            df_X.loc[p, :] = all_dfsX_Sygen[ind].loc[p,:]
            df_allRMSEY_Sygen.loc[p, 'best_models_mean'] =df_allRMSEY_Sygen.loc[p, best_models[0]]
            df_allRMSEX_Sygen.loc[p, 'best_models_mean'] = df_allRMSEX_Sygen.loc[p, best_models[0]]
            df_allDelLEMSY_Sygen.loc[p, 'best_models_mean'] = df_allDelLEMSY_Sygen.loc[p, best_models[0]]
            df_allDelLEMSX_Sygen.loc[p, 'best_models_mean'] = df_allDelLEMSX_Sygen.loc[p, best_models[0]]

    all_dfsY_Sygen.append(df_Y)
    all_dfsX_Sygen.append(df_X)

    return all_dfsY_Sygen,all_dfsX_Sygen,df_allRMSEY_Sygen, df_allRMSEX_Sygen,df_allDelLEMSY_Sygen,df_allDelLEMSX_Sygen



    # df_cluster_goodMatchX = pd.DataFrame()
    # all_rmse_sygen = []
    # all_rmse_emsci = []
    # all_lems_sygen = []
    # all_lems_emsci = []
    # for c in np.unique(adata.obs[cl]):
    #
    #     # SYGEN
    #     pats_thiscluster_Sygen_all = [s for s in X_add_Sygen[X_add_Sygen['cluster'] == c].index]
    #     m = df_cluster.loc[c, 'model']
    #     pats_thiscluster_Sygen = list(df_allRMSEX_Sygen.loc[pats_thiscluster_Sygen_all, m][df_allRMSEX_Sygen.loc[pats_thiscluster_Sygen_all, m]<=0.5].index)
    #
    #     if len(pats_thiscluster_Sygen)>0:
    #         df_cluster_goodMatchX.loc[c, 'plegia_para_Sygen'] = Counter(X_add_Sygen.loc[pats_thiscluster_Sygen].plegia)['para'] / len(
    #             pats_thiscluster_Sygen)
    #
    #         df_cluster_goodMatchX.loc[c, 'n_pats_Sygen'] = len(pats_thiscluster_Sygen)
    #
    #         df_cluster_goodMatchX.loc[c, 'rmse_med_Sygen'] = df_allRMSEY_Sygen.loc[pats_thiscluster_Sygen, m].median()
    #         df_cluster_goodMatchX.loc[c, 'rmse_2.5_Sygen'] = np.percentile(df_allRMSEY_Sygen.loc[pats_thiscluster_Sygen, m], 2.5)
    #         df_cluster_goodMatchX.loc[c, 'rmse_97.5_Sygen'] = np.percentile(df_allRMSEY_Sygen.loc[pats_thiscluster_Sygen, m], 97.5)
    #
    #         df_cluster_goodMatchX.loc[c, 'LEMS_med_Sygen'] = df_allDelLEMSY_Sygen.loc[pats_thiscluster_Sygen, m].median()
    #         df_cluster_goodMatchX.loc[c, 'LEMS_2.5_Sygen'] = np.percentile(df_allDelLEMSY_Sygen.loc[pats_thiscluster_Sygen, m], 2.5)
    #         df_cluster_goodMatchX.loc[c, 'LEMS_97.5_Sygen'] = np.percentile(df_allDelLEMSY_Sygen.loc[pats_thiscluster_Sygen, m], 97.5)
    #
    #         df_cluster_goodMatchX.loc[c, 'AISA_A_Sygen'] = Counter(X_add_Sygen.loc[pats_thiscluster_Sygen].aisa)['A'] / len(
    #             pats_thiscluster_Sygen)
    #         df_cluster_goodMatchX.loc[c, 'AISA_B_Sygen'] = Counter(X_add_Sygen.loc[pats_thiscluster_Sygen].aisa)['B'] / len(
    #             pats_thiscluster_Sygen)
    #         df_cluster_goodMatchX.loc[c, 'AISA_C_Sygen'] = Counter(X_add_Sygen.loc[pats_thiscluster_Sygen].aisa)['C'] / len(
    #             pats_thiscluster_Sygen)
    #         df_cluster_goodMatchX.loc[c, 'AISA_D_Sygen'] = Counter(X_add_Sygen.loc[pats_thiscluster_Sygen].aisa)['D'] / len(
    #             pats_thiscluster_Sygen)
    #
    #     # EMSCI
    #     pats_thiscluster_int_all = [int(s) for s in adata[adata.obs[cl] == c].obs.index]
    #     pats_thiscluster_int = list(df_allRMSEX.loc[pats_thiscluster_int_all, m]
    #                                   [df_allRMSEX.loc[pats_thiscluster_int_all, m] <= 0.5].index)
    #     pats_thiscluster = [str(p) for p in  pats_thiscluster_int]
    #
    #
    #     df_cluster_goodMatchX.loc[c, 'n_pats'] = len(pats_thiscluster)
    #
    #     if len(pats_thiscluster) > 0:
    #         df_cluster_goodMatchX.loc[c, 'rmse_med'] = adata.obs['rmseyMerge' + cl].loc[pats_thiscluster].median()
    #         df_cluster_goodMatchX.loc[c, 'rmse_2.5'] = np.percentile(adata.obs['rmseyMerge' + cl].loc[pats_thiscluster], 2.5)
    #         df_cluster_goodMatchX.loc[c, 'rmse_97.5'] = np.percentile(adata.obs['rmseyMerge' + cl].loc[pats_thiscluster], 97.5)
    #
    #         df_cluster_goodMatchX.loc[c, 'LEMS_med'] = adata.obs['LEMSyMerge' + cl].loc[pats_thiscluster].median()
    #         df_cluster_goodMatchX.loc[c, 'LEMS_2.5'] = np.percentile(adata.obs['LEMSyMerge' + cl].loc[pats_thiscluster], 2.5)
    #         df_cluster_goodMatchX.loc[c, 'LEMS_97.5'] = np.percentile(adata.obs['LEMSyMerge' + cl].loc[pats_thiscluster], 97.5)
    #
    #         df_cluster_goodMatchX.loc[c, 'AISA_A'] = Counter(X_add_toMatch.loc[pats_thiscluster_int].aisa)['A'] / len(
    #             pats_thiscluster_int)
    #         df_cluster_goodMatchX.loc[c, 'AISA_B'] = Counter(X_add_toMatch.loc[pats_thiscluster_int].aisa)['B'] / len(
    #             pats_thiscluster_int)
    #         df_cluster_goodMatchX.loc[c, 'AISA_C'] = Counter(X_add_toMatch.loc[pats_thiscluster_int].aisa)['C'] / len(
    #             pats_thiscluster_int)
    #         df_cluster_goodMatchX.loc[c, 'AISA_D'] = Counter(X_add_toMatch.loc[pats_thiscluster_int].aisa)['D'] / len(
    #             pats_thiscluster_int)
    #
    #         df_cluster_goodMatchX.loc[c, 'plegia_para'] = Counter(X_add_toMatch.loc[pats_thiscluster_int].plegia)['para'] / len(
    #             pats_thiscluster_int)
def get_eval_scores(this_Y, this_pred,this_ind_nliMS):

    # RMSE
    this_rmse = mean_squared_error(this_Y[this_ind_nliMS:], this_pred[this_ind_nliMS:], squared=False)

    # LEMS
    lems_true = this_Y[10:].sum()
    lems_pred = this_pred[10:].sum()
    this_deltaLEMS = (lems_true - lems_pred)

    # non linear score
    #this_nonlin = nonlineScore(this_pred[this_ind_nliMS:], this_Y[this_ind_nliMS:])
    this_nonlin = np.nan

    return this_rmse, this_deltaLEMS, this_nonlin
def softplus(x):
    y = np.log(1 + np.exp(x))
    return y
def linSc(sc):
    b = 0.4
    a = 99.63
    y = a * sc/ (sc + b)
    y[sc < 0] = 0
    return y
def nonlineScore(y_pred, y_true):
    target_low = y_true
    delta_top = linSc(y_pred) -  linSc(target_low+0.5)
    delta_bot = linSc((target_low-0.5)*np.heaviside(target_low-0.5, 0.5)) - linSc(y_pred)
    nonlinSc = np.sum((softplus(delta_top) + softplus(delta_bot)))

    return nonlinSc
def getNumerOfTwins(pats, all_names_short, folder, normalizeCounts = True):

    # get the twins identified for each patient
    df_neighs = pd.DataFrame(index=pats, columns=all_names_short)
    for m in tqdm.tqdm(all_names_short):
        neighs = pd.read_csv(os.path.join(folder, m, 'run_predictions', m + '_patsNeigh.csv'), index_col=0)
        for p in pats:
            try:
                df_neighs.loc[p, m] = neighs.loc[p, 'n_neighbour']
            except:
                df_neighs.loc[p, m] = np.nan

    # Get counts
    df_sum_counts = pd.DataFrame(columns=['Type', 'subgroup', 'count'])
    for m_i in tqdm.tqdm(all_names_short):

        if 'median' in m_i:
            continue

        # get type
        if 'RMSE' in m_i:
            this_type = 'Type 3'
        elif 'LEMS' in m_i:
            this_type = 'Type 1'
        elif 'MeanSc' in m_i:
            this_type = 'Type 2'
        elif 'ClosestTwin' in m_i:
            if 'SS' in m_i:
                this_type = 'Type 4B'
            else:
                # continue
                this_type = 'Type 4A'
        else:
            continue

        if 'aisa' in m_i:
            if 'nli' in m_i:
                if 'age' in m_i:
                    this_subset = 'AIS, NLI, Age/Sex'
                else:
                    this_subset = 'AIS, NLI'
            else:
                if 'age' in m_i:
                    this_subset = 'AIS, Age/Sex'
                else:
                    this_subset = 'AIS'
        else:
            if 'nli' in m_i:
                if 'age' in m_i:
                    this_subset = 'NLI, Age/Sex'
                else:
                    this_subset = 'NLI'
            else:
                if 'age' in m_i:
                    this_subset = 'Age/Sex'
                else:
                    this_subset = 'None'

        df_sub = df_neighs.loc[:, m_i].to_frame()
        df_sub.columns = ['count']
        df_sub['Type'] = this_type
        df_sub['subgroup'] = this_subset
        df_sum_counts = df_sum_counts.append(df_sub)

    # Summarize the models
    df_subset = pd.DataFrame(columns=['Type', 'subset', 'agglom', 'k'], index=all_names_short)
    for m_i in tqdm.tqdm(all_names_short):

        if 'median' in m_i:
            this_agglom = 'median'
        else:
            this_agglom = 'mean'

        # get type
        k = 'N.A.'
        if 'RMSE' in m_i:
            this_type = 'Type 3'
        elif 'LEMS' in m_i:
            this_type = 'Type 1'
        elif 'MeanSc' in m_i:
            this_type = 'Type 2'
        elif 'KNN' in m_i:
            if 'MS_SS' in m_i:
                this_type = 'Type 4B'
            else:
                this_type = 'Type 4A'

            if 'ClosestTwin' in m_i:
                k = 'cl'
            else:
                k = m_i.split('KNN_')[1].split('_')[0]
        else:
            continue

        if 'aisa' in m_i:
            if 'nli' in m_i:
                if 'age' in m_i:
                    this_subset = 'AIS, NLI, Age/Sex'
                else:
                    this_subset = 'AIS, NLI'
            else:
                if 'age' in m_i:
                    this_subset = 'AIS, Age/Sex'
                else:
                    this_subset = 'AIS'
        else:
            if 'nli' in m_i:
                if 'age' in m_i:
                    this_subset = 'NLI, Age/Sex'
                else:
                    this_subset = 'NLI'
            else:
                if 'age' in m_i:
                    this_subset = 'Age/Sex'
                else:
                    this_subset = 'None'

        df_subset.loc[m_i, 'Type'] = this_type
        df_subset.loc[m_i, 'subset'] = this_subset
        df_subset.loc[m_i, 'agglom'] = this_agglom
        df_subset.loc[m_i, 'k'] = k

    # normalize
    if normalizeCounts:
        df_sum_counts['count'] = df_sum_counts['count'] / len(pats)

    return df_sum_counts, df_subset


# Functional endpoints
def getFunctionalOutcomePreds(m, folderFunc, scores, pats, aisa_dic,emsci_data, asia_data):
    cols = []
    for s in scores:
        cols.append(s+ '_true')
        cols.append(s + '_twin')
    df_functional = pd.DataFrame(index=pats, columns=cols)
    #neighs = pd.read_csv(os.path.join(folder, m, 'run_predictions', m + '_patsNeigh.csv'), index_col=0)

    df_ais = pd.DataFrame(index=pats, columns=aisa_dic.keys())
    for p in tqdm.tqdm(pats):
        with open(os.path.join(folderFunc,m,'twins',str(p)), "rb") as fp:  # Unpickling
            twins = pickle.load(fp)
        # twins = []
        # for t in neighs.loc[p, 'Neighbour'].split(', '):
        #     if t[0] == '[':
        #         twins.append(int(t[1:]))
        #     elif t[:12] == 'Int64Index([':
        #         twins.append(int(t[12:]))
        #
        #     elif t[-1] == ']':
        #         twins.append(int(t[:-1]))
        #     elif ('length' in t) or ('dtype' in t):
        #         continue
        #     else:
        #         if '\n' in t:
        #             for i_t in t.split(',\n'):
        #                 if ('length' in i_t) or ('dtype' in i_t):
        #                     continue
        #                 else:
        #                     if i_t[-1] == ']':
        #                         twins.append(int(i_t[:-1]))
        #                     else:
        #                         twins.append(int(i_t))
        #
        #         else:
        #             twins.append(int(t))

        for s in scores:
            if s in ['AIS']:
                true_AIS = asia_data.loc[p, s]
                try:
                    np.isnan(true_AIS)
                    df_functional.loc[p, s + '_true'] = np.nan
                except:
                    if true_AIS == 'NT':
                        df_functional.loc[p, s + '_true'] = np.nan
                    else:
                        df_functional.loc[p, s + '_true'] = aisa_dic[true_AIS]  # .values[0]

                coll_ais = []
                coll_ais_class = []
                for this_twin in twins:
                    this_ais = asia_data.loc[this_twin, s]
                    try:
                        np.isnan(this_ais)
                        coll_ais.append(np.nan)
                    except:
                        if this_ais == 'NT':
                            coll_ais.append(np.nan)
                        else:
                            coll_ais.append(aisa_dic[this_ais])
                            coll_ais_class.append(this_ais)
                try:
                    df_functional.loc[p, s + '_twin'] = aisa_dic[Counter(coll_ais_class).most_common(1)[0][0]]#np.nanmedian(coll_ais)
                except:
                    df_functional.loc[p, s + '_twin'] = np.nan

                for this_AIS in aisa_dic.keys():
                    if len(coll_ais_class)>0:
                        df_ais.loc[p,this_AIS] = Counter( coll_ais_class)[this_AIS]/len(coll_ais_class)
                    else:
                        df_ais.loc[p, this_AIS] = np.nan

                # .value_counts().idxmax()[0]
            elif s in ['Walker']:
                df_functional.loc[p, s + '_true'] = emsci_data.loc[p, s]
                df_functional.loc[p, s + '_twin'] = emsci_data.loc[twins, s].mean()  # value_counts().idxmax()
            elif s in ['Self-Carer']:
                df_functional.loc[p, s + '_true'] = emsci_data.loc[p, s]
                df_functional.loc[p, s + '_twin'] = emsci_data.loc[twins, s].mean()  # value_counts().idxmax()
            else:
                df_functional.loc[p, s + '_true'] = emsci_data.loc[p, s]  # .values[0]
                df_functional.loc[p, s + '_twin'] = np.nanmedian(emsci_data.loc[twins, s].values)

    return df_functional, df_ais
def getFunctionalOutcomePreds_Cluster(folderFunc, scores, pats, aisa_dic,emsci_data, asia_data, df_cluster,df_pats_cluster ):
    cols = []
    for s in scores:
        cols.append(s+ '_true')
        cols.append(s + '_twin')
    df_functional = pd.DataFrame(index=pats, columns=cols)
    #neighs = pd.read_csv(os.path.join(folder, m, 'run_predictions', m + '_patsNeigh.csv'), index_col=0)

    df_ais = pd.DataFrame(index=pats, columns=aisa_dic.keys())
    for p in tqdm.tqdm(pats):

        # get the model for this patient
        m = df_cluster.loc[df_pats_cluster.loc[str(p)],'model']#.values[0]
        with open(os.path.join(folderFunc,m,'twins',str(p)), "rb") as fp:  # Unpickling
            twins = pickle.load(fp)
        # twins = []
        # for t in neighs.loc[p, 'Neighbour'].split(', '):
        #     if t[0] == '[':
        #         twins.append(int(t[1:]))
        #     elif t[:12] == 'Int64Index([':
        #         twins.append(int(t[12:]))
        #
        #     elif t[-1] == ']':
        #         twins.append(int(t[:-1]))
        #     elif ('length' in t) or ('dtype' in t):
        #         continue
        #     else:
        #         if '\n' in t:
        #             for i_t in t.split(',\n'):
        #                 if ('length' in i_t) or ('dtype' in i_t):
        #                     continue
        #                 else:
        #                     if i_t[-1] == ']':
        #                         twins.append(int(i_t[:-1]))
        #                     else:
        #                         twins.append(int(i_t))
        #
        #         else:
        #             twins.append(int(t))

        for s in scores:
            if s in ['AIS']:
                true_AIS = asia_data.loc[p, s]
                try:
                    np.isnan(true_AIS)
                    df_functional.loc[p, s + '_true'] = np.nan
                except:
                    if true_AIS == 'NT':
                        df_functional.loc[p, s + '_true'] = np.nan
                    else:
                        df_functional.loc[p, s + '_true'] = aisa_dic[true_AIS]  # .values[0]

                coll_ais = []
                coll_ais_class = []
                for this_twin in twins:
                    this_ais = asia_data.loc[this_twin, s]
                    try:
                        np.isnan(this_ais)
                        coll_ais.append(np.nan)
                    except:
                        if this_ais == 'NT':
                            coll_ais.append(np.nan)
                        else:
                            coll_ais.append(aisa_dic[this_ais])
                            coll_ais_class.append(this_ais)
                try:
                    df_functional.loc[p, s + '_twin'] = aisa_dic[Counter(coll_ais_class).most_common(1)[0][0]]#np.nanmedian(coll_ais)
                except:
                    df_functional.loc[p, s + '_twin'] = np.nan

                for this_AIS in aisa_dic.keys():
                    if len(coll_ais_class)>0:
                        df_ais.loc[p,this_AIS] = Counter( coll_ais_class)[this_AIS]/len(coll_ais_class)
                    else:
                        df_ais.loc[p, this_AIS] = np.nan

                # .value_counts().idxmax()[0]
            elif s in ['Walker']:
                df_functional.loc[p, s + '_true'] = emsci_data.loc[p, s]
                df_functional.loc[p, s + '_twin'] = emsci_data.loc[twins, s].mean()  # value_counts().idxmax()
            elif s in ['Self-Carer']:
                df_functional.loc[p, s + '_true'] = emsci_data.loc[p, s]
                df_functional.loc[p, s + '_twin'] = emsci_data.loc[twins, s].mean()  # value_counts().idxmax()
            else:
                df_functional.loc[p, s + '_true'] = emsci_data.loc[p, s]  # .values[0]
                df_functional.loc[p, s + '_twin'] = np.nanmedian(emsci_data.loc[twins, s].values)

    return df_functional, df_ais
def getFunctionalOutcomePreds_best(df_allRMSEX, folderFunc, scores, pats, aisa_dic, emsci_data, asia_data):
    df_allRMSEX_use = df_allRMSEX.drop('best_models_mean', axis=1)
    cols = []
    for s in scores:
        cols.append(s + '_true')
        cols.append(s + '_twin')
    df_functional = pd.DataFrame(index=pats, columns=cols)
    # neighs = pd.read_csv(os.path.join(folder, m, 'run_predictions', m + '_patsNeigh.csv'), index_col=0)

    df_ais = pd.DataFrame(index=pats, columns=aisa_dic.keys())
    for p in tqdm.tqdm(pats):

        # get the model for this patient
        m = df_allRMSEX_use.loc[p, :][df_allRMSEX_use.loc[p, :] == df_allRMSEX_use.loc[p, :].min()].index[-1]
        with open(os.path.join(folderFunc, m, 'twins', str(p)), "rb") as fp:  # Unpickling
            twins = pickle.load(fp)

        for s in scores:
            if s in ['AIS']:
                true_AIS = asia_data.loc[p, s]
                try:
                    np.isnan(true_AIS)
                    df_functional.loc[p, s + '_true'] = np.nan
                except:
                    if true_AIS == 'NT':
                        df_functional.loc[p, s + '_true'] = np.nan
                    else:
                        df_functional.loc[p, s + '_true'] = aisa_dic[true_AIS]  # .values[0]

                coll_ais = []
                coll_ais_class = []
                for this_twin in twins:
                    this_ais = asia_data.loc[this_twin, s]
                    try:
                        np.isnan(this_ais)
                        coll_ais.append(np.nan)
                    except:
                        if this_ais == 'NT':
                            coll_ais.append(np.nan)
                        else:
                            coll_ais.append(aisa_dic[this_ais])
                            coll_ais_class.append(this_ais)
                try:
                    df_functional.loc[p, s + '_twin'] = aisa_dic[
                        Counter(coll_ais_class).most_common(1)[0][0]]  # np.nanmedian(coll_ais)
                except:
                    df_functional.loc[p, s + '_twin'] = np.nan

                for this_AIS in aisa_dic.keys():
                    if len(coll_ais_class) > 0:
                        df_ais.loc[p, this_AIS] = Counter(coll_ais_class)[this_AIS] / len(coll_ais_class)
                    else:
                        df_ais.loc[p, this_AIS] = np.nan

                # .value_counts().idxmax()[0]
            elif s in ['Walker']:
                df_functional.loc[p, s + '_true'] = emsci_data.loc[p, s]
                df_functional.loc[p, s + '_twin'] = emsci_data.loc[
                    twins, s].mean()  # value_counts().idxmax()
            elif s in ['Self-Carer']:
                df_functional.loc[p, s + '_true'] = emsci_data.loc[p, s]
                df_functional.loc[p, s + '_twin'] = emsci_data.loc[
                    twins, s].mean()  # value_counts().idxmax()
            else:
                df_functional.loc[p, s + '_true'] = emsci_data.loc[p, s]  # .values[0]
                df_functional.loc[p, s + '_twin'] = np.nanmedian(emsci_data.loc[twins, s].values)

    return df_functional, df_ais
def plotSummaryScoresFunctional(all_df_functional,names,s,title = 'Walker', colors = ['black','red','blue'],
                                linestyles = ['-','-','-']):

    plt.subplots(1, figsize=(4, 4))
    plt.title(title)

    for i_m, df_functional in enumerate(all_df_functional):
        y_true = df_functional.loc[:,s+'_true'].dropna().astype(int)
        y_pred_prob = df_functional.loc[df_functional.loc[:, s + '_true'].dropna().index, s + '_twin'].dropna()
        y_true = y_true.loc[y_pred_prob.index]
        #precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)

        roc_auc = roc_auc_score(y_true,y_pred_prob)
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_true,y_pred_prob)
        plt.plot(false_positive_rate1, true_positive_rate1, color = colors[i_m], linestyle=linestyles[i_m],
                 label=names[i_m]+': AUC = %.2f ' %roc_auc)

    #plt.plot([0, 1], ls="--",c = 'k')
    plt.ylabel('True Positive Rate', fontsize = 14)
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.legend()
    return 0
def getSummaryScoresFunctional(df_functional,scores, m, df_ais,X_add_in,aisa_dic):

    cols = getColsSummaryDF(scores)
    df_functional_all = pd.DataFrame(index=[m], columns=cols)
    for s in scores:

        if s == 'Walker':
            y_true = df_functional.loc[:,s+'_true'].dropna().astype(int)
            y_pred_prob = df_functional.loc[df_functional.loc[:, s + '_true'].dropna().index, s + '_twin'].dropna()
            y_true = y_true.loc[y_pred_prob.index]


            precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)
            threshold = thresholds[np.abs(recalls - 0.9).argmin()]
            y_pred = ( y_pred_prob<threshold).astype(int)

            roc_auc = roc_auc_score(y_true,y_pred_prob)
            confusion_matrix(y_true,y_pred)
            acc = accuracy_score(y_true,y_pred)
            f1 = f1_score(y_true,y_pred)

            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_true,y_pred_prob)
            prev = np.mean(y_true)
            aps = average_precision_score(y_true, y_pred_prob)
            rel_APS = aps/prev



            plt.subplots(1, figsize=(5, 5))
            plt.title('Receiver Operating Characteristic')
            #plt.plot(false_positive_rate6m, true_positive_rate6m,'r',label = '6 months, AUC = %.3f ' %roc_auc6m )
            plt.plot(false_positive_rate1, true_positive_rate1, 'b',label='AUC = %.3f ' %roc_auc)
            plt.plot([0, 1], ls="--",c = 'k')
            #plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend()

            df_functional_all.loc[m, s + ' ACC'] = acc
            df_functional_all.loc[m, s + ' F1'] = f1
            df_functional_all.loc[m, s + ' roc_AUC'] = roc_auc
            df_functional_all.loc[m, s + ' rel_APS'] = rel_APS
            df_functional_all.loc[m, s + ' prev'] = prev
        elif s == 'Self-Carer':
            try:
                y_true = df_functional.loc[:,s+'_true'].dropna().astype(int)
                y_pred_prob = df_functional.loc[df_functional.loc[:, s + '_true'].dropna().index, s + '_twin'].dropna()
                y_true = y_true.loc[y_pred_prob.index]


                precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)
                threshold = thresholds[np.abs(recalls - 0.9).argmin()]
                y_pred = ( y_pred_prob<threshold).astype(int)

                roc_auc = roc_auc_score(y_true,y_pred_prob)
                confusion_matrix(y_true,y_pred)
                acc = accuracy_score(y_true,y_pred)
                f1 = f1_score(y_true,y_pred)

                false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_true,y_pred_prob)
                prev = np.mean(y_true)
                aps = average_precision_score(y_true, y_pred_prob)
                rel_APS = aps/prev



                plt.subplots(1, figsize=(5, 5))
                plt.title('Receiver Operating Characteristic')
                #plt.plot(false_positive_rate6m, true_positive_rate6m,'r',label = '6 months, AUC = %.3f ' %roc_auc6m )
                plt.plot(false_positive_rate1, true_positive_rate1, 'b',label='AUC = %.3f ' %roc_auc)
                plt.plot([0, 1], ls="--",c = 'k')
                #plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.legend()

                df_functional_all.loc[m, s + ' ACC'] = acc
                df_functional_all.loc[m, s + ' F1'] = f1
                df_functional_all.loc[m, s + ' roc_AUC'] = roc_auc
                df_functional_all.loc[m, s + ' rel_APS'] = rel_APS
                df_functional_all.loc[m, s + ' prev'] = prev
            except:
                df_functional_all.loc[m, s + ' ACC'] = np.nan
                df_functional_all.loc[m, s + ' F1'] = np.nan
                df_functional_all.loc[m, s + ' roc_AUC'] = np.nan
                df_functional_all.loc[m, s + ' rel_APS'] = np.nan
                df_functional_all.loc[m, s + ' prev'] = np.nan
        elif s == 'AIS':
            # multiclass problem
            all_AIS_roc = []
            for this_ais in range(len(df_functional.loc[:,s+'_true'].dropna().unique())):
                # for each class calculate the one-vs-rest AUC and APS
                y_true = (df_functional.loc[:, s + '_true'].dropna().astype(int)==this_ais).astype(int)
                y_pred_proba = df_ais.loc[y_true.index,list(aisa_dic.keys())[this_ais]].dropna()
                y_true = y_true.loc[y_pred_proba.index]

                this_roc = roc_auc_score(y_true, y_pred_proba)
                all_AIS_roc.append(this_roc)

                df_functional_all.loc[m, s + ' '+list(aisa_dic.keys())[this_ais]] = this_roc
            df_functional_all.loc[m, s + ' Mean'] = np.mean(all_AIS_roc)

            y_true_val = df_functional.loc[:,s+'_true'].dropna()
            y_pred_val = df_functional.loc[df_functional.loc[:,s+'_true'].dropna().index,s+'_twin'].dropna()
            y_true_val = y_true_val.loc[y_pred_val.index]

            for p in y_true_val.index:
                df_ais.loc[p,'BL'] = aisa_dic[X_add_in.loc[p,'aisa']]
                this_y_true = aisa_dic[X_add_in.loc[p,'aisa']]
                if np.isnan(this_y_true):
                    df_ais.loc[p,'impr_min1'] = np.nan
                    df_ais.loc[p,'impr_0'] = np.nan
                    df_ais.loc[p, 'impr_1'] = np.nan
                    df_ais.loc[p, 'impr_2'] = np.nan
                    df_ais.loc[p, 'impr_3'] = np.nan
                    df_ais.loc[p, 'impr_4'] = np.nan
                else:
                    df_ais.loc[p,'impr_min1'] = df_ais.loc[p].iloc[this_y_true+1:5].sum()
                    df_ais.loc[p,'impr_0'] = df_ais.loc[p].iloc[this_y_true]

                    if (this_y_true + 1)<5:
                        df_ais.loc[p, 'impr_1'] = df_ais.loc[p].iloc[this_y_true + 1]
                    else:
                        df_ais.loc[p, 'impr_1'] = np.nan

                    if (this_y_true + 2)<5:
                        df_ais.loc[p, 'impr_2'] = df_ais.loc[p].iloc[this_y_true + 2]
                    else:
                        df_ais.loc[p, 'impr_2'] = np.nan

                    if (this_y_true + 3)<5:
                        df_ais.loc[p, 'impr_3'] = df_ais.loc[p].iloc[this_y_true + 3]
                    else:
                        df_ais.loc[p, 'impr_3'] = np.nan

                    if (this_y_true + 4)<5:
                        df_ais.loc[p, 'impr_4'] = df_ais.loc[p].iloc[this_y_true + 4]
                    else:
                        df_ais.loc[p, 'impr_4'] = np.nan

            # Get prediction performance for binary task of AIS conversion
            y_impr_true = y_true_val-df_ais.loc[y_true_val.index,'BL']
            y_impr_binary_true = (y_impr_true >0).astype(int)
            y_impr_binary_pred_prob = df_ais.loc[y_impr_binary_true.index,'impr_min1']

            precisions_impr, recalls_impr, thresholds_impr = precision_recall_curve(y_impr_binary_true,
                                                                                    y_impr_binary_pred_prob)
            threshold_impr = thresholds_impr[np.abs(recalls_impr - 0.9).argmin()]

            # Check if multiple classes are present:
            if len(np.unique(y_impr_binary_true))>1:
                roc_auc_impr = roc_auc_score(y_impr_binary_true, y_impr_binary_pred_prob)
                #acc_impr = accuracy_score(y_impr_binary_true, y_impr_binary_pred)
                #f1_impr = f1_score(y_impr_binary_true, y_impr_binary_pred)

                #false_positive_rate1_impr, true_positive_rate1_impr, threshold1_impr = roc_curve(y_impr_binary_true, y_impr_binary_pred_prob)
                prev_impr = np.mean(y_impr_binary_true)
                aps_impr = average_precision_score(y_impr_binary_true, y_impr_binary_pred_prob)
                rel_APS_impr = aps_impr / prev_impr
            else:
                print('Only this class present for improvement:'+str(np.unique(y_impr_binary_true)))
                roc_auc_impr = np.nan
                prev_impr = np.mean(y_impr_binary_true)
                rel_APS_impr = np.nan


            df_res_Ais = pd.DataFrame(index=['all'] +list(range(0, 3)), columns=['AUC', 'relAPS'])
            df_res_Ais.loc['all', 'AUC'] = roc_auc_impr
            df_res_Ais.loc['all', 'relAPS'] = rel_APS_impr

            # FOR AIS A-C
            for this_ais in range(0,3):
                pats_ais = list(set(y_impr_true.index)&set(df_ais[df_ais['BL']==this_ais].index))

                # Just predict AIS grade (stays the same)
                # y_true = (df_functional.loc[pats_ais, s + '_true'].dropna().astype(int)==this_ais).astype(int)
                #
                # y_pred = (df_functional.loc[y_true.index, s + '_twin'].astype(int)==this_ais).astype(int)
                # confusion_matrix(y_true, y_pred)
                #
                # y_pred_proba = df_ais.loc[y_true.index,list(aisa_dic.keys())[this_ais]].dropna()
                # y_true = y_true.loc[y_pred_proba.index]

                #this_roc = roc_auc_score(y_true, y_pred_proba)


                try:
                    df_res_Ais.loc[this_ais,'AUC'] = roc_auc_score(y_impr_binary_true.loc[pats_ais],
                                                                   y_impr_binary_pred_prob.loc[pats_ais])
                    df_functional_all.loc[m, s + '_AUC_min1impr_'+str(this_ais)] = roc_auc_score(y_impr_binary_true.loc[pats_ais],
                                                                   y_impr_binary_pred_prob.loc[pats_ais])
                    df_res_Ais.loc[this_ais, 'relAPS'] = average_precision_score(y_impr_binary_true.loc[pats_ais],
                                                                    y_impr_binary_pred_prob.loc[pats_ais])/\
                                                         np.mean(y_impr_binary_true.loc[pats_ais])

                    df_functional_all.loc[m, s + '_relAPS_min1impr_' + str(this_ais)] = average_precision_score(y_impr_binary_true.loc[pats_ais],
                                                                    y_impr_binary_pred_prob.loc[pats_ais])/\
                                                         np.mean(y_impr_binary_true.loc[pats_ais])
                except:
                    df_res_Ais.loc[this_ais, 'AUC'] = np.nan
                    df_functional_all.loc[m, s + '_AUC_min1impr_' + str(this_ais)] = np.nan
                    df_functional_all.loc[m, s + '_relAPS_min1impr_' + str(this_ais)] = np.nan
                    df_res_Ais.loc[this_ais, 'relAPS'] = np.nan


                df_res_Ais.loc[this_ais, 'prev'] = np.mean(y_impr_binary_true.loc[pats_ais])

            rmse  = mean_squared_error(y_true_val,y_pred_val, squared=False)
            abs_err = y_true_val-y_pred_val

            bins =np.arange(-4,5)
            plt.figure()
            plt.hist(abs_err, bins = bins,color='black')
            plt.xlabel(s + ': True - Twin')
            #
            # print(np.nanmean(delta))
            # print(np.nanstd(delta))
            df_functional_all.loc[m,s+'_2.5']= np.percentile(abs_err.values, 2.5)
            df_functional_all.loc[m,s+'_med']=np.percentile(abs_err.values, 50)
            df_functional_all.loc[m,s+'_97.5']=np.percentile(abs_err.values, 97.5)
        else:
            rmse  = mean_squared_error(df_functional.loc[:,s+'_true'].dropna(),
                                       df_functional.loc[df_functional.loc[:,s+'_true'].dropna().index,s+'_twin'], squared=False)
            abs_err = df_functional.loc[:,s+'_true'].dropna()-\
                      df_functional.loc[df_functional.loc[:,s+'_true'].dropna().index,s+'_twin']
            # delta = list((df_functional.loc[:,s+'_true'] - df_functional.loc[:,s+'_twin']).values)

            plt.figure()
            plt.hist(abs_err,color='black')
            plt.xlabel(s + ': True - Twin')
            #
            # print(np.nanmean(delta))
            # print(np.nanstd(delta))
            df_functional_all.loc[m,s+'_2.5']= np.percentile(abs_err.values, 2.5)
            df_functional_all.loc[m,s+'_med']=np.percentile(abs_err.values, 50)
            df_functional_all.loc[m,s+'_97.5']=np.percentile(abs_err.values, 97.5)
    return df_functional_all, df_res_Ais
def getFunctionalOutcomePreds_Sygen(m,folderFunc_sygen,scores,pats_Sygen,aisa_dic,emsci_data,sygen_data,time_point,
                                    aisa_data,asia_data_sygen):

    #neighs = pd.read_csv(os.path.join(output_path_Sygen, m, 'run_predictions', m + '_patsNeigh.csv'), index_col=0)
    cols = []
    for s in scores:
        cols.append(s + '_true')
        cols.append(s + '_twin')
    df_functional = pd.DataFrame(index =pats_Sygen,columns = cols )
    df_ais = pd.DataFrame(index=pats_Sygen, columns=aisa_dic.keys())
    for p in tqdm.tqdm(pats_Sygen):

        with open(os.path.join(folderFunc_sygen,m,'twins',p), "rb") as fp:
            twins = pickle.load(fp)

        for s in scores:
            if s in ['AIS']:

                true_AIS = asia_data_sygen.loc[p, s]
                try:
                    true_AIS = aisa_dic[true_AIS[-1]]
                except:
                    true_AIS = np.nan
                df_functional.loc[p, s + '_true'] = true_AIS

                coll_ais = []
                coll_ais_class = []
                for this_twin in twins:
                    this_ais = aisa_data.loc[this_twin, s]
                    try:
                        np.isnan(this_ais)
                        coll_ais.append(np.nan)
                    except:
                        if this_ais == 'NT':
                            coll_ais.append(np.nan)
                        else:
                            coll_ais.append(aisa_dic[this_ais])
                            coll_ais_class.append(this_ais)
                df_functional.loc[p, s + '_twin'] = np.nanmedian(coll_ais)

                for this_AIS in aisa_dic.keys():
                    if len(coll_ais_class)==0:
                        df_ais.loc[p, this_AIS] = np.nan
                    else:
                        df_ais.loc[p, this_AIS] = Counter(coll_ais_class)[this_AIS] / len(coll_ais_class)

            elif s in ['Walker']:
                df_functional.loc[p, s + '_true'] = (int(sygen_data[sygen_data.ptid==p]['modben'+str(time_point)].values>4))
                df_functional.loc[p, s + '_twin'] = emsci_data.loc[twins, s].mean()  # value_counts().idxmax()
            else:
                print('Undefined score for Sygen!')

    return df_functional, df_ais
def getFunctionalOutcomePreds_ClusterSygen(folderFunc, scores, pats, aisa_dic,emsci_data,
                                           asia_data,asia_data_sygen, df_cluster,df_pats_cluster,
                                           sygen_data, time_point):
    cols = []
    for s in scores:
        cols.append(s+ '_true')
        cols.append(s + '_twin')
    df_functional = pd.DataFrame(index=pats, columns=cols)

    df_ais = pd.DataFrame(index=pats, columns=aisa_dic.keys())
    for p in tqdm.tqdm(pats):

        # get the model for this patient
        m = df_cluster.loc[df_pats_cluster.loc[str(p)],'model']
        with open(os.path.join(folderFunc,m,'twins',str(p)), "rb") as fp:
            twins = pickle.load(fp)
        # twins = []
        # for t in neighs.loc[p, 'Neighbour'].split(', '):
        #     if t[0] == '[':
        #         twins.append(int(t[1:]))
        #     elif t[:12] == 'Int64Index([':
        #         twins.append(int(t[12:]))
        #
        #     elif t[-1] == ']':
        #         twins.append(int(t[:-1]))
        #     elif ('length' in t) or ('dtype' in t):
        #         continue
        #     else:
        #         if '\n' in t:
        #             for i_t in t.split(',\n'):
        #                 if ('length' in i_t) or ('dtype' in i_t):
        #                     continue
        #                 else:
        #                     if i_t[-1] == ']':
        #                         twins.append(int(i_t[:-1]))
        #                     else:
        #                         twins.append(int(i_t))
        #
        #         else:
        #             twins.append(int(t))

        for s in scores:
            if s in ['AIS']:
                true_AIS = asia_data_sygen.loc[p, s]
                try:
                    np.isnan(true_AIS)
                    df_functional.loc[p, s + '_true'] = np.nan
                except:
                    if true_AIS == 'NT':
                        df_functional.loc[p, s + '_true'] = np.nan
                    else:
                        df_functional.loc[p, s + '_true'] = aisa_dic[true_AIS]  # .values[0]

                coll_ais = []
                coll_ais_class = []
                for this_twin in twins:
                    this_ais = asia_data.loc[this_twin, s]
                    try:
                        np.isnan(this_ais)
                        coll_ais.append(np.nan)
                    except:
                        if this_ais == 'NT':
                            coll_ais.append(np.nan)
                        else:
                            coll_ais.append(aisa_dic[this_ais])
                            coll_ais_class.append(this_ais)
                try:
                    df_functional.loc[p, s + '_twin'] = aisa_dic[Counter(coll_ais_class).most_common(1)[0][0]]#np.nanmedian(coll_ais)
                except:
                    df_functional.loc[p, s + '_twin'] = np.nan

                for this_AIS in aisa_dic.keys():
                    if len(coll_ais_class)>0:
                        df_ais.loc[p,this_AIS] = Counter( coll_ais_class)[this_AIS]/len(coll_ais_class)
                    else:
                        df_ais.loc[p, this_AIS] = np.nan

                # .value_counts().idxmax()[0]
            elif s in ['Walker']:
                df_functional.loc[p, s + '_true'] = (int(sygen_data[sygen_data.ptid==p]['modben'+str(time_point)].values>4))
                df_functional.loc[p, s + '_twin'] = emsci_data.loc[twins, s].mean()  # value_counts().idxmax()
            else:
                print('Undefined score for Sygen!')

    return df_functional, df_ais
def getFunctionalOutcomePreds_bestSygen(df_allRMSEX, folderFunc, scores, pats,
                                        aisa_dic, emsci_data, asia_data,asia_data_sygen,
                                        sygen_data, time_point):
    df_allRMSEX_use = df_allRMSEX.drop('best_models_mean', axis=1)
    cols = []
    for s in scores:
        cols.append(s + '_true')
        cols.append(s + '_twin')
    df_functional = pd.DataFrame(index=pats, columns=cols)

    df_ais = pd.DataFrame(index=pats, columns=aisa_dic.keys())
    for p in tqdm.tqdm(pats):

        # get the model for this patient
        m = df_allRMSEX_use.loc[p, :][df_allRMSEX_use.loc[p, :] == df_allRMSEX_use.loc[p, :].min()].index[-1]
        with open(os.path.join(folderFunc, m, 'twins', str(p)), "rb") as fp:  # Unpickling
            twins = pickle.load(fp)

        for s in scores:
            if s in ['AIS']:
                true_AIS = asia_data_sygen.loc[p, s]
                try:
                    np.isnan(true_AIS)
                    df_functional.loc[p, s + '_true'] = np.nan
                except:
                    if true_AIS == 'NT':
                        df_functional.loc[p, s + '_true'] = np.nan
                    else:
                        df_functional.loc[p, s + '_true'] = aisa_dic[true_AIS]  # .values[0]

                coll_ais = []
                coll_ais_class = []
                for this_twin in twins:
                    this_ais = asia_data.loc[this_twin, s]
                    try:
                        np.isnan(this_ais)
                        coll_ais.append(np.nan)
                    except:
                        if this_ais == 'NT':
                            coll_ais.append(np.nan)
                        else:
                            coll_ais.append(aisa_dic[this_ais])
                            coll_ais_class.append(this_ais)
                try:
                    df_functional.loc[p, s + '_twin'] = aisa_dic[
                        Counter(coll_ais_class).most_common(1)[0][0]]  # np.nanmedian(coll_ais)
                except:
                    df_functional.loc[p, s + '_twin'] = np.nan

                for this_AIS in aisa_dic.keys():
                    if len(coll_ais_class) > 0:
                        df_ais.loc[p, this_AIS] = Counter(coll_ais_class)[this_AIS] / len(coll_ais_class)
                    else:
                        df_ais.loc[p, this_AIS] = np.nan

                # .value_counts().idxmax()[0]
            elif s in ['Walker']:
                df_functional.loc[p, s + '_true'] = (int(sygen_data[sygen_data.ptid==p]['modben'+str(time_point)].values>4))
                df_functional.loc[p, s + '_twin'] = emsci_data.loc[
                    twins, s].mean()  # value_counts().idxmax()
            else:
                print('Undefined score for Sygen!')

    return df_functional, df_ais
def getColsSummaryDF(scores):
    cols = []
    for s in scores:
        if s == 'AIS':
            cols.append('AIS A')
            cols.append('AIS B')
            cols.append('AIS C')
            cols.append('AIS D')
            cols.append('AIS E')
            cols.append('AIS Mean')
            cols.append('AIS_2.5')
            cols.append('AIS_med')
            cols.append('AIS_97.5')
            cols.append('AIS_mean')
            cols.append('AIS_std')
            for this_ais in range(0,3):
                cols.append('AIS_AUC_min1impr_' + str(this_ais))
                cols.append('AIS_relAPS_min1impr_' + str(this_ais))

        elif s in ['Walker','Self-Carer','Other']:
            cols.append(s + ' roc_AUC')
            cols.append(s + ' ACC')
            cols.append(s + ' F1')
            cols.append(s + ' rel_APS')
            cols.append(s + ' prev')

        else:
            cols.append(s + '_med')
            cols.append(s + '_2.5')
            cols.append(s + '_97.5')
    return cols




# For plotting
def plotClusters(adata, output_path,cl= 'kmeans'):
    sc.pl.umap(adata, color='kmeans', size=100, palette=sc.pl.palettes.vega_10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'figures', 'DimRed', 'adata_all' + cl + '_clusters_' + cl + '.png'))
    plt.savefig(os.path.join(output_path, 'figures', 'DimRed', 'adata_all' + cl + '_clusters_' + cl + '.pdf'),
                format='pdf')

    # Heatmap characterizing the clusters
    adata.obs[cl] = adata.obs[cl].astype('category')
    plt.figure()
    sc.tl.dendrogram(adata, groupby=cl)
    ax = sc.pl.heatmap(adata, adata.var_names, groupby=cl, figsize=(7, 3),
                       var_group_positions=[(0, 9), (10, 19), (20, 47), (47, 75), (76, 103), (104, 131)], use_raw=False,
                       cmap='bwr',
                       var_group_labels=['l.MS', 'r.MS', 'l.LTS', 'r.LTS', 'l.PPS', 'r.PPS'], var_group_rotation=0,
                       dendrogram='dendrogram_' + cl)
    plt.savefig(os.path.join(output_path, 'figures', 'DimRed', 'adata_all' + cl + '_heatmap_' + cl + '.png'))
    plt.savefig(os.path.join(output_path, 'figures', 'DimRed', 'adata_all' + cl + '_heatmap_' + cl + '.pdf'),
                format='pdf')
def plotNumberOfTwins(df_sum_counts,output_path, all_names_short):

    # Prepare plot
    m_types = []
    m_isAISA = []
    m_isNLI = []
    m_isNone = []
    m_isAGE = []
    for m_i in all_names_short:
        if 'RMSE' in m_i:
            m_types.append('Type 3')
        elif 'LEMS' in m_i:
            m_types.append('Type 1')
        elif 'MeanSc' in m_i:
            m_types.append('Type 2')
        else:
            if 'SS' in m_i:
                m_types.append('Type 4B')
            else:
                if 'best' in m_i or 'multi' in m_i:
                    m_types.append('other')
                else:
                    m_types.append('Type 4A')

        if 'aisa' in m_i:
            m_isAISA.append(True)
        else:
            m_isAISA.append(False)

        if '_nli_' in m_i:
            m_isNLI.append(True)
        else:
            m_isNLI.append(False)

        if '_age_' in m_i:
            m_isAGE.append(True)
        else:
            m_isAGE.append(False)

        if ('_aisa_' in m_i) or ('_nli_' in m_i) or ('_age_' in m_i):
            m_isNone.append(False)
        else:
            m_isNone.append(True)

        # Cluster all EMSCI patients and split into train/test sets

    my_pal = \
        {'Age/Sex': 'lemonchiffon',
         'None': 'white',
         'AIS, Age/Sex': 'sandybrown',
         'AIS, NLI': 'mediumpurple',
         'AIS, NLI, Age/Sex': 'grey',
         'NLI, Age/Sex': 'darkseagreen',
         'AIS': 'lightcoral',
         'NLI': 'lightsteelblue'}


    sns.set_style("whitegrid")
    plt.figure(figsize=(5, 2.5))
    ax = sns.boxplot(x=df_sum_counts['Type'],
                     y=df_sum_counts['count'],
                     hue=df_sum_counts['subgroup'],
                     order=['Type 1', 'Type 2', 'Type 3'],  # ,'Type 4A','Type 4B'
                     hue_order=['None', 'AIS', 'NLI', 'Age/Sex', 'AIS, NLI', 'AIS, Age/Sex',
                                'NLI, Age/Sex', 'AIS, NLI, Age/Sex'],
                     palette=my_pal,
                     whis=[0, 100],
                     boxprops=dict(edgecolor="k"))
    plt.xlabel('')
    plt.ylabel('Number of twins [%]', fontsize=16)
    # plt.ylim([0,100])
    plt.tick_params(axis='both', which='major', labelsize=16)
    n = 5  # this was for tinkering

    # Loop over the bars
    plt.legend('', frameon=False)
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(os.path.join(output_path, 'top_n_neigh.png'))
    plt.savefig(os.path.join(output_path, 'top_n_neigh.pdf'))

    hatches = ['', '\\', '//', '', '\\', '\\', '', '\\',
               '', '\\', '', '', '\\', '\\', '', '\\',
               '', '\\', '', '', '\\', '\\', '', '\\']
    for i, thisbar in enumerate(ax.artists):

        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[i])

        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color('k')
            line.set_mfc('k')
            line.set_mec('k')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    for i, legpatch in enumerate(ax.get_legend().get_patches()):
        # col_edge = cols[i]#legpatch.get_edgecolor()
        # legpatch.set_edgecolor(col_edge)

        col_face = legpatch.get_facecolor()
        legpatch.set_facecolor(col_face)
        # legpatch.set_facecolor(my_pal)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'top_n_neigh.png'))

    plt.figure(figsize=(5, 2.5))
    ax = sns.boxplot(x=df_sum_counts['Type'],
                     y=df_sum_counts['count'],
                     hue=df_sum_counts['subgroup'],
                     order=['Type 1', 'Type 2', 'Type 3'],  # ,'Type 4A','Type 4B'
                     hue_order=['None', 'AIS', 'NLI', 'Age/Sex', 'AIS, NLI', 'AIS, Age/Sex',
                                'NLI, Age/Sex', 'AIS, NLI, Age/Sex'],
                     palette=my_pal,
                     whis=[0, 100],
                     boxprops=dict(edgecolor="k"))
    plt.xlabel('')
    plt.ylabel('Number of twins', fontsize=16)
    plt.ylim([100, 850])
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Loop over the bars
    plt.legend('', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'bottom_n_neigh.png'))
    plt.savefig(os.path.join(output_path, 'bottom_n_neigh.pdf'))

    hatches = ['', '\\', '//', '', '\\', '\\', '', '\\',
               '', '\\', '', '', '\\', '\\', '', '\\',
               '', '\\', '', '', '\\', '\\', '', '\\']
    cols = ['k', 'k', 'r', 'k', 'r', 'k', 'r', 'r',
            'k', 'k', 'r', 'k', 'r', 'k', 'r', 'r',
            'k', 'k', 'r', 'k', 'r', 'k', 'r', 'r']
    for i, thisbar in enumerate(ax.artists):

        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[i])

        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color('k')
            line.set_mfc('k')
            line.set_mec('k')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    for i, legpatch in enumerate(ax.get_legend().get_patches()):
        # col_edge = cols[i]#legpatch.get_edgecolor()
        # legpatch.set_edgecolor(col_edge)

        col_face = legpatch.get_facecolor()
        legpatch.set_facecolor(col_face)
        # legpatch.set_facecolor(my_pal)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'bottom_n_neigh.png'))
    plt.savefig(os.path.join(output_path, 'bottom_n_neigh.pdf'))

    return 0
def plotIndividualPatTraj_1side_recovery(p, ms_use, Y_MS,
                                         df_med_lst, df_low_lst, df_top_lst,
                                         labels, savepath, name='x', doSave=True):
    if name == 'x':
        name = p
    colors = ['k', 'green', 'blue', 'yellow', 'orange', 'purple', 'lightblue', 'coral']
    ms_use_l = [m for m in ms_use if m[-1] == 'l']
    ms_use_r = [m for m in ms_use if m[-1] == 'r']
    ms_plt_label = [m[:-2] for m in ms_use if m[-1] == 'l']

    plt.figure(figsize=(2.8, 2.5))

    plt.plot(ms_plt_label, Y_MS.loc[p, ms_use_l], 'o-', alpha=0.75, color='r', label='True',
             markersize=10)
    for i in range(0, len(df_med_lst)):
        plt.plot(ms_plt_label, df_med_lst[i].loc[p, ms_use_l], 'o-', color=colors[i],
                 label=labels[i], alpha=0.75, markersize=10)

        try:
            plt.fill_between(ms_plt_label, df_low_lst[i].loc[p, ms_use_l].values.astype(float),
                             df_top_lst[i].loc[p, ms_use_l].values.astype(float), alpha=0.25,
                             linewidth=0,
                             zorder=3,
                             color=colors[i])
        except:
            continue

    plt.ylim((-0.2, 5.8))
    plt.grid(False)
    plt.ylabel('MS left', fontsize=14)
    # plt.title('recovered')
    plt.xticks(np.arange(len(ms_plt_label),step = 2),
               [ms_plt_label[i] for i in np.arange(len(ms_plt_label),step = 2)],rotation=0, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)

    plt.tight_layout()
    if doSave:
        plt.savefig(os.path.join(savepath, name + '_MS.png'))
        plt.savefig(os.path.join(savepath, name + '_MS.pdf'), format='pdf')
def plotIndividualPatTraj_2sidesManyModels(p, ms_use, Y_MS, X_MS,
                                 df_med_lst, df_medX_lst,use_thick,
                                 labels, savepath, name='x', doSave=True):
    if name == 'x':
        name = p

    ms_use_l = [m for m in ms_use if m[-1] == 'l']
    ms_use_r = [m for m in ms_use if m[-1] == 'r']
    colors = ['red','black','blue','orange','green', 'yellow']

    plt.figure(figsize=(8, 8))
    plt.subplot(2,2,1)
    plt.plot(ms_use_l, X_MS.loc[p, ms_use_l], 'ko-', label='True')
    for i in range(0, len(df_medX_lst)):
        plt.plot(ms_use_l, df_medX_lst[i].loc[p, ms_use_l], '-', color=colors[use_thick[i]], alpha=0.25)


    plt.ylim((0, 6))
    plt.ylabel('MS left - acute')


    plt.subplot(2, 2, 2)
    plt.plot(ms_use_r, X_MS.loc[p, ms_use_r], 'ko-', label='True')
    for i in range(0, len(df_medX_lst)):
        plt.plot(ms_use_r, df_medX_lst[i].loc[p, ms_use_r], '-', color=colors[use_thick[i]], alpha=0.25)



    plt.ylim((0, 6))
    plt.ylabel('MS right - acute')

    plt.subplot(2,2,3)
    plt.plot(ms_use_l, Y_MS.loc[p, ms_use_l], 'ko-', label='True')
    for i in range(0, len(df_med_lst)):
        plt.plot(ms_use_l, df_med_lst[i].loc[p, ms_use_l], '-', color=colors[use_thick[i]], alpha=0.25)


    plt.ylim((0, 6))
    plt.ylabel('MS left - recovered')


    plt.subplot(2, 2, 4)
    plt.plot(ms_use_r, Y_MS.loc[p, ms_use_r], 'ko-', label='True')
    for i in range(0, len(df_med_lst)):
        plt.plot(ms_use_r, df_med_lst[i].loc[p, ms_use_r], '-', color=colors[use_thick[i]], alpha=0.25)


    plt.ylim((0, 6))
    plt.ylabel('MS right - recovered')
    plt.legend()

    plt.suptitle(p, fontsize=14)
    plt.tight_layout()
    if doSave:
        plt.savefig(os.path.join(savepath,name + '_MS.png'))
def plotIndividualPatTraj_1side(p, ms_use, Y_MS, X_MS,
                                 df_med_lst, df_low_lst, df_top_lst,
                                 df_medX_lst, df_lowX_lst, df_topX_lst,
                                 labels, savepath, name='x', doSave=True):
    if name == 'x':
        name = p
    colors = ['royalblue', 'green', 'blue', 'yellow', 'orange', 'purple', 'lightblue', 'coral']
    ms_use_l = [m for m in ms_use if m[-1] == 'l']
    ms_use_r = [m for m in ms_use if m[-1] == 'r']
    ms_plt_label = [m[:-2] for m in ms_use if m[-1] == 'l']

    plt.figure(figsize=(3.5, 2))
    plt.subplot(1,2,1)
    plt.plot(ms_plt_label, X_MS.loc[p, ms_use_l], 'o-',color = 'darkorange', label='True')
    for i in range(0, len(df_medX_lst)):
        plt.plot(ms_plt_label, df_medX_lst[i].loc[p, ms_use_l], 'o-', color=colors[i], label=labels[i], alpha=0.5)

        try:
            plt.fill_between(ms_plt_label, df_lowX_lst[i].loc[p, ms_use_l].values.astype(float), df_topX_lst[i].loc[p, ms_use_l].values.astype(float), alpha=0.25, linewidth=0,
                             zorder=3,
                             color=colors[i])
        except:
            continue


    plt.ylim((-0.2, 5.8))
    plt.ylabel('MS left')
    plt.xticks(rotation=90)
    plt.title('acute')


    # plt.subplot(2, 2, 2)
    # plt.plot(ms_use_r, X_MS.loc[p, ms_use_r], 'ko-', label='True')
    # for i in range(0, len(df_medX_lst)):
    #     plt.plot(ms_use_r, df_medX_lst[i].loc[p, ms_use_r], 'o-', color=colors[i], label=labels[i], alpha=0.5)
    #
    #     try:
    #         plt.fill_between(ms_use_r, df_lowX_lst[i].loc[p, ms_use_r].values.astype(float),
    #                          df_topX_lst[i].loc[p, ms_use_r].values.astype(float), alpha=0.25, linewidth=0,
    #                          zorder=3,
    #                          color=colors[i])
    #     except:
    #         continue
    #
    # plt.ylim((0, 6))
    # plt.ylabel('MS right - acute')

    plt.subplot(1,2,2)
    plt.plot(ms_plt_label, Y_MS.loc[p, ms_use_l], 'o-', color = 'darkorange', label='True')
    for i in range(0, len(df_med_lst)):
        plt.plot(ms_plt_label, df_med_lst[i].loc[p, ms_use_l], 'o-', color=colors[i], label=labels[i], alpha=0.5)

        try:
            plt.fill_between(ms_plt_label, df_low_lst[i].loc[p, ms_use_l].values.astype(float), df_top_lst[i].loc[p, ms_use_l].values.astype(float), alpha=0.25, linewidth=0,
                             zorder=3,
                             color=colors[i])
        except:
            continue


    plt.ylim((-0.2, 5.8))
    plt.ylabel('MS left')
    plt.title('recovered')
    plt.xticks(rotation=90)


    # plt.subplot(2, 2, 4)
    # plt.plot(ms_use_r, Y_MS.loc[p, ms_use_r], 'ko-', label='True')
    # for i in range(0, len(df_med_lst)):
    #     plt.plot(ms_use_r, df_med_lst[i].loc[p, ms_use_r], 'o-', color=colors[i], label=labels[i], alpha=0.5)
    #
    #     try:
    #         plt.fill_between(ms_use_r, df_low_lst[i].loc[p, ms_use_r].values.astype(float),
    #                          df_top_lst[i].loc[p, ms_use_r].values.astype(float), alpha=0.25, linewidth=0,
    #                          zorder=3,
    #                          color=colors[i])
    #     except:
    #         continue
    #
    # plt.ylim((0, 6))
    # plt.ylabel('MS right - recovered')
    # plt.legend()

    #plt.suptitle(p, fontsize=14)
    plt.tight_layout()
    if doSave:
        plt.savefig(os.path.join(savepath,name + '_MS.png'))
def plotIndividualPatTraj_2sides(p, ms_use, Y_MS, X_MS,
                                 df_med_lst, df_low_lst, df_top_lst,
                                 df_medX_lst, df_lowX_lst, df_topX_lst,
                                 labels, savepath, name='x', doSave=True):
    if name == 'x':
        name = p
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'lightblue', 'coral']
    ms_use_l = [m for m in ms_use if m[-1] == 'l']
    ms_use_r = [m for m in ms_use if m[-1] == 'r']

    plt.figure(figsize=(8, 8))
    plt.subplot(2,2,1)
    plt.plot(ms_use_l, X_MS.loc[p, ms_use_l], 'ko-', label='True')
    for i in range(0, len(df_medX_lst)):
        plt.plot(ms_use_l, df_medX_lst[i].loc[p, ms_use_l], 'o-', color=colors[i], label=labels[i], alpha=0.5)

        try:
            plt.fill_between(ms_use_l, df_lowX_lst[i].loc[p, ms_use_l].values.astype(float), df_topX_lst[i].loc[p, ms_use_l].values.astype(float), alpha=0.25, linewidth=0,
                             zorder=3,
                             color=colors[i])
        except:
            continue


    plt.ylim((0, 6))
    plt.ylabel('MS left - acute')


    plt.subplot(2, 2, 2)
    plt.plot(ms_use_r, X_MS.loc[p, ms_use_r], 'ko-', label='True')
    for i in range(0, len(df_medX_lst)):
        plt.plot(ms_use_r, df_medX_lst[i].loc[p, ms_use_r], 'o-', color=colors[i], label=labels[i], alpha=0.5)

        try:
            plt.fill_between(ms_use_r, df_lowX_lst[i].loc[p, ms_use_r].values.astype(float),
                             df_topX_lst[i].loc[p, ms_use_r].values.astype(float), alpha=0.25, linewidth=0,
                             zorder=3,
                             color=colors[i])
        except:
            continue

    plt.ylim((0, 6))
    plt.ylabel('MS right - acute')

    plt.subplot(2,2,3)
    plt.plot(ms_use_l, Y_MS.loc[p, ms_use_l], 'ko-', label='True')
    for i in range(0, len(df_med_lst)):
        plt.plot(ms_use_l, df_med_lst[i].loc[p, ms_use_l], 'o-', color=colors[i], label=labels[i], alpha=0.5)

        try:
            plt.fill_between(ms_use_l, df_low_lst[i].loc[p, ms_use_l].values.astype(float), df_top_lst[i].loc[p, ms_use_l].values.astype(float), alpha=0.25, linewidth=0,
                             zorder=3,
                             color=colors[i])
        except:
            continue


    plt.ylim((0, 6))
    plt.ylabel('MS left - recovered')


    plt.subplot(2, 2, 4)
    plt.plot(ms_use_r, Y_MS.loc[p, ms_use_r], 'ko-', label='True')
    for i in range(0, len(df_med_lst)):
        plt.plot(ms_use_r, df_med_lst[i].loc[p, ms_use_r], 'o-', color=colors[i], label=labels[i], alpha=0.5)

        try:
            plt.fill_between(ms_use_r, df_low_lst[i].loc[p, ms_use_r].values.astype(float),
                             df_top_lst[i].loc[p, ms_use_r].values.astype(float), alpha=0.25, linewidth=0,
                             zorder=3,
                             color=colors[i])
        except:
            continue

    plt.ylim((0, 6))
    plt.ylabel('MS right - recovered')
    plt.legend()

    plt.suptitle(p, fontsize=14)
    plt.tight_layout()
    if doSave:
        plt.savefig(os.path.join(savepath,name + '_MS.png'))
def plotAllScores_Compare3New(all_df, model_names, title,output_folder,
                              scores = ['RMSEblNLI','LEMS_delta','nonlinSc_blNLI' ],
                              titles = ['RMSE bl.NLI', 'LEMS difference', 'linSc bl.NLI'],
                              bins_use = [np.linspace(0, 6, 13), np.linspace(-50,50,26), np.linspace(0, 500, 21)],
                              this_ais_str = ['A', 'B', 'C', 'D'],
                              plegia_use = ['tetra', 'para']):



    colors = ['red', 'green','blue','yellow','black','orange']
    for this_plegia in plegia_use:

        for isc, sc in enumerate(scores):

            ais_dict = dict(zip(['A', 'B', 'C', 'D'], ['1','2','3','4']))

            this_ais_use = this_ais_str
            fig, axs, = plt.subplots(1, len(this_ais_use), figsize=(2.5*len(this_ais_use), 3))
            for i,this_ais in enumerate(this_ais_use):
            #try:
                if len(this_ais_use)>1:
                    axs_use = axs[i]
                else:
                    axs_use = axs

                for idf in range(0,len(all_df)):

                    try:
                        this_df = all_df[idf]
                        df_sub_pl = this_df[this_df.plegia == this_plegia]

                        df_sub = df_sub_pl[df_sub_pl.AIS==this_ais]
                        if len(df_sub)<1:
                            df_sub = df_sub_pl[df_sub_pl.AIS == ais_dict[this_ais]]

                        if sc == 'RMSEblNLI':
                            axs_use.hist(df_sub[sc], alpha=0.5,
                                        label=model_names[idf] + " %.1f(%.1f,%.1f)" % (np.percentile(df_sub[sc], 50),
                                                                                       np.percentile(df_sub[sc], 2.5),
                                                                                       np.percentile(df_sub[sc], 97.5)),
                                        density=True, color=colors[idf], bins=bins_use[isc])
                        else:
                            axs_use.hist(df_sub[sc],alpha = 0.5,
                                          label=model_names[idf]+" %.0f(%.0f,%.0f)" % (np.percentile(df_sub[sc],50),
                                                                             np.percentile(df_sub[sc],2.5),
                                                                             np.percentile(df_sub[sc],97.5)),
                                          density=True,color = colors[idf],bins = bins_use[isc])
                    except:
                        continue

                # if i == 0:
                #     axs[i].set_title(titles[isc])
                axs_use.set_title(this_ais_str[i])
                axs_use.set_xlabel(titles[isc])
                axs_use.legend(fontsize=8)#loc="lower center", bbox_to_anchor=(0.5, -0.5)

            plt.suptitle('Plegia: '+ this_plegia +'   Score: '+ sc, fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder,title+ this_plegia + sc + '.png'))