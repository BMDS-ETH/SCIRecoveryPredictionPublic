import os
import pandas as pd
import numpy as np
import tqdm
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path


from utilitis_Twins import getaddData, generate_pat_uncert,getPatCar, getTwin, get_eval_scores,\
    plotIndividualPatTraj_2sides,plotAllScores_Compare3New, getName


#######################################################################################
# INPUT SECTION
#######################################################################################


# 1. Scores to be calculated based on given motor scores
# Options: 'LEMS', 'MeanScBlNLI', 'RMSEblNLI', 'NN_MS_SS', 'NN_MS'
motorfunction_score = 'NN_MS_SS'
sc_tol              = 0.5     # Score tolerance for 'LEMS', 'MeanScBlNLI', otherwise unused
                            # Suggestion: LEMS: 5, MeanScBlNLI: 0.5
allowScoreDoubling  = False # if True: in case no matching patient could be found that meets the score criterion,
                            # also patients with a score deviation of 2xsc_tol will be considered

# Define the type of nearest neighbour to be used if 'NN_MS_SS' or 'NN_MS' chosen above - otherwise this is ignored
n_NN           = 10    # number of NN to be used (max)
useClosestTwin = False # Use all twins at minimum distance or (if False) take average/median of the n_NN nearest neighbours
useMean        = False # Use the mean or median over all NN as prediction: True: use mean, False: use median

# 2. Which parameters to include for prior to optional additional KNN based on sensory and motor scores IN ORDER!:
# Options: 'aisa', 'nli', 'age', 'nli_coarse', 'cause', motorfunction_score
# Age match by 10 years - not enforced (if not possible to match it is ignored)
useToSubset = ['aisa', 'nli', 'age', motorfunction_score]


# 3. Use MS uncertainty distribution to bootstrap - median values will be reported
# Number of bootstrape (int) - if zero no bootstrap will be performed - use at least 50
n_bootstrap = 0


# 4. Reference Data
# Reference Database: Input/output_files contain motor and (optionally) sensory function at the input, i.e. time of matching,
# and (optional) time of evaluation. Files should be csv with columns indicating all motor and (optional) sensory scores as floats
outcome_file_ref  = '/data/Example_MS_26weeks.csv'
inputMS_file_ref  = '/data/Example_MS_2weeks.csv'
inputLTS_file_ref = '/data/Example_LTS_2weeks.csv'
inputPPS_file_ref = '/data/Example_PPS_2weeks.csv'



# Indicate field names for MS and SS for reference database
ms_fields_ref  = ['C5_l','C5_r','C6_l','C6_r','C7_l','C7_r','C8_l','C8_r','T1_l','T1_r','L2_l','L2_r','L3_l','L3_r',
                       'L4_l','L4_r','L5_l','L5_r','S1_l','S1_r'] # These should be 20 (!)
dermatomes_ref = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'T1', 'T2', 'T3',
                  'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                  'L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'S2', 'S3', 'S45']
sc_TMS_ref     = ['C5', 'C6', 'C7', 'C8', 'T1', 'L2', 'L3', 'L4', 'L5', 'S1']
derm_fields_ref = ['C2_l', 'C3_l', 'C4_l', 'C5_l', 'C6_l', 'C7_l', 'C8_l', 'T1_l', 'T2_l',
                       'T3_l', 'T4_l', 'T5_l', 'T6_l', 'T7_l', 'T8_l', 'T9_l', 'T10_l',
                       'T11_l', 'T12_l', 'L1_l', 'L2_l', 'L3_l', 'L4_l', 'L5_l', 'S1_l',
                       'S2_l', 'S3_l', 'S45_l', 'C2_r', 'C3_r', 'C4_r', 'C5_r', 'C6_r', 'C7_r',
                       'C8_r', 'T1_r', 'T2_r', 'T3_r', 'T4_r', 'T5_r', 'T6_r', 'T7_r', 'T8_r',
                       'T9_r', 'T10_r', 'T11_r', 'T12_r', 'L1_r', 'L2_r', 'L3_r', 'L4_r',
                       'L5_r', 'S1_r', 'S2_r', 'S3_r', 'S45_r']

# Additional information for each patient
X_addfile_ref =  '/data/Example_addData_2weeks.csv'

# Indicate data fields for additional information
aisa_grade_field_ref = 'AIS'
plegia_field_ref     = 'plegia'
nli_field_ref        = 'NLI'
age_field_ref        = 'AgeAtDOI'
sex_field_ref        = 'Sex'
cause_field_ref      = 'Cause'
nlicoarse_field_ref  = 'NLI_level'


# 5. Patient Data to be matched - as described above for reference data
# Patients to find twins for - optional outcomes as well to evaluate matching:
refEqToMatch = True

# Optional: Specify which patients should be matched (index ids) - if empty all patients will be matched
pats_toMatch = [] # For example: 100136

if refEqToMatch:
    outcome_file_toMatch  = outcome_file_ref
    inputMS_file_toMatch  = inputMS_file_ref
    inputLTS_file_toMatch = inputLTS_file_ref
    inputPPS_file_toMatch = inputPPS_file_ref
    ms_fields_toMatch     = ms_fields_ref.copy()
    dermatomes_toMatch    = dermatomes_ref.copy()
    sc_TMS_toMatch        = sc_TMS_ref
    derm_fields_toMatch   = derm_fields_ref
    X_addfile_toMatch     = X_addfile_ref
    aisa_grade_field_toMatch = aisa_grade_field_ref
    plegia_field_toMatch  = plegia_field_ref
    nli_field_toMatch     = nli_field_ref
    age_field_toMatch     = age_field_ref
    sex_field_toMatch     = sex_field_ref
    cause_field_toMatch   = cause_field_ref
    nlicoarse_field_toMatch = nlicoarse_field_ref
else:
    outcome_file_toMatch  = '/data/ExampleToMatch_MS_26weeks.csv'
    inputMS_file_toMatch  = '/data/ExampleToMatch_MS_2weeks.csv'
    inputLTS_file_toMatch = '/data/ExampleToMatch_LTS_2weeks.csv'
    inputPPS_file_toMatch = '/data/ExampleToMatch_PPS_2weeks.csv'

    # These should be 20 (!) and be added in order  ['C5_left','C5_right', 'C6_left', 'C6_right', 'C7_left', 'C7_right',
    # 'C8_left','C8_right','T1_left','T1_right','L2_left', L2_right', 'L3_left', 'L3_right', 'L4_left' 'L4_right',
    # 'L5_left', 'L5_right', 'S1_left', 'S1_right']
    ms_fields_toMatch   = ['C5_l','C5_r','C6_l','C6_r','C7_l','C7_r','C8_l','C8_r','T1_l','T1_r','L2_l','L2_r','L3_l','L3_r',
                           'L4_l','L4_r','L5_l','L5_r','S1_l','S1_r']
    dermatomes_toMatch  = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'T1', 'T2', 'T3',
                          'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                          'L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'S2', 'S3', 'S45']
    sc_TMS_toMatch      = ['C5', 'C6', 'C7', 'C8', 'T1', 'L2', 'L3', 'L4', 'L5', 'S1']
    derm_fields_toMatch = ['C2_l', 'C3_l', 'C4_l', 'C5_l', 'C6_l', 'C7_l', 'C8_l', 'T1_l', 'T2_l',
                           'T3_l', 'T4_l', 'T5_l', 'T6_l', 'T7_l', 'T8_l', 'T9_l', 'T10_l',
                           'T11_l', 'T12_l', 'L1_l', 'L2_l', 'L3_l', 'L4_l', 'L5_l', 'S1_l',
                           'S2_l', 'S3_l', 'S45_l', 'C2_r', 'C3_r', 'C4_r', 'C5_r', 'C6_r', 'C7_r',
                           'C8_r', 'T1_r', 'T2_r', 'T3_r', 'T4_r', 'T5_r', 'T6_r', 'T7_r', 'T8_r',
                           'T9_r', 'T10_r', 'T11_r', 'T12_r', 'L1_r', 'L2_r', 'L3_r', 'L4_r',
                           'L5_r', 'S1_r', 'S2_r', 'S3_r', 'S45_r']
    X_addfile_toMatch  = '/data/ExampleToMatch_addData_2weeks.csv'
    aisa_grade_field_toMatch = 'AIS'
    plegia_field_toMatch     = 'plegia'
    nli_field_toMatch        = 'NLI'
    age_field_toMatch        = 'AgeAtDOI'
    sex_field_toMatch        = 'Sex'
    cause_field_toMatch      = 'Cause'
    nlicoarse_field_toMatch  = 'NLI_level'



# 6. Score uncertainties (provided)
useBootstrap   = False
bootstrap_file = '/data/uncertMS.csv'


# 7. Output path and details
output_path  = '/output'
plot_refData = False # Plot a dimensionality reduction of the used scores for matching (if using a form of NN)
plotIndPats  = False # Plot the patients' true and matched MS at baseline and recovery
plotSummary  = True  # Plot summary metrics as histograms subset by AISA grade and plegia to evaluate the overall
                     # matching of a patient population - this only makes sense if multiple patients are matched!
#######################################################################################
# END OF INPUT
#######################################################################################

#######################################################################################
# END OF INPUT
#######################################################################################

# Check inputs
if motorfunction_score in useToSubset:
    if len(ms_fields_ref) != 20:
        raise('Indicate 10 fields for motor scores - `ms_fields_ref`!')

    if len(ms_fields_toMatch) != 20:
        raise('Indicate 10 fields for motor scores - `ms_fields_toMatch`!')

if ('NN_MS_SS' in useToSubset) or ('NN_MS' in useToSubset):
    useKNN = True
else:
    useKNN = False


# Prepare output folder
name = getName(useToSubset,sc_tol, useKNN, n_NN, useClosestTwin,allowScoreDoubling,useMean)
output = os.path.join(output_path, name)
Path(output).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output, 'run_predictions')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output, 'figures')).mkdir(parents=True, exist_ok=True)

# Load ref data
Y_ref     = pd.read_csv(outcome_file_ref, index_col=0)
X_MS_ref  = pd.read_csv(inputMS_file_ref, index_col=0)
X_LTS_ref = pd.read_csv(inputLTS_file_ref, index_col=0)
X_PPS_ref = pd.read_csv(inputPPS_file_ref, index_col=0)
df_uncert = pd.read_csv(bootstrap_file, index_col=0)
X_add_ref = getaddData(X_addfile_ref,aisa_grade_field_ref,plegia_field_ref, nli_field_ref,age_field_ref, sex_field_ref,
                         cause_field_ref,nlicoarse_field_ref,useToSubset, ms_fields_ref,X_MS_ref, list(Y_ref.index),
                       dermatomes_ref,sc_TMS_ref)


# Load data for matching
Y_toMatch     = pd.read_csv(outcome_file_toMatch, index_col=0)
X_MS_toMatch  = pd.read_csv(inputMS_file_toMatch, index_col=0)
X_LTS_toMatch = pd.read_csv(inputLTS_file_toMatch, index_col=0)
X_PPS_toMatch = pd.read_csv(inputPPS_file_toMatch, index_col=0)
if len(pats_toMatch)>0:
    pats = pats_toMatch
else:
    pats = X_MS_toMatch.index
X_add_toMatch = getaddData(X_addfile_toMatch,aisa_grade_field_toMatch,plegia_field_toMatch, nli_field_toMatch,
                           age_field_toMatch, sex_field_toMatch,cause_field_toMatch,nlicoarse_field_toMatch,
                           useToSubset,ms_fields_toMatch,X_MS_toMatch, pats,dermatomes_toMatch,sc_TMS_toMatch)




# Standardize data for NN search
if np.any([(s in useToSubset) for s in['NN_MS_SS', 'NN_MS'] ]) and useKNN:
    pats_ref = X_MS_ref.index

    if 'NN_MS_SS' in useToSubset:
        data_Knn = pd.concat([X_MS_ref.loc[pats_ref,ms_fields_ref]/5.,X_LTS_ref.loc[pats_ref,derm_fields_ref]/2.,X_PPS_ref.loc[pats_ref,derm_fields_ref]/2.], axis = 1)
    else:
        data_Knn = X_MS_ref.loc[pats_ref,ms_fields_ref]/5.

    adata  = sc.AnnData(X=data_Knn.loc[pats_ref].values, obs=X_add_ref.loc[pats_ref])

    if plot_refData:
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.tsne(adata)
        sc.pl.tsne(adata, color=useToSubset, wspace=1)
        plt.savefig(os.path.join(output,'Figures','refData_tsne.png'))
else:

    # Using fixed scores only
    data_Knn = X_MS_ref.loc[:,ms_fields_ref]/5.


# Initialize
df_realY      = pd.DataFrame(columns = ms_fields_toMatch)
df_realX      = pd.DataFrame(columns = ms_fields_toMatch)
dists         = []
df_median     = pd.DataFrame(columns=ms_fields_ref)
df_median_low = pd.DataFrame(columns=ms_fields_ref)
df_median_top = pd.DataFrame(columns=ms_fields_ref)
df_medianX    = pd.DataFrame(columns=ms_fields_ref)
df_medianX_low = pd.DataFrame(columns=ms_fields_ref)
df_medianX_top = pd.DataFrame(columns=ms_fields_ref)


# Match each patient
for p in tqdm.tqdm(pats):

    # Get patient characteristics
    this_char = getPatCar(useToSubset,X_add_toMatch.loc[p,:])
    try:
        this_ind_nli = dermatomes_toMatch.index(X_add_toMatch.loc[p, 'nli'])
    except:
        try:
            if np.isnan(X_add_toMatch.loc[p, 'nli']):
                thisthis_ind_nli = 0
        except:
            if X_add_toMatch.loc[p, 'nli'] == 'C1' or X_add_toMatch.loc[p, 'nli'] == 'INT' :
                this_ind_nli = 0

    # Convert NLI to MS scale
    ms_blnli = [value for value in dermatomes_toMatch[this_ind_nli:] if value in sc_TMS_toMatch]
    this_ind_nliMS = 2 * sc_TMS_toMatch.index(ms_blnli[0])

    # Generate bootstrap (if needed)
    this_X_MS_use    = X_MS_toMatch.loc[p, ms_fields_toMatch]
    thisX_MS_uncert  = generate_pat_uncert(df_uncert, this_X_MS_use, n_bootstrap, this_ind_nliMS)
    if 'NN_MS_SS' in useToSubset:
        this_X_use   = pd.concat([X_MS_toMatch.loc[p,:]/5.,X_LTS_toMatch.loc[p,:]/2.,X_PPS_toMatch.loc[p,:]/2.], axis = 0)
    else:
        this_X_use   = this_X_MS_use/5.
    thisX_uncert = thisX_MS_uncert

    # Exclude patient from matching if reference and toMatch agree
    if refEqToMatch:
        pats_forref = [p_r for p_r in X_add_ref.index if p_r!=p]
    else:
        pats_forref = X_add_ref.index

    # Actual matiching
    real_y_med, dist, pats_neigh, real_x_med = getTwin(useToSubset, data_Knn.loc[pats_forref,:], this_X_use,
                                                       X_add_ref.loc[pats_forref,:], this_char,useKNN,n_NN,
                                                       useClosestTwin,useMean,
                                                       Y_ref.loc[pats_forref,ms_fields_ref],X_MS_ref.loc[pats_forref,ms_fields_ref],
                                                       allowScoreDoubling,sc_tol,this_ind_nliMS)

    # Only if a match was found
    if len(real_y_med)>0:

        # Save results at baseline (X, time of matching) and recovery (Y)
        dists.append(dist)
        df_realY.loc[p, :] = real_y_med
        df_realX.loc[p, :] = real_x_med

        # Bootstrap to estimate uncertainty
        if n_bootstrap > 0:
            y_hat_med = pd.DataFrame(columns=df_realY.columns, index = np.arange(n_bootstrap))
            x_hat_med = pd.DataFrame(columns=df_realY.columns, index=np.arange(n_bootstrap))
            for i in range(0, n_bootstrap):

                if 'LEMS' in useToSubset:
                    this_char[useToSubset.index('LEMS')] = thisX_uncert.loc[i, ms_fields_toMatch[10:]].sum()

                if 'MeanScBlNLI' in useToSubset:
                    this_char[useToSubset.index('MeanScBlNLI')] = thisX_uncert.loc[i, ms_fields_toMatch[2 * this_ind_nliMS:]].mean()


                if 'NN_MS_SS' in useToSubset:
                    this_X_BS = pd.concat(
                        [thisX_uncert.loc[i,ms_fields_toMatch]/5., X_LTS_toMatch.loc[p, derm_fields_toMatch] / 2.,
                         X_PPS_toMatch.loc[p, derm_fields_toMatch] / 2.], axis=0)
                else:
                    this_X_BS = thisX_uncert.loc[i,ms_fields_toMatch]


                bs_y_med, dist_bs, pats_neigh_bs , bs_x_med = getTwin(useToSubset, data_Knn.loc[pats_forref, :], this_X_BS,
                                                       X_add_ref.loc[pats_forref, :], this_char, useKNN, n_NN,
                                                       useClosestTwin, useMean, Y_ref.loc[pats_forref,ms_fields_ref],
                                                       X_MS_ref.loc[pats_forref,ms_fields_ref],allowScoreDoubling,sc_tol,this_ind_nliMS)
                if len(bs_y_med)>0:
                    y_hat_med.loc[i,:]  = bs_y_med
                    x_hat_med.loc[i, :] = bs_x_med
                else:
                    continue

            # Get median and 95% CI - dropping any nans!!!
            if len(y_hat_med.dropna())>0:
                max_y_med = np.percentile(y_hat_med.dropna(), 97.5, axis=0)
                min_y_med = np.percentile(y_hat_med.dropna(), 2.5, axis=0)
                med_y_med = np.percentile(y_hat_med.dropna(), 50, axis=0)
                df_median.loc[p, :] = med_y_med
                df_median_low.loc[p, :] = min_y_med
                df_median_top.loc[p, :] = max_y_med

                max_x_med = np.percentile(x_hat_med.dropna(), 97.5, axis=0)
                min_x_med = np.percentile(x_hat_med.dropna(), 2.5, axis=0)
                med_x_med = np.percentile(x_hat_med.dropna(), 50, axis=0)
                df_medianX.loc[p, :] = med_x_med
                df_medianX_low.loc[p, :] = min_x_med
                df_medianX_top.loc[p, :] = max_x_med

        # Plot individual patient (if wanted)
        if plotIndPats:
            plotIndividualPatTraj_2sides(p, ms_fields_toMatch, Y_toMatch,X_MS_toMatch,
                                         [df_realY], [df_median_low], [df_median_top],
                                         [df_realX], [df_medianX_low], [df_medianX_top],
                                         [name],os.path.join(output,'figures'), name=str(p)+ '_'+name, doSave=True)

    else:

        # no match was found
        dists.append(np.nan)
        df_realY.loc[p, :] = np.nan
        df_realX.loc[p, :] = np.nan


# Overall result
print('Found matches for '+str(len(df_realY.dropna().index)) + ' out of ' + str(len(df_realY)) + ' patients')


# Save to csv
df_realY.to_csv(os.path.join(output,'run_predictions',name+'_realY.csv'))
df_realX.to_csv(os.path.join(output,'run_predictions',name+'_realX.csv'))
df_dists = pd.DataFrame(dists, index = df_realY.index)
df_dists.to_csv(os.path.join(output,'run_predictions',name+'_distsTwins.csv'))
if n_bootstrap>0:
    df_medianX.to_csv(os.path.join(output,'run_predictions',name+'_BS_medX.csv'))
    df_medianX_low.to_csv(os.path.join(output,'run_predictions',name+'_BS_lowX.csv'))
    df_medianX_top.to_csv(os.path.join(output,'run_predictions',name+'_BS_topX.csv'))

    df_median.to_csv(os.path.join(output,'run_predictions',name+'_BS_medY.csv'))
    df_median_low.to_csv(os.path.join(output,'run_predictions',name+'_BS_lowY.csv'))
    df_median_top.to_csv(os.path.join(output,'run_predictions',name+'_BS_topY.csv'))



# Evaluate and plot overall matching performance
# Calculate RMSE, nonlinSc, LEMS deviation for all models
if len(df_realY.dropna().index) > 10 and plotSummary:
    print('Calculating summary scores')
    all_dfsY = [df_realY]
    all_dfsX = [df_realX]
    df_lstY = []
    df_lstX = []
    for i in range(0,len(all_dfsY)):
        df_lstY.append(pd.DataFrame(columns=['RMSEblNLI','LEMS_delta','nonlinSc_blNLI']))
        df_lstX.append(pd.DataFrame(columns=['RMSEblNLI', 'LEMS_delta', 'nonlinSc_blNLI']))


    for p in tqdm.tqdm(df_realY.dropna().index):
        try:
            this_ind_nli = dermatomes_toMatch.index(X_add_toMatch.loc[p, 'nli'])
        except:
            try:
                if np.isnan(X_add_toMatch.loc[p, 'nli']):
                    thisthis_ind_nli = 0
            except:
                if X_add_toMatch.loc[p, 'nli'] == 'C1' or X_add_toMatch.loc[p, 'nli'] == 'INT':
                    this_ind_nli = 0

        # Convert NLI to MS scale
        ms_blnli = [value for value in dermatomes_toMatch[this_ind_nli:] if value in sc_TMS_toMatch]
        this_ind_nliMS = 2 * sc_TMS_toMatch.index(ms_blnli[0])

        this_Y = Y_toMatch.loc[p, ms_fields_toMatch]
        this_X = X_MS_toMatch.loc[p, ms_fields_toMatch]

        for i,this_dfY in enumerate(all_dfsY):


            this_rmseY, this_deltaLEMSY, this_nonlinY = get_eval_scores(this_Y.values, this_dfY.loc[p, :].values.astype(float),this_ind_nliMS)
            df_lstY[i].loc[p, :] = [this_rmseY, this_deltaLEMSY, this_nonlinY]

            this_dfX = all_dfsX[i]
            this_rmseX, this_deltaLEMSX, this_nonlinX = get_eval_scores(this_X.values, this_dfX.loc[p, :].values.astype(float),this_ind_nliMS)
            df_lstX[i].loc[p, :] = [this_rmseX, this_deltaLEMSX, this_nonlinX]



    for i, this_df_ev in enumerate(df_lstY):
        this_df_ev['AIS']    = X_add_toMatch.loc[this_df_ev.index,'aisa']
        this_df_ev['plegia'] = X_add_toMatch.loc[this_df_ev.index, 'plegia']
        df_lstY[i] = this_df_ev

    for i, this_df_ev in enumerate(df_lstX):
        this_df_ev['AIS']    = X_add_toMatch.loc[this_df_ev.index,'aisa']
        this_df_ev['plegia'] = X_add_toMatch.loc[this_df_ev.index, 'plegia']
        df_lstX[i] = this_df_ev

    plotAllScores_Compare3New([df_lstY[0]],
                              ['Twin_recovery'],
                              name+ '_recovery',
                              os.path.join(output,'figures'))

    plotAllScores_Compare3New([df_lstX[0]],
                              ['Twin_BL'],
                              name+ '_BL',
                              os.path.join(output,'figures'))

print('FINISHED!!')




