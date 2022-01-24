import os
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from utilitis_Twins import getaddData, get_eval_scores,plotAllScores_Compare3New


#######################################################################################
# INPUT SECTION
#######################################################################################

folders = ['Twin_aisa_nli_age_NN_MS_KNN_10_ClosestTwin_',
           'Twin_aisa_nli_age_NN_MS_KNN_10_',
           'Twin_aisa_nli_age_NN_MS_SS_KNN_10_ClosestTwin_'
           ]
all_names_short = ['NN_MS_Cl','NN_MS','NN_MS_SS_Cl']



# Patient Data to be matched - as described above for reference data
# Patients to find twins for - optional outcomes as well to evaluate matching:
outcome_file_toMatch  = '/data/ExampleToMatch_MS_26weeks.csv'
inputMS_file_toMatch  = '/data/ExampleToMatch_MS_2weeks.csv'


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



# Output path and details
output_path  = '/output'



#######################################################################################
# END OF INPUT
#######################################################################################

# Load matched data
all_dfsX = []
all_dfsY = []
for f in folders:
    this_dfX = pd.read_csv(os.path.join(output_path,f,'run_predictions',f+'_realX.csv'), index_col=0)
    this_dfY = pd.read_csv(os.path.join(output_path, f, 'run_predictions', f + '_realY.csv'), index_col=0)
    all_dfsX.append(this_dfX)
    all_dfsY.append(this_dfY)


# Get overlapping patients
pats = all_dfsX[0].dropna().index
for i in range(1,len(all_dfsX)):
    pats = list(set(pats) & set(all_dfsX[i].dropna().index))


# Load baseline and recovery MS for the patients to be matched (true results)
Y_toMatch     = pd.read_csv(outcome_file_toMatch, index_col=0)
X_MS_toMatch  = pd.read_csv(inputMS_file_toMatch, index_col=0)
X_add_toMatch = getaddData(X_addfile_toMatch,aisa_grade_field_toMatch,plegia_field_toMatch, nli_field_toMatch,
                           age_field_toMatch, sex_field_toMatch,cause_field_toMatch,nlicoarse_field_toMatch,
                           ['aisa', 'nli', 'age','plegia','LEMS','MeanScBlNLI'],ms_fields_toMatch,X_MS_toMatch,
                           pats,dermatomes_toMatch,sc_TMS_toMatch)


# Calculate summary scores to compare the different predictions
print('Calculating summary scores')
df_lstY = []
df_lstX = []
for i in range(0, len(all_dfsY)):
    df_lstY.append(pd.DataFrame(columns=['RMSEblNLI', 'LEMS_delta', 'nonlinSc_blNLI']))
    df_lstX.append(pd.DataFrame(columns=['RMSEblNLI', 'LEMS_delta', 'nonlinSc_blNLI']))

for p in tqdm.tqdm(pats):
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

    for i, this_dfY in enumerate(all_dfsY):
        this_rmseY, this_deltaLEMSY, this_nonlinY = get_eval_scores(this_Y.values,
                                                                    this_dfY.loc[p, :].values.astype(float),
                                                                    this_ind_nliMS)
        df_lstY[i].loc[p, :] = [this_rmseY, this_deltaLEMSY, this_nonlinY]

        this_dfX = all_dfsX[i]
        this_rmseX, this_deltaLEMSX, this_nonlinX = get_eval_scores(this_X.values,
                                                                    this_dfX.loc[p, :].values.astype(float),
                                                                    this_ind_nliMS)
        df_lstX[i].loc[p, :] = [this_rmseX, this_deltaLEMSX, this_nonlinX]


# Visualize the summary scores for patients subset by AISA grade and plegia
for i, this_df_ev in enumerate(df_lstY):
    this_df_ev['AIS'] = X_add_toMatch.loc[this_df_ev.index, 'aisa']
    this_df_ev['plegia'] = X_add_toMatch.loc[this_df_ev.index, 'plegia']
    df_lstY[i] = this_df_ev

for i, this_df_ev in enumerate(df_lstX):
    this_df_ev['AIS'] = X_add_toMatch.loc[this_df_ev.index, 'aisa']
    this_df_ev['plegia'] = X_add_toMatch.loc[this_df_ev.index, 'plegia']
    df_lstX[i] = this_df_ev

plotAllScores_Compare3New(df_lstY,
                          all_names_short,
                          '-'.join(all_names_short) + '_recovery',
                          os.path.join(output_path, 'figures'))

plotAllScores_Compare3New(df_lstX,
                          all_names_short,
                          '-'.join(all_names_short) + '_BL',
                          os.path.join(output_path, 'figures'))



# Show how many patients were matched with the relevant method
n_matched = [len(all_dfsX[i].dropna().index) for i in range(0,len(all_dfsX))]
plt.figure()
plt.bar(all_names_short, n_matched)
plt.savefig(os.path.join(output_path, 'figures','-'.join(all_names_short)+'n_matched.png'))
