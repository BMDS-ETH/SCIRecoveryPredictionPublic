import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances


#######################################################################################
# Auxillary functions
#######################################################################################

# Get data
def getName(useToSubset,sc_tol, useKNN, n_NN, useClosestTwin,allowScoreDoubling,useMean):
    name = 'Twin_'

    # all subsets
    for sub in useToSubset:
        name += sub+'_'

    # score tolerance
    if np.any(['LEMS' in useToSubset,'MeanScBlNLI' in useToSubset,'RMSE' in useToSubset]):
        name += 'SCtol'+ str(sc_tol) + '_' + 'scDouble'+str(allowScoreDoubling)

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
def getaddData(X_addfile_ref,aisa_grade_field,plegia_field,nli_field,age_field, sex_field,cause_field,nlicoarse_field,
               useToSubset, ms_fields,X,pats,dermatomes_toMatch,sc_TMS_toMatch):

    X_add_load = pd.read_csv(X_addfile_ref, index_col=0)

    if type(pats) == str:
        pats_use =  X_add_load.index
    else:
        pats_use = pats.copy()


    X_add = pd.DataFrame(columns = useToSubset, index = pats_use)


    if len(plegia_field)>0:
        X_add.loc[pats_use,'plegia'] = X_add_load.loc[pats_use, plegia_field]
    else:
        raise('plegia_field needs to be given!')

    if len(aisa_grade_field)>0:
        X_add.loc[pats_use,'aisa'] = X_add_load.loc[pats_use, aisa_grade_field]
    else:
        raise('aisa_grade_field needs to be given!')

    if len(nli_field)>0:
        X_add.loc[pats_use, 'nli'] = X_add_load.loc[pats_use, nli_field]
    else:
        raise('nli_field needs to be given!')
    if len(age_field)>0:
        X_add.loc[pats_use,'age'] = X_add_load.loc[pats_use, age_field]
    if len(sex_field)>0:
        X_add.loc[pats_use,'sex'] = X_add_load.loc[pats_use, sex_field]
    if len(cause_field)>0:
        X_add.loc[pats_use, 'cause'] = X_add_load.loc[pats_use, cause_field]
    if len(nlicoarse_field)>0:
        X_add.loc[pats_use, 'nli_coarse'] = X_add_load.loc[pats_use, nlicoarse_field]

    for sub in useToSubset:
        # Summary scores
        if sub == 'LEMS':
            X_add.loc[pats_use, 'LEMS'] = X.loc[pats_use, ms_fields[10:]].sum(axis = 1)

        if sub == 'MeanScBlNLI':
            for p in pats_use:
                try:
                    this_ind_nli = dermatomes_toMatch.index(X_add_load.loc[p, nli_field])
                except:
                    try:
                        if np.isnan(X_add_load.loc[p, nli_field]):
                            this_ind_nli = 0
                    except:
                        if X_add_load.loc[p, nli_field] == 'C1' or X_add_load.loc[p, nli_field] == 'INT':
                            this_ind_nli = 0

                ms_blnli = [value for value in dermatomes_toMatch[this_ind_nli:] if value in sc_TMS_toMatch]
                this_ind_nliMS = 2 * sc_TMS_toMatch.index(ms_blnli[0])
                X_add.loc[p, 'MeanScBlNLI'] = X.loc[p, ms_fields[this_ind_nliMS:]].mean()



    return X_add

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
        this_char.append(this_X_add.loc[sub])

    return this_char
def getTwin(useToSubset, data_Knn_use, this_X_use,X_add_ref_use, this_char,useKNN,n_NN,useClosestTwin,useMean,
            Y_MS_ref, X_MS_ref,allowScoreDoubling,sc_tol, this_ind_nliMS):


    # First generate patient subset
    data_sub = X_add_ref_use.copy()
    for i_sc,sub in enumerate(useToSubset):
        this_sc = this_char[i_sc]

        if sub in ['RMSEblNLI']:
            this_X_MS = this_X_use[:20]
            this_rmse = [mean_squared_error(X_MS_ref.loc[i,:][this_ind_nliMS:].values,this_X_MS[this_ind_nliMS:].values, squared=False) for i in list(data_sub.index)]
            inds_keep =np.where(np.array(this_rmse)<=sc_tol)[0]
            data_sub_sc = data_sub.iloc[inds_keep]
            if data_sub_sc.shape[0] >= 1:
                data_sub = data_sub_sc
            else:
                print(sub + ' match not possible')
                return [], None, None, []

        if sub in ['LEMS', 'meanScBlNLI']:

            data_sub_sc = data_sub[data_sub[sub] >= this_sc - sc_tol]
            data_sub_sc = data_sub_sc[data_sub_sc[sub] <= this_sc + sc_tol]

            if data_sub_sc.shape[0] >= 1:
                data_sub = data_sub_sc
            else:
                if allowScoreDoubling:
                    print('Doubled score tolerance!')
                    data_sub_sc = data_sub[data_sub[sub] >= this_sc - sc_tol*2]
                    data_sub_sc = data_sub_sc[data_sub_sc[sub] <= this_sc + sc_tol*2]
                    if data_sub_sc.shape[0] >= 1:
                        data_sub = data_sub_sc
                    else:
                        print(sub+' match not possible')
                        return [], None, None, []
                else:
                    print(sub + ' match not possible')
                    return [], None, None, []

        elif sub == 'age':
            data_sub_age = data_sub[data_sub['age'] >= this_sc - 10]
            data_sub_age = data_sub_age[data_sub_age['age'] <= this_sc + 10]

            if data_sub_age.shape[0]>=1:
                data_sub = data_sub_age.copy()
            else:
                print('Age match not possible but I continue')

        elif sub in ['aisa','nli_coarse', 'cause', 'nli']:
            data_sub_ais = data_sub[data_sub[sub] == this_sc]
            if data_sub_ais.shape[0] > 1:
                data_sub = data_sub_ais
            else:
                print(sub + ' match not possible!')
                return [], None, None, []
        else:
            continue

    # Check if any possible matches remain
    if data_sub.shape[0] > 0:

        thisdist = (pairwise_distances(data_Knn_use.loc[data_sub.index],
                                       np.expand_dims(np.array(this_X_use), axis=0), metric='hamming') *
                    data_Knn_use.loc[data_sub.index].shape[1]).astype(int)


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


        return twin_mean, min_dist, pats_neigh, twin_mean_X

    else:
        return [], None, None, []

# Evaluation
def get_eval_scores(this_Y, this_pred,this_ind_nliMS):

    # RMSE
    this_rmse = mean_squared_error(this_Y[this_ind_nliMS:], this_pred[this_ind_nliMS:], squared=False)

    # LEMS
    lems_true = this_Y[10:].sum()
    lems_pred = this_pred[10:].sum()
    this_deltaLEMS = (lems_true - lems_pred)

    # non linear score
    this_nonlin = nonlineScore(this_pred[this_ind_nliMS:], this_Y[this_ind_nliMS:])

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

# For plotting
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