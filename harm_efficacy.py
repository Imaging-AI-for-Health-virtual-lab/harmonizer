#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:45:24 2022

@author: chiaramarzi
"""

from Harmonizer import harmonizer
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from scipy.stats import wilcoxon

def efficacy(data, MRI_features, covars_features = ['SITE'], smooth_terms = [], smooth_term_bounds=(None, None), covar_cat = None):
    """Compute the harmonization efficacy
    
    Parameters
    ----------
    data : Pandas DataFrame
        Contains the entire set of data, including MRI-derived features, imaging site information, and biological covariates (if needed)
    MRI_features: list
        Contains the MRI-derived features used for the SITE prediction and the efficacy computation
    covars_features : list, default = ['SITE']
        Contains the imaging site, named 'SITE', and (optionally) the biological features used in the harmonization process. 
    smooth_terms : list, default = []
        Contains the covariates that present a nonlinear effects on the MRI-derived features. For example, age presents a nonlinear relationship with most of the brain MRI-derived features.
    smooth_term_bounds : tuple, default = (None, None)
        Controls the boundary knots for nonlinear estimation. You should specify boundaries that contain the limits of the entire dataset, including the test data.
    covar_cat : list, default = None
        Labels to constrain permutation within groups, i.e. y values are permuted among samples with the same group identifier. When not specified, y values are permuted among all samples (details in scikit-learn permutation_test_score page).
        
    Output
    --------
    perm_p : float64
        The permutations test p-value. 
        If perm_p >= 0.5, the imaging site prediciton is not significantly different from a random prediction. Thus, the imaging site has been removed from the MRI-derived features.
    wilc_p : float64
        The one-sided Wilcoxon signed-rank test p-value. 
        If wilc_p < 0.5, the median balanced accuracy obtaining in imaging site prediction using raw MRI-derived features is significantly higher than that estimated using harmonized MRI-derived features.
        Thus, the imaging site has been reduced from the MRI-derived features.
        
    Examples
    --------
    >>> from harm_efficacy import efficacy
    
    >>> data = pd.read_csv('data.csv')
    >>> NI_features = data.columns.tolist()[3::] # MRI-derived features
    >>> covars_features = data.columns.tolist()[0:3] # covariates features
    >>> perm_pvalue, wilcoxon_pvalue = efficacy(data, NI_features, covars_features=covars_features, smooth_terms = ['Age'], smooth_term_bounds=(0, 100))
    
    """
    
    y = np.array(data["SITE"])
    lc = LabelEncoder() 
    lc_y = lc.fit_transform(y)
    columns_to_take = covars_features + MRI_features
    my_data= data[columns_to_take]
    
    Nrep = 100
    score = 'balanced_accuracy'
    bal_acc_raw = []
    bal_acc_harm = []
    
    Nperm = 5000
    
    for rep in range(0, Nrep):
        print('# Repetition', rep)
        skf = StratifiedKFold(n_splits = 5, random_state=rep, shuffle=True)
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=rep)
        print('Computing balanced accuray of SITE prediction (within stratified 5-fold CV) using raw data')
        raw_scores = cross_val_score(clf, X=my_data[MRI_features], y=lc_y, groups=None, scoring=score, cv=skf, n_jobs=-1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score=np.nan)
        bal_acc_raw.append(np.mean(raw_scores))
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=rep)
        harm = harmonizer(feature_names = MRI_features, covariates_names = covars_features, eb = True, smooth_terms = smooth_terms, smooth_term_bounds=smooth_term_bounds) 
        pipe = make_pipeline(harm, clf)
        print('Computing balanced accuray of SITE prediction (within stratified 5-fold CV) using harmonized data')
        harm_scores = cross_val_score(pipe, X=my_data, y=lc_y, groups=None, scoring=score, cv=skf, n_jobs=-1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score=np.nan)
        bal_acc_harm.append(np.mean(harm_scores))
    median_bal_acc_raw = np.median(bal_acc_raw)
    median_bal_acc_harm = np.median(bal_acc_harm)
    ave_bal_acc_harm = np.mean(bal_acc_harm)
    print("The median balanced accuray in predicting imaging site using raw data is", median_bal_acc_raw)
    print("The median balanced accuray in predicting imaging site using hamonized data is", median_bal_acc_harm)
    print()

    print('Permutations')
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=rep)
    harm = harmonizer(feature_names = MRI_features, covariates_names = covars_features, eb = True, smooth_terms = smooth_terms, smooth_term_bounds=smooth_term_bounds) 
    pipe = make_pipeline(harm, clf)
    score_fake, perm_scores, pvalue_fake = permutation_test_score(pipe, X=my_data, y=lc_y, groups=covar_cat, scoring=score, cv=skf, n_permutations=Nperm, n_jobs=-1, random_state=rep)   
    df_perm_scores = pd.DataFrame(data=perm_scores, columns=['Permutation scores'])
    df_perm_scores.to_csv('permutation_scores.csv', index=None)
    perm_p = ((perm_scores >= ave_bal_acc_harm).sum()+1)/(Nperm+1)
    perm_p = np.round(perm_p,4)
    
    w, wilc_p = wilcoxon(bal_acc_raw, bal_acc_harm, alternative='greater')
    wilc_p = np.round(wilc_p,4)
    print()
    print('### The efficacy of harmonization is: (' + str(perm_p) + ', ' + str(wilc_p) + ')')
    
    return perm_p, wilc_p

####################################################################    
