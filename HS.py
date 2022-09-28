#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:49:47 2022

@author: chiaramarzi
"""
from Harmonizer import harmonizer
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def HS(data, MRI_features, harmonization = False, covars_features = [], smooth_terms = [], smooth_term_bounds=(None, None)):
    """Compute the harmonization score (HS) of MRI-derived features.
    
    Parameters
    ----------
    data : Pandas DataFrame
        Contains the entire set of data, including MRI-derived features and biological covariates
    MRI_features: list
        Contains the MRI-derived features used for the SITE prediction and the HS computation
    harmonization : bool, default = False
        If True data wil be harmonized (using the harmonizer transformer within the stratified 5-fold CV used for the SITE prediction task)
        If False (default) data will not be harmonized (data can be raw or already harmonized externally)
    covars_features : list, default = []
        Contains the imaging site, named "SITE", and (optionally) the biological features used in the harmonization process. covars_features is mandatory if harmonization = True
    smooth_terms : list, default = []
        Contains the covariates that present a nonlinear effects on the MRI-derived features. For example, age presents a nonlinear relationship with most of the brain MRI-derived features.
    smooth_term_bounds : tuple, default = (None, None)
        Controls the boundary knots for nonlinear estimation. You should specify boundaries that contain the limits of the entire dataset, including the test data
        
    Output
    --------
    HS : float64
        The harmonization score
        
    Examples
    --------
    >>> from HS import HS
    
    >>> data = pd.read_csv('data.csv')
    >>> NI_features = data.columns.tolist()[3::] # MRI-derived features
    >>> covars_features = data.columns.tolist()[0:3] # covariates features
    >>> raw_data_HS = HS(data, MRI_features, harmonization = False, covars_features = [], smooth_terms = [], smooth_term_bounds=(None, None))
    >>> harm_data_HS = HS(data, MRI_features, harmonization = True, covars_features = ['SITE', 'Age'], smooth_terms = ['Age'], smooth_term_bounds=(0, 100))
    
    """
    
    y = np.array(data["SITE"])
    lc = LabelEncoder() 
    lc_y = lc.fit_transform(y)
    if harmonization is True:
        columns_to_take = covars_features + MRI_features
    else:
        columns_to_take = MRI_features
    my_data= data[columns_to_take]
    
    Nrep = 1
    score = 'balanced_accuracy'
    bal_acc = []
    
    Nperm = 1
    
    for rep in range(0, Nrep):
        print('# Repetition', rep)
        skf = StratifiedKFold(n_splits = 5, random_state=rep, shuffle=True)
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=rep)
        if harmonization is True:
            harm = harmonizer(feature_names = MRI_features, covariates_names = covars_features, eb = True, smooth_terms = ["Age"], smooth_term_bounds=(0, 100)) 
            pipe = make_pipeline(harm, clf)
        else:
            pipe = clf
        print('Computing balanced accuray of SITE prediction (within stratified 5-fold CV) for the repetition', rep)
        scores = cross_val_score(pipe, X=my_data, y=lc_y, groups=None, scoring=score, cv=skf, n_jobs=-1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score=np.nan)
        bal_acc.append(np.mean(scores))
    ave_bal_acc = np.mean(bal_acc)

    print('Permutations')
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=rep)
    if harmonization is True:
        harm = harmonizer(feature_names = MRI_features, covariates_names = covars_features, eb = True, smooth_terms = ["Age"], smooth_term_bounds=(0, 100)) 
        pipe = make_pipeline(harm, clf)
    else:
        pipe = clf
    score_fake, perm_scores, pvalue_fake = permutation_test_score(pipe, X=my_data, y=lc_y, scoring=score, cv=skf, n_permutations=Nperm, n_jobs=1, random_state=rep)   
    df_perm_scores = pd.DataFrame(data=perm_scores, columns=['Permutation scores'])
    df_perm_scores.to_csv('permutation_scores.csv', index=None)
    
    p = ((perm_scores >= ave_bal_acc).sum()+1)/(Nperm+1)
    
    HS = 0
    if p != 0:
        HS = -1/np.log10(p)
        
    
    return HS

####################################################################    
'''
data = pd.read_csv('data.csv')

NI_features = data.columns.tolist()[3::]
covars_features = data.columns.tolist()[0:3]
HS = HS(data, NI_features, harmonization = True, covars_features=covars_features)
'''