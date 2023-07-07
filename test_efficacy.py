#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:55:07 2023

@author: chiaramarzi
"""
import numpy as np
import pandas as pd
from harm_efficacy import efficacy


print('# ----------------------------- Harmonization efficacy ----------------------------- #')
data = pd.read_csv('multicenter_CT-FD_features_k3_n25_age2.csv')
NI_features = data.columns.tolist()[2::]
covars_features = data.columns.tolist()[0:2]

delta = 5 # each bin spans 5 years
bins = list(np.arange(np.floor(np.min(data.age)), np.max(data.age)+delta, delta))
bins = [0] + bins
labels = list(np.arange(len(bins)-1))
age_category = list(pd.cut(data.age,bins=bins,labels=labels)) # age discretization based on bins and labels

perm_pvalue, wilcoxon_pvalue = efficacy(data, NI_features, covars_features=covars_features, smooth_terms = ['age'], smooth_term_bounds=(0, 100), covar_cat=age_category)
print('The permutations test pvalue is', perm_pvalue)
print('The one-sided Wilcoxon signed-rank test pvalue is', wilcoxon_pvalue)