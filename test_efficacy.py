#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:55:07 2023

@author: chiaramarzi
"""

import pandas as pd
from harm_efficacy import efficacy


print('# ----------------------------- Harmonization efficacy ----------------------------- #')
data = pd.read_csv('multicenter_CT-FD_features_k3_n25.csv')
NI_features = data.columns.tolist()[2::]
covars_features = data.columns.tolist()[0:2]
perm_pvalue, wilcoxon_pvalue = efficacy(data, NI_features, covars_features=covars_features, smooth_terms = ['age'], smooth_term_bounds=(0, 100))
print('The permutations test pvalue is', perm_pvalue)
print('The one-sided Wilcoxon signed-rank test pvalue is', wilcoxon_pvalue)