#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:02:05 2022

@author: chiaramarzi
"""

import numpy as np
import pandas as pd
from Harmonizer import harmonizer
from harm_efficacy import efficacy
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

data = pd.read_csv('data.csv')

NI_features = data.columns.tolist()[3::]
covars_features = data.columns.tolist()[0:3]

print('# --------------------------------- harmonizer --------------------------------- #')
### Harmonize the entire dataset ###
print('### Harmonize the entire dataset ###')
harm = harmonizer(feature_names = NI_features, covariates_names = covars_features, eb = True, smooth_terms = ["Age"], smooth_term_bounds=(0, 100))
data_adj = harm.fit_transform(data)
print()

### Learn the harmonization model in the training set and applied it in the tes set, by using a hold-out validation scheme ###
print('### Learn the harmonization model in the training set and applied it in the tes set, by using a hold-out validation scheme ###')
harm = harmonizer(feature_names = NI_features, covariates_names = covars_features, eb = True, smooth_terms = ["Age"], smooth_term_bounds=(0, 100))
X_train, X_test = train_test_split(data, test_size=0.20, random_state=42)
harm.fit_transform(X_train)
test_data_adj = harm.transform(X_test)
test_data_adj_df = pd.DataFrame(data=test_data_adj, columns = NI_features)
print()

### Age prediction using the harmonizer transformer and the Support Vector regressor estimator within a 5-fold cross validation ###
print('### Age prediction using the harmonizer transformer and the Support Vector regressor estimator within a 5-fold cross validation ###')
X = data
y = data['Age']
kf = KFold(n_splits = 5, random_state=42, shuffle=True)
harm = harmonizer(feature_names = NI_features, covariates_names = covars_features, eb = True, smooth_terms = ["Age"], smooth_term_bounds=(0, 100)) 
clf = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
pipe = make_pipeline(harm, clf)
cv_results = cross_validate(pipe, X, y=y, groups=None, scoring='neg_mean_absolute_error', cv=kf, n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=True, return_estimator=True)
print('Average MAE in the test set:', np.round(np.abs(np.mean(cv_results['test_score'])), 2))
print()


print('# ----------------------------- Harmonization efficacy ----------------------------- #')
### Compute the HS of raw data ###
data = pd.read_csv('data.csv')
NI_features = data.columns.tolist()[3::]
covars_features = data.columns.tolist()[0:3]
perm_pvalue, wilcoxon_pvalue = efficacy(data, NI_features, covars_features=covars_features, smooth_terms = ['Age'], smooth_term_bounds=(0, 100))
print('The permutations test pvalue is', perm_pvalue)
print('The one-sided Wilcoxon signed-rank test pvalue is', wilcoxon_pvalue)






