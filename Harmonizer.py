#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:50:48 2021
@author: chiaramarzi
"""
from neuroHarmonize import harmonizationLearn, harmonizationApply
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#Custom Transformer that extracts columns passed as argument to its constructor 
class harmonizer( BaseEstimator, TransformerMixin ):
    """Harmonize MRI-derived features, by removing the imaging site effect using the ComBat algorithm [1]. This transform extends the functionality 
        of the package neuroHarmonize [2].
    
        Parameters
        ----------
        feature_names : list
            Contains the MRI-derived features to harmonize
        covariates_names : list
            Contains the imaging site, named "SITE", and (optionally) the biological features used in the harmonization process
        eb: bool, default=True
            If True (default), ComBat uses Empirical Bayes to fit a prior distribution for the site effects for each site
        smooth_terms : list, default = []
            Contains the covariates that present a nonlinear effects on the MRI-derived features. 
            For example, age presents a nonlinear relationship with most of the brain MRI-derived features
        smooth_term_bounds : tuple, default = (None, None)
            Controls the boundary knots for nonlinear estimation. You should specify boundaries that contain the limits of the entire dataset, 
            including the test data
            
            
        Attributes
        ----------
        feature_names : list
            Contains the MRI-derived features to harmonize
        covariates_names : list
            Contains the imaging site and the biological covariates used in the harmonization process
        eb: bool
            If True, ComBat uses Empirical Bayes to fit a prior distribution for the site effects for each site
        smooth_terms : list
            Contains the covariates that present a nonlinear effects on the MRI-derived features
        smooth_term_bounds : tuple
            Contains the boundary knots for nonlinear estimation
        model : dictionary
        data_adj : numpy N-dimensional array, with shape = (N_samples, N_features)
        
            
        Notes
        -----
        All the covariates used to control for during harmonization, must be encoded numerically.
        The imaging site information must be reported in a single column called "SITE" 
        with labels that identify sites (the labels in "SITE" need not be numeric).
        
        References
        ----------
        [1] Johnson, W.E., Li, C., Rabinovic, A., 2007. 
        Adjusting batch effects in microarray expression data 
        using empirical Bayes methods. Biostat. Oxf. Engl. 8, 118–127. 
        https://doi.org/10.1093/biostatistics/kxj037
        
        [2] Pomponio, R., Erus, G., Habes, M., Doshi, J., Srinivasan, D., Mamourian, E.,
        Bashyam, V., Nasrallah, I.M., Satterthwaite, T.D., Fan, Y., Launer, L.J.,
        Masters, C.L., Maruff, P., Zhuo, C., Völzke, H., Johnson, S.C., Fripp, J.,
        Koutsouleris, N., Wolf, D.H., Gur, Raquel, Gur, Ruben, Morris, J., Albert, M.S.,
        Grabe, H.J., Resnick, S.M., Bryan, R.N., Wolk, D.A., Shinohara, R.T., Shou, H.,
        Davatzikos, C., 2020. Harmonization of large MRI datasets for the analysis
        of brain imaging patterns throughout the lifespan. NeuroImage 208, 116450. 
        https://doi.org/10.1016/j.neuroimage.2019.116450
        
        
        Example
        --------
        >>> from Harmonizer import harmonizer
        >>> data = pd.read_csv('data.csv')

        >>> NI_features = data.columns.tolist()[3::]
        >>> covars_features = data.columns.tolist()[0:3]
        >>> harm = harmonizer(feature_names = NI_features, covariates_names = covars_features, eb = True, smooth_terms = ["Age"], smooth_term_bounds=(0, 100))
        >>> X_train, X_test = train_test_split(data, test_size=0.20, random_state=42)
        >>> harm.fit_transform(X_train)
        >>> test_data_adj = harm.transform(X_test)
        """
        
        
    def __init__( self, feature_names, covariates_names, eb = True, smooth_terms = [], smooth_term_bounds=(None, None) ):
        self.feature_names = feature_names 
        self.covariates_names = covariates_names
        self.eb = eb
        self.smooth_terms = smooth_terms
        self.smooth_term_bounds = smooth_term_bounds
        self.model = None
        self.data_adj = None

    
    def fit( self, X, y = None ):
        data = X[self.feature_names]
        data = np.array(data)
        covars = X[self.covariates_names]
        my_model, my_data_adj = harmonizationLearn(data, covars, eb=self.eb, smooth_terms=self.smooth_terms, smooth_term_bounds=self.smooth_term_bounds)
        self.model = my_model
        self.data_adj = my_data_adj
        return self 

    
    def transform( self, X, y = None ):
        data = X[self.feature_names]
        data = np.array(data)
        covars = X[self.covariates_names]
        my_holdout_data_adj = harmonizationApply(data, covars, self.model)
        return my_holdout_data_adj

