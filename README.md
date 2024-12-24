# harmonizer
[![DOI](https://zenodo.org/badge/539387936.svg)](https://doi.org/10.5281/zenodo.14550914)

The *harmonizer* repository contains two main tools:

* *Harmonizer.py* which includes the harmonizer transformer, i.e., a Python transformer that encapsulates the *neuroHarmonize* procedure [1] among the preprocessing steps of a machine learning pipeline. The harmonizer transformer works with the [*Scikit-learn*](https://scikit-learn.org/stable/) library, a popular, open-source, well-documented, and easy-to-learn machine learning package that implements a vast number of machine learning algorithms. The harmonizer transformer can be easily included in a pipeline to learn the harmonization procedure parameters on the training data only and apply the harmonization procedure (with parameters learned in the training set) to the test data. This prevents data leakage in the harmonization procedure independently of the chosen validation scheme.
* *harm_efficacy.py* contains a function able to estimate the harmonization efficacy in removing and/or reducing the unwanted imaging site effect from the MRI-derived features. 

Please read the [LICENSE.md](./LICENSE.md) file before using it.

## Getting Started
### Create a new Python virtual environment using conda
We suggest to create a new virtual environment as follows:

1. Open a terminal window (for Unix users) or Anaconda Prompt (for Windows users) and type:

`conda create --name env_name`

`conda activate env_name`

`conda install python=3.10.4`

`conda install -c conda-forge py-xgboost=1.6.2`

`pip install neuroharmonize==2.1.0`

`pip install nibabel==4.0.2`

`pip install statsmodels==0.13.2`

To run the examples included in *test.py* file, please install *pandas* library: `conda install pandas=1.5.0`

### harmonizer
The source file [Harmonizer.py](./Harmonizer.py) contains the *harmonizer* transformer. It is a *Scikit-learn* custom transformer, that encapsulated the *neuroHarmonize* package.

		class harmonizer(*, features_names, covariates_names, eb = True, smooth_terms = [], smooth_term_bounds = (None, None))

		Harmonize MRI-derived features, by removing the imaging site effect using the ComBat algorithm [1]. 
		This transform extends the functionality of the package neuroHarmonize [2].
            
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

        >>> NI_features = data.columns.tolist()[3::] # MRI-derived features
        >>> covars_features = data.columns.tolist()[0:3] # covariates features
        >>> harm = harmonizer(feature_names = NI_features, covariates_names = covars_features, eb = True, smooth_terms = ["Age"], smooth_term_bounds=(0, 100))
        >>> X_train, X_test = train_test_split(data, test_size=0.20, random_state=42)
        >>> harm.fit_transform(X_train)
        >>> test_data_adj = harm.transform(X_test)

### Harmonization efficacy
The source file [harm_efficacy.py](./harm_efficacy.py) contains the *efficacy* function. It computes the harmonization efficacy in removing and/or reducing the unwanted imaging site effect by using the performance of an XGBoost classifier trained to predict the imaging site from the MRI-derived features. Specifically, training an XGBoost classifier through N=100 repetitions of a stratified 5-fold cross-validation (CV), we estimated the median balanced accuracy in predicting imaging site from MRI-derived features (raw and harmonized with the *harmonizer* within the CV). The removal of the imaging site effect was evaluated by permutations test (5000 permutations). Thus, 5000 new models were created using a random permutation of the target labels (i.e., the imaging site), such that the explanatory MRI-derived variables were dissociated from their corresponding imaging site to simulate the null distribution of the performance measure against which the observed value was tested. If the permutations test p-value is >= 0.05, then the 
average balanced accuracy obtatining in predicting imaging site using harmonized MRI-derived features is not different from a chance-level one, and the imaging site effect can be considered removed from MRI-derived features. In some cases, the imaging site effect cannot be completely removed (permutations test p-value < 0.05) but only reduced. We measured the imaging site effect reduction by computing the one-sided Wilcoxon signed-rank between balanced accuracy obtained using raw data and balanced accuracy estimated using harmonized data. If the imaging site effect has been reduced by the harmonization step, then the balanced accuracy obtained using raw should be greater than that estimated using harmonized data, with a one-sided Wilcoxon signed-rank p-value < 0.05.


	def efficacy(data, MRI_features, covars_features = ['SITE'], smooth_terms = [], smooth_term_bounds=(None, None), covar_cat = None):
    
    Compute the harmonization efficacy
       
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
        Controls the boundary knots for nonlinear estimation. You should specify boundaries that contain the limits of the entire dataset, including the test data
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
    
 
## Testing
The file [test\_harmonizer.py](./test\_harmonizer.py) uses the *harmonizer* transformer in different configurations and the file [test\_efficacy.py](./test\_efficacy.py) computes the harmonization efficacy. They use the file [multicenter\_CT-FD\_features\_k3\_n25_age2.csv](./multicenter_CT-FD_features_k3_n25_age2.csv), which includes simulated MRI-derived features (i.e., cortical thickness, CT and fractal dimension, FD) and simulated age values of 75 healthy subjects belonged to three different single-center dataset (25 subjects per each single-center dataset). The relationship between age and neuroimaging features is quadratic. A detail descritption of MRI-derived features and how to compute them has been reported in our previous studies [2-7]. 

The file [multicenter\_CT-FD\_features\_k3\_n25_age2.csv](./multicenter_CT-FD_features_k3_n25_age2.csv) contains the following columns:

* SITE: label of each imaging site (i.e., a, b, c, ..., j)
* age: each subject's simulated age, expressed in years
* cortex\_CT (FD): CT (or FD) of the cerebral cortical gray matter (GM).
* lh\_cortex\_CT (FD), rh\_cortex\_CT (FD): CT (or FD) of the left (lh) and right (rh) cerebral cortical GM.
* lh\_frontal\_cortex_CT (FD), rh\_frontal\_cortex\_CT (FD): CT (or FD) of the left (lh) and right (rh) cerebral cortical GM of the frontal lobe.
* lh\_temporal\_cortex_CT (FD), rh\_temporal\_cortex\_CT (FD): CT (or FD) of the left (lh) and right (rh) cerebral cortical GM of the temporal lobe.
* lh\_parietal\_cortex_CT (FD), rh\_parietal\_cortex\_CT (FD): CT (or FD) of the left (lh) and right (rh) cerebral cortical GM of the parietal lobe.
* lh\_occipital\_cortex_CT (FD), rh\_occipital\_cortex\_CT (FD): CT (or FD) of the left (lh) and right (rh) cerebral cortical GM of the occipital lobe.

From the terminal window (for Unix users) or Anaconda Prompt (for Windows users), run [test\_harmonizer.py](./test\_harmonizer.py) and [test\_efficacy.py](./test\_efficacy.py): 

`python test_harmonizer.py`

`python test_efficacy.py`

Additional simulated neuroimaging features datasets can be found on Zenodo ([10.5281/zenodo.8119042](https://zenodo.org/record/8119042#.ZKfD-S9Bz0o), [10.5281/zenodo.7848840](https://zenodo.org/record/7848840#.ZKZ0zy9BzfY)). *In vivo* neuroimaging features multicenter datasets are freely available at [10.5281/zenodo.7845311](https://zenodo.org/record/7845311#.ZKfCry9BxhE) and [10.5281/zenodo.7845361](https://zenodo.org/record/7845361#.ZKVhwi9BzfY).

## Authors
* [**Chiara Marzi**](https://www.unibo.it/sitoweb/chiara.marzi3/en) - *Junior Assistant Professor (fixed-term) at the Department of Statistics, Computer Science and Applications "Giuseppe Parenti", University of Florence, Florence, Italy.* <chiara.marzi@unifi.it>, <chiara.marzi3@unibo.it>

* [**Stefano Diciotti**](https://www.unibo.it/sitoweb/stefano.diciotti/en) - *Associate Professor in Biomedical Engineering, Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi", University of Bologna, Bologna, Italy.*  <stefano.diciotti@unibo.it>

## Contribution, help, bug reports, feature requests
The authors welcome contributions to the *harmonizer* repository. Please contact the authors if you would like to contribute code, or for any questions and comments.
Bug reports should include sufficient information to reproduce the problem.

## References
[1] Pomponio R, Erus G, Habes M, Doshi J, Srinivasan D, Mamourian E, Bashyam V, Nasrallah IM, Satterthwaite TD, Fan Y, Launer LJ, Masters CL, Maruff P, Zhuo C, Völzke H, Johnson SC, Fripp J, Koutsouleris N, Wolf DH, Gur R, Gur R, Morris J, Albert MS, Grabe HJ, Resnick SM, Bryan RN, Wolk DA, Shinohara RT, Shou H, Davatzikos C. Harmonization of large MRI datasets for the analysis of brain imaging patterns throughout the lifespan. Neuroimage. 2020 Mar;208:116450. doi: 10.1016/j.neuroimage.2019.116450. Epub 2019 Dec 9. PMID: 31821869; PMCID: PMC6980790.

[2] Marzi C, Giannelli M, Tessa C, Mascalchi M, Diciotti S. Toward a more reliable characterization of fractal properties of the cerebral cortex of healthy subjects during the lifespan. Sci Rep. 2020 Oct 12;10(1):16957. doi: 10.1038/s41598-020-73961-w. PMID: 33046812; PMCID: PMC7550568.

[3] Pantoni L, Marzi C, Poggesi A, Giorgio A, De Stefano N, Mascalchi M, Inzitari D, Salvadori E, Diciotti S. Fractal dimension of cerebral white matter: A consistent feature for prediction of the cognitive performance in patients with small vessel disease and mild cognitive impairment. Neuroimage Clin. 2019;24:101990. doi: 10.1016/j.nicl.2019.101990. Epub 2019 Aug 22. PMID: 31491677; PMCID: PMC6731209.

[4] Pani J, Marzi C, Stensvold D, Wisløff U, Håberg AK, Diciotti S. Longitudinal study of the effect of a 5-year exercise intervention on structural brain complexity in older adults. A Generation 100 substudy. Neuroimage. 2022 Aug 1;256:119226. doi: 10.1016/j.neuroimage.2022.119226. Epub 2022 Apr 18. PMID: 35447353.

[5] Marzi C, Ciulli S, Giannelli M, Ginestroni A, Tessa C, Mascalchi M, Diciotti S. Structural Complexity of the Cerebellum and Cerebral Cortex is Reduced in Spinocerebellar Ataxia Type 2. J Neuroimaging. 2018 Nov;28(6):688-693. doi: 10.1111/jon.12534. Epub 2018 Jul 5. PMID: 29975004.

[6] Marzi C., Giannelli M., Tessa C., Mascalchi M., and Diciotti S., "Fractal Analysis of MRI Data at 7 T: How Much Complex Is the Cerebral Cortex?," in IEEE Access, vol. 9, pp. 69226-69234, 2021, doi: 10.1109/ACCESS.2021.3077370.

[7] Marzi, C., Giannelli, M., Barucci, A., Tessa, C., Mascalchi, M., & Diciotti, S. (2022). Efficacy of MRI data harmonization in the age of machine learning. A multicenter study across 36 datasets. arXiv preprint arXiv:2211.04125.
