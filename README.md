# harmonizer

The *harmonizer-HS* repository contains two main tools:

* *Harmonizer.py* which includes the harmonizer transformer, i.e., a Python transformer that encapsulates the *neuroHarmonize* procedure [1] among the preprocessing steps of a machine learning pipeline. The harmonizer transformer works with the *Scikit-learn* library, a popular, open-source, well-documented, and easy-to-learn machine learning package that implements a vast number of machine learning algorithms. The harmonizer transformer can be easily included in a pipeline to learn the harmonization procedure parameters on the training data only and apply the harmonization procedure (with parameters learned in the training set) to the test data. This prevents data leakage in the harmonization procedure independently of the chosen validation scheme.
* *HS.py* contains a function able to compute the *harmonization score* (HS), i.e., a quantitative index to measure the efficacy of data harmonization in reducing the unwanted site effect by using the performance of an XGBoost classifier trained to predict the imaging site from the MRI-derived features. 

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
[Harmonizer.py](./Harmonizer.py) contains the *harmonizer* transformer. It is a *Scikit-learn* custom transformer, that encapsulated the *neuroHarmonize* package.

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

### HS
[HS.py](./HS.py) contains the *HS* function. The harmonization score (HS)is a quantitative index to measure the efficacy of data harmonization in reducing the unwanted site effect by using the performance of an XGBoost classifier trained to predict the imaging site from the MRI-derived features. Specifically, training an XGBoost classifier through N=10 repetitions of a stratified 5-fold CV, we estimated the average balanced accuracy. The statistical significance of prediction performance was determined via permutation analysis. Thus, 500 new models were created using a random permutation of the target labels (i.e., the imaging site), such that the explanatory MRI-derived variables were dissociated from their corresponding imaging site to simulate the null distribution of the performance measure against which the observed value was tested. Average balanced accuracy was considered significant when the p-value computed using permutation tests was < 0.05. Then, we defined the harmonization score (HS) to measure the efficacy of harmonization in reducing the site effect as HS = -1/log10(p) if p != 0, otherwise HS = 0, where p is the p-value resulting from the permutation test. HS assumes values in the range [0, +∞[, with high values indicating more effective harmonization. 


	def HS(data, MRI_features, harmonization = False, covars_features = [], smooth_terms = [], smooth_term_bounds=(None, None))

    Compute the harmonization score (HS) of MRI-derived features.
    
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
    
 
## Testing
The file [test.py](./test.py) uses the *harmonizer* transformer and computes the *HS* in different configurations (details within [test.py](./test.py) code). It uses the file [data.csv](./data.csv), which includes the MRI-derived fractal descriptors of healthy subjects belonged the [*International Consortium for Brain Mapping (ICBM)*](http://fcon_1000.projects.nitrc.org/fcpClassic/FcpTable.html), [*Nathan Kline Institute - Rockland Sample Pediatric Multimodal Imaging Test-Retest Sample (NKI2)*](http://fcon_1000.projects.nitrc.org/indi/CoRR/html/nki_2.html), and [*Information eXtraction from Images (IXI)*](https://brain-development.org/ixi-dataset/) international and public studies. A detail descritption of neuroimaging fractal descriptors and how to compute them has been reported in our previous studies [2-6].

The file [data.csv](./data.csv) contains the following columns:

* SITE: label of each imaging site (i.e., ICBM, NKI2, Guys, HH, IOP (the last three belong to IXI))
* Age: each subject's age, expressed in years
* Sex: 0=male; 1=female
* cerebralGM\_FD, cerebralWM\_FD: fractal dimension (FD) of the cerebral cortical gray matter (GM) and white matter (WM)
* lh\_cerebralGM\_FD, lh\_cerebralWM\_FD: FD of the left hemisphere of the cerebral cortical GM and WM
* rh\_cerebralGM\_FD, rh\_cerebralWM\_FD: FD of the right hemisphere of the cerebral cortical GM and WM
* lh\_frontalGM\_FD, lh\_temporalGM\_FD, lh\_parietalGM\_FD, lh\_occipitalGM\_FD: FD of the left cerebral GM lobes	
* rh\_frontalGM\_FD, rh\_temporalGM\_FD, rh\_parietalGM\_FD, rh\_occipitalGM\_FD: FD of the right cerebral GM lobes	

From the terminal window (for Unix users) or Anaconda Prompt (for Windows users), run [test.py](./test.py): 

`python test.py`


## Authors
* [**Chiara Marzi**](https://www.unibo.it/sitoweb/chiara.marzi3/en) - *Post-doctoral reserach fellow at the Institute of Applied Physics "Nello Carrara" (IFAC), National Council of Research (CNR), Sesto Fiorentino, Firenze, Italy.* <c.marzi@ifac.cnr.it>, <chiara.marzi3@unibo.it>

* [**Stefano Diciotti**](https://www.unibo.it/sitoweb/stefano.diciotti/en) - *Associate Professor in Biomedical Engineering, Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi", University of Bologna, Bologna, Italy.*  <stefano.diciotti@unibo.it>

## Contribution, help, bug reports, feature requests
The authors welcome contributions to the *harmonizer-HS* repository. Please contact the authors if you would like to contribute code, or for any questions and comments.
Bug reports should include sufficient information to reproduce the problem.

## References
[1] Pomponio R, Erus G, Habes M, Doshi J, Srinivasan D, Mamourian E, Bashyam V, Nasrallah IM, Satterthwaite TD, Fan Y, Launer LJ, Masters CL, Maruff P, Zhuo C, Völzke H, Johnson SC, Fripp J, Koutsouleris N, Wolf DH, Gur R, Gur R, Morris J, Albert MS, Grabe HJ, Resnick SM, Bryan RN, Wolk DA, Shinohara RT, Shou H, Davatzikos C. Harmonization of large MRI datasets for the analysis of brain imaging patterns throughout the lifespan. Neuroimage. 2020 Mar;208:116450. doi: 10.1016/j.neuroimage.2019.116450. Epub 2019 Dec 9. PMID: 31821869; PMCID: PMC6980790.

[2] Marzi C, Giannelli M, Tessa C, Mascalchi M, Diciotti S. Toward a more reliable characterization of fractal properties of the cerebral cortex of healthy subjects during the lifespan. Sci Rep. 2020 Oct 12;10(1):16957. doi: 10.1038/s41598-020-73961-w. PMID: 33046812; PMCID: PMC7550568.

[3] Pantoni L, Marzi C, Poggesi A, Giorgio A, De Stefano N, Mascalchi M, Inzitari D, Salvadori E, Diciotti S. Fractal dimension of cerebral white matter: A consistent feature for prediction of the cognitive performance in patients with small vessel disease and mild cognitive impairment. Neuroimage Clin. 2019;24:101990. doi: 10.1016/j.nicl.2019.101990. Epub 2019 Aug 22. PMID: 31491677; PMCID: PMC6731209.

[4] Pani J, Marzi C, Stensvold D, Wisløff U, Håberg AK, Diciotti S. Longitudinal study of the effect of a 5-year exercise intervention on structural brain complexity in older adults. A Generation 100 substudy. Neuroimage. 2022 Aug 1;256:119226. doi: 10.1016/j.neuroimage.2022.119226. Epub 2022 Apr 18. PMID: 35447353.

[5] Marzi C, Ciulli S, Giannelli M, Ginestroni A, Tessa C, Mascalchi M, Diciotti S. Structural Complexity of the Cerebellum and Cerebral Cortex is Reduced in Spinocerebellar Ataxia Type 2. J Neuroimaging. 2018 Nov;28(6):688-693. doi: 10.1111/jon.12534. Epub 2018 Jul 5. PMID: 29975004.

[6] C. Marzi, M. Giannelli, C. Tessa, M. Mascalchi and S. Diciotti, "Fractal Analysis of MRI Data at 7 T: How Much Complex Is the Cerebral Cortex?," in IEEE Access, vol. 9, pp. 69226-69234, 2021, doi: 10.1109/ACCESS.2021.3077370.
