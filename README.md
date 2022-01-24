# SCIRecoveryPredictionPublic

We provide a matching algorithm to identify digital twins for spinal cord injury patients in the acute injury phase to predict recovery trajectories. 

## Data required
This scrip required input data for: 
A) Reference data base (acute and recovery phase) - the data base from which matches are taken
B) Information on patients to match (acute phase, for validation purposes: recovery phase)
For each data base the following information is required (a subset of scores may be sufficient for some of the matching algorithms): 
- light touch and pin prich scores at each of the 28 dermatome levels (score range 0-2, saved in individual .csv files)
- motor scores for each of 10 myotomes (score range 0-5, saved in a separate .csv file)
- Additional data (saved in a separate .csv file): AISA grade (A,B,C,D,E), neurological level of injury (NLI, in the format 'C4'),patien age in years (or age group), plegia (tetra/para), optional addition of other parameters such as cause of injury or treatment centre. 

Input and reference data should be saved separately, but for testing it is also possible to use the same data base for matching and reference, too. All information should be provided as columns, patients as rows separately for acute MS, LTS, PPS and additional data, as well as recovery MS (reference data only).See included dummy datasets for details. 

### Bootstrapping
Physical examination to assess mytomal integrity is subject to uncertainty stemming from the physician's level of experience and personal variation, as well as the patient's compliance to the examination and natural variation of physical fitness. It is hence crucial to account for these sources of uncertainty during model training and evaluation. We base our estimation of MS uncertainty distributions on an investigation of [Bye et al. (2021)](https://pubmed.ncbi.nlm.nih.gov/31674263/) who assessed the inter-observer variability of manual MS assessments as a function of MS level for two myotomes (elbow flexor, wrist extensor). We compile a probability density function for each score level (0 to 5) to serve as the basis for bootstrap analysis during model training and evaluation. We derived a probability density function based on their assessments for each motor score level. The relevant information is provided in data/uncertMS.csv. 
Based on this estimation it is possible to bootstrap the relevant motor scores used for matching. The final result will be saved as median values and 95\% percentiles (upper and lower bounds), if seleced. 


## Matching options
Patients can be matched using the getDigitalTwins.py script based on different choices: 

1) Matching based on k nearest neighbours (based on sensory and/or motor scores)
2) Matching based on patient subgrouping (by AISA grade, NLI (exact, coarse), LEMS, meanMS below NLI, age, cause of injury)
3) Matching based on a combination of 1) and 2)

Depending on the choice of matching algorithm, if possible, one or multipe matched patients will be identified. The final recovery prediction can be based on either the mean or median of the matched patient trajectories. Motor score assessment uncertainty can be adressed via bootstrapping. 

## Evaluation of matching performance
The matching can be evaluated at the individual patient or patient population level using three metrics (evalDigitalTwins.py): 
1) LEMS deviation: The difference of the lower extremity motor score (i.e. the sum over all motor scores of the lower extremities) between predicted and true recovery
2) RMSE below the NLI: The root mean squared error of all motor scores below the NLI between predicted and true recovery
3) Mean Linearized Score below the NLI: A linearized score, accounting for the nonlinearity of the motor score scale, averaged over all myotomes below the NLI. The difference between true and prediced recovery scores is given.
All evaluation scores are presented in the form of histograms over all patients matched by all evaluated models. Median values and 95\% confidence bounds are reported in figure legends. Histograms are currently plotted separately for each score, AISA grade and type of plegia (para/tetra). Summary histograms are also plotted for each matching model if the option "plotSummary = True" in line 145 of getDigitalTwins.py.

In addition to summary metrics it is also possible to visualize individual patient motor scores in the actute and recovery phase (use option "plotIndPats = True" in line 146 of getDigitalTwins.py).  

## Installation
Please install the required packages by following the instructions for [installation of the scanpy package] (https://scanpy.readthedocs.io/en/stable/) - this fulfills all requirements. 
