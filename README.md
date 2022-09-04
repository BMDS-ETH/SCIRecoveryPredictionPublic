# SCIRecoveryPredictionPublic

We provide a matching algorithm to identify digital twins for spinal cord injury patients in the acute injury phase (e.g. 1-2weeks) to predict recovery (e.g. 26 weeeks) trajectories. The algorithm is flexible to handle different time points

## Data required
This scrip required input data for: 
A) Reference data base (acute and recovery phase) - the data base from which matches are taken
B) Information on patients to match (acute phase, for validation purposes: recovery phase)
For each data base the following information is required (a subset of scores may be sufficient for some of the matching algorithms): 
- Light touch and pin prich scores at each of the 28 dermatome levels (score range 0-2, saved in individual .csv files)
- Motor scores for each of 10 myotomes (score range 0-5, saved in a separate .csv file)
- Additional data (saved in a separate .csv file): AISA grade (A,B,C,D,E), neurological level of injury (NLI, in the format 'C4'),patien age in years (or age group), plegia (tetra/para), optional addition of other parameters such as cause of injury or treatment centre. 

Input and reference data should be saved separately, but for testing it is also possible to use the same data base for matching and reference, too. All information should be provided as columns, patients as rows separately for acute MS, LTS, PPS and additional data, as well as recovery MS (reference data only).See included dummy datasets for details. 

### Bootstrapping
Physical examination to assess mytomal integrity is subject to uncertainty stemming from the physician's level of experience and personal variation, as well as the patient's compliance to the examination and natural variation of physical fitness. It is hence crucial to account for these sources of uncertainty during model training and evaluation. We base our estimation of MS uncertainty distributions on an investigation of [Bye et al. (2021)](https://pubmed.ncbi.nlm.nih.gov/31674263/) who assessed the inter-observer variability of manual MS assessments as a function of MS level for two myotomes (elbow flexor, wrist extensor). We compile a probability density function for each score level (0 to 5) to serve as the basis for bootstrap analysis during model training and evaluation. We derived a probability density function based on their assessments for each motor score level. The relevant information is provided in data/uncertMS.csv. 
Based on this estimation it is possible to bootstrap the relevant motor scores used for matching. The final result will be saved as median values and 95\% percentiles (upper and lower bounds), if seleced. 


## Matching options:
Patients can be matched using the getDigitalTwins.py script based on different choices. Each matching comprises three steps: 1) Optional reference pool subgrouping, 2) Motor score matching by one of 4 types, 3) Agglomeration of the (multiple) identified twins to arrive at the final prediciton. 

Step 1: 
Matching based on patient subgrouping (by AISA grade, NLI, age & sex, cause of injury) - by choosing one or multipe of these subgrouping options the reference pool is resitricted to patients meeting these criteria with respect to the patient oof interest. 

Step 2: 
Neurological motor function can be matched using one of four different methods: 
1) Matching based on LEMS (lower extremity motor score) within a given score window (e.g. +/- 5 points)
2) Matching based on the mean motor score below the NLI (meanMS) within a given score window (e.g. +/- 0.5 points)
3) Matching based on the RMSE in the acute phase between the patient of interest and all possible reference patients within a given score (e.g. +/- 0.5 points)
4) Matching based on k nearest neighbours (based on motor scores only (type 4A), or motor and sensory scores (type 4B)

Step 3: 
Twins can be agglomerated by either mean or median calculation accross all motor score sequences of the identified twins. 

Motor score assessment uncertainty can be adressed via bootstrapping (optional).

## Evaluation of matching performance
The matching can be evaluated at the individual patient or patient population level using three metrics (evalDigitalTwins.py): 
1) LEMS deviation: The difference of the lower extremity motor score (i.e. the sum over all motor scores of the lower extremities) between predicted and true recovery
2) RMSE below the NLI: The root mean squared error of all motor scores below the NLI between predicted and true recovery
3) Functional scores: AIS grade conversion, walking and self-care ability

In addition to summary metrics it is also possible to visualize individual patient motor scores in the actute and recovery phase.  

## Installation
Please install the required packages by following the instructions for [installation of the scanpy package] (https://scanpy.readthedocs.io/en/stable/). Addionally we require the tqdm, pandas, pickle, numpy and matplotlib libraries. 
