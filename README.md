# SCIRecoveryPredictionPublic
We provide a matching algorithm to identify digital twins for spinal cord injury patients in the acute injury phase to predict recovery trajectories. This requires the following input information: 
A) Reference data base (acute and recovery phase)
B) Information on patients to match (acute phase, for validation purposes: recovery phase)
For each data base the following information is required (a subset of scores may be sufficient for some of the matching algorithms): light touch and pin prich scores at each of the 28 dermatome levels (score range 0-2), motor scores for each of 10 myotomes (score range 0-5), AISA grade (A,B,C,D,E), neurological level of injury (NLI, in the format 'C4'),patien age in years (or age group), plegia (tetra/para), optional addition of other parameters such as cause of injury or treatment centre. 
Input and reference data need to be saved in .csv format with relevant information as columns, patients as rows separately for acute MS,LTS,PPS and additional data, as well as recovery MS (reference data only).See included dummy datasets for details. 

Patients can be matched using the getDigitalTwins.py script based on different choices: 
1) Matching based on k nearest neighbours (based on sensory and/or motor scores)
2) Matching based on patient subgrouping (by AISA grade, NLI (exact, coarse), LEMS, meanMS below NLI, age, cause of injury)
3) Matching based on a combination of 1) and 2)

Depending on the choice of matching algorithm, if possible, one or multipe matched patients will be identified. The final recovery prediction can be based on either the mean or median of the matched patient trajectories. Motor score assessment uncertainty can be adressed via bootstrapping. 

The matching can be evaluated at the individual patient or patient population level using three metrics (evalDigitalTwins.py): 
1) LEMS deviation: The difference of the lower extremity motor score (i.e. the sum over all motor scores of the lower extremities) between predicted and true recovery
2) RMSE below the NLI: The root mean squared error of all motor scores below the NLI between predicted and true recovery
3) Mean Linearized Score below the NLI: A linearized score, accounting for the nonlinearity of the motor score scale, averaged over all myotomes below the NLI. The difference between true and prediced recovery scores is given.
All scores are presented in the form of histograms over all patients matched by all evaluated models. Median values and 95\% confidence bounds are reported in figure legends. 
