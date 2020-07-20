# COVID-19 Bayesian ODE Model (county-level forecasts)
## How to use: 
The model is based on a Bayesian inference framework that utilizes the Johns Hopkins Center for Systems Engineering and Science (JHU-CSSE) U.S. county-level confirmed case count to estimate parameters and forecast infections.  To estimate the parameters using the Markov Chain Monte Carlo sampling simply execute the following command.
```
python getAllData.py
```
The multi-processing computations and can be scaled up to harness high-performance computing (HPC) resources such as Amazon Web Services or large computational clusters. Once the parameter estimation procedure is finished. The posterior ranges of the paramters are used to conduct a forecast for a prescribed time, the forecast is produced using the following command.
```
python CountyModelProjectionCounty.py    # (or using CountyModelProjectionCountyMP.py for running faster)
```
For technical details, refer to *Model Documentation.pdf* in the *documents* folder.

##  Note: 
  - getAllData.py file is to get the lastest real time-series data in each county (cases&deaths data go to data/countiesData folder, and all U.S. County names (totally 3064) go to USCountiesList.txt)
  - CountyModelProjectionCounty.py file is to get the 500 days projected time-series data (write to results folder) 
  - Projection starting date: 2020-01-22 
  - Sources of real time-series data: The New York Times and Johns Hopkins University's Center For Systems Science and Engineering.

