# ISL_PK-PD-PrEP

This repository contains programs and data related to the Islatravir (ISL) pharmacokinetics and pharmacodynamics (PK and PD) modeling. It includes clinical data, fitting programs, and results.

## Table of Contents

- [Folder Structure](#folder-structure)
- [Usage](#usage)
  - [Data](#data)
  - [Fitting procedure](#fitting-procedure)
  - [Code](#code)
    - [PK Fitting](#pk-fitting)
    - [PD Fitting](#pd-fitting)
    - [PGS](#pgs)
- [Results](#results)

## Folder Structure

```
project-root/
│
├── code/
│   ├── pk_fitting/
│   ├── pk_bootstrap/
│   ├── steady_state/
│   ├── pd_fitting/
│   └── pgs/      
│
├── data/
│   ├── PK/
│   │   ├── oral_plasma/
│   │   ├── oral_intracellular/
│   │   ├── implant_plasma/
│   │   └── implant_intracellular/
│   └── PD/
│       └── viral_load/
│
├── _ini_/
│   └── pk_parameters.csv
│
├── results/
│   ├── pk_boundaries.csv
│   ├── pk_bootstrap_estimates.csv
│   └── pd_fitting.csv
│
├── requirements.txt
│
└── README.md
```

## Usage

### Data

(Pre-)clinical data, manually extracted using Engauge Digitizer, include concentration-time profiles for ISL (administered as tablet or subdermal implant) in plasma and intracellular ISL-TP. The dataset also contains viral load (VL) data related to oral administration.

We have 10 datasets in plasma and 11 intracellular datasets in PBMCs for oral administration with doses ranging from 0.5 to 400 mg. For implant formulation, we examined 5 loading doses with ISL levels ranging from 48 to 62 mg, including two doses (54 and 62 mg) that report both plasma ISL and intracellular ISL-TP and three doses (48, 52 and 56 mg) reporting intracellular ISL-TP only.

The extracted data includes mean concentrations ($\mu$), time points, and error measurements (if available). Errors, standardized to standard deviations ($\sigma$), were used for parametric bootstrap. The bootstrapped data consisted of mean and standard error values. For missing values without obtainable errors, interpolation was performed using the coefficient of variation (CV = $\frac{\sigma}{\mu}$) within or between datasets.

### Fitting procedure
Optimization was performed by least-squares minimization with the trust region reflective method, considering the squared error between measured and predicted data points.

PK parameters for ISL were determined using a two-step approach. Parameter estimation could either follow random sampling combined with multiple local optimizations or global optimization with parameter initialization, executed through pk_optimization.py. Alternatively, local optimization with bootstrapped data could be performed using pk_optimization_bootstrap.py. Random sampling was applied over a relatively large parameter space and then refined through multiple local optimizations until convergence. The resulting parameter range served as initial values for local optimization of bootstraped data.

Using an established HIV-1 viral dynamics model, we estimated the log-change in VL by fitting the drug potency $IC_{50}$ through the execution of pd_optimization.py. The baseline, representing the pre-treatment condition of the viral dynamics model in the absence of ISL, can be computed using steady_state.py and is automatically executed by pd_optimization.py.

To assess the prophylactic efficacy of ISL before (PrEP) or after (PEP) a viral challenge, we used a new numerical approach called PGS, published from our research group. The program was customized to meet our specific requirements.

## Code

### PK Fitting

Parameter estimation following random sampling + local optimization or global optimization.

```bash
usage: ./pk_optimization.py [-h] [optional arguments]

#Run the pharmacokinetic model for ISL with mean concentration-time data.

positional arguments:
  data_path   Path to the directory containing input data. Default: '../data/PK/'.
  out_path    Path to the directory for output data. Default: '../results/'.
  glbl        Runs global optimization (initial values specified in the script). Otherwise, random parameter sampling and local optimization are performed. Default: False.
  max_count   Number of iterations (=1 if --glbl=True). Default: 5.
  filename    Name of the output file. Default: 'pk_solution.xlsx'.

  -h, --help  show this help message and exit

```

Parameter estimation with bootstraped data and local optimization.

```bash
usage: ./pk_optimization_bootstrap.py [-h] [optional arguments]

#Run the pharmacokinetic model for ISL with bootstrapped data.

positional arguments:
  data_path   Path to the directory containing input data. Default: '../data/PK/'.
  out_path    Path to the directory for output data. Default: '../results/'.
  filename    Name of the output file. If executed multiple times, the results are added to the existing file (not overwritten). Default: 'pk_bootstrap_solution.xlsx'.

  -h, --help  show this help message and exit

```

### PD Fitting

Obtain steady-state levels in the absence of ISL used as baseline (pre-treatment condition) to estimate the change in VL (executing pd_optimization.py automatically runs steady_state.py).

```bash
usage: ./steady_state.py [-h] [optional arguments]

#Compute steady-state levels of HIV-1 viral dynamics system without ISL (baseline for PD optimization).

positional arguments:
  pk_params   Path to file with PK parameters. Default: '../_ini_/PK_default.csv'.
  plot        Plot the Output. Default: False

  -h, --help  show this help message and exit

```

Execute the coupled PK and PD (viral dynamics system) model for ISL to estimate $IC_{50}$ by local optimization.

```bash
usage: ./pd_optimization.py [-h] [optional arguments]

#Estimate the antiviral potency of ISL IC50.

positional arguments:
  data_path   Path to the directory containing input data. Default: '../data/PD/'.
  out_path    Path to the directory for output data. Default: '../results/'.
  max_count   Number of iterations. Default: 10.
  filename    Name of the output file. Default: 'pd_solution.xlsx'.

```

### PGS

Compute the prophylactic efficacy of ISL for both dosing forms, orally as tablet or as subdermal implant formulation for PEP/PrEP events.

```bash
usage: ./pgs.py [-h] [optional arguments]

#Compute the prophylactic efficacy of ISL using PGS.

positional arguments:
  pk_params   Path to file with PK parameters. Default: '../_ini_/PK_default.csv'.

  -h, --help  show this help message and exit

```

## Results

The parameters obtained from each step are provided in the folder 'results'.

