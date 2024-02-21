#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
from lmfit import minimize, Parameters
import os
import argparse
from scipy.integrate import solve_ivp
import steady_state

# Function to extract data and doses from the given path.
def get_pd_data(path):
    data = []; doses = []
    for root, _, files in os.walk(path):
        files = [os.path.join(root, f) for f in files]
        # Sort files based on dosage, handling '0' prefixes like 05mg.
        sorted_files = sorted(files, key=lambda x: int(x.split('/')[-1].split('mg')[0])
        if x.split('/')[-1].split('mg')[0][0] != '0' else float(int(x.split('/')[-1].split('mg')[0]) / 10))
        for f in sorted_files:
            df = pd.read_csv(f, decimal=',').astype(float)
            df = df.dropna(axis='columns')
            df.columns = ['time', 'change']
            data.append(df)
            doses.append(f.split('/')[-1].split('mg')[0])
    return data, doses

# Function to solve the ordinary differential equations (ODEs).
def solve_ode(z, params, doses, t_list):
    deltaCV = np.array([])

    for j in range(len(z)):
        solution = solve_ivp(steady_state.coupled_ode, (0, t_list[j][-1]), z[j], t_eval=t_list[j], args=(doses[j], params,))
        V_total = solution.y[3]+solution.y[7] # Total number of viruses (free infectious + non-infectious viruses).
        deltaCV = np.append(deltaCV, np.log10(V_total/ V_total[0]))  # Total number divided by initial number of viruses.
    return deltaCV

# Function to calculate the residual between observed data and model predictions
def residual_CV(params, z0, data, dosages):
    data_mat = []; data_time = []

    for i in range(len(z0)):
        data_mat += data[i].change.tolist()
        data_time.append(data[i].time.tolist())

    deltaCV = solve_ode(z0, params, dosages, data_time)
    return np.subtract(deltaCV, data_mat)

def pd_optimization(max_count, parameters, z0, data, dose):
    solution = []
    for count in range(max_count):
        result = minimize(residual_CV, parameters, args=(z0, data, dose), method='least_squares')
        solution.append([result.params['IC50'].value, result.aic, np.sum(np.power(result.residual, 2))])
    return solution

# Main function to run the HIV-1 dynamics model with viral load data to estimate IC50.
def main(dir_path, out_path, filename, params, max_count):
    output_path = out_path + '/' + filename
    # Convert units to 1/day
    hr = 24
    ka, kC_0, kC_P1, kP1_C, kC_P2, kP2_C = [value[0]*hr for value in list(params.values())[1:7]]
    kE_0 = params['kE_0'][0] * hr
    # Maximal value (upper boundary).
    ic50_max = 600

    data, doses = get_pd_data(dir_path)

    parameters = Parameters()
    parameters.add('ka', value=ka, vary=False)
    parameters.add('kC_0', value=kC_0, vary=False)
    parameters.add('kC_P1', value=kC_P1, vary=False)
    parameters.add('kP1_C', value=kP1_C, vary=False)
    parameters.add('kC_P2', value=kC_P2, vary=False)
    parameters.add('kP2_C', value=kP2_C, vary=False)
    parameters.add('Vc', value=3.5, vary=False)
    parameters.add('kE_0', value=kE_0, vary=False)
    parameters.add('IC50', value=300, min=0, max=ic50_max)
    [parameters.add('kC_E_' + d[:-2], value=params[d][0]*hr, vary=False) for d in list(params.keys())[11:] if d[:-2] in doses]

    # Steady state of HIV-1 viral dynamics system without ISL (baseline).
    Tu0, T10, T20, V0, Mu0, M10, M20, VNI0 = steady_state.main(params, False)[:8]
    #print('steady_state', Tu0, T10, T20, V0, Mu0, M10, M20, VNI0)

    z0 = [] # Generate initial state of ODE-system, handling '0' prefixes like 05mg.
    [z0.append([Tu0, T10, T20, V0, Mu0, M10, M20, VNI0] + [float(d)*3410, 0, 0, 0, 0]) if d[0] != '0' else z0.append([Tu0, T10, T20, V0, Mu0, M10, M20, VNI0] + [float(d)/10 *3410, 0, 0, 0, 0]) for d in doses]

    # Run optimization.
    results = pd_optimization(max_count, parameters, z0, data, doses)

    # Save results.
    dfresult = pd.DataFrame(results, columns=['IC50', 'AIC', 'RSS'])
    df = pd.concat([dfresult[-1:], dfresult[:-1].sort_values(by=['AIC'])])  # Sort by AIC values.

    # Check if output file exists.
    if os.path.isfile(output_path):
        ans = input('\nOverwrite existing file {}? [Y/N] '.format(filename))
        if ans in 'Nn':
            sys.stderr.write('Program breaks\n')
            exit(1)

    df.to_csv(output_path)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the HIV-1 dynamics model model with viral load data.", formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument("--data_path", nargs='?', type=str, default='../data/PD/',
                        help="Path to the directory containing input data. Default: %(default)s.")
    parser.add_argument("--out_path", nargs='?', type=str, default='../results/',
                        help="Path to the directory for output data. Default: %(default)s.")
    parser.add_argument("--filename", nargs='?', type=str, default='pd_solution.xlsx',
                        help="Name of the output file. Default: %(default)s.")
    parser.add_argument("--maxcount", nargs='?', type=int, default=5,
                        help="Number of runs. Default: %(default)s'.")
    parser.add_argument("--pk_params", nargs='?', type=str, default='../_ini_/PK_default.csv',
                        help="Path to file with PK parameters. Default: %(default)s.")

    args = parser.parse_args()
    df_params = pd.read_csv(args.pk_params, header=0)
    df_params.columns = df_params.columns.str.strip()
    params = df_params.to_dict(orient='list')

    dir_path = os.path.abspath(args.data_path)
    out_path = os.path.abspath(args.out_path)
    max_count = args.maxcount
    filename = args.filename
    main(dir_path, out_path, filename, params, max_count)
