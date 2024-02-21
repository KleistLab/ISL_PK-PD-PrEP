#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from lmfit import Parameters, minimize
import argparse

def param_bootstrap(files, data_type):
    data = []
    n_len = [0]  # Initialize as a list for easier appending.
    for file in files:
        df = pd.read_csv(file, decimal=',').astype(float)
        n = len(df)
        n_len.append(n_len[-1] + n) # Cumulative lengths of the data sets.

        samples = []
        for _, row in df.iterrows():  # Iterate over rows
            time, conc, std = row['time'], row['mean'], row['std']
            se = std / np.sqrt(n)  # Standard error
            sample = np.random.normal(conc, se) # Normal distribution.
            if data_type != 'implant_plasma':
                sample = np.log10(sample*5555.6) if 'intra' in data_type else np.log10(sample) # Convert nM to pmol/million cells (intracellular concentration).
            samples.extend([time, sample])
        data.append(samples)
    return data, n_len

def sample_data(dir_path):
    data = []; admin = []; ind_n = []
    for root, dirs, files in os.walk(dir_path):
        # Read files and sorts them based on dosage.
        dirs.sort(reverse=True)
        data_type = os.path.basename(root) #i.e. oral_plasma, oral_intra, implant_plasma, implant_intra.
        files = [os.path.join(root, f) for f in files]
        # Sort files based on dosage, handling '0' prefixes like 05mg.
        sorted_files = sorted(files, key=lambda x: int(x.split('/')[-1].split('mg')[0]) if x.split('/')[-1].split('mg')[0][0] != '0' else float(int(x.split('/')[-1].split('mg')[0])/10))
        df, n = param_bootstrap(sorted_files, data_type) # Returns sampled data set and their cumulative lengths for a data_type.
        ind_n.append(n)
        # Extract and sort dosage labels from file names, considering '0' prefixes (values < 1).
        admin.append((data_type, sorted([float(os.path.splitext(os.path.basename(f))[0][:-2]) if os.path.splitext(os.path.basename(f))[0][0] != '0' else float(os.path.splitext(os.path.basename(f))[0][:-2])/10 for f in files])))
        data.append(df)
        # Convert ind_n to list with arrays.
        ind_n = [np.array(arr) for arr in ind_n]
    return data[1:], admin[1:], ind_n[1:]


def get_params():
    # Dictionary containing values for dose-specific parameters.
    dose_specific_values = {'05': 36.879, '1': 27.803, '2': 16.572, '5': 12.584, '10': 17.52, '15': 13.96,
          '30': 17.571, '100': 8.877, '200': 10.632, '400': 9.162, '48': 25.138,
          '52': 31.06, '54': 19.616, '56': 41.426, '62': 35.24}

    # Create a Parameters object.
    parameter = Parameters()
    # Fixed implant-specific parameters (obtained from prior optimization).
    # parameter contains: variable name, value, vary=bool, minimal and maximal values.
    parameter.add_many(('kR', 0.005241/24, False), ('k_48', 0.004606/24, False), ('k_52', 0.0066447/24, False),
                       ('k_54', 0.007367/24, False), ('k_56', 0.008288/24, False), ('k_62', 0.011049/24, False),
                       ('p_1', 0.0412, False), ('p_2', 0.0583, False), ('ka', 0.415, True, 0.1, 1),
                       ('kC_0', 7.495, True, 0, 15), ('kC_P1', 1.741, True, 1, 2), ('kP1_C', 0.0016, True, 0, 0.1),
                       ('kC_P2', 4.896, True, 3, 7), ('kP2_C', 0.0258, True, 0, 1), ('Vc', 3.5, False),
                       ('kSC_C', 0.0156, True, 0, 0.1), ('kSC_0', 0.1590, True, 0, 1), ('kE_0', 0.0079, True, 0, 0.1))
    # Add parameters for kC_E(dose) from kl dictionary.
    [parameter.add('kC_E_' + key, value, min=value-3, max=value+3) for key, value in dose_specific_values.items()]
    return parameter

def pk_compartment_model(t, z, dose, med, params):
    # Based on the current state, time, dose, medication type, and parameter values.
    kC_0 = params['kC_0'].value
    kC_P1 = params['kC_P1'].value
    kP1_C = params['kP1_C'].value
    Vc = params['Vc'].value
    kC_P2 = params['kC_P2'].value
    kP2_C = params['kP2_C'].value
    kE_0 = params['kE_0'].value

    if int(dose) == 20: kC_E = 1 # For 20 mg only plasma concentration is considered.
    else: kC_E = params['kC_E_' + str(dose)].value # Get dose-specific transfer rates (C -> E).

    # Distinguish between oral and implant administration.
    if 'implant' in med:
        ka = 0
        kR = params['kR'].value
        k_dose = params['k_' + str(dose)].value
        kSC_C = params['kSC_C'].value
        kSC_0 = params['kSC_0'].value
    else:
        kR, k_dose, kSC_C, kSC_0 = [0]*4
        ka = params['ka'].value

    # Impl: Implant
    # SC: Subcutaneous
    # D: Dosing
    # C: Plasma (central compartment)
    # P1: Peripheral I
    # P2: Peripheral II
    # E: Effect side (intracellular compartment)
    Impl, SC, D, C, P1, P2, E = z[0], z[1], z[2], z[3], z[4], z[5], z[6]
    dImpl = - np.exp(- kR * t) * k_dose * Impl
    if t >= 2016: dImpl = 0 # Stop drug release after implant removal.
    dSC = - dImpl - kSC_C * SC - kSC_0 * SC
    dD = - ka * D
    dC = kSC_C * SC - dD / Vc - kC_P1 * C + kP1_C * P1 - kC_0 * C - kC_P2 * C + kP2_C * P2
    dP1 = kC_P1 * C - kP1_C * P1
    dP2 = kC_P2 * C - kP2_C * P2
    dE = kC_E * C - kE_0 * E
    return [dImpl, dSC, dD, dC, dP1, dP2, dE]

def solve_ode(z0, params, t_list):
    # Function solves the ordinary differential equations (ODEs) of ISL's PK model and returns solutions for plasma and intracellular concentrations over time (t_list).
    sol_plasma, sol_intra = np.array([]), np.array([])
    for z in z0:
        # z: [administration_type, dose_mg, initial_conditions].
        solution = solve_ivp(pk_compartment_model, (0, int(t_list[-1])+1), z[2], t_eval=t_list, args=(z[1], z[0], params))
        if z[0] == 'oral':
            sol_plasma = np.append(sol_plasma, solution.y[3])
        else:
            # Append the solutions for plasma concentrations.
            if z[1] == '54' or z[1] == '62':
                sol_plasma = np.append(sol_plasma, solution.y[3])

        # Append the solutions for intracellular concentrations, excluding 20mg doses.
        if str(z[1]) != '20': sol_intra = np.append(sol_intra, solution.y[6])
    return sol_plasma, sol_intra

def residual(params, z0, data_time, data_conc, ind_t, ind_n):
    # Compute residuals for the given parameters, initial conditions, and observed data.
    residual_array = np.array([])

    # Iterate over administration type in plasma and intracellular.
    for i in range(2):
        # Initialize arrays to store plasma and intracellular solutions, and indices.
        fp, fi, ind_p, ind_i = np.array([]), np.array([]), np.array([]), np.array([])

        sol_plas, sol_intr = solve_ode(z0[i], params, data_time[i]) # Solution of ODE.

        n_plasma = ind_n[i*2]
        n_intra = ind_n[i*2+1]
        offset = len(data_time[i]) # offset: Length of data for a single dose.

        # Get Solution for each dose in plasma compartment.
        for x in range(len(n_plasma)-1):
            ip = [np.where(ind_t[i] == e)[0][0] for e in range(n_plasma[x], n_plasma[x+1])]
            ind_p = np.append(ind_p, ip)
            fp = np.append(fp, sol_plas[offset*x:offset*(x+1)][ip])

        # Get Solution for each dose in plasma compartment.
        for y in range(len(n_intra) - 1):
            ii = [np.where(ind_t[i] == e)[0][0] for e in range(n_intra[y], n_intra[y+1])]
            ind_i = np.append(ind_i, ii)
            fi = np.append(fi, sol_intr[offset * y:offset * (y + 1)][ii])

        # Extract time and concentration data for plasma and intracellular compartments.
        ind_p = ind_p.astype(int)
        ind_i = ind_i.astype(int)

        t = data_time[i][ind_p]
        conc_p = data_conc[i][ind_p]
        conc_i = data_conc[i][ind_i]

        # Calculate residuals based on administration type (oral or implant).
        if z0[i][0][0] == 'oral':
            log_fp = [np.log10(x) if x > 0 else 1e-6 for x in fp]
            log_fi = [np.log10(y) if y > 0 else 1e-6 for y in fi]
            fy = np.concatenate([np.subtract(conc_p, log_fp), np.subtract(conc_i, log_fi)]).ravel()
        else:
            # For implant plasma, set negative residuals to 0 if time exceeds 24 hours and measured minus predicted data point is negative.
            f = [max(0, np.subtract(conc_p[x], fp[x])) if t[x] >= 24 else np.subtract(conc_p[x], fp[x]) for x in range(len(t))]
            log_fi = [np.log10(y) if y > 0 else 1e-6 for y in fi]
            fy = np.concatenate([f, np.subtract(conc_i, log_fi)]).ravel()

        residual_array = np.append(residual_array, fy)
    return residual_array

def per_iteration(pars, iteration, resid, *args, **kws):
    # Print iteration information including parameter names and values, and the sum of squared residuals (RSS).
    print(" ITER ", iteration, [f"{p.name} = {p.value:.5f}" for p in pars.values()])
    print("RSS", round(np.sum(resid**2),3))
    print(" ")

def main(input_dir, output_dir):
    # Main function to perform parameter estimation and save results.
    data, admin, ind_n = sample_data(input_dir)
    params = get_params()

    data_time, data_conc, ind_t, n_list = [], [], [], []

    z0 = []
    conv = 1e6 / 293.258  # Convert mg to nM.
    for i in range(0, 3, 2):
        data_i = data[i + 1]  # Intracellular data.
        data_p = data[i]  # Plasma data.
        doses = sorted(list(set(np.concatenate([admin[i][1], admin[i + 1][1]]))))
        ind_n[i + 1] = ind_n[i + 1] + ind_n[i][-1]

        z = [] # Create initial condition for the ODE system.
        for d in doses:
            if 'oral' in admin[i][0]:
                z += [('oral', str(int(d)) if d >= 1 else '0' + str(int(d * 10)), [0, 0, d * conv, 0, 0, 0, 0])]
            else:
                # Implants are initialized with a specific concentration (percentage) in the subcutaneous compartment.
                if d <= 54:
                    p = 0.0412
                else:
                    p = 0.0583
                z += [('implant', str(int(d)), [d * conv * (1 - p), d * conv * p, 0, 0, 0, 0, 0])]
        z0.append(z)

        data_t = np.concatenate([dpi[::2] for dpi in data_p] + [dti[::2] for dti in data_i])
        ind_t.append(np.argsort(data_t))
        data_time.append(data_t[ind_t[-1]])
        data_c = np.concatenate([dpc[1::2] for dpc in data_p] + [dtc[1::2] for dtc in data_i])
        data_conc.append(data_c[ind_t[-1]])

    # Start optimization. Optional arguments: max_nfev=int #maximal number of iterations, iter_cb=per_iteration #show results per iteration.
    solution = minimize(residual, params, args=(z0, data_time, data_conc, ind_t, ind_n), method='least_squares')

    # Save results.
    results_df = pd.DataFrame({'Parameter': [p.name for p in solution.params.values()],
                               'Value': [p.value for p in solution.params.values()]})
    transposed_df = results_df.set_index('Parameter').T.reset_index(drop=True)
    filename = 'pk_bootstrap_solution.xlsx'
    output_path = os.path.join(output_dir, filename)

    # Check if the file already exists.
    if os.path.isfile(output_path):
        existing_df = pd.read_excel(output_path)
        # If the file already exists, the results are appended, not overwritten.
        df =  pd.concat([existing_df, transposed_df.iloc[-1].to_frame().T], ignore_index=True)
        df.to_excel(output_path, index=False)
    else:
        transposed_df.to_excel(output_path, index=False) # Save as rows not columns.
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pharmacokinetic model for ISL with bootstrapped data.", formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument("--data_path", nargs='?', type=str, default='../data/PK/',
                        help="Path to the directory containing input data. Default: %(default)s.")
    parser.add_argument("--out_path", nargs='?', type=str, default='../results/',
                        help="Path to the directory for output data. Default: %(default)s.")
    parser.add_argument("--filename", nargs='?', type=str, default='pk_bootstrap_solution.xlsx',
                        help="Name of the output file. If executed multiple times, the results are added to the existing file, not replaced. Default: %(default)s.")

    args = parser.parse_args()
    input_dir = os.path.abspath(args.data_path)
    output_dir = os.path.abspath(args.out_path)
    main(input_dir, output_dir)
