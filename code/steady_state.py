#!/usr/bin/env python3

import numpy as np
from lmfit import Parameters
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.integrate import solve_ivp

# Global HIV-1 viral dynamics parameters.
params_vd = {
    'CL': 23, 'lambdaT': 2e9, 'deltaT': 0.02, 'deltaT1': 0.5, 'deltaT2': 1.4, 'deltaPICT': 0.35,
    'kT': 0.35, 'betaT0': 8e-12, 'NThat': 1000, 'NT': 670, 'NTdiff': 330,
    'lambdaM': 6.9e7, 'deltaM': 0.0069, 'deltaM1': 0.0069, 'deltaM2': 0.09, 'deltaPICM': 0.0035,
    'kM': 0.07, 'betaM0': 1e-13, 'NMhat': 100, 'NM': 67, 'NMdiff': 33,
    'm': 1  # Fixed for RTIs.
}

# Direct drug effect.
def eta(E, m=1):
    return E ** m / (IC50 ** m + E ** m)

def betaT(E):
    return params_vd['betaT0'] * (1 - eta(E))


def CLT(E):
    return (1 + eta(E)) * params_vd['betaT0']


def betaM(E):
    return params_vd['betaM0'] * (1 - eta(E))


def CLM(E):
    return (1 + eta(E)) * params_vd['betaM0']


def coupled_ode(t, z, dose, params):
    global IC50
    IC50 = params['IC50'].value
    Tu, T1, T2, V, Mu, M1, M2, VN, D, C, P1, P2, E = z

    ka = params['ka'].value
    kC_0 = params['kC_0'].value
    kC_P1 = params['kC_P1'].value
    kP1_C = params['kP1_C'].value
    kC_P2 = params['kC_P2'].value
    kP2_C = params['kP2_C'].value
    Vc = params['Vc'].value

    kE_0 = params['kE_0'].value
    kC_E = params['kC_E_' + str(dose)].value

    # Viral dynamics model:
    # Tu, Mu: Uninfected T-cells and macrophages.
    # T1, M1: Early infected T-cells and macrophages.
    # T2, M2: Productively infected T-cells and macrophages.
    # V, VN: Free infectious and non-infectious viruses.
    dTu = params_vd['lambdaT'] - params_vd['deltaT'] * Tu - betaT(E) * V * Tu + params_vd['deltaPICT'] * T1
    dMu = params_vd['lambdaM'] - params_vd['deltaM'] * Mu - betaM(E) * V * Mu + params_vd['deltaPICM'] * M1
    dT1 = betaT(E) * V * Tu - (params_vd['deltaT1'] + params_vd['kT'] + params_vd['deltaPICT']) * T1
    dM1 = betaM(E) * V * Mu - (params_vd['deltaM1'] + params_vd['kM'] + params_vd['deltaPICM']) * M1
    dT2 = params_vd['kT'] * T1 - params_vd['deltaT2'] * T2
    dM2 = params_vd['kM'] * M1 - params_vd['deltaM2'] * M2
    dV = params_vd['NM'] * M2 + params_vd['NT'] * T2 - V * (params_vd['CL'] + (betaT(E) + CLT(E)) * Tu + (CLM(E) + betaM(E)) * Mu)
    dVN = (params_vd['NTdiff'] * T2 + params_vd['NMdiff'] * M2) - params_vd['CL'] * VN

    # ISL's PK model for oral administration:
    # D: Dosing
    # C: Plasma (central compartment)
    # P1: Peripheral I
    # P2: Peripheral II
    # E: Effect side (intracellular compartment)
    dD = - ka * D
    dC = - dD / Vc - kC_P1 * C + kP1_C * P1 - kC_0 * C - kC_P2 * C + kP2_C * P2
    dP1 = kC_P1 * C - kP1_C * P1
    dP2 = kC_P2 * C - kP2_C * P2
    dE = kC_E * C - kE_0 * E
    return [dTu, dT1, dT2, dV, dMu, dM1, dM2, dVN, dD, dC, dP1, dP2, dE]

def solve_ode_single(z, t_obs, dose, params):
    # Solve the system of ordinary differential equations (ODEs) for a single dose.
    res = solve_ivp(coupled_ode, (0, t_obs[-1]), z, t_eval=t_obs, args=(dose, params,))
    return res

def main(params, show_plot):
    # Convert units to 1/day.
    hr = 24
    ka, kC_0, kC_P1, kP1_C, kC_P2, kP2_C = [value[0]*hr for value in list(params.values())[1:7]]

    kE_0 = params['kE_0'][0] * hr
    kC_E = params['05mg'][0] * hr # Arbitrary value for initialization.

    z0 = [2e9 / 0.02, 0, 0, 10] + [0] * 9  # Initial state of the coupled ODE-system without ISL (dose=0).
    times = np.linspace(0, 200, num=1000)  # Considered time span.

    # PK parameters for oral dosing.
    parameters = Parameters()
    parameters.add('ka', value=ka, vary=False)
    parameters.add('kC_0', value=kC_0, vary=False)
    parameters.add('kC_P1', value=kC_P1, vary=False)
    parameters.add('kP1_C', value=kP1_C, vary=False)
    parameters.add('kC_P2', value=kC_P2, vary=False)
    parameters.add('kP2_C', value=kP2_C, vary=False)
    parameters.add('Vc', value=3.5, vary=False)
    parameters.add('kC_E_0', value=kC_E, vary=False)
    parameters.add('kE_0', value=kE_0, vary=False)
    # PD parameter initialized with arbitratry value.
    parameters.add('IC50', value=100, vary=False)

    # Get solution of ODE-system by solving once.
    solution = solve_ode_single(z0, times, 0, parameters)

    # Steady state of ODE-system (baseline, (pre)-treatment condition).
    Tucells, T1cells, T2cells, Viruses, Mucells, M1cells, M2cells, VNIs = solution.y[:8]

    if show_plot:
        plt.plot(solution.t, Tucells, 'c-')
        plt.plot(solution.t, T1cells, 'b-')
        plt.plot(solution.t, T2cells, 'r-')
        plt.plot(solution.t, Viruses, 'g-')
        plt.plot(solution.t, VNIs, 'k-')
        plt.xlabel('Time in days', fontsize=16)
        plt.ylabel('Viral dynamics', fontsize=16)
        #plt.show()
        plt.savefig('../results/steady_state.png') # Save result as figure.

    print('Steady state')
    print('Tu cells: ', Tucells[-1])
    print('T1 cells: ', T1cells[-1])
    print('T2 cells: ', T2cells[-1])
    print('Infectious Viruses: ', Viruses[-1])
    print('Mu cells: ', Mucells[-1])
    print('M1 cells: ', M1cells[-1])
    print('M2 cells: ', M2cells[-1])
    print('Non-infectious Viruses: ', VNIs[-1])

    return Tucells[-1], T1cells[-1], T2cells[-1], Viruses[-1], Mucells[-1], M1cells[-1], M2cells[-1], VNIs[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute steady state levels of HIV-1 viral dynamics system without the presence of ISL (baseline for PD optimization).",
                                     formatter_class=argparse.MetavarTypeHelpFormatter)

    parser.add_argument("--pk_params", nargs='?', type=str, default='../_ini_/PK_default.csv',
                            help="Path to file with PK parameters. Default: %(default)s.")
    parser.add_argument("--plot", nargs='?', type=bool, default=False,
                            help="Plot the Output. Default: %(default)s.")


    args = parser.parse_args()

    args = parser.parse_args()
    df_params = pd.read_csv(args.pk_params, header=0)
    df_params.columns = df_params.columns.str.strip()
    params = df_params.to_dict(orient='list')
    main(params, args.plot)
