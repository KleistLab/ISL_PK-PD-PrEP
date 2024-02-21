#!/usr/bin/env python3
#adapted from PMCID: PMC8741042 DOI: 10.1371/journal.pcbi.1009295

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import argparse
import os

class Sample:
    M_ISL = 293.258 # mol mass of ISL

    def __init__(self, k, ka, k10, kC1, k1C, kC2, k2C, Vc, kIM, k20, k30, kce, ki, ini):
        self.__k = k
        self.__ka = ka
        self.__k10 = k10
        self.__kC1 = kC1
        self.__k1C = k1C
        self.__kC2 = kC2
        self.__k2C = k2C
        self.__Vc = Vc
        self.__k30 = k30
        self.__kce = kce

        self.__kIM = kIM
        self.__k20 = k20

        self.__ki = ki if ki is not None else 0  # Assign a default value of 0 if ki is None
        self.__ini_conc = ini if ini is not None else 0  # Assign a default value of 0 if ini is None

        self.day_array = list()
        self.solutions = list()
        self.dose = 0
        self.adh = 0
        self.countdose = 0
        self.Dmax_array = list()

    def reset(self):
        self.day_array = list()
        self.solutions = list()
        self.dose = 0
        self.adh = 0
        self.countdose = 0
        self.Dmax_array = list()

    def __pkModel(self, t, y):
        R, IM, D, C, P1, P2, IC = y[0], y[1], y[2], y[3], y[4], y[5], y[6]

        #if t >= 30*24: R = 0 # Removal of Implant.
        dydt = [- np.exp(- self.__k * t) * self.__ki * R,
                - (- np.exp(- self.__k * t) * self.__ki * R) - self.__kIM * IM - self.__k20 * IM,
                - self.__ka * D,
                self.__kIM * IM + self.__ka * D / self.__Vc - self.__kC1 * C + self.__k1C * P1 - self.__k10 * C - self.__kC2 * C + self.__k2C * P2,
                self.__kC1 * C - self.__k1C * P1,
                self.__kC2 * C - self.__k2C * P2,
                self.__kce * C - self.__k30 * IC]
        return dydt

    def __solveODE(self, t_start, t_end, n_release, n_intramusc, n_dosing, c_plasma, c_peri1, c_peri2, c_intra):
        t = [t_start, t_end]
        y0 = [n_release, n_intramusc, n_dosing, c_plasma, c_peri1, c_peri2, c_intra]
        sol = solve_ivp(self.__pkModel, t, y0, dense_output=True)
        return sol

    def getConcentration(self, m_ISL, tp, adherence, count_doses, admin, adh_pattern=None, day_upperbound=100):
        # day_max: maximum days for simulation, must be larger than count_doses
        day_max = count_doses + day_upperbound

        if tp < 0: return 0
        #recalculate the concentrations if one of the inputs is changed
        if not self.dose or self.dose != m_ISL or self.adh != adherence or self.countdose != count_doses:
            self.reset()
            self.dose = m_ISL
            self.adh = adherence
            self.countdose = count_doses
            if adh_pattern is not None:
                self.day_array = adh_pattern
            else:
                self.day_array = np.random.choice(2, size=count_doses, p=[1 - adherence, adherence])
                self.day_array[0] = 1
            self.day_array = list(self.day_array) + [0] * (day_max - count_doses)

            # Compute the ODE solution for the first day (implant or oral dosing).
            if admin == 'implant':
                n_release, n_intramusc, n_dosing, c_plasma, c_peri1, c_peri2, c_intra = m_ISL/self.M_ISL * (1 - self.__ini_conc), m_ISL/self.M_ISL * self.__ini_conc, 0, 0, 0, 0, 0
            else:
                n_release, n_intramusc, n_dosing, c_plasma, c_peri1, c_peri2, c_intra = 0, 0, m_ISL / self.M_ISL, 0, 0, 0, 0

            solution_0 = self.__solveODE(0, 24, n_release, n_intramusc, n_dosing, c_plasma, c_peri1, c_peri2, c_intra).sol
            self.solutions.append(solution_0)
            day = 1
            while day < day_max:
                if self.day_array[day]:
                    n_release, n_intramusc, n_dosing, c_plasma, c_peri1, c_peri2, c_intra = self.solutions[-1](day * 24)[:7]
                    n_dosing += self.dose / self.M_ISL

                    solution_new = self.__solveODE(day * 24, (day + 1) * 24, n_release, n_intramusc, n_dosing, c_plasma, c_peri1, c_peri2, c_intra).sol
                    self.solutions.append(solution_new)
                    day += 1
                else:
                    beginday = day
                    nodrugdays = 0
                    while day < day_max and not self.day_array[day]:
                        nodrugdays += 1
                        day += 1
                    n_release, n_intramusc, n_dosing, c_plasma, c_peri1, c_peri2, c_intra = self.solutions[-1](beginday * 24)[:7]
                    solution_new = self.__solveODE(beginday * 24, day * 24, n_release, n_intramusc, n_dosing, c_plasma, c_peri1, c_peri2, c_intra).sol
                    self.solutions += [solution_new] * nodrugdays

        daycount = int(tp // 24)

        if daycount >= day_max:
            return 0
        else:
            return self.solutions[daycount](tp)[6] # Solution of intracellular compartment.

def getPropensities(Y, D):
    IC50 = 429.707
    m = 1
    CL = 2.3
    beta = 8e-12
    lambda_T = 2e9
    delta_T = 0.02
    delta_T1 = 0.5
    delta_T2 = 1.4
    delta_PIC = 0.35
    k = 0.35
    N_T = 670

    eta = D ** m / (IC50 ** m + D ** m) # Direct effect
    CL_T = (1 + eta) * beta     # Clearance rate of unsuccessful infected virus.
    T_u = lambda_T / delta_T    # Steady state level of uninfected T-cells.
    a = [0] * 7                 # List of propensities.
    a[1] = (CL + CL_T * T_u) * Y[0] / 24                # V -> *
    a[2] = (delta_PIC + delta_T1) * Y[1] / 24           # T1 -> *
    a[3] = delta_T2 * Y[2] / 24                         # T2 -> *
    a[4] = (1 - eta) * beta * T_u * Y[0] / 24           # V -> T1
    a[5] = k * Y[1] / 24                                # T1 -> T2
    a[6] = N_T * Y[2] / 24                              # T2 -> T2 + V
    return a

def deterministics_output(p1, p2, p3, file, tspanres, timesteps, ts, te):
    # Process the output arrays and output a csv file.
    t = np.arange(ts, te, tspanres)
    step = int(tspanres/timesteps[0]*60)
    p0 = getExtinctionProbability([1,0,0], 0)[1][0]
    p01 = getExtinctionProbability([1, 0, 0], 0)[1][1]
    p02 = getExtinctionProbability([1, 0, 0], 0)[1][2]

    efficacy_V =  [1 - (1 - p1[i]) / (1 - p0) for i in range(0, len(p1), step)]
    efficacy_T1 = [1 - (1 - p2[j]) / (1 - p01) for j in range(0, len(p2), step)]
    efficacy_T2 = [1 - (1 - p3[k]) / (1 - p02) for k in range(0, len(p3), step)]
    p1 = p1[::step]
    p2 = p2[::step]
    p3 = p3[::step]
    if len(efficacy_V) > len(t):
        efficacy_V = efficacy_V[:len(t)]
        efficacy_T1 = efficacy_T1[:len(t)]
        efficacy_T2 = efficacy_T2[:len(t)]
        p1 = p1[:len(t)]
        p2 = p2[:len(t)]
        p3 = p3[:len(t)]
    df = pd.DataFrame([t, p1, p2, p3, efficacy_V, efficacy_T1, efficacy_T2])
    df.index = ['time point', 'PE_V', 'PE_T1', 'PE_T2', 'efficacy_V', 'efficacy_T1', 'efficacy_T2']
    df.to_csv(file)
    return df

def deterministics_plot(df, file, dose):
    t = df.loc['time point', :] / 24
    days = int(t[len(t)-1] - t[0]) + 1
    if days <= 1:
        ticks = 2
        t *= 24
    elif days < 10:
        ticks = 1
    else:
        ticks = days // 10
    fig, axs = plt.subplots(1, 2, figsize=[15,6])
    df_ = df.loc['efficacy_V', :]*100
    axs[0].plot(t, df_, lw=2, c='crimson')
    axs[0].plot(t, [85]*len(t), 'k--')
    axs[0].plot(t, [90] * len(t), color='darkgray', linestyle=':')
    axs[0].set_ylabel('Prophylactic efficacy [%]')
    axs[0].set_ylim([0, 100])
    axs[0].set_title('Prophylactic efficacy for {} mg ISL'.format(dose))
    axs[0].set_xticks(np.arange(int(t[0]), int(t[len(t)-1])+2, ticks))
    axs[0].legend(('{} mg'.format(dose), '85%', '90%'), fontsize=15) #, '$\phi(\hat T_1)$', '$\phi(\hat T_2)$'
    if days <= 1:
        axs[0].set_xlabel('Timing of viral exposure relative to first dose [hr]')
    else:
        axs[0].set_xlabel('Timing of viral exposure relative to first dose [day]')
    axs[1].plot(t, df.loc['PE_V', :]*100, lw=2, c='crimson')
    axs[1].plot(t, df.loc['PE_T1', :]*100, lw=2, c='forestgreen')
    axs[1].plot(t, df.loc['PE_T2', :]*100, lw=2, c='dodgerblue')
    axs[1].set_xticks(np.arange(int(t[0]), int(t[len(t)-1])+2, ticks))
    axs[1].set_ylabel('Extinction probability [%]')
    axs[1].set_ylim([0, 100])
    axs[1].set_title('Extinction probability for {} mg ISL'.format(dose))
    axs[1].legend(('$P_{E\_V}$', '$P_{E\_T_1}$', '$P_{E\_T_2}$'), fontsize=15)
    if days <= 1:
        axs[1].set_xlabel('Timing of viral exposure relative to first dose [hr]')
    else:
        axs[1].set_xlabel('Timing of viral exposure relative to first dose [day]')
    fig.savefig('{}'.format(file))

def getExtinctionProbability(Y, D):

    def solution_quadratic_equation(a, b, c):
        # solve ax^2 + bx + c = 0
        tmp = b * b - 4 * a * c
        assert tmp >= 0
        tmp_sqr = tmp ** 0.5
        return (-b - tmp_sqr) / (2 * a)

    a = getPropensities([1, 1, 1], D)
    alpha1 = a[1] / (a[1] + a[4])
    alpha3 = a[4] / (a[1] + a[4])
    beta1 = a[2] / (a[2] + a[5])
    beta3 = a[5] / (a[2] + a[5])
    gamma = a[6] / (a[3] + a[6])
    e_a = alpha3 * beta3 * gamma
    e_b = gamma - 1 - e_a
    e_c = 1 - gamma
    PE_t2 = solution_quadratic_equation(e_a, e_b, e_c)
    PE_t1 = beta1 + beta3 * PE_t2
    PE_v = alpha1 + alpha3 * PE_t1

    # Extinction probability.
    PE = PE_v ** Y[0] * PE_t1 ** Y[1] * PE_t2 ** Y[2]

    return PE, [PE_v, PE_t1, PE_t2]

def get_a(sample, t, m_dose, count_doses, adherence, a_i, admin):
    D = sample.getConcentration(m_dose, t, adherence, count_doses, admin)
    return getPropensities([1,1,1], D)[a_i]


def peModel(t, y, a, sample, m_dose, count_doses, adherence, admin):
    a1 = get_a(sample, t, m_dose, count_doses, adherence, 1, admin)
    a2 = a[2]
    a3 = a[3]
    a4 = get_a(sample, t, m_dose, count_doses, adherence, 4, admin)
    a5 = a[5]
    a6 = a[6]
    P1, P2, P3 = y[0], y[1], y[2]
    dydt = [-a1*(1 - P1) - a4*(P2 - P1),
            - a2*(1 - P2) - a5*(P3 - P2),
            - a3*(1 - P3) - a6*(P3*P1 - P3)]
    return dydt


def solvePE_pgs(sample, t_start, t_end, m_dose, count_doses, adherence, admin, p10=0.94226, p20=0.71852, p30=0.03493, t_extend=100):
    # Reset the sample, in case the concentration profile already exists.
    sample.reset()
    m_dose *= 1e6     # Convert the mass into ng.
    t_end += t_extend
    a = getPropensities([1,1,1], 0)
    t = [t_end, t_start]
    y0 = [p10, p20, p30]
    sol = solve_ivp(peModel, t, y0, method='LSODA', args=[a, sample, m_dose, count_doses, adherence, admin], dense_output=True)
    return sol


def run_pgs(dose, mkce, ki, ini, cdose, adh, inputfile, ts, te, tspanres, admin):
    df_sample = pd.read_csv(inputfile)
    if df_sample.shape[0] != 1:
        sys.stderr.write('This method takes only one sample, more than one '
                         'are given \n')
        exit(1)
    sample = Sample(*df_sample.loc[0, :].values[:11], mkce, ki, ini)
    solution = solvePE_pgs(sample, ts, te+100, dose, cdose, adh, admin).sol
    p1 = [solution(i/60)[0] for i in np.arange(ts*60, te*60, int(tspanres*60))]
    p2 = [solution(i/60)[1] for i in np.arange(ts*60, te*60, int(tspanres*60))]
    p3 = [solution(i/60)[2] for i in np.arange(ts*60, te*60, int(tspanres*60))]
    return p1, p2, p3


def main(argv):
    parser = argparse.ArgumentParser(description='Run PGS for ISL for oral and implant dosing.', formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default='../_ini_/PK_default.csv', help='Input file.')
    parser.add_argument('-o', '--output', type=str, default='test_run', help='Output file name.')
    parser.add_argument('--admin', type=str, default='oral', help='Administration type (oral or implant).')
    parser.add_argument('--dose', type=float, default=56, help='ISL in mg.')
    parser.add_argument('--cdose', type=int, default=3, help='Count of doses in a regimen.')
    parser.add_argument('--adh', type=float, default=1, help='Adherence of drug taken.')
    parser.add_argument('--ts', type=float, default=0, help='Start time point to run the computation (hr).')
    parser.add_argument('--te', type=float, default=240, help='End time point of the computation (hr).')
    parser.add_argument('--tspanres', type=float, default=1, help='Timespan in the output data (hr).')
    parser.add_argument('--ifpdf', type=bool, default=False, help='Output the graphic of results in pdf')

    args = parser.parse_args(argv)

    dose = args.dose
    admin = args.admin

    # Get drug load specific implant values.
    implant_map = {
        54: (0.00030696, 0.0412),
        62: (0.00046038, 0.0583),
        48: (0.004606 / 24, 0.0412),
        52: (0.0066447 / 24, 0.0412),
        56: (0.008288 / 24, 0.0583)
    }

    # Get dose-specific kC_E value.
    dose_map = {
        0.5: 36.88,
        1: 27.80,
        2: 16.57,
        5: 12.58,
        10: 17.52,
        15: 13.96,
        30: 17.57,
        100: 8.88,
        200: 10.63,
        400: 9.16,
        48: 25.14,
        52: 31.06,
        54: 19.62,
        56: 41.43,
        62: 35.24
    }

    if admin == 'implant':
        if dose not in implant_map:
            print('Unknown drug load for implant administration.')
            sys.exit(1)

        ki, ini = implant_map[dose]
        c_dose = 1

    else:
        ki = ini = None  # Not needed for oral administration.
        c_dose = args.cdose

    if dose not in dose_map:
        print('Unknown dose.')
        sys.exit(1)

    kce = dose_map[dose]

    outputfile = os.path.join('../results/pgs/', args.output + '.csv')

    p1, p2, p3 = run_pgs(dose, kce, ki, ini, c_dose, args.adh, args.input, args.ts, args.te,
                         args.tspanres, admin)

    df = deterministics_output(p1, p2, p3, outputfile, args.tspanres, [args.tspanres * 60], args.ts, args.te)

    if args.ifpdf:
        filename = outputfile[:-4] + '.svg'
        deterministics_plot(df, filename, dose)

    if admin == 'implant':
        print(rf'Implant administration with a dose of {dose} mg (kImpl_SC={ki:.5f}, $\pi$ ={ini:.4f}, kC_E={kce:.2f}) is completed.')
    else:
        print(f'Oral administration with a dose of {dose} mg (kC_E={kce:.2f}) is completed.')

if __name__ == "__main__":
    main(sys.argv[1:])
