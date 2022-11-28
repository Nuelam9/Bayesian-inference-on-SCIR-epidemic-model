# check difference between plot with median param e median of sol

import glob
import numpy as np
import pandas as pd
from datetime import timedelta
import sys
sys.path.append('../../../modules/')
from utils import get_data
import matplotlib.pyplot as plt
from dataclasses import dataclass


datatimeIndex_type = pd.core.indexes.datetimes.DatetimeIndex

@dataclass
class Plot_solutions:
    samples: 
    def __init__(self, samples):
        self.samples = samples

    def sol_plot(self, file_path: str, plot_path: str, lin: bool,
                 dates: datatimeIndex_type,
                 sol_dates: datatimeIndex_type) -> None:
        
        sol = np.load(file_path) # [:, :10]
        var = file_path.split('_')[-2]
        t0 = samples['t0'] - 1
        tf = samples['tf']

        if not lin:
            sol = np.log(sol) / np.log(10)
            mode = 'log'
            if var in ['I', 'X']:
                obs = samples[var][:tf] / np.log(10)
                plt.plot(dates[:tf], obs, 'ro', label='Observed data')
                plt.legend()      
        else:
            mode = 'lin'
            if var in ['I', 'X']:
                obs = np.exp(samples[var][:tf])
                plt.scatter(dates[:tf], obs, c='r', label='Observed data')
                plt.legend()   

        ci = 95.
        sol1 = np.percentile(sol, (100. + ci) / 2., axis=1)
        sol2 = np.percentile(sol, (100. - ci) / 2., axis=1)
        sol_med = np.median(sol, axis=1)

        plt.fill_between(sol_dates, sol1, sol2, color='orange', alpha=0.3)
        plt.plot(sol_dates, sol_med, color='orange')
        plt.xticks(sol_dates[np.arange(t0, tf * 100 , 20*100)])
        plt.ylabel(f'${var}(t)$', fontsize=14, weight='bold')
        plt.grid()
        plt.savefig(f'{plot_path}plot_{mode}_{var}.png', dpi=400)
        # plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    path = '../../../Results/ita/first_wave/'
    simul_path = f'{path}simul_res/'
    sol_path = f'{path}solve_res/'

    tick_size: int = 10

    file_name = glob.glob1(simul_path, '*.pkl')[0]
    samples = get_data(simul_path + file_name)
    files = glob.glob1(sol_path, '*.npy')
    dates = pd.to_datetime(samples['date'],
                           format='%Y.%m.%d').dt.strftime('%d/%m/%Y')
    step = 0.01
    start = dates.iloc[0]
    end = dates.iloc[samples['tf']]
    dates = pd.date_range(start, end, freq='D')[:-1]
    frac = 86400. * step  # Seconds corresponding to Day / (1 / step)
    sol_dates = pd.date_range(start, end, freq=f'{frac}S')[:-1]

    plot_sol = Plot_solutions(samples)

    for lin in [True, False]:
        for file in files:
            plot_sol.sol_plot(
                sol_path + file, f'{path}plot/ode_sol/', lin, dates, sol_dates)
