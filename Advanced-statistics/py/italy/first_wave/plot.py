# check difference between plot with median param e median of sol

import glob
import numpy as np
import pandas as pd
import sys

sys.path.append('../../../modules/')
from utils import get_data
import matplotlib.pyplot as plt


class Plot_solutions:

    def __init__(self, tf):
        self.tf = tf

    def sol_plot(self, file_path: str, plot_path: str, 
                 dates: pd.core.indexes.datetimes.DatetimeIndex,
                 t0: int, tf: int) -> None:
        sol = np.load(file_path)[:, :10]
        var = file_path.split('_')[-2]
        ci = 95.
        sol1 = np.percentile(sol, (100. + ci) / 2., axis=1)
        sol2 = np.percentile(sol, (100. - ci) / 2., axis=1)
        sol_med = np.median(sol, axis=1)
        t = np.arange(0, self.tf, 0.01)
        plt.fill_between(t, sol1, sol2, color='orange', alpha=0.3)
        plt.plot(t, sol_med, color='orange')
        #plt.xticks(dates[np.arange(t0, tf, 20)])
        plt.ylabel(f'${var}(t)$', fontsize=14, weight='bold')
        plt.savefig(plot_path + f'plot_{var}.png', dpi=400)#transparent=True, dpi=400)
        plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    path = '../../../Results/ita/first_wave/'
    simul_path = path + 'simul_res/'
    sol_path = path + 'solve_res/'

    tick_size: int = 10

    file_name = glob.glob1(simul_path, '*.pkl')[0]
    samples = get_data(simul_path + file_name)
    files = glob.glob1(sol_path, '*.npy')
    dates = pd.to_datetime(samples['date'],
                           format='%Y.%m.%d').dt.strftime('%d/%m/%Y')

    plot_sol = Plot_solutions(tf = samples['tf'])

    for file in files:
        plot_sol.sol_plot(sol_path + file, path + 'plot/ode_sol/',
                          dates, t0=samples['t0'] - 1, tf = samples['tf'] - 1)
