# To do:
# 1. func plot_results()
#    1.1 synthesize


import numpy as np
import pandas as pd
import pyjags as pj
from scipy import stats
from utils import *
from os import cpu_count
import itertools as it
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from datetime import datetime as dt # in end_epidemic

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import seaborn as sns

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8

colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta',
          'xkcd:dull purple', 'xkcd:teal', 'xkcd:goldenrod', 'xkcd:cadet blue', 'xkcd:scarlet']
cmap_big = cm.get_cmap('Spectral', 512)
cmap = mcolors.ListedColormap(cmap_big(np.linspace(0.7, 0.95, 256)))

bbox_props = dict(boxstyle="round,pad=0.3", fc=colors[0], alpha=.5)


def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def autocorrelation_time_plot(self):
    self.times = self.autocorrelation_time()

    fig, ax = plt.subplots(len(self.names), figsize=(10, 10), constrained_layout=True)
    for i in range(len(self.names)):
        ax[i].plot(self.times[:, i], '-ob')
        ax[i].set_title(f'Autocorrelation time for {self.names[i]}', weight='bold')
        ax[i].grid()

def autocorr_fit_plot(self, xmin=-10, xmax=200, var=None):
    # Acf fit plot of all variables and chains
    if not var:
        for i in range(len(self.varname)):
            for j in range(self.nchains):
                y = self.samples[self.varname[i]][0, :, j]
                t = np.arange(len(y))
                acf_data = acf(y, fft=True, nlags=len(t))
                tau = self.autocorrelation_time()[j][i]
                plt.plot(acf_data, 'b--', label='acf')
                plt.plot(np.exp(-t / tau), 'r', lw=1, label=r'fit $e^{\frac{-t}{\tau}}$' + f' ({tau:.2f})')
                plt.grid()
                plt.legend()
                plt.xlim(xmin, xmax)
                plt.title(f'Autocorrelation fit for {self.names[i]} of Chain {j + 1}', weight='bold')
                plt.show()
    # Acf fit plot of one variable and all chains
    else:
        if self.nchains == 12:
            rows, cols = 4, 3
        elif self.nchains == 8:
            rows, cols = 4, 2

        fig, axs = plt.subplots(rows, cols, figsize=(16, 8), constrained_layout=True)
        axs = trim_axs(axs, self.nchains)
        for ax, j in zip(axs, range(self.nchains)):
            ind = np.where(self.varname == var)[0][0]
            y = self.samples[var][0, :, j]
            t = np.arange(len(y))
            acf_data = acf(y, fft=True, nlags=len(t))
            tau = self.autocorrelation_time()[j][ind]
            ax.plot(acf_data, 'b--', label='acf')
            ax.plot(np.exp(-t / tau), 'r', lw=1, label=r'fit $e^{\frac{-t}{\tau}}$' + f' ({tau:.2f})')
            ax.grid()
            ax.legend()
            ax.set_xlim(xmin, xmax)
            ax.set_title(f'Autocorrelation fit for {self.names[ind]} of Chain {j + 1}', weight='bold')

def peak_posterior(self, nthreads=cpu_count() - 2, binwidth=10, offset=3, second_wave=False):
    data = self.traces.to_numpy()
    t0 = self.data['t0'] - 1
    beta, rmu, p, q = data.T
    # Filter out parameters with no confinement regime
    data = data[rmu * (p + q) > beta * p]
    beta, rmu, p, q = data.T

    # Compute t_peak from the analytical expression for the epidemic's peak
    if len(data) >= 1e6:
        # Numba parallelized function
        func_nb_mt = make_multithread(peak_time_nb, nthreads)
        t_peak = np.int32(func_nb_mt(beta, rmu, p, q))
    else:
        # Numpy standard function
        t_peak = np.int32(func_np(beta, rmu, p, q))

    # Actual peak time
    act_peak = np.where(self.date == self.peak)[0][0] - t0
    # Analytical expression for the peak of the epidemic
    # Probability of actual peak
    p_act_peak = len(t_peak[t_peak == act_peak]) / len(t_peak)
    # Percentage of simulation in which confinement measures succeed at inhibiting the epidemic
    p_conf = len([rmu * (p + q) > beta * p]) / len(data) * 100

    string = 'First day of confinement'
    # Percentage of cases actually peak
    if (self.country == 'Italy') & (not second_wave):
        string = 'DPCM lockdown'
    elif (self.country == 'Italy') & second_wave:
        string = 'DPCM red zones'

    # Posterior distribution on the time to reach the peak pf the epidemic
    sns.histplot(t_peak, binwidth=binwidth, stat='density', color='b')
    ax = plt.gca()
    ax.axvline(act_peak, color='r', linestyle='-', label=f'Actual epidemic peak (p={p_act_peak:.3f})')
    ax.axvline(self.data['tq'], color='gray', linestyle='--')
    ax.axvline(self.data['tmax'], color='gray', linestyle='--')
    # vertical dashed lines for confinement and last data fitted
    y_lim = ax.get_ylim()[1] / 3
    ax.text(self.data['tq'] - offset, y_lim, string, rotation=90)
    ax.text(self.data['tmax'] - offset, y_lim, 'Last data point fitted', rotation=90)
    ax.set_xlabel('Days since first confirmed case')
    ax.set_ylabel('Distribution of peak')
    plt.legend()
    plt.show()
    # Add peak times series to Analysis attribute
    self.t_peak = t_peak

def end_epidemic_plot(self, tf=380, threshold=1000.):
    tq = self.data['tq'] - 1
    tmax = self.data['tmax'] - 1
    Iq = self.data['Iq']
    beta = self.samples['beta']
    rmu = self.samples['rmu']
    p = self.samples['p']
    q = self.samples['q']
    # Filter out parameters with no confinement regime
    mask = rmu * (p + q) > beta * p
    beta = beta[mask]
    rmu = rmu[mask]
    p = p[mask]
    q = q[mask]

    t = np.array([np.arange(tmax, tf)] * len(beta)).T
    I = Iq + ((beta * q) / (p + q) ** 2 * (1 - np.exp(-(p + q) * (t - tq))) +
              (beta - rmu - beta * q / (q + p)) * (t - tq))
    # Compute times until the number of confirmed cases falls below threshold for the first time
    # Numba function

    length = I.shape[1]
    times = np.empty(length, dtype=np.float64)
    # Numba function
    epidemic_end(times, I, threshold, float(tmax))
    # Remove times corresponding at not satisfied condition (first time < threshold)
    times = times[times != tmax]

    # print(np.median(times), np.std(times))
    med, std = rounding(np.median(times), np.std(times))
    # plot distribution of times
    sns.histplot(times, binwidth=10, color='b', stat='density')
    textstr = f'({med:d}' + r' $\pm$ ' + f'{std:d}) days'
    ax = plt.gca()
    ax.axvline(np.median(times), c='r', label=textstr)
    plt.legend()
    plt.grid()
    plt.xlabel('Days since first confirmed case')
    plt.ylabel(f'Distribution of confirmed < {threshold}')

def plot_results(self, CI=95, Y=False, Z=False, observed=False):
    t0 = self.data['t0'] - 1
    tmax = self.data['tmax'] - 1
    tX0 = self.data['tX0'] - 1
    tf = self.data['tf']
    I_exact = self.infected_exact() / np.log(10)
    _, b, c = self.samples['y'].shape

    fig, ax = plt.subplots(figsize=(16, 8))
    plt.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if Y:
        I = self.data["I"] / np.log(10)
        y = self.samples["y"].reshape(tf, b * c)
        y1 = np.percentile(y, (100 + CI) / 2, axis=1) / np.log(10)
        y2 = np.percentile(y, (100 - CI) / 2, axis=1) / np.log(10)

        t = np.arange(t0, tmax) - t0
        plt.scatter(t, I[t0:tmax], fc='w', ec='k', label='Fitted data')
        plt.plot(I_exact, c='orange', label='Fit+prediction')
        t = np.arange(t0, tf) - t0
        plt.fill_between(t, y1[t0:tf], y2[t0:tf], alpha=0.3, color='orange', label=f'{CI}% CI')

        if observed:
            # prevision beyond dataset's time
            if tf - 1 > len(I):
                t = np.arange(tmax, len(I)) - t0
                plt.scatter(t, I[tmax:], fc='g', ec='k', alpha=0.5, label='Observed data')
            else:
                t = np.arange(tmax, tf) - t0
                plt.scatter(t, I[tmax:tf], fc='g', ec='k', alpha=0.5, label='Observed data')
                # actual peak data of a chosen epidemic wave
                tpeak = np.where(self.date == self.peak)[0][0] - t0
                plt.scatter(tpeak - t0, I[tpeak], fc='r', ec='k', label='Actual peak')

        y_true = I[t0:tmax]
        y_pred = I_exact[t0:tmax]

        r2 = (stats.linregress(y_true, y_pred)[2]) ** 2
        plt.annotate(r'$r^{2}$=' + f'{r2:.6f}', xy=(0.85, 0.97), xycoords='axes fraction')
        plt.xlabel(f'Days since first confirmed case ({self.date[t0]})')
        plt.ylabel("Active cases ($\mathbf{log_{10}}$)")
        plt.title(f"Active cases {self.country:s}", fontsize=16, fontweight='bold')
        plt.legend(loc='upper left')

    elif Z:
        X = self.data["X"] / np.log(10)

        if type(self.samples["z"]) == np.ma.core.MaskedArray:
            z = self.samples["z"].data.reshape(tf, b * c)

        elif type(self.samples["z"]) == np.ndarray:
            z = self.samples["z"].reshape(tf, b * c)

        z1 = np.percentile(z, (100 + CI) / 2, axis=1) / np.log(10)
        z2 = np.percentile(z, (100 - CI) / 2, axis=1) / np.log(10)
        z_med = np.median(z, axis=1) / np.log(10)

        t = np.arange(tX0, tf) - tX0
        plt.plot(t, z_med[tX0:], c='orange', label='Fit+prediction')
        t = np.arange(tX0, tmax) - tX0
        plt.scatter(t, X[tX0:tmax], fc='w', ec='k', label='Fitted data')
        t = np.arange(tX0, tf) - tX0
        plt.fill_between(t, z1[tX0:tf], z2[tX0:tf], alpha=0.3, color='orange', label=f'{CI}% CI')

        if observed:
            if tf > len(X):
                t = np.arange(tmax, len(X)) - tX0
                plt.scatter(t, X[tmax:], fc='g', ec='k', alpha=0.5, label='Observed data')
            else:
                t = np.arange(tmax, tf) - tX0
                plt.scatter(t, X[tmax:tf], fc='g', ec='k', alpha=0.5, label='Observed data')

        z_true = X[tX0:tmax]
        z_pred = z_med[tX0:tmax]

        # Goodness of fit, r-square (square of pearson's coefficient)
        r2 = (stats.linregress(z_true, z_pred)[2]) ** 2
        plt.annotate(r'$r^{2}=%.6f$' % r2, xy=(0.85, 0.97), xycoords='axes fraction')
        plt.xlabel(f'Days since first death+recovered case ({self.date[tX0]})')
        plt.ylabel("Variation of death+recovered cases ($\mathbf{log_{10}}$)")
        plt.legend(loc='upper left')
        plt.title("Daily variation of death+recovered cases %s" % self.country, fontsize=16, fontweight='bold')
        plt.legend(loc='upper left')

def trace_plot(self, var=None):
    ind = np.where(self.varname == var)[0][0]
    # Trace plot of one parameter, all chains togheter
    if not var:
        for i in range(self.nchains):
            plt.plot(self.samples[var][0, :, i], lw=1)
            plt.title(f'Chain {i + 1}', weight='bold')
            plt.xlabel('MCMC step')
            plt.tick_params(axis='x', which='major', labelsize=8.5)
            plt.title(f'Trace of {self.names[ind]}', fontsize=16, fontweight='bold')
    # Trace plot of one parameter, so number of plots as #chains
    else:
        # this is for bigger PC
        if self.nchains == 12:
            rows, cols = 4, 3
        # this is for portable pc
        elif self.nchains == 8:
            rows, cols = 4, 2

        fig, axs = plt.subplots(rows, cols, figsize=(15, 8), constrained_layout=True)
        axs = trim_axs(axs, self.nchains)

        for i, ax in zip(np.arange(self.nchains), axs):
            ax.plot(self.samples[var][0, :, i], c='b', lw=1)
            ax.plot([np.median(self.samples[var].ravel())] * (int(self.niter * self.burn_in)), c='r', lw=1)
            ax.grid()
            # removing top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xlabel('MCMC step')
            ax.tick_params(axis='x', which='major', labelsize=8.5)
            ax.set_title(f"Chain {(i + 1):d}", fontweight='bold')
        fig.suptitle(f'Trace of {self.names[ind]}', fontsize=16, fontweight='bold')
        plt.show()

def posteriors(self, var=None):
    ind = np.where(self.varname == var)[0][0]
    exp = 1
    if (var == 'tauX') | (var == 'tauI'):
        exp = - 1 / 2

    if not var:
        sns.histplot((np.power(self.samples[var].ravel(), exp)), kde=True, stat='density', color='b')
        plt.grid()
        ax = plt.gca()
        # removing top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_title(f'Posterior of all chains for {self.names[ind]}', weight='bold')
    else:
        if self.nchains == 12:
            rows, cols = 4, 3
        elif self.nchains == 8:
            rows, cols = 4, 2

        fig, axs = plt.subplots(rows, cols, figsize=(16, 8), constrained_layout=True)
        axs = trim_axs(axs, self.nchains)

        for ax, i in zip(axs, range(self.nchains)):
            sns.histplot((np.power(self.samples[var][0, :, i], exp)), kde=True, stat='density', color='b', ax=ax)
            ax.tick_params(axis='x', which='major', labelsize=7.5)
            ax.grid()
            # removing top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_title(f'Chain {i + 1}', weight='bold')

        fig.suptitle(f'Posterior distributions of {self.names[ind]}', fontsize=16, fontweight='bold')
        plt.show()

def plot_summary(self):
    """
    Plot of trace, posterior, acf, for all parameters and chains
    """
    fig, ax = plt.subplots(6, 3, figsize=(15, 12), constrained_layout=True)
    sample_size = self.samples['beta'].shape[1]

    for i in range(len(self.varname)):
        exp = 1
        if (self.varname[i] == 'tauI') | (self.varname[i] == 'tauX'):
            exp = - 1 / 2
        for j in range(self.nchains):
            variable_chain = np.power(self.samples[self.varname[i]][0, :, j], exp)
            variable_chains = np.power(self.samples[self.varname[i]].ravel(), exp)
            ax[i, 0].plot(variable_chain, label=f'Chain {j + 1}', lw=1)
            sns.distplot(variable_chain, hist=False, kde=True, ax=ax[i, 1])
            ax[i, 2].plot(acf(variable_chain, fft=True, nlags=100))

        ax[i, 0].plot([np.median(variable_chains)] * sample_size, c='r', lw=2, label='Median')
        ax[i, 1].axvline(np.median(variable_chains), color='r', lw=1, label='Actual epidemic peak')
        ax[i, 1].set_ylabel('')
        ax[i, 2].set_xlabel('Lag')
        titles = [f'Trace of {self.names[i]}', f'Posterior distribution of {self.names[i]}',
                  f'Acf of {self.names[i]}']
        for k in range(3):
            ax[i, k].set_xlabel('MCMC step')
            ax[i, k].set_title(titles[k], weight='bold')
            ax[i, k].grid()
