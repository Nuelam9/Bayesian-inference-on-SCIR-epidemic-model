# To do:
# 1. func plot_results()
#    1.1 synthesize
# 2. change in plot or where it comes:
#    2.1 the evaluation of same things (waste of time)
# 3. try to use numba also for I_exact (new version in prova_numba)
# 2. Add type annotation


import numpy as np
import pandas as pd
import pyjags as pj
from scipy import stats
from utils import *
from os import cpu_count
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
from datetime import datetime as dt  # in end_epidemic
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

colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red',
          'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple',
          'xkcd:teal', 'xkcd:goldenrod', 'xkcd:cadet blue', 'xkcd:scarlet']
cmap_big = cm.get_cmap('Spectral', 512)
cmap = mcolors.ListedColormap(cmap_big(np.linspace(0.7, 0.95, 256)))

bbox_props = dict(boxstyle="round,pad=0.3", fc=colors[0], alpha=.5)


def trim_axs(axs: np.ndarray) -> np.array:
    axs = axs.flat
    n = len(axs)
    for ax in axs[n:]:
        ax.remove()
    return axs[:n]


def autocorrelation_time(samples: dict, nlags: int = 1000) -> np.ndarray:
    nchains = samples['nchains']
    nvars = len(samples['varname'])
    times = np.zeros((nchains, nvars))
    x = np.arange(nlags + 1)
    for i in range(nvars):
        for j in range(nchains):
            y = acf(samples[samples['varname'][i]][0, :, j], 
                    fft=True, nlags=nlags)
            popt, _ = curve_fit(fit_time, x, y)
            times[j, i] = popt
    return times


def autocorrelation_time_plot(samples: dict) -> None:
    times = autocorrelation_time(samples)
    n = len(samples['names'])
    _, ax = plt.subplots(n, figsize=(10, 10), constrained_layout=True)
    for i in range(n):
        ax[i].plot(times[:, i], '-ob')
        ax[i].grid()
        ax[i].set_title(f"Autocorrelation time for {samples['names'][i]}",
                        weight='bold')


def autocorr_fit_plot(samples: dict, var: str, total: bool = True, 
                      xmin: int = -10, xmax: int = 200) -> None:
    # Acf fit plot of one variable and all chains
    if samples['nchains'] == 12:
        rows, cols = 4, 3
    elif samples['nchains'] == 10:
        rows, cols = 5, 2
    elif samples['nchains'] == 8:
        rows, cols = 4, 2

    _, axs = plt.subplots(rows, cols, figsize=(16, 8), constrained_layout=True)
    axs = trim_axs(axs)
    for ax, j in zip(axs, range(samples['nchains'])):
        ind = np.where(samples['varname'] == var)[0][0]
        y = samples[var][0, :, j]
        t = np.arange(len(y))
        acf_data = acf(y, fft=True, nlags=len(t))
        tau = autocorrelation_time(samples)[j][ind]
        ax.plot(acf_data, 'b--', label='acf')
        ax.plot(np.exp(-t / tau), 'r', lw=1,
                label=r'fit $e^{\frac{-t}{\tau}}$' + f' ({tau:.2f})')
        ax.set_xlim(xmin, xmax)
        ax.legend()
        ax.grid()
        ax.set_title(f"Autocorrelation fit for {samples['names'][ind]}"
                     + " of Chain {j + 1}", weight='bold')


def peak_posterior_plot(samples: dict, nthreads: int = cpu_count() - 2, 
                   binwidth: int = 10, offset: int = 3, 
                   second_wave: bool = False, label_size: int = 12,
                   tick_size: int = 10) -> None:
    """
    
    """

    beta, rmu, p, q = samples['beta'], samples['rmu'], \
                      samples['p'], samples['q']
    # Filter out parameters with no confinement regime
    mask = rmu * (p + q) > beta * p
    t0 = samples['t0'] - 1
    dates = pd.to_datetime(samples['date'],
                           format='%Y.%m.%d').dt.strftime('%d/%m/%Y')
    
    # Compute t_peak from the analytical expression for the epidemic's peak
    if len(mask) >= 1e6:
        # Numba parallelized function
        func_nb_mt = make_multithread(peak_posterior_nb, nthreads)
        t_peak = np.int32(func_nb_mt(beta[mask], rmu[mask], p[mask], q[mask]))
    else:
        # Numpy standard function
        t_peak = np.int32(peak_posterior_np(beta[mask], rmu[mask], 
                                            p[mask], q[mask]))

    # Actual peak time
    act_peak = np.where(samples['date'] == samples['peak'])[0][0] - t0
    # Analytical expression for the peak of the epidemic
    # Probability of actual peak
    p_act_peak = len(t_peak[t_peak == act_peak]) / len(t_peak)
    # Percentage of simulation in which confinement measures succeed at 
    # inhibiting the epidemic
    p_conf = len(mask) / beta.size * 100

    string = 'First day of confinement'
    # Percentage of cases actually peak
    if (samples['country'] == 'Italy') & (not second_wave):
        string = 'DPCM lockdown'
    elif (samples['country'] == 'Italy') & second_wave:
        string = 'DPCM red zones'

    # Posterior distribution on the time to reach the peak pf the epidemic
    sns.histplot(t_peak, binwidth=binwidth, stat='density', color='b')
    ax = plt.gca()
    ax.axvline(act_peak, color='r', linestyle='-', 
               label=f'Actual epidemic peak (p={p_act_peak:.3f})')
    ax.axvline(samples['tq'], color='gray', linestyle='--')
    ax.axvline(samples['tmax'], color='gray', linestyle='--')
    # vertical dashed lines for confinement and last data fitted
    _, y_lim = ax.get_ylim() / 3
    ax.text(samples['tq'] - offset, y_lim, string, rotation=90)
    ax.text(samples['tmax'] - offset, y_lim,
            'Last data point fitted', rotation=90)
    ax.set_xlabel(f'Days since first confirmed case ({dates[t0]})',
                  fontsize=label_size)
    ax.set_ylabel('Probability of peak', fontsize=label_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    plt.legend()
    plt.grid()


def end_epidemic_plot(samples: dict, tlast: str, threshold: int = None, 
                      label_size: int = 12, tick_size: int = 10) -> None:
    """
 
    """
    if threshold is None:
        if samples['country'] == 'Italy':
            threshold = 1200.
        elif samples['country'] == 'Spain':
            threshold = 1000.
        elif samples['country'] == 'France':
            threshold = 1300.

    t0 = samples['t0'] - 1
    tq = samples['tq'] - 1
    tmax = samples['tmax'] - 1
    fmt = '%Y.%m.%d'
    tlast = (dt.strptime(tlast, fmt) - 
             dt.strptime(samples['date'][0], fmt)).days
    dates = pd.to_datetime(samples['date'],
                           format='%Y.%m.%d').dt.strftime('%d/%m/%Y')
    Iq = samples['Iq']
    beta, rmu, p, q = samples['beta'], samples['rmu'], \
                      samples['p'], samples['q']
    # Filter out parameters with no confinement regime
    mask = rmu * (p + q) > beta * p
    beta, rmu, p, q = beta[mask], rmu[mask], p[mask], q[mask]

    t = np.array([np.arange(tmax, tlast)] * len(beta)).T
    I = Iq + ((beta * q) / (p + q) ** 2 * (1 - np.exp(-(p + q) * (t - tq))) +
              (beta - rmu - beta * q / (q + p)) * (t - tq))
    # Compute times until the number of confirmed cases falls below threshold 
    # for the first time 

    length = I.shape[1]
    times = np.empty(length, dtype=np.float64)
    # Numba function
    epidemic_end(times, I, threshold, float(tmax))
    # Remove times corresponding to not satisfied condition 
    # (first time < threshold)
    times = times[times != tmax]

    # print(np.median(times), np.std(times))
    med, std = rounding(np.median(times), np.std(times))
    # plot distribution of times
    sns.histplot(times, binwidth=10, color='b', stat='density')
    textstr = f'({med:d}' + r' $\pm$ ' + f'{std:d}) days'
    ax = plt.gca()
    ax.axvline(np.median(times), c='r', label=textstr)
    plt.xlabel(f'Days since first confirmed case ({dates[t0]})', 
               fontsize=label_size)
    plt.ylabel(f'Probability of confirmed < {int(threshold)}', 
               fontsize=label_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    plt.legend()
    plt.grid()


def plot_results(samples: dict, ci: int = 95, Y: bool = False,
                 Z: bool = False, observed: bool = False, 
                 label_size: int = 14, tick_size: int = 10,
                 title_size: int = 16) -> None:
    t0 = samples['t0'] - 1
    tmax = samples['tmax'] - 1
    tX0 = samples['tX0'] - 1
    tf = samples['tf']
    I_exact = infected_exact(samples) / np.log(10)
    _, b, c = samples['y'].shape
    dates = pd.to_datetime(samples['date'],
                           format='%Y.%m.%d').strftime('%d/%m/%Y')

    fig, ax = plt.subplots(figsize=(16, 8))
    plt.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if Y:
        I = samples["I"] / np.log(10)
        y = samples["y"].reshape(tf, b * c)
        y1 = np.percentile(y, (100. + ci) / 2., axis=1) / np.log(10)
        y2 = np.percentile(y, (100. - ci) / 2., axis=1) / np.log(10)

        t = np.arange(t0, tmax) - t0
        plt.scatter(dates[t], I[t0:tmax], fc='w', ec='k', label='Fitted data')
        plt.plot(I_exact, c='orange', label='Fit+prediction')
        t = np.arange(t0, tf) - t0
        plt.fill_between(dates[t], y1[t0:tf], y2[t0:tf], alpha=0.3, 
                         color='orange', label=f'{ci}% CI')

        if observed:
            if tf - 1 > len(I):
                t = np.arange(tmax, len(I)) - t0
                plt.scatter(dates[t], I[tmax:], fc='g', ec='k',
                            alpha=0.5, label='Observed data')
            else:
                t = np.arange(tmax, tf) - t0
                plt.scatter(dates[t], I[tmax:tf], fc='g', ec='k', 
                            alpha=0.5, label='Observed data')
                # actual peak data of a chosen epidemic wave
                tpeak = np.where(samples['date'] == samples['peak'])[0][0] - t0
                plt.scatter(dates[tpeak - t0], I[tpeak], 
                            fc='r', ec='k', label='Actual peak')

        r2 = _extracted_from_plot_results_47(I, t0, tmax, I_exact)
        plt.annotate(r'$r^{2}$=' + f'{r2:.6f}', xy=(0.85, 0.97),
                     xycoords='axes fraction')
        plt.xlabel(f"Days since first confirmed case ({dates[t0]})",
                   fontsize=label_size)
        plt.ylabel("Active cases ($\mathbf{log_{10}}$)", 
                   fontsize=label_size)
        plt.xticks(dates[range(t0, tf, 20)])
        plt.tick_params(axis='x', labelsize=tick_size)
        plt.tick_params(axis='y', labelsize=tick_size)
        plt.title(f"Active cases {samples['country']}", 
                  fontsize=title_size, fontweight='bold')
        plt.legend(loc='upper left')

    elif Z:
        X = samples["X"] / np.log(10)

        if type(samples["z"]) == np.ma.core.MaskedArray:
            z = samples["z"].data.reshape(tf, b * c)

        elif type(samples["z"]) == np.ndarray:
            z = samples["z"].reshape(tf, b * c)

        z1 = np.percentile(z, (100 + ci) / 2, axis=1) / np.log(10)
        z2 = np.percentile(z, (100 - ci) / 2, axis=1) / np.log(10)
        z_med = np.median(z, axis=1) / np.log(10)

        t = np.arange(tX0, tf) - tX0
        plt.plot(dates[t], z_med[tX0:], c='orange', label='Fit+prediction')
        t = np.arange(tX0, tmax) - tX0
        plt.scatter(dates[t], X[tX0:tmax], fc='w', ec='k', label='Fitted data')
        t = np.arange(tX0, tf) - tX0
        plt.fill_between(dates[t], z1[tX0:tf], z2[tX0:tf], alpha=0.3, 
                         color='orange', label=f'{ci}% CI')

        if observed:
            if tf > len(X):
                t = np.arange(tmax, len(X)) - tX0
                plt.scatter(dates[t], X[tmax:], fc='g', ec='k', 
                            alpha=0.5, label='Observed data')
            else:
                t = np.arange(tmax, tf) - tX0
                plt.scatter(dates[t], X[tmax:tf], fc='g', ec='k', 
                            alpha=0.5, label='Observed data')

        r2 = _extracted_from_plot_results_47(X, tX0, tmax, z_med)
        plt.annotate(r'$r^{2}=%.6f$' % r2, xy=(0.85, 0.97), 
                     xycoords='axes fraction')
        plt.xlabel(f"Days since first recovered+dead case ({dates[tX0]})", 
                   fontsize=label_size)
        plt.ylabel("Number of new recovered+dead cases ($\mathbf{log_{10}}$)", 
                   fontsize=label_size)
        plt.xticks(dates[np.arange(tX0, tf, 20)])
        plt.tick_params(axis='x', labelsize=tick_size)
        plt.tick_params(axis='y', labelsize=tick_size)
        plt.legend(loc='upper left')
        plt.title(f"Daily number of new dead and recovered cases \
                  {samples['country']}", fontsize=title_size, 
                  fontweight='bold')


# TODO Rename this here and in `plot_results`
def _extracted_from_plot_results_47(arg0, arg1, tmax, arg3):
    y_true = arg0[arg1:tmax]
    y_pred = arg3[arg1:tmax]

    return (stats.linregress(y_true, y_pred)[2]) ** 2


def trace_plot(samples: dict, var: str, total: bool = True,
               title_size: int = 16) -> None:
    ind = np.where(samples['varname'] == var)[0][0]
    samples_size = int(samples['niter'] * samples['burn_in'])
    if not total:
        for i in range(samples['nchains']):
            plt.plot(samples[var][0, :, i], lw=1, label=f"Chain {i + 1}")
        plt.plot([np.median(samples[var])] * samples_size, 
                  c='r', lw=2, label='Median')
        plt.grid()
        plt.legend()
        ax = plt.gca()
        # removing top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.xlabel('MCMC step')
        plt.title(f"Trace of {samples['names'][ind]}", weight='bold')
        # Trace plot of one parameter
    else:
        # this is for bigger PC
        if samples['nchains'] == 12:
            rows, cols = 4, 3
        elif samples['nchains'] == 10:
            rows, cols = 5, 2
        # this is for portable pc
        elif samples['nchains'] == 8:
            rows, cols = 4, 2

        fig, axs = plt.subplots(rows, cols, figsize=(15, 8), 
                                constrained_layout=True)
        axs = trim_axs(axs)

        for i, ax in zip(np.arange(samples['nchains']), axs):
            ax.plot(samples[var][0, :, i], c='b', lw=1)
            ax.plot([np.median(samples[var])] * samples_size, c='r', lw=1)
            ax.grid()
            # removing top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xlabel('MCMC step')
            ax.tick_params(axis='x', which='major', labelsize=8.5)
            ax.set_title(f"Chain {(i + 1):d}", fontweight='bold')
        fig.suptitle(f"Trace of {samples['names'][ind]}", 
                     fontsize=title_size, fontweight='bold')
        plt.show()

    
def posteriors(samples: dict, var: str, total: bool = True) -> None:
    ind = np.where(samples['varname'] == var)[0][0]
    exp = 1
    if (var == 'tauX') | (var == 'tauI'):
        exp = - 1 / 2

    if not total:
        for i in range(samples['nchains']): 
            sns.distplot((np.power(samples[var][0, :, i], exp)), 
                         hist=False, kde=True, label=f"Chain {i + 1}")
        plt.grid()
        plt.legend()
        ax = plt.gca()
        # removing top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_title(f"Posterior of all chains for {samples['names'][ind]}",
                     weight='bold')
    else:
        if samples['nchains'] == 12:
            rows, cols = 4, 3
        elif samples['nchains'] == 10:
            rows, cols = 5, 2
        elif samples['nchains'] == 8:
            rows, cols = 4, 2

        fig, axs = plt.subplots(rows, cols, figsize=(16, 8),
                                constrained_layout=True)
        axs = trim_axs(axs)

        for ax, i in zip(axs, range(samples['nchains'])):
            sns.histplot((np.power(samples[var][0, :, i], exp)), 
                         kde=True, stat='density', color='b', ax=ax)
            ax.tick_params(axis='x', which='major', labelsize=7.5)
            ax.grid()
            # removing top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_title(f'Chain {i + 1}', weight='bold')

        fig.suptitle(f"Posterior distributions of {samples['names'][ind]}", 
                     fontsize=16, fontweight='bold')
        plt.show()


def plot_summary(samples: dict, without_acf: bool = True, label_size: int = 14, 
                 tick_size: int = 10, title_size: int = 16) -> None:
    # Plot of trace, posterior for all parameters and chains
    if without_acf:
        _, ax = plt.subplots(6, 2, figsize=(12, 12), constrained_layout=True)
        samples_size = int(samples['niter'] * samples['burn_in'])
        for i in range(len(samples['varname'])):
            exp = 1
            if samples['varname'][i] in ['tauI', 'tauX']:
                exp = - 1 / 2
            for j in range(samples['nchains']):
                var_chain = np.power(samples[samples['varname'][i]][0, :, j],
                                     exp)
                var_chains = np.power(samples[samples['varname'][i]].ravel(),
                                      exp)
                ax[i, 0].plot(var_chain, label=f'Chain {j + 1}', lw=1)
                sns.distplot(var_chain, hist=False, kde=True, ax=ax[i, 1])

            ax[i, 0].plot([np.median(var_chains)] * samples_size, 
                          c='r', lw=2, label='Median')
            ax[i, 1].axvline(np.median(var_chains), color='r', 
                             lw=1, label='Actual epidemic peak')
            ax[i, 1].set_ylabel('')
            titles = [f"Trace of {samples['names'][i]}",
                      f"Posterior distribution of {samples['names'][i]}"]
            for k in range(2):
                ax[i, k].set_title(titles[k], weight='bold', fontsize=title_size)
                ax[i, k].tick_params(axis='x', labelsize=tick_size)
                ax[i, k].tick_params(axis='y', labelsize=tick_size)
                ax[i, k].grid()
        ax[-1, 0].set_xlabel('MCMC step', fontsize=label_size)
    # Plot of trace, posterior, acf, for all parameters and chains
    else:
        _, ax = plt.subplots(6, 3, figsize=(15, 12), constrained_layout=True)
        samples_size = int(samples['niter'] * samples['burn_in'])
        for i in range(len(samples['varname'])):
            exp = 1
            if samples['varname'][i] in ['tauI', 'tauX']:
                exp = - 1 / 2
            for j in range(samples['nchains']):
                var_chain = np.power(samples[samples['varname'][i]][0, :, j], 
                                     exp)
                var_chains = np.power(samples[samples['varname'][i]].ravel(), 
                                      exp)
                ax[i, 0].plot(var_chain, label=f'Chain {j + 1}', lw=1)
                sns.distplot(var_chain, hist=False, kde=True, ax=ax[i, 1])
                ax[i, 2].plot(acf(var_chain, fft=True, nlags=100))

            ax[i, 0].plot([np.median(var_chains)] * samples_size, 
                          c='r', lw=2, label='Median')
            ax[i, 1].axvline(np.median(var_chains), color='r', 
                             lw=1, label='Actual epidemic peak')
            ax[i, 1].set_ylabel('')
            ax[i, 2].set_xlabel('Lag')
            titles = [f"Trace of {samples['names'][i]}", 
                      f"Posterior distribution of {samples['names'][i]}",
                      f"Acf of {samples['names'][i]}"
                      ]
            for k in range(3):
                ax[i, k].set_xlabel('MCMC step')
                ax[i, k].set_title(titles[k], weight='bold')
                ax[i, k].grid()        
