# To do:
# 1. func plot_results()
#    1.1 synthesize
#    1.2 in other module
# 2. change in plot or where it comes
#    2.1 the evaluation of same things (waste of time)
# 3. in peak_posterior:
#    3.1 change dataframe in ndarray (use numba)
# 4. all code:
#    4.1 create small modules


import numpy as np
import pandas as pd
import pyjags as pj
import arviz as az
from scipy import stats
from utils import *
from os import cpu_count
import itertools as it
from multiprocessing import Pool, cpu_count
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from datetime import datetime as dt

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

jags_model = '''
model {
    # Regression before any confinement measure (populations are in log scale)
    for(t in (t0 + 1):(tq)) {
        I[t] ~ dnorm(I0 + (beta - rmu) * (t - t0), tauI) # Active cases
        y[t] ~ dnorm(I0 + (beta - rmu) * (t - t0), tauI) # Posterior predictive
    }

    # Regression for active cases post-confinement (populations are in log scale)
    for(t in (tq + 1):tmax) {
        I[t] ~ dnorm(Iq + ((beta * q) / (p + q)^2 * (1 - exp(-(p + q) * (t - tq))) +
                    (beta - rmu - beta * q / (q + p))*(t - tq)), tauI)
    }

    # Posterior predictive for active cases post-confinement (extended until tf)
    for(t in (tq + 1):tf) {
        y[t] ~ dnorm(y[tq] + ((beta * q)/(p + q)^2 * (1 - exp(-(p + q) * (t - tq))) +
                    (beta - rmu - beta * q / (q + p)) * (t - tq)), tauI)
    }

    # Regression for new death+recovered cases
    for(t in (tX0):tmax) {
        X[t] ~ dnorm(log(rmu) + I[t], tauX)  # New Deaths + Recovered
    }

    # Posterior predictive for new deaths+recovered (extended until tf)
    for(t in tX0:tf) {
        z[t] ~ dnorm(log(rmu) + y[t], tauX)
    }

    # Priors for parameters
    p ~ dunif(p0, p1)  # Non-informative prior
    q ~ dunif(q0, q1)  # Non-informative prior
    beta ~ dunif(b0, b1)  # Doubling time is less than 1 per day (informative prior)
    rmu ~ dunif(r0, r1)  # rmu is lower than beta (so R0>1) (informative prior)

    # Priors for precisions (inverse of variance)
    tauI ~ dgamma(tauI0, tauI1)  # Non-informative prior
    tauX ~ dgamma(tauX0, tauX1)  # Non-informative prior
    y[t0] <- I0
}'''


class Analysis:

    def __init__(self, date, confirmed, recovered_death, quarantine, last_data,
                 last_projection, peak, beta, rmu, q, p, tauI, tauX, country):
        self.date = date
        self.confirmed = confirmed
        self.recovered_death = recovered_death
        self.quarantine = quarantine
        self.last_data = last_data
        self.last_projection = last_projection
        self.peak = peak
        self.beta = beta
        self.rmu = rmu
        self.q = q
        self.p = p
        self.tauI = tauI
        self.tauX = tauX
        self.country = country
        self.varname = np.array(['beta', 'rmu', 'p', 'q', 'tauX', 'tauI'])
        self.names = np.array([r'$ \mathbf{\beta} $', r'$ \mathbf{r + \mu} $',
                               '$ \mathbf{p} $', '$ \mathbf{q} $', r'$ \mathbf{\sigma_{X}} $',
                               r'$ \mathbf{\sigma_{I}} $'])

    def data_processing(self):
        # Active cases in logarithmic scale
        I = np.log(self.confirmed)
        # New deaths+recovered (daily derivative) in logarithmic scale (prepend add 0 at first)
        X = np.log(self.recovered_death)

        # Remove infinites and NaNs (set to 0)
        X[np.isinf(X) | np.isnan(X)] = 0
        I[np.isinf(I) | np.isnan(I)] = 0

        # Setting parameters
        t0 = (I > 0).argmax()  # First day with more than 1 confirmed case
        tX0 = (X > 0).argmax()  # First day with more than 1 new death or recovered
        tq = np.where(self.date == self.quarantine)[0][0]  # Begin quarantine
        tmax = np.where(self.date == self.last_data)[0][0]  # Last data-point used for estimation
        FMT = '%Y.%m.%d'
        tf = (dt.strptime(self.last_projection, FMT) - dt.strptime(self.date[0], FMT)).days  # Last day to project the data
        I0 = I[t0]  # First datum in the Active cases series
        Iq = I[tq]  # First datum after quarantine in the Active cases series

        # return time+1 because in JAGS arrays are indexed from 1
        self.data = dict(b0=self.beta[0], b1=self.beta[1], r0=self.rmu[0], r1=self.rmu[1],
                         q0=self.q[0], q1=self.q[1], p0=self.p[0], p1=self.p[1],
                         tauI0=self.tauI[0], tauI1=self.tauI[1], tauX0=self.tauX[0], tauX1=self.tauX[1],
                         I=I, X=X, I0=I0, Iq=Iq, t0=t0+1, tX0=tX0+1, tq=tq+1, tmax=tmax+1, tf=tf+1)

    def sampler(self, nchains, nthreads, nchains_per_thread=1, niter=10000, nadapt=0, thin=1, burn_in=0.5):
        # Create data attribute
        self.data_processing()

        # Construct JAGS Model
        model = pj.Model(code=jags_model,
                         data=self.data,
                         chains=nchains,
                         adapt=nadapt,
                         threads=nthreads,
                         chains_per_thread=nchains_per_thread
                         )
        # Create samples
        samples = model.sample(niter, thin=thin, monitor_type="trace")
        # Initial portion of a Markov chain that is not stationary and is still affected by its initial value
        samples = pj.discard_burn_in_samples(samples, burn_in=round(burn_in * niter))
        self.niter = niter
        self.burn_in = burn_in
        self.nchains = nchains
        self.samples = samples

    def summary(self):
        def median_sd(x):
            median = np.percentile(x, 50)
            sd = np.sqrt(np.mean((x - median) ** 2))
            return sd

        func_dict = {
            "median": lambda x: np.percentile(x, 50),
            "median_std": median_sd,
            "2.5%_hdi": lambda x: np.percentile(x, 2.5),
            "97.5%_hdi": lambda x: np.percentile(x, 97.5),
        }

        idata = az.from_pyjags(self.samples)
        param = az.summary(idata, round_to=4, var_names=['beta', 'rmu', 'q', 'p', 'tauI', 'tauX'],
                           stat_funcs=func_dict)

        beta = self.samples['beta'].ravel()
        rmu = self.samples['rmu'].ravel()
        p = self.samples['p'].ravel()
        q = self.samples['q'].ravel()
        traces = pd.DataFrame((beta, rmu, p, q), index=['beta', 'rmu', 'p', 'q']).T
        self.traces = traces
        self.idata = idata
        self.param = param
        return param[['median', 'sd', '2.5%_hdi', '97.5%_hdi', 'r_hat']]

    def infected_exact(self):
        I0 = self.data['I0']
        Iq = self.data['Iq']
        t0 = self.data['t0'] - 1
        tq = self.data['tq'] - 1
        tf = self.data['tf'] - 1

        # compute median of parameters for all chains
        beta = np.median(self.samples['beta'].ravel())
        rmu = np.median(self.samples['rmu'].ravel())
        p = np.median(self.samples['p'].ravel())
        q = np.median(self.samples['q'].ravel())

        t1 = np.arange(t0, tq)
        t2 = np.arange(tq, tf+1)
        I1 = I0 + (beta - rmu) * (t1 - t0)
        I2 = Iq + ((beta * q) / (p + q) ** 2 * (1 - np.exp(-(p + q) * (t2 - tq))) +
                   (beta - rmu - beta * q / (q + p)) * (t2 - tq))
        return np.concatenate((I1, I2))

    def autocorrelation_time(self, nlags=1000):
        def fit_time(x, tau):
            return np.exp(-x / tau)

        times = np.zeros((self.nchains, len(self.varname)))
        x = np.arange(nlags + 1)
        for i in range(len(self.varname)):
            for j in range(self.nchains):
                y = acf(self.samples[self.varname[i]][0, :, j], fft=True, nlags=nlags)
                popt, pcov = curve_fit(fit_time, x, y)
                times[j, i] = popt
        return times

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
            axs = self.trim_axs(axs, self.nchains)
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
        # beta = data[:, 0], rmu = data[:, 1], p = data[:, 2], q = data[:, 3]
        tmp = data[data[:, 1] * (data[:, 2] + data[:, 3]) > data[:, 0] * data[:, 2]]

        beta = tmp[:, 0]
        rmu = tmp[:, 1]
        p = tmp[:, 2]
        q = tmp[:, 3]

        # Compute t_peak from the analytical expression for the epidemic's peak
        if len(tmp) >= 1e6:
            # Numba parallelized function
            func_nb_mt = make_multithread(inner_func_nb, nthreads)
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
        p_conf = len([rmu * (p + q) > beta * p]) / len(tmp) * 100

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

    def end_epidemic_plot(self, threshold=1000):
        y = self.samples['y']
        tmax = self.data['tmax'] - 1

        nchains = y.shape[2]
        niters = y.shape[1]
        length = nchains * niters
        times = np.empty(length, dtype=np.int32)
        # Compute times until the number of confirmed cases falls below 1000 for the first time
        i = 0
        for k in range(nchains):
            for j in range(niters):
                times[i] = np.argmax(y[tmax:, j, k] < np.log(threshold)) + tmax
                i += 1

        # Remove times corresponding at not satisfied condition (first time < 1000)
        times = times[times != tmax]

        #print(np.median(times), np.std(times))
        med, std = rounding(np.median(times), np.std(times))
        # plot distribution of times
        sns.histplot(times, binwidth=10, color='b', stat='density')
        textstr = f'({med:d}' + r' $\pm$ ' + f'{std:d}) days'
        ax = plt.gca()
        ax.axvline(np.median(times), c='r', label=textstr)
        plt.legend()
        plt.grid()
        plt.xlabel('Days since first confirmed case')
        plt.ylabel('Distribution of confirmed < 1000')

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
            plt.scatter(t, I[t0:tmax], facecolor='w', edgecolor='k', label='Fitted data')
            plt.plot(I_exact, c='orange', label='Fit+prediction')
            t = np.arange(t0, tf) - t0
            plt.fill_between(t, y1[t0:tf], y2[t0:tf], alpha=0.3, color='orange', label=f'{CI}% CI')

            if observed:
                # prevision beyond dataset's time
                if tf-1 > len(I):
                    t = np.arange(tmax, len(I)) - t0
                    plt.scatter(t, I[tmax:], c='g', alpha=0.5, label='Observed data')
                else:
                    t = np.arange(tmax, tf) - t0
                    plt.scatter(t, I[tmax:tf], c='g', alpha=0.5, label='Observed data')
                    # actual peak data of a chosen epidemic wave
                    tpeak = np.where(self.date == self.peak)[0][0] - t0
                    plt.plot(tpeak - t0, I[tpeak], 'ro', label='Actual peak')

            y_true = I[t0:tmax]
            y_pred = I_exact[t0:tmax]

            r2 = (stats.linregress(y_true, y_pred)[2]) ** 2
            plt.annotate(r'$r^{2}$=' + f'{r2:.6f}', xy=(0.85, 0.97), xycoords='axes fraction')
            plt.xlabel(f'Days since first confirmed case ({self.date[t0]})')
            plt.ylabel("Active cases ($\mathbf{log_{10}}$)")
            plt.title(f"Active cases {self.country:s}", fontsize=16, fontweight='bold')
            plt.legend(loc='upper left')
            plt.show()

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
            plt.scatter(t, X[tX0:tmax], facecolor='w', edgecolor='k', label='Fitted data')
            t = np.arange(tX0, tf) - tX0
            plt.fill_between(t, z1[tX0:tf], z2[tX0:tf], alpha=0.3, color='orange', label=f'{CI}% CI')

            if observed:
                if tf > len(X):
                    t = np.arange(tmax, len(X)) - tX0
                    plt.scatter(t, X[tmax:], c='g', alpha=0.5, label='Observed data')
                else:
                    t = np.arange(tmax, tf) - tX0
                    plt.scatter(t, X[tmax:tf], c='g', alpha=0.5, label='Observed data')

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
            plt.show()

    def trim_axs(self, axs, N):
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]

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
            axs = self.trim_axs(axs, self.nchains)

            for i, ax in zip(np.arange(self.nchains), axs):
                ax.plot(self.samples[var][0, :, i], c='b', lw=1)
                ax.plot([np.median(self.samples[var].ravel())] * (int(self.niter*self.burn_in)), c='r', lw=1)
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
            axs = self.trim_axs(axs, self.nchains)

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
            titles = [f'Trace of {self.names[i]}', f'Posterior distribution of {self.names[i]}', f'Acf of {self.names[i]}']
            for k in range(3):
                ax[i, k].set_xlabel('MCMC step')
                ax[i, k].set_title(titles[k], weight='bold')
                ax[i, k].grid()

    def solve_SCIR(self, step=0.01):
        """
        Numerical solution of SCIR model with chosen parameters (median posteriors).
        This data is fitted to show less (but still big) width area of predictive
        posterior interval
        """
        # Total population, N
        if self.country == 'Spain':
            N = 46754783.
        elif self.country == 'Italy':
            N = 60461828.

        # The SCIR model differential equations
        def SCIR(state, t, N, beta, q, p, rmu):
            """
            return: dSdt, dCdt, dIdt, dXdt (Derivatives)
            """
            # Unpack the state vector
            S, C, I, X = state
            return (-beta / N * I * S - q * S + p * C,
                    q * S - p * C,
                    beta / N * I * S - rmu * I,
                    rmu * I)

        # Contact rate (beta), mean recovery+death rate, rmu,
        # rate of specific measures restricting mobility and contacts (q),
        # rate of individuals that leave the confinement measure (p) (all in 1/days)
        beta, rmu, q, p = self.summary().loc[['beta', 'rmu', 'q', 'p'], 'median']
        t0 = self.data['t0'] - 1
        tX0 = self.data['tX0'] - 1
        tq = self.data['tq'] - 1
        tf = self.data['tf'] - 1
        # Initial conditions for the first regime
        I0 = np.exp(self.data['I'][0])
        X0 = np.exp(self.data['X'][0])
        S0 = N - I0 - X0
        C0 = 0.
        state0 = np.array([S0, C0, I0, X0])
        # Integrate the SCIR equations over a time grid before confinement (first regime)
        t = np.arange(0, tf, step)
        ret1 = odeint(SCIR, state0, t[:int(tq / step)], args=(N, beta, q * 0, p * 0, rmu))
        # Integrate the SCIR equations over a time grid, after confinement (second regime)
        ret2 = odeint(SCIR, ret1[-1, :], t[int(tq / step):], args=(N, beta, q, p, rmu))
        ret = np.concatenate((ret1, ret2))

        return np.column_stack((t, ret))
