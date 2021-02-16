# To do:
# 1. func plot_results()
#     a. sintesi

import numpy as np
import pandas as pd
import pyjags as pj
import arviz as az
from scipy import stats
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
    p ~ dunif(0, 5)  # Non-informative prior
    q ~ dunif(0, 5)  # Non-informative prior
    beta ~ dunif(0, 1)  # Doubling time is less than 1 per day (informative prior)
    rmu ~ dunif(0, 1)  # rmu is lower than beta (so R0>1) (informative prior)

    # Priors for precisions (inverse of variance)
    tauI ~ dgamma(0.01, 0.01)  # Non-informative prior
    tauX ~ dgamma(0.01, 0.01)  # Non-informative prior
    y[t0] <- I0
}'''


class Analysis:

    def __init__(self, date, confirmed, recovered_death, quarantine, last_data, last_projection, country):
        self.date = date
        self.confirmed = confirmed
        self.recovered_death = recovered_death
        self.quarantine = quarantine
        self.last_data = last_data
        self.last_projection = last_projection
        self.country = country

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

        # returning time+1 because in JAGS arrays are indexed from 1
        self.data = dict(I=I, X=X, I0=I0, Iq=Iq, t0=t0+1,
                         tX0=tX0+1, tq=tq+1, tmax=tmax+1, tf=tf+1)

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
        param = az.summary(idata, round_to=4, var_names=['beta', 'rmu', 'q', 'p', 'tauI', 'tauX'], stat_funcs=func_dict)
        self.idata = idata
        self.param = param
        return param[['median', 'sd', '2.5%_hdi', '97.5%_hdi', 'r_hat']]

    def infected_exact(self):
        I0 = self.data['I0']
        Iq = self.data['Iq']
        t0 = self.data['t0'] - 1
        tq = self.data['tq'] - 1
        tf = self.data['tf']   # not -1 cause np.arange(a, b) ends in b-1

        # compute median of parameters for all chains
        beta = np.median(self.samples['beta'].ravel())
        rmu = np.median(self.samples['rmu'].ravel())
        p = np.median(self.samples['p'].ravel())
        q = np.median(self.samples['q'].ravel())

        t1 = np.arange(t0, tq)
        t2 = np.arange(tq, tf)
        I1 = I0 + (beta - rmu) * (t1 - t0)
        I2 = Iq + ((beta * q) / (p + q) ** 2 * (1 - np.exp(-(p + q) * (t2 - tq))) +
                   (beta - rmu - beta * q / (q + p)) * (t2 - tq))
        return np.concatenate((I1, I2))

    def plot_results(self, CI=95, Y=False, Z=False, observed=False):
        t0 = self.data['t0'] - 1
        tmax = self.data['tmax']
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
            plt.scatter(t, I[t0:tmax], facecolor='w', edgecolor='k', label='fitted data')
            plt.plot(I_exact, c='orange', label='fit+prediction')
            t = np.arange(t0, tf) - t0
            plt.fill_between(t, y1[t0:tf], y2[t0:tf], alpha=0.3, color='orange', label=f'{CI}% CI')

            if observed:
                # prevision beyond dataset's time
                if tf > len(I):
                    t = np.arange(tmax, tf) - t0
                    plt.scatter(t, I[tmax:], c='g', alpha=0.5, label='observed data')
                else:
                    t = np.arange(tmax, tf) - t0
                    plt.scatter(t, I[tmax:tf], c='g', alpha=0.5, label='observed data')

            y_true = I[t0:tmax]
            y_pred = I_exact[t0:tmax]

            r2 = (stats.linregress(y_true, y_pred)[2]) ** 2
            plt.annotate(r'$r^{2}$=' + f'{r2:.6f}', xy=(0.85, 0.97), xycoords='axes fraction')
            plt.xlabel(f'Days since first confirmed case ({self.date[self.data["t0"]-1]})')
            plt.ylabel("Active cases ($\mathbf{log_{10}}$)")
            plt.legend(loc='upper left')
            plt.title(f"Active cases {self.country:s}", fontsize=16, fontweight='bold')
            plt.legend(loc='upper left');

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
            plt.plot(t, z_med[tX0:], c='orange', label='fit+prediction')
            t = np.arange(tX0, tmax) - tX0
            plt.scatter(t, X[tX0:tmax], facecolor='w', edgecolor='k', label='fitted data')
            t = np.arange(tX0, tf) - tX0
            plt.fill_between(t, z1[tX0:tf], z2[tX0:tf], alpha=0.3, color='orange', label=f'{CI}% CI')

            if observed:
                t = np.arange(tmax, tf) - tX0
                plt.scatter(t, X[tmax:tf], c='g', alpha=0.5, label='observed data')

            z_true = X[tX0:tmax]
            z_pred = z_med[tX0:tmax]

            # Goodness of fit, r-square (square of pearson's coefficient)
            r2 = (stats.linregress(z_true, z_pred)[2]) ** 2
            plt.annotate(r'$r^{2}=%.6f$' % r2, xy=(0.85, 0.97), xycoords='axes fraction')
            plt.xlabel(f'Days since first death+recovered case ({self.date[self.data["tX0"]-1]})')
            plt.ylabel("Variation of death+recovered cases ($\mathbf{log_{10}}$)")
            plt.legend(loc='upper left')
            plt.title("Daily variation of death+recovered cases %s" % self.country, fontsize=16, fontweight='bold')
            plt.legend(loc='upper left');

    def trim_axs(self, axs, N):
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]

    def trace_plot(self, var=None, total=True):
        if total:
            #az.plot_trace(self.idata, var_names=['beta', 'rmu', 'q', 'p', 'tauI', 'tauX']);
            for i in range(self.nchains):
                plt.plot(self.samples[var][0, :, i], c='b', lw=1)

        else:
            varname = np.array(['beta', 'rmu', 'p', 'q', 'tauX', 'tauI'])
            names = np.array([r'$ \mathbf{\beta} $', r'$ \mathbf{r + \mu} $', '$ \mathbf{p} $',
                                    '$ \mathbf{q} $', r'$ \mathbf{\tau_{X}} $', r'$ \mathbf{\tau_{I}} $'])

            ind = np.where(varname == var)[0][0]

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
                #ax.axhline(np.median(self.samples[var].ravel()), xmin=0, xmax=(self.niter - self.burn_in), c='r', lw=1)
                ax.plot(np.arange(self.niter - self.burn_in), [np.median(self.samples[var].ravel())] * (self.niter - self.burn_in), c='r', lw=1)
                ax.grid()
                # removing top and right borders
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_title(f'Chain {i + 1}', weight='bold')
                ax.set_xlabel('MCMC step')
                ax.tick_params(axis='x', which='major', labelsize=8.5)
                ax.set_title(f"Chain {(i + 1):d}", fontweight='bold')
            fig.suptitle(f'Trace of {names[ind]}', fontsize=16, fontweight='bold');

    def posteriors(self, var, total=0):
        varname = np.array(['beta', 'rmu', 'p', 'q', 'tauX', 'tauI'])
        names = np.array([r'$ \mathbf{\beta} $', r'$ \mathbf{r + \mu} $',
                               '$ \mathbf{p} $', '$ \mathbf{q} $', r'$ \mathbf{\sigma_{X}} $',
                               r'$ \mathbf{\sigma_{I}} $'])

        ind = np.where(varname == var)[0][0]
        exp = 1
        if (var == 'tauX') | (var == 'tauI'):
            exp = -1 / 2

        if total:
            sns.histplot((np.power(self.samples[var].ravel(), exp)), kde=True, stat='density', color='b')
            plt.grid()
            ax = plt.gca()
            # removing top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_title(f'Posterior of all chains for {names[ind]}', weight='bold')
        else:
            if self.nchains == 12:
                rows, cols = 4, 3
            elif self.nchains == 8:
                rows, cols = 4, 2

            fig, axs = plt.subplots(rows, cols, figsize=(16, 8), constrained_layout=True)
            axs = self.trim_axs(axs, self.nchains)

            for ax, i in zip(axs, np.arange(self.nchains)):
                sns.histplot((np.power(self.samples[var][0, :, i], exp)), kde=True, stat='density', color='b', ax=ax)
                ax.tick_params(axis='x', which='major', labelsize=7.5)
                ax.grid()
                # removing top and right borders
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_title(f'Chain {i + 1}', weight='bold')

            fig.suptitle(f'Posterior distributions of {names[ind]}', fontsize=16, fontweight='bold');

    def simulation(self):
        # Total population, N
        if self.country == 'Spain':
            N = 46754783.

        elif self.country == 'Italy':
            N = 60461828.

        # Contact rate (beta), mean recovery+death rate, rmu,
        # rate of specific measures restricting mobilty and contacts (q),
        # rate of individuals that leave the confinement measure (p) (all in 1/days)
        beta, rmu, q, p = self.summary().loc[['beta', 'rmu', 'q', 'p'], 'median']

        tX0 = self.data['tX0']
        tq = self.data['tq']
        tf = self.data['tf']

        # Initial number of infected and recovered+death individuals for the first regime
        I1 = np.round(np.exp(self.data['I'][0]))
        X1 = np.round(np.exp(self.data['X'][0]))
        # Susceptibles individuals to infection initially for the first regime
        S1 = N - I1 - X1
        C1 = 0

        # A grid of time points for the first regime
        t1 = np.linspace(0, tq, 1000)

        # The SCIR model differential equations
        def SCIR(y, t, N, beta, q, p, rmu):
            S, C, I, X = y
            dSdt = -beta / N * I * S - q * S + p * C
            dCdt = q * S - p * C
            dIdt = beta / N * I * S - rmu * I
            dXdt = rmu * I

            return dSdt, dCdt, dIdt, dXdt

        # Initial conditions vector for the first regime
        y1 = S1, C1, I1, X1
        # Integrate the SCIR equations over the time grid, t1 (first regime, before confinement)
        ret1 = odeint(SCIR, y1, t1, args=(N, beta, q * 0, p * 0, rmu))
        S1, C1, I1, X1 = ret1.T

        # Initial number of infected and recovered+death individuals for the second regime
        I2 = I1[-1]
        X2 = X1[-1]
        # Susceptible to infection initially for the second regime
        S2 = N - I2 - X2
        C2 = 0

        # A grid of time points for the second regime
        t2 = np.linspace(tq, tf - 1, 1000)

        # Initial conditions vector for the second regime
        y2 = S2, C2, I2, X2
        # Integrate the SCIR equations over the time grid, t2 (second regime, after confinement)
        ret2 = odeint(SCIR, y2, t2, args=(N, beta, q, p, rmu))
        S2, C2, I2, X2 = ret2.T

        t = np.concatenate([t1, t2])
        S = np.concatenate([S1, S2])
        C = np.concatenate([C1, C2])
        I = np.concatenate([I1, I2])
        X = np.concatenate([X1, X2])

        return t, S, C, I, X

##' Save class object (Italy)
# file_ita = open('results_ita.obj', 'wb')
# pickle.dump(analysis_ita, file_ita)
# file_ita.close()
## Load class object (Italy)
# filehandler_ita = open('results_ita.pickle', 'rb')
# object_ita = pickle.load(filehandler_ita)
## Check if two dict samples are the same (print elements different)
# {k: object_try.samples[k] for k in object_try.samples if k in analysis_ita.samples and np.all(object_try.samples[k] != analysis_ita.samples[k])}