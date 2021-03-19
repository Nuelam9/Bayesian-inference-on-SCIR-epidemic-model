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
# 5 plot functions in other modules
# 6. use numba also for I_exact (new version in prova_numba)


import numpy as np
import pandas as pd
import pyjags as pj
import arviz as az
from utils import *
from analysis_plot import *
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from datetime import datetime as dt


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

        # Remove infinities and NaNs (set to 0)
        X[np.isinf(X) | np.isnan(X)] = 0
        I[np.isinf(I) | np.isnan(I)] = 0

        # Setting parameters
        t0 = (I > 0).argmax()  # First day with more than 1 confirmed case
        tX0 = (X > 0).argmax()  # First day with more than 1 new death or recovered
        tq = np.where(self.date == self.quarantine)[0][0]  # Begin quarantine
        tmax = np.where(self.date == self.last_data)[0][0]  # Last data-point used for estimation
        FMT = '%Y.%m.%d'
        tf = (dt.strptime(self.last_projection, FMT) - dt.strptime(self.date[0],
                                                                   FMT)).days  # Last day to project the data
        I0 = I[t0]  # First datum in the Active cases series
        Iq = I[tq]  # First datum after quarantine in the Active cases series

        # return time+1 because in JAGS arrays are indexed from 1
        self.data = dict(b0=self.beta[0], b1=self.beta[1], r0=self.rmu[0], r1=self.rmu[1],
                         q0=self.q[0], q1=self.q[1], p0=self.p[0], p1=self.p[1],
                         tauI0=self.tauI[0], tauI1=self.tauI[1], tauX0=self.tauX[0], tauX1=self.tauX[1],
                         I=I, X=X, I0=I0, Iq=Iq, t0=t0 + 1, tX0=tX0 + 1, tq=tq + 1, tmax=tmax + 1, tf=tf + 1)

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

    # Here use numba
    def infected_exact(self):
        I0 = self.data['I0']
        Iq = self.data['Iq']
        t0 = self.data['t0'] - 1
        tq = self.data['tq'] - 1
        tf = self.data['tf']  # cause in np.arange tf - 1

        # compute median of parameters for all chains
        beta = np.median(self.samples['beta'].ravel())
        rmu = np.median(self.samples['rmu'].ravel())
        p = np.median(self.samples['p'].ravel())
        q = np.median(self.samples['q'].ravel())

        t = np.arange(t0, tf)
        I = np.zeros(tf - t0, dtype=np.float64)
        I[:tq] = I0 + (beta - rmu) * (t[:tq] - t0)
        I[tq:] = Iq + ((beta * q) / (p + q) ** 2 * (1 - np.exp(-(p + q) * (t[tq:] - tq))) +
                       (beta - rmu - beta * q / (q + p)) * (t[tq:] - tq))
        return I

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

        # Contact rate (beta), mean recovery+death rate, rmu,
        # rate of specific measures restricting mobility and contacts (q),
        # rate of individuals that leave the confinement measure (p) (all in 1/days)
        beta, rmu, q, p = self.summary().loc[['beta', 'rmu', 'q', 'p'], 'median']
        t0 = self.data['t0'] - 1
        tq = self.data['tq'] - 1
        tf = self.data['tf'] - 1
        # Initial conditions for the first regime
        I0 = np.exp(self.data['I'][0])
        X0 = np.exp(self.data['X'][0])
        S0 = N - I0 - X0
        C0 = 0.
        state0 = np.array([S0, C0, I0, X0])
        # Integrate the SCIR equations over a time grid before confinement (first regime)
        t = np.arange(t0, tf, step)
        ret1 = odeint(SCIR, state0, t[:int(tq / step)], args=(N, beta, q * 0, p * 0, rmu))
        # Integrate the SCIR equations over a time grid, after confinement (second regime)
        ret2 = odeint(SCIR, ret1[-1, :], t[int(tq / step):], args=(N, beta, q, p, rmu))
        ret = np.concatenate((ret1, ret2))

        return np.column_stack((t, ret)).T
    
    def autocorrelation_time_plot(self):
        autocorrelation_time_plot(self)

    def autocorr_fit_plot(self, xmin=-10, xmax=200, var=None):
        autocorr_fit_plot(self, xmin, xmax, var)

    def peak_posterior(self, nthreads=cpu_count() - 2, binwidth=10, offset=3, second_wave=False):
        peak_posterior(self, nthreads, binwidth, offset, second_wave)

    def end_epidemic_plot(self, tf=380, threshold=1000):
        end_epidemic_plot(self, tf, threshold)

    def plot_results(self, CI=95, Y=False, Z=False, observed=False):
        plot_results(self, CI, Y, Z, observed)

    def trace_plot(self, var=None):
        trace_plot(self, var)

    def posteriors(self, var=None):
        posteriors(self, var)

    def plot_summary(self):
        plot_summary(self)

