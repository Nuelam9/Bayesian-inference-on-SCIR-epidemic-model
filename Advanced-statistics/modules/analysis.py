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
# 7. Add type annotation (with pycharm)


import numpy as np
import pyjags as pj
import arviz as az
from datetime import datetime as dt


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
        fmt = '%Y.%m.%d'
        tf = (dt.strptime(self.last_projection, fmt) - dt.strptime(self.date[0],
                                                                   fmt)).days  # Last day to project the data
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
        jags_model = open('../../modules/model', 'r').read()
        model = pj.Model(code=jags_model,
                         data=self.data,
                         chains=nchains,
                         adapt=nadapt,
                         threads=nthreads,
                         chains_per_thread=nchains_per_thread
                         )
        # Create samples
        samples = model.sample(niter, vars=['beta', 'p', 'q', 'rmu', 'tauI', 'tauX', 'y', 'z'],
                               thin=thin, monitor_type='trace')
        # Discard initial portion of a Markov chain that is not stationary and is still affected by its initial value
        samples = pj.discard_burn_in_samples(samples, burn_in=round(burn_in * niter))
        self.niter = niter
        self.burn_in = burn_in
        self.nchains = nchains
        self.samples = samples

    @property
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
        self.idata = idata
        self.param = param
        return param[['median', 'sd', '2.5%_hdi', '97.5%_hdi', 'r_hat']]

# import pickle
##' Save class object (Italy)
# file_ita = open('results_ita.pkl', 'wb')
# pickle.dump(analysis_ita, file_ita)
# file_ita.close()
## Load class object (Italy)
# filehandler_ita = open('results_ita.pickle', 'rb')
# object_ita = pickle.load(filehandler_ita)
## Check if two dict samples are the same (print elements different)
# {k: object_try.samples[k] for k in object_try.samples if k in analysis_ita.samples and
# np.all(object_try.samples[k] != analysis_ita.samples[k])}
