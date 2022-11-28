# When implemented numerical solution with parameteres infered
# track on y, z can be removed, so saving memory iters can be higher

import numpy as np
import pyjags as pj
import arviz as az
from datetime import datetime as dt
import pandas as pd
from dataclasses import dataclass


@dataclass
class Analysis:
    date: str
    confirmed: float
    recovered_death: float
    confinement: str
    last_data: str
    last_projection: str
    peak
    beta
    rmu
    q
    p
    tauI
    tauX
    varname = np.array(['beta', 'rmu', 'p', 'q', 'tauX', 'tauI'])
    names = np.array([r'$ \mathbf{\beta} $',
                            r'$ \mathbf{r + \mu} $',
                            '$ \mathbf{p} $',
                            '$ \mathbf{q} $', 
                            r'$ \mathbf{\sigma_{X}} $',
                            r'$ \mathbf{\sigma_{I}} $'])   

    def data_processing(self) -> None:
        # Active cases in logarithmic scale
        I = np.log(self.confirmed)
        # New deaths+recovered (daily derivative) in logarithmic scale
        # (prepend add 0 at first)
        X = np.log(self.recovered_death)

        # Remove infinities and NaNs (set to 0)
        X[np.isinf(X) | np.isnan(X)] = 0
        I[np.isinf(I) | np.isnan(I)] = 0

        # Setting parameters
        # First day with more than 1 confirmed case
        t0 = (I > 0).argmax()
        # First day with more than 1 new death or recovered
        tX0 = (X > 0).argmax()
        # Begin confinement
        tq = np.where(self.date == self.confinement)[0][0]
        # Last data-point used for estimation
        tmax = np.where(self.date == self.last_data)[0][0]
        fmt = '%Y.%m.%d'
        # Last day to project the data
        tf = (dt.strptime(self.last_projection, fmt) 
              - dt.strptime(self.date[0], fmt)).days
        I0 = I[t0]  # First datum in the Active cases series
        Iq = I[tq]  # First datum after confinement in the Active cases series

        # return time+1 because in JAGS arrays are indexed from 1
        self.data = dict(b0=self.beta[0], b1=self.beta[1], r0=self.rmu[0], 
                         r1=self.rmu[1], q0=self.q[0], q1=self.q[1], 
                         p0=self.p[0], p1=self.p[1], tauI0=self.tauI[0], 
                         tauI1=self.tauI[1], tauX0=self.tauX[0], 
                         tauX1=self.tauX[1], I=I, X=X, I0=I0, Iq=Iq, t0=t0 + 1, 
                         tX0=tX0 + 1, tq=tq + 1, tmax=tmax + 1, tf=tf + 1)       

    def sampler(self, nchains: int, nthreads: int, nchains_per_thread: int = 1, 
                niter: int = 10000, nadapt: int = 0, thin: int=1, 
                burn_in: float = 0.5) -> None:
        # Create data attribute
        self.data_processing()

        # Import JAGS Model from .txt file
        jags_model = open('../../../modules/model', 'r').read()
        model = pj.Model(code=jags_model,
                         data=self.data,
                         chains=nchains,
                         adapt=nadapt,
                         threads=nthreads,
                         chains_per_thread=nchains_per_thread
                         )
        # Create samples
        samples = model.sample(niter, vars=['beta', 'p', 'q', 'rmu',
                                            'tauI', 'tauX', 'y', 'z'],
                               thin=thin, monitor_type='trace')
        # Discard initial portion of a Markov chain that is not stationary 
        # and is still affected by its initial value
        samples = pj.discard_burn_in_samples(samples, 
                                             burn_in=round(burn_in * niter))
        self.niter = niter
        self.burn_in = burn_in
        self.nchains = nchains
        self.samples = samples

    @property
    def summary(self) -> pd.DataFrame:

        def median_sd(x: np.array) -> float:
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
        param = az.summary(idata, round_to=6, var_names=['beta', 'rmu', 'q', 
                                                         'p', 'tauI', 'tauX'],
                           stat_funcs=func_dict)
        self.idata = idata # To do: remove from attributes? (prob yes) 
        self.param = param # same question
        return param[['median', 'sd', '2.5%_hdi', '97.5%_hdi', 'r_hat']]

## Check if two dict samples are the same (print elements different)
# {k: object_try.samples[k] for k in object_try.samples if k in analysis_ita.samples and
# np.all(object_try.samples[k] != analysis_ita.samples[k])}
