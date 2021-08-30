import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.integrate import odeint
from typing import Tuple, Callable
import glob 
import pickle


def get_data(file: str) -> dict:
        """
        Load data from .pkl file and extract only useful data
        """
        file_handler = open(file, 'rb')
        samples = pickle.load(file_handler)
        var = [
                'country',
                'beta',
                'rmu', 
                'p', 
                'q', 
                't0', 
                'tq', 
                'tf',
                'I',
                'X'
                ]
        samp_new = {key: samples[key] for key in var}
        return samp_new


# The SCIR model differential equations
def SCIR(state: np.array, t: float, N: float, beta: float, q: float, p: float, 
         rmu: float) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Compute dSdt, dCdt, dIdt, dXdt (Derivatives)
        """
        # Unpack the state vector
        S, C, I, X = state
        return (
                -beta / N * I * S - q * S + p * C,
                q * S - p * C,
                beta / N * I * S - rmu * I,
                rmu * I
                )


def solve_SCIR(param: tuple, res_path: str = '../../../Results/ita/first_wave'
               + '/solve_res/',
               step: float = 0.01) -> None:
        beta, rmu, p, q = param
        # Initial conditions for the first regime
        S0 = N - I0 - X0
        C0 = 0.
        state0 = np.array([S0, C0, I0, X0])
        # Integrate the SCIR equations over a time grid before confinement 
        # (first regime)
        t = np.arange(t0, tf, step)
        ret1 = odeint(SCIR, state0, t[:int(tq / step)], 
                        args=(N, beta, q * 0, p * 0, rmu))
        # Integrate the SCIR equations over a time grid, after confinement 
        # (second regime)
        ret2 = odeint(SCIR, ret1[-1, :], t[int(tq / step):], 
                        args=(N, beta, q, p, rmu))
        ret = np.concatenate((ret1, ret2))
        # t, S, C, I, R
        np.save(res_path + f'ode_sol{beta}_{rmu}_{p}_{q}.npy', ret)


if __name__ == '__main__':
        path = '../../../Results/ita/first_wave/simul_res/'
        file_name = glob.glob1(path, '*.pkl')[0]
        samples = get_data(path + file_name)
        # Total population, N
        if samples['country'] == 'Spain':
                N = 46754783.
        elif samples['country'] == 'Italy':
                N = 60461828.
        elif samples['country'] == 'France':
                N = 65273511.

        t0 = samples['t0'] - 1
        tq = samples['tq'] - 1
        tf = samples['tf']
        # Data is in log scale but ode system not
        I0 = np.exp(samples['I'][0])
        X0 = np.exp(samples['X'][0])
        # get epidemic parameters
        param = samples['beta'].flat, samples['rmu'].flat, \
                samples['p'].flat, samples['q'].flat

        parameters = [(a, b, c, d) for a, b, c, d in zip(*param)]
        p = Pool(cpu_count())
        p.map(solve_SCIR, parameters)
        p.close()
