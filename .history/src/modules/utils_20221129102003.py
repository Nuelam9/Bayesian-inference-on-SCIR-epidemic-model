# To do
# 1. Add function param type annotation
#   1.1 make_singlethread, make_multithread
# 2. Change how compute epidemic end, root finding is better, cause 
#    in future solve numerically ode system 


import threading  # change trading cause is obsolete 
import math
import numpy as np
from numba import jit
import pickle
from scipy.integrate import odeint
from typing import Tuple, Callable
from timeit import Timer


def fit_time(x: np.array, tau: float) -> np.array:
    return np.exp(-x / tau)


def peak_posterior_np(beta: np.ndarray, rmu: np.ndarray, 
                      p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Control function using Numpy.
    """
    return 1 / (p + q) * np.log(beta * q / (rmu * (p + q) - beta * p))


@jit('void(double[:], double[:], double[:], double[:], double[:])',
     nopython=True, nogil=True)
def peak_posterior_nb(result: float, beta: float, 
                      rmu: float, p: float, q: float) -> None:
    """
    nopython: (compilation mode) compile the decorated function 
              so that it will run entirely without the involvement of 
              the Python interpreter (best performance)
    nogil: release GIL (global interpreter lock) allowing you to take 
           advantage of multi-core systems

    Function under test.
    """
    for i in range(len(result)):
        result[i] = 1 / (p[i] + q[i]) * math.log(beta[i] * q[i] /
                                                 (rmu[i] * (p[i] + q[i]) 
                                                 - beta[i] * p[i]))


@jit('void(double[:], double[:,:], double, double)',
      nopython=True, nogil=True)
def epidemic_end(times: np.array, I: np.ndarray, threshold: float, tmax: float):
    """
    Function under test.
    """
    lenght = len(times)
    for i in range(lenght):
        times[i] = np.argmax(I[:, i] < np.log(threshold)) + tmax


def make_singlethread(inner_func: Callable) -> Callable:
    """
    Run the given function inside a single thread.
    """
    def func(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        inner_func(result, *args)
        return result
    return func


def make_multithread(inner_func: Callable, numthreads: int) -> Callable:
    """
    Run the given function inside *numthreads* threads, splitting
    its arguments into equal-sized chunks.
    """
    def func_mt(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        args = (result,) + args
        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk
        chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in
                   args] for i in range(numthreads)]
        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result
    return func_mt


def rounding(med: float, std: float) -> Tuple[float, float]:
    dim = len(str(int(med)))
    prec = len(str(int(std)))
    med_round = int(round(med / 10 ** dim, prec) * 10 ** dim)
    std_round = int(round(std / 10 ** prec, 1) * 10 ** prec)
    return med_round, std_round


def infected_exact(samples: dict) -> np.array:
    I0 = samples['I0']
    Iq = samples['Iq']
    t0 = samples['t0'] - 1
    tq = samples['tq'] - 1
    tf = samples['tf']  # cause in np.arange tf - 1

    # compute median of parameters for all chains
    beta = np.median(samples['beta'])
    rmu = np.median(samples['rmu'])
    p = np.median(samples['p'])
    q = np.median(samples['q'])

    t = np.arange(t0, tf)
    I = np.zeros(tf - t0, dtype=np.float64)
    I[:tq] = I0 + (beta - rmu) * (t[:tq] - t0)
    I[tq:] = Iq + ((beta * q) / (p + q) ** 2
                   * (1 - np.exp(-(p + q) * (t[tq:] - tq))) 
                   + (beta - rmu - beta * q / (q + p)) * (t[tq:] - tq))
    return I


def get_data(file: str) -> dict:
    """
        Load data from .pkl file and extract only useful data
        """
    file_handler = open(file, 'rb')
    return pickle.load(file_handler)    


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


def solve_SCIR(samples: dict, step: float = 0.01, 
               tf: int = None) -> np.ndarray:
    """
    Numerical solution of SCIR model with chosen parameters 
    (median posteriors).
    This data is fitted to show less (but still big) width area of 
    predictive posterior interval.
    """
    # Total population, N
    if samples['country'] == 'Spain':
        N = 46754783.
    elif samples['country'] == 'Italy':
        N = 60461828.
    elif samples['country'] == 'France':
        N = 65273511.

    # Contact rate (beta), mean recovery+death rate, rmu,
    # rate of specific measures restricting mobility and contacts (q),
    # rate of individuals that leave the confinement measure (p) (all in 1/days)
    beta = np.median(samples['beta'])
    rmu = np.median(samples['rmu'])
    p = np.median(samples['p'])
    q = np.median(samples['q'])
    t0 = samples['t0'] - 1
    tq = samples['tq'] - 1
    if tf is None:
        tf = samples['tf'] - 1
    # Initial conditions for the first regime
    I0 = np.exp(samples['I'][0])
    X0 = np.exp(samples['X'][0])
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
    return np.column_stack((t, ret)).T

def time_check(func, n):
    t = Timer(lambda: func)
    print(t.timeit(number=n))
