import threading
import math
import numpy as np
from numba import jit


def func_np(beta, rmu, p, q):
    """
    Control function using Numpy.
    """
    return 1 / (p + q) * np.log(beta * q / (rmu * (p + q) - beta * p))


# nopython: (compilation mode) compile the decorated function so that it will run 
# entirely without the involvement of the Python interpreter (best performance)
# nogil: release GIL (global interpreter lock) allowing you to take advantage of multi-core systems
@jit('void(double[:], double[:], double[:], double[:], double[:])',
     nopython=True, nogil=True)
def peak_time_nb(result, beta, rmu, p, q):
    """
    Function under test.
    """
    for i in range(len(result)):
        result[i] = 1 / (p[i] + q[i]) * math.log(beta[i] * q[i] / (rmu[i] * (p[i] + q[i]) - beta[i] * p[i]))

@jit('void(double[:], double[:,:], double, double)',
      nopython=True, nogil=True)
def epidemic_end(times, I, threshold, tmax):
    """
    Function under test.
    """
    for i in range(len(times)):
        times[i] = np.argmax(I[:, i] < np.log(threshold)) + tmax

def make_singlethread(inner_func):
    """
    Run the given function inside a single thread.
    """
    def func(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        inner_func(result, *args)
        return result
    return func

def make_multithread(inner_func, numthreads):
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

def rounding(med, std):
    dim = len(str(int(med)))
    prec = len(str(int(std)))
    med_round = int(round(med / 10 ** dim, prec) * 10 ** dim)
    std_round = int(round(std / 10 ** prec, 1) * 10 ** prec)
    return med_round, std_round

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