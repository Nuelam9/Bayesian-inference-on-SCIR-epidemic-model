import threading
import math
import numpy as np
from numba import jit


def func_np(beta, rmu, p, q):
    """
    Control function using Numpy.
    """
    return 1 / (p + q) * np.log(beta * q / (rmu * (p + q) - beta * p))


@jit('void(double[:], double[:], double[:], double[:], double[:])',
     nopython=True, nogil=True)
def inner_func_nb(result, beta, rmu, p, q):
    """
    Function under test.
    """
    for i in range(len(result)):
        result[i] = 1 / (p[i] + q[i]) * math.log(beta[i] * q[i] / (rmu[i] * (p[i] + q[i]) - beta[i] * p[i]))


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


"""    def end_epidemic(self, param, threshold=1000):
        y = globals()['y']
        tmax = globals()['tmax']
        k, j = param
        return np.argmax(y[tmax:, j, k] < np.log(threshold)) + tmax

    def end_epidemic_plot2(self):
        y = self.samples['y']
        tmax = self.data['tmax'] - 1

        nchains = y.shape[2]
        niters = y.shape[1]
        paramlist = list(it.product(range(nchains), range(niters)))

        # Compute times until the number of confirmed cases falls below 1000 for the first time
        p = Pool(cpu_count() - 1)
        times = np.asarray(p.map(self.end_epidemic, paramlist))
        p.close()
        # Remove times corresponding at not satisfied condition (first time < 1000)
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
        plt.ylabel('Distribution of confirmed < 1000')"""
