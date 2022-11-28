"""
Launch this code on windows.
Try to upgrade to wsl2 and check the pbm
"""


import numpy as np
import sys
sys.path.append('../../../modules/')
from utils import get_data, SCIR
from multiprocessing import Pool, cpu_count
from scipy.integrate import odeint
from typing import Tuple
from dataclasses import dataclass
import glob 


@dataclass
class Ode_system_solver:
                N: flota = N
                t0 = t0
                tq = tq
                tf = tf
                I0 = I0
                X0 = X0
                var = var
                
        def __init__(self, N: float, t0: int, tq: int, tf: int, 
                     I0: float, X0: float, var: str) -> None:
                self.N = N
                self.t0 = t0
                self.tq = tq
                self.tf = tf
                self.I0 = I0
                self.X0 = X0
                self.var = var
                if var == 'S':
                        self.ind = 0
                elif var == 'C':
                        self.ind = 1
                elif var == 'I':
                        self.ind = 2
                elif var == 'X':
                        self.ind = 3
        

        def solve_SCIR(self, param: tuple, step: float = 0.01) -> np.ndarray:
                beta, rmu, p, q = param
                # Initial conditions for the first regime
                N, t0, tq, tf, I0, X0, ind = (
                                         self.N, 
                                         self.t0, 
                                         self.tq,
                                         self.tf,
                                         self.I0,
                                         self.X0,
                                         self.ind
                                        )
                S0 = N - I0 - X0
                C0 = 0.
                state0 = np.array([S0, C0, I0, X0])
                # Integrate the SCIR equations over a time grid before 
                # confinement (first regime)
                t = np.arange(t0, tf, step)
                ret1 = odeint(SCIR, state0, t[:int(tq / step)], 
                              args=(N, beta, q * 0, p * 0, rmu))
                # Integrate the SCIR equations over a time grid, after
                # confinement (second regime)
                ret2 = odeint(SCIR, ret1[-1, :], t[int(tq / step):], 
                              args=(N, beta, q, p, rmu))
                ret = np.concatenate((ret1, ret2))
                # S, C, I, X
                return ret[:, ind]        
        

if __name__ == '__main__':
        path = '../../../Results/ita/first_wave/simul_res/'
        res_path = '../../../Results/ita/first_wave/solve_res/'
        file_name = glob.glob1(path, '*.pkl')[0]
        samples = get_data(path + file_name)
        # Total population, N
        if samples['country'] == 'Spain':
                N = 46754783.
        elif samples['country'] == 'Italy':
                N = 60461828.
        elif samples['country'] == 'France':
                N = 65273511.

        # Data is in log scale but ode system not
        # instance
        solver_ist = Ode_system_solver(
                                       N = N, 
                                       t0 = samples['t0'] - 1, 
                                       tq = samples['tq'] - 1, 
                                       tf = samples['tf'], 
                                       I0 = np.exp(samples['I'][0]), 
                                       X0 = np.exp(samples['X'][0]), 
                                       var = 'X' 
                                      )

        # get epidemic parameters
        param = samples['beta'].flat, samples['rmu'].flat, \
                samples['p'].flat, samples['q'].flat

        parameters = [(a, b, c, d) for a, b, c, d in zip(*param)]
        p = Pool(cpu_count())
        res = p.map(solver_ist.solve_SCIR, parameters)
        p.close()

        res = np.asarray(res).T
        np.save(res_path + f'sol_{solver_ist.var}_.npy', res)
