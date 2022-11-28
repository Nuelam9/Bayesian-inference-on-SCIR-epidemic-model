import sys
import pandas as pd
import numpy as np
sys.path.append('../../../modules/')
from analysis import Analysis
import pickle
from time import time
import warnings
warnings.filterwarnings('ignore')


if len(sys.argv) < 5:
    print("nchain (10), nthreads (10), niters (200000), burn_in (0.5)")
else:
    # getting number of iterations as command-line arguments 
    nchains = int(sys.argv[1])
    nthreads = int(sys.argv[2])
    niter = int(sys.argv[3])
    burn_in = float(sys.argv[4])

    # get data to fit
    df = pd.read_csv('../../../Data/dataset_ita.csv')
    df = df[df.Day > '2020.08.18'].reset_index(drop=True)

    # instantiating an analysis object
    analysis = Analysis(date=df['Day'],
                        confirmed=df['Active_cases_smooth'].to_numpy(),
                        recovered_death=df['Recovered_Death_smooth'].to_numpy(),
                        confinement='2020.10.28',
                        last_data='2020.11.27',   # motivate choise
                        last_projection='2021.03.09',
                        peak='2020.11.27',
                        beta=[0,1],
                        rmu=[0,1],
                        q=[0,5],
                        p=[0,5],
                        tauI=[0.01, 0.01],
                        tauX=[0.01, 0.01])

    # call sampler analysis' method
    analysis.sampler(nchains=nchains, nthreads=nthreads, niter=niter, burn_in=burn_in)
    results = { **analysis.data, **analysis.samples,
            'date': analysis.date,
            'peak': analysis.peak,
            'nchains': analysis.nchains,
            'niter': analysis.niter,
            'burn_in': analysis.burn_in,
            'varname': analysis.varname,
            'names': analysis.names,
            'country': 'Italy' }

    print('\n')
    print('Summary:')
    print(analysis.summary)

    print("\nSaving simulation's results...")
    t1 = time()
    filepath = "../../../Results/ita/second_wave/"
    filename = "results_after_peak_ita.pkl"
    with open(filepath + filename, 'wb') as file:
        pickle.dump(results, file)
    print(f'{time() - t1:.4f}s')
