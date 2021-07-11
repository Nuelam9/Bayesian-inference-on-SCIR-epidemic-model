import sys
import pandas as pd
import numpy as np
sys.path.append('../../modules/')
from analysis import Analysis
import matplotlib.pyplot as plt
from time import time
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../../Data/dataset_esp.csv')

if len(sys.argv) < 5:
    print("nchain (10), nthreads (10), niters (100000), burn_in (0.5)")
else:
    # getting number of iterations as command-line arguments 
    nchains = int(sys.argv[1])
    nthreads = int(sys.argv[2])
    niter = int(sys.argv[3])
    burn_in = float(sys.argv[4])

    # instantiating an analysis object
    analysis = Analysis(date=df['Day'].to_numpy(),
                        confirmed=df['Confirmed'].to_numpy(),
                        recovered_death=df['Recovered_Death'].to_numpy(),
                        quarantine='2020.03.09',
                        last_data='2020.04.18',
                        last_projection='2020.05.17',
                        peak='2020.04.18',
                        beta=[0,1],
                        rmu=[0,1],
                        q=[0,5],
                        p=[0,5],
                        tauI=[0.01, 0.01],
                        tauX=[0.01, 0.01],
                        country='Spain')

    # call sampler analysis' method
    analysis.sampler(nchains=nchains, nthreads=nthreads, niter=niter, burn_in=burn_in)
    results = { **analysis.data, **analysis.samples,
            'date': analysis.date,
            'peak': analysis.peak,
            'nchains': analysis.nchains,
            'niter': analysis.niter,
            'burn_in': analysis.burn_in,
            'country': analysis.country,
            'varname': analysis.varname,
            'names': analysis.names }


    print('\n')
    print('Summary:')
    print(analysis.summary)

    print("\nSaving simulation's results...")
    t1 = time()
    # Save dictionary to file
    import pickle
    file = open('../../Results/esp/results_after_peak_esp.pkl', 'wb')
    pickle.dump(results, file)
    file.close()
    print(f'{time() - t1:.4f}s')
