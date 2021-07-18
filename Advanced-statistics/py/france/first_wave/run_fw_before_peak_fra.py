import sys
import pandas as pd
import numpy as np
sys.path.append('../../../modules/')
from analysis import Analysis
import matplotlib.pyplot as plt
from time import time
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../../../Data/dataset_fra.csv')
df = df[df.Day >= '2020.03.03']

if len(sys.argv) < 5:
    print("nchain (10), nthreads (10), niters (200000), burn_in (0.5)")
else:
    # getting number of iterations as command-line arguments 
    nchains = int(sys.argv[1])
    nthreads = int(sys.argv[2])
    niter = int(sys.argv[3])
    burn_in = float(sys.argv[4])

    # instantiating an analysis object
    analysis = Analysis(date=df['Day'].to_numpy(),
                        confirmed=df['Active_cases_smooth'].to_numpy(),
                        recovered_death=df['Recovered_Death_smooth'].to_numpy(),
                        quarantine='2020.03.12',
                        last_data='2020.04.12',
                        last_projection='2020.05.17', 
                        peak='2020.04.28',
                        beta=[0,1],
                        rmu=[0,1],
                        q=[0,5],
                        p=[0,5],
                        tauI=[0.01, 0.01],
                        tauX=[0.01, 0.01])

    # call sampler analysis' method
    analysis.sampler(nchains=nchains, nthreads=nthreads, niter=niter, burn_in=burn_in)

    print('\n')
    print('Summary:')
    print(analysis.summary)

    print("\nSaving simulation's results...")
    t1 = time()

    results = { **analysis.data, **analysis.samples,
            'date': analysis.date,
            'peak': analysis.peak,
            'nchains': analysis.nchains,
            'niter': analysis.niter,
            'burn_in': analysis.burn_in,
            'varname': analysis.varname,
            'names': analysis.names,
            'country'='France'}

    # Save dictionary to file
    import pickle
    file = open('../../../Results/fra/first_wave/results_before_peak_fra.pkl', 'wb')
    pickle.dump(results, file)
    file.close()
    print(f'{time() - t1:.4f}s')

# chains = analysis_esp.nchains
# iters = int(analysis_esp.niter * analysis_esp.burn_in)

# sub_param = {key: value for key, value in samples.items() if value.shape == (1,)}  # all parameters
# sub_trace = {key: value for key, value in samples.items() if value.shape == (1, iters, chains)}  # beta, rmu, p, q
# sub_fit = {key: value for key, value in samples.items() if np.sum(value.shape) > iters + chains + 1}  # y, z 
# sub_data = {key: value for key, value in samples.items() if 1 < np.sum(value.shape) < iters}  # I, X
