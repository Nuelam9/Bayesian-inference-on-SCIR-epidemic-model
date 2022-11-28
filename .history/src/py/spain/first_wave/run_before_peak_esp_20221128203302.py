import sys
import pandas as pd
import numpy as np
sys.path.append('../../../modules/')
from analysis import Analysis
from time import time
import pickle
import warnings
warnings.filterwarnings('ignore')


if len(sys.argv) < 5:
    print("nchain (10), nthreads (10), niters (200000), thin(1), burn_in (0.5)")
else:
    # getting number of iterations as command-line arguments 
    nchains = int(sys.argv[1])
    nthreads = int(sys.argv[2])
    niter = int(sys.argv[3])
    thin = int(sys.argv[4])
    burn_in = float(sys.argv[5])

    # get data to fit
    df = pd.read_csv('../../../Data/dataset_esp.csv')
    df = df[df.Day >= '2020.03.01'].reset_index(drop=True)

    # instantiating an analysis object
    analysis = Analysis(date=df['Day'].to_numpy(),
                        confirmed=df['Active_cases'].to_numpy(),
                        recovered_death=df['Recovered_Death'].to_numpy(),
                        confinement='2020.03.09',
                        last_data='2020.03.29',
                        last_projection='2020.05.17', 
                        peak='2020.04.18',
                        beta=[0,1],
                        rmu=[0,1],
                        q=[0,5],
                        p=[0,5],
                        tauI=[0.01, 0.01],
                        tauX=[0.01, 0.01])

    # call sampler analysis' method
    analysis.sampler(nchains=nchains, nthreads=nthreads, niter=niter, thin=thin, burn_in=burn_in)
    results = { **analysis.data, **analysis.samples,
            'date': analysis.date,
            'peak': analysis.peak,
            'nchains': analysis.nchains,
            'niter': analysis.niter,
            'burn_in': analysis.burn_in,
            'varname': analysis.varname,
            'names': analysis.names,
            'country': 'Spain' }


    print('\n')
    print('Summary:')
    #print(analysis.summary)

    print("\nSaving simulation's results...")
    t1 = time()
    # Save dictionary to file
    filepath = "../../../Results/esp/first_wave/"
    filename = f"results_after_peak_esp_{niter}.pkl"
    with open(f'../../../Results/esp/first_wave/results_before_peak_esp_{niter}_{thin}.pkl', 'wb') as file:
        pickle.dump(results, file)
    print(f'{time() - t1:.4f}s')

# chains = analysis_esp.nchains
# iters = int(analysis_esp.niter * analysis_esp.burn_in)

# sub_param = {key: value for key, value in samples.items() if value.shape == (1,)}  # all parameters
# sub_trace = {key: value for key, value in samples.items() if value.shape == (1, iters, chains)}  # beta, rmu, p, q
# sub_fit = {key: value for key, value in samples.items() if np.sum(value.shape) > iters + chains + 1}  # y, z 
# sub_data = {key: value for key, value in samples.items() if 1 < np.sum(value.shape) < iters}  # I, X