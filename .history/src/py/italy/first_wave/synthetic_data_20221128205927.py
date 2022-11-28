import sys
import pandas as pd
sys.path.append('../../../modules/')
from analysis import Analysis
from utils import *
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

    # get data with time
    df = pd.read_csv('../../../Data/dataset_ita.csv')

    # get the dictonary containing all parameters values
    filehandler = open('../../../Results/ita/first_wave/results_before_peak_ita_200000.pkl', 'rb')
    res = pickle.load(filehandler)
    # numeric solutino of SCIR system with median of parameters fitted 
    t, *states = solve_SCIR(res)
    # get the synthetic data for Active cases and new Death+Recovered 
    I, X = states[2][::100], states[3][::100]

    # instantiating an analysis object
    analysis = Analysis(date=df['Day'],
                        confirmed=I,
                        recovered_death=X,
                        confinement='2020.03.09',
                        last_data='2020.04.23',   # motivate choice
                        last_projection='2020.07.23',
                        peak='2020.04.23',
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
    # Save dictionary to file
    filepath = "../../../Results/ita/first_wave/"
    filename = "results_synthetic_ita.pkl"
    with open(filepath + filename, 'wb') as file:
        pickle.dump(results, file)
    print(f'{time() - t1:.4f}s')




