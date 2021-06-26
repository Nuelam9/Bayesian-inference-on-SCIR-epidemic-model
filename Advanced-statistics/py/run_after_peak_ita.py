import sys
import pandas as pd
import numpy as np
sys.path.append('../modules/')
from analysis import Analysis
import matplotlib.pyplot as plt
from time import time
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../Data/dataset_ita.csv')


# instantiating an analysis object
analysis = Analysis(date=df['Day'],
                    confirmed=df['Confirmed_smooth'].to_numpy(),
                    recovered_death=df['Recovered_Death_smooth'].to_numpy(),
                    quarantine='2020.03.09',
                    last_data='2020.04.23',   # motivate choise
                    last_projection='2020.07.23',
                    peak='2020.04.23',
                    beta=[0,1],
                    rmu=[0,1],
                    q=[0,5],
                    p=[0,5],
                    tauI=[0.01, 0.01],
                    tauX=[0.01, 0.01],
                    country='Italy')

# call sampler analysis' method
analysis.sampler(nchains=10, nthreads=10, niter=10000, burn_in=0.5)
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
file = open('../Results/results_after_peak_ita.pkl', 'wb')
pickle.dump(results, file)
file.close()
print(f'{time() - t1:.4f}s')