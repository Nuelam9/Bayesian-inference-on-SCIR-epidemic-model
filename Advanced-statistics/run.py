import sys
import pandas as pd
import numpy as np
sys.path.append('./modules/')
from analysis import Analysis
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Data/dataset_esp.csv')

# instantiating an analysis object
analysis_esp = Analysis(date=df['Day'].to_numpy(),
                        confirmed=df['Confirmed'].to_numpy(),
                        recovered_death=df['Recovered_Death'].to_numpy(),
                        quarantine='2020.03.09',
                        last_data='2020.03.29',
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
analysis_esp.sampler(nchains=12, nthreads=12, niter=10000, burn_in=0.5)
samples = analysis_esp.samples

print("\nSaving simulation's results...")
# Save dictionary to file
import pickle
file = open('Results/samples_esp.pkl', 'wb')
pickle.dump(analysis_esp, file)
file.close()

# chains = analysis_esp.nchains
# iters = int(analysis_esp.niter * analysis_esp.burn_in)

# sub_param = {key: value for key, value in samples.items() if value.shape == (1,)}  # all parameters
# sub_trace = {key: value for key, value in samples.items() if value.shape == (1, iters, chains)}  # beta, rmu, p, q
# sub_fit = {key: value for key, value in samples.items() if np.sum(value.shape) > iters + chains + 1}  # y, z 
# sub_data = {key: value for key, value in samples.items() if 1 < np.sum(value.shape) < iters}  # I, X