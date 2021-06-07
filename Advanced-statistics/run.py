import sys
import pandas as pd
import numpy as np
sys.path.append('./modules/')
from analysis import Analysis
from time import time
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Data/dataset_esp.csv')
# df = pd.read_csv('Data/dataset_ita.csv') 

# instantiating an analysis object
analysis = Analysis(date=df['Day'].to_numpy(),
                    confirmed=df['Confirmed'].to_numpy(),
                    recovered_death=df['Recovered_Death'].to_numpy(),
                    quarantine='2020.03.09',  # change
                    last_data='2020.03.29',  # change
                    last_projection='2020.05.17',  # change 
                    peak='2020.04.18',  # change
                    beta=[0,1],
                    rmu=[0,1],
                    q=[0,5],
                    p=[0,5],
                    tauI=[0.01, 0.01],
                    tauX=[0.01, 0.01],
                    country='Spain')  # change

# call sampler analysis' method
analysis.sampler(nchains=10, nthreads=10, niter=30000, burn_in=0.5)
samples = analysis.samples

print("\nSaving simulation's results...")
t1 = time()
# Save dictionary to file
import pickle
file = open('Results/samples_esp_1.pkl', 'wb')  # change
pickle.dump(analysis, file)
file.close()
print(time() - t1)

# chains = analysis_esp.nchains
# iters = int(analysis_esp.niter * analysis_esp.burn_in)

# sub_param = {key: value for key, value in samples.items() if value.shape == (1,)}  # all parameters
# sub_trace = {key: value for key, value in samples.items() if value.shape == (1, iters, chains)}  # beta, rmu, p, q
# sub_fit = {key: value for key, value in samples.items() if np.sum(value.shape) > iters + chains + 1}  # y, z 
# sub_data = {key: value for key, value in samples.items() if 1 < np.sum(value.shape) < iters}  # I, X