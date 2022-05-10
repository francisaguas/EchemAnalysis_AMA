# <--- Example code to plot frequency map for F-COM1, F-COM2, F-COM3 dataset --->

import frequencyMaps as s
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
time_zero = time.time()  # start time of execution

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

"""Data Analysis"""
# Settings
folders = ['F-COM1','F-COM2','F-COM3'] # List all electrode folders of interest
experimentlabels = ['MCH', 'HSA']
save = True

# Load Excel file of saved raw peaks
filename = 'samplePeaksFreq.csv'
rawPeaks = pd.read_csv(os.path.join(os.getcwd(), filename),sep=',')
print('Raw Peaks \n', rawPeaks)

"""Normalization Calculation
        Three options: normalizeCol(), normalizeRows(), customNormalize() 
        --> see 'frequencyMaps.py' for documentation
"""
c=len(experimentlabels)
norm = s.customNormalize(rawPeaks, c)
print(' Normalized Change [%] \n', norm)
averages = s.stats(norm)
print(' Averages, SE, STD \n', averages)

"""Plot Frequency Maps"""
# Go to functions in 'frequencyMaps.py' to customize figure formatting
# Raw Peaks
s.plotFreq(rawPeaks,experimentlabels,save)
# Normalized Change
n=len(folders)
s.plotFreqN(averages,n,experimentlabels,save,1) # Last parameter indicates error type: 1=standard error, 2 = standard deviation

# Save to File
if save: # save to Excel files
    filename1 = 'sampleNormalizedData.csv' # choose file name
    norm.to_csv(os.path.join(os.getcwd(), filename1),index=None,mode='a')
    filename2 = 'sampleAverages.csv'
    averages.to_csv(os.path.join(os.getcwd(), filename2), index=None, mode='a')

t1 = time.time()  # end time of execution
print('-'*80)
print("Took "+ str(t1-time_zero)+' seconds to complete execution.')
print('-'*80)

plt.show()