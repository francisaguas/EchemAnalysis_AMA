# <--- Example code to calculate peak heights for an Excel dataset --->

import concentrationCurves as s
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import numpy as np

t0 = time.time()
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
plt.rcParams.update({'figure.max_open_warning': 0})

# Settings
folder = 'Jan_28' # Name of main folder with Excel files
electrodeNames = [['E1 ','E2 ','E7 ','E9 ','F25 '],['F48 ','F49 ','F50 ','F51 ','F52 '],
                  ['F2 ','F4 ','F45 ','F46 ','F47 '],['F1 ','F41 ','F42 ','F43 ','F44 '],
                  ['F27 ','F36 ','F37 ','F38 ','F40 ']] # Electrode names of interest (case-sensitive), grouped according to experiment condition
frequency = 10 # Integer form of frequency of interest
experimentConditions = ['saliva', '1 GEmL','1E3 GEmL','1E5 GEmL', '1E7 GEmL']
labels = [0, 1, 1000,100000,10000000] # Interger form of experimentConditions
save = False # Set to True to save peak heights/areas in a file
resolution = 10 # Value is the percent of the voltammogram length you would like to skip by --> 100 means no skipping

# Extract files of interest
ti = time.time()
xIndex, yIndex = s.parseFreq(frequency) # identify columns associated with frequency
electrodeFiles = [] # keep track of all electrode files of interest
for file in os.listdir(folder): # loop through all files in folder
    if file.endswith('.xlsx'):
            electrodeFiles.append(file) # store all file paths of electrodes of interest
print(electrodeFiles)

# Store final peaks/areas
# Here data was organized into separate matrices for MCH and Inactivated Virus condition for further analysis
mchPeaks = []
mchAreas = []
concPeaks = []
concAreas = []
for batch in electrodeNames:
    # Store peaks/areas for each group of electrodes temporarily
    mPeaks = []
    mAreas = []
    cPeaks = []
    cAreas = []
    for electrode in batch:
        count =0
        for file in electrodeFiles:
            if 'mch' in file and electrode in file: # for MCH conditions
                [p,A] = s.peaksExcel(file, xIndex, yIndex, folder, resolution)
                mPeaks.append(p)
                mAreas.append(float(A))
                # electrodeFiles.remove(file)
                count +=1
            elif 'mch' not in file and electrode in file: # for Inactivated Virus conditions
                [p, A] = s.peaksExcel(file, xIndex, yIndex, folder, resolution)
                cPeaks.append(p)
                cAreas.append(float(A))
                # electrodeFiles.remove(file)
                count+=1
            if count ==2: break
    mchPeaks.append(mPeaks)
    mchAreas.append(mAreas)
    concPeaks.append(cPeaks)
    concAreas.append(cAreas)
print(mchPeaks)
print(mchAreas)
print(concPeaks)
print(concAreas)

# Convert all matriced to dataframes
# mchPeaks= np.transpose(mchPeaks) # if needed
refPeaks = pd.DataFrame(mchPeaks)
print('MCH Peaks\n', refPeaks)

refAreas = pd.DataFrame(mchAreas)
print('MCH Areas\n', refAreas)

# concPeaks=np.transpose(concPeaks) # if needed
allPeaks = pd.DataFrame(concPeaks)
print('Concentration Peaks\n', allPeaks)

allAreas = pd.DataFrame(concAreas)
print('Concentration Areas\n', allAreas)

if save: # save to Excel files
    filename1 = 'MCH Peaks Data.csv' # choose file name for peak heights
    refPeaks.to_csv(os.path.join(os.getcwd(), filename1),index=None,mode='a')
    filename2 = 'All Peak Data.csv' # choose file name for areas
    allPeaks.to_csv(os.path.join(os.getcwd(), filename2),index=None,mode='a')
    # Repeat for Area data if needed

print('Time taken for baseline calculation at ' + str(frequency) + ':', time.time() - ti)
t1 = time.time()

print('-'*80)
print("Took "+ str(t1-t0)+' seconds to complete execution.')
print('-'*80)

plt.show()