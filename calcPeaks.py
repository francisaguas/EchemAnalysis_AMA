# <--- Example code to calculate peak heights for a dataset --->
# Sample data for Concentration Curve: C-COM1, C-COM2, C-COM3 dataset
# Sample data for Frequency Map: F-COM1,F-COM2, F-COM3

import frequencyMaps as f
import concentrationCurves as c
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

time_zero = time.time()  # start time of execution

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Settings
folders = ['F-COM1','F-COM2','F-COM3'] # List all electrode folders of interest in quotes, separated by commas
dataType = 'SWV' # String identifier for selecting files of interest, case-sensitive
save = True # Set to True to save peak heights/areas in a file
resolution = 20 # Value is the percent of the voltammogram length you would like to skip by --> 100 means no skipping

# Import directory
electrodesFolders = []
for electrode in folders: # Gets the folder paths of all electrodes of interest
    electrodesFolders.append(os.path.join(os.getcwd(), electrode))

# Choose to analyze data for a Concentration Curve (multiple concentrations at one frequency)
"""Concentration Curve""" # comment out if not being used
# frequency = "10Hz" # Indicate which frequency to analyze
# experimentConditions = ['0 fg/mL S1','0.1 fg/mL S1','1 fg/mL S1','10 fg/mL S1','50 fg/mL S1','100 fg/mL S1'] # String identifiers for selecting files of interest, case-sensitive
# labels = [ 0,0.1,1,10,50,100] # integer form of experimentConditions variable
#
# # Choose peaksMatrix function based on input file type (for TXT, include parameter for index splicing e.g. [0,1] --> [xIndex,yIndex])
# # [rawPeaks,areas] = c.peaksMatrixTXT(dataType, frequency,electrodesFolders,experimentConditions,[0,1],resolution,save) # Calculates the peaks for every electrode
# [rawPeaks,areas] = c.peaksMatrixDTA(dataType, frequency,electrodesFolders,experimentConditions,resolution,save) # Calculates the peaks for every electrode
#
# print('Raw Peaks\n',rawPeaks)
# print('Areas\n',areas)

# Choose to analyze data for a Frequency Map (multiple frequencies with concentration overlayed)
"""Frequency Map""" # comment out if not being used
# Calculate Raw Peaks
identifiers = ['MCH','HSA'] # String identifiers for experiment conditions , case-sensitive
rawPeaks = f.justPeaksFreq(electrodesFolders, dataType, identifiers, resolution, save)
print('Raw Peaks \n', rawPeaks)

if save: # save to Excel files
    filename1 = 'samplePeaksFreq.csv' # choose file name for peak heights
    rawPeaks.to_csv(os.path.join(os.getcwd(), filename1),index=None,mode='a')
    # filename2 = 'sampleAreas.csv' # choose file name for areas
    # areas.to_csv(os.path.join(os.getcwd(), filename2),index=None,mode='a')

t1 = time.time()  # end time of execution
print('-'*80)
print("Took "+ str(t1-time_zero)+' seconds to complete execution.')
print('-'*80)

# plt.show() # uncomment to view figures after code runs. WARNING: > 50 figures won't appear