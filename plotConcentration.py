# <--- Example code to plot concentration curve for C-COM1, C-COM2, C-COM3 dataset --->

import concentrationCurves as s
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

time_zero = time.time()  # start time of execution

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

"""Data Analysis"""
folders = ['C-COM1','C-COM2','C-COM3'] # List all electrode folders of interest
labels = [ 0,0.1,1,10,50,100]
frequency = '10Hz' # for labeling
experimentConditions = ['0 S1','0.1 fg/mL S1', '1 fg/mL S1','10 fg/mL S1','50 fg/mL S1', '100 fg/mL S1']
save = True

# Load saved excel files
filename1 = 'samplePeaksCon.csv'
rawPeaks = pd.read_csv(os.path.join(os.getcwd(), filename1),sep=',')
print('Raw Peaks \n', rawPeaks)

filename2 = 'sampleAreasCon.csv'
areas = pd.read_csv(os.path.join(os.getcwd(), filename2),sep=',')
print('Areas \n', areas)

# Normalization
referenceMeasurement = 0  # Indicate index of measurement number to normalize to [NOTE: Python indexing starts at 0]
normSignal = s.normalizeCol(rawPeaks, referenceMeasurement, experimentConditions)
print('Normalized Change \n',normSignal)

averages = s.stats(normSignal)
print('Averages')
print(averages)

"""Plot Concentration Curve"""
s.plotPeaks(rawPeaks,labels,folders,save) # Peak Height Concentration Curve
s.plotAreas(areas,labels,folders,save) # Peak Area Concentration Curve
s.plotConcentration(averages, labels,folders,frequency,save) # Normalized, Averaged Concentration Curve

t1 = time.time()  # end time of execution
print('-'*80)
print("Took "+ str(t1-time_zero)+' seconds to complete execution.')
print('-'*80)

plt.show()