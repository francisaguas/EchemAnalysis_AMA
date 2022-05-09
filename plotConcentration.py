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
save = False

# Load saved excel files
filename1 = 'samplePeaksCon.csv'
rawPeaks = pd.read_csv(os.path.join(os.getcwd(), filename1),sep=',')
print('Raw Peaks \n', rawPeaks)

filename2 = 'Amanda Peak Areas.csv'
areas = pd.read_csv(os.path.join(os.getcwd(), filename2),sep=',')
print('Areas \n', areas)

"""Plot Concentration Curve"""
s.plotPeaks(rawPeaks,labels,filename1,save)
s.plotX(combo,labels,filename3,save)

t1 = time.time()  # end time of execution
print('-'*80)
print("Took "+ str(t1-time_zero)+' seconds to complete execution.')
print('-'*80)

plt.show()