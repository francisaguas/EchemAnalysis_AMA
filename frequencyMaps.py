import pandas as pd
import gamry_parser as parser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.ticker as ticker
import os
from scipy.signal import argrelmin
from scipy.signal import savgol_filter
import scipy
import natsort
from _collections import defaultdict
from mpmath import mp
import math

# <---------------- Analysis Functions ------------>

"""
justPeaksFreq -- calculates peaks across frequencies (columns) and electrodes (rows)
inputs - dataType --> string of measurement type (e.g. DPV, SWV, ACV) used to filter file names, case-sensitive
        electrodesFolders --> list of electrode file paths
        identifiers --> string identifiers/conditions for selecting files of interest
        resolution --> integer used for AMA sampling resolution
output - dataframe of peak heights where CONDITIONS are collated by rows
        ex)  F-COM1, MCH -- row 1
             F-COM2, MCH -- row 2
             F-COM3, MCH -- row 3
             F-COM1, HSA -- row 4
             F-COM2, HSA -- row 5
             F-COM3, HSA -- row 6
"""
def justPeaksFreq(electrodesFolders,dataType,identifiers,resolution,save):
    peaksMatrix =[] # Store the peaks for each electrode
    for element in identifiers: # loop through data for each identifier/condition
        for electrode in electrodesFolders: # search through each electrode folder
            # Extract files of interest from each electrode folder
            fileList = extractConcentration(electrode, dataType, element)
            print(fileList)
            peaksMatrix.append(peaksDTA(fileList, electrode, resolution, save)) # calculate peaks
        freq_list = frequencies(fileList, electrode, dataType) # extract frequencies from Gamry files for labeling
    print(freq_list)
    # print(peaksMatrix)
    peaksMatrix = pd.DataFrame(peaksMatrix, columns=freq_list)
    return peaksMatrix

"""
justPeaksTemp -- calculates peaks across temperatures (columns) and electrodes (rows)
inputs - freq --> string of frequency of interest used to filter file names, case-sensitive
        electrodesFolders --> list of electrode file paths
        identifiers --> string identifiers/conditions for selecting files of interest
        temps --> string list of temperatures for labeling
        resolution --> integer used for AMA sampling resolution
output - dataframe of peak heights where CONDITIONS are collated by rows
"""
def justPeaksTemp(electrodesFolders,freq,identifiers,temps,resolution,save):
    peaksMatrix = [] # Store the peaks for each electrode
    for element in identifiers: # loop through data for each identifier/condition
        for electrode in electrodesFolders: # search through each electrode folder
            # Extract files of interest from each electrode folder
            fileList = extractConcentration(electrode, freq,element)
            print(fileList)
            peaksMatrix.append(peaksDTA(fileList, electrode, resolution,save))
    peaksMatrix = pd.DataFrame(peaksMatrix, columns=temps)
    return peaksMatrix

# Generate file list of all frequencies of a single identifier/condition/concentration
def extractConcentration(electrodeFolder, dataType,concentration):
    allFiles = os.listdir(electrodeFolder) # All the files in folder
    filesList = [] # Store files of interest
    for file in allFiles:  # Loop through each file in folder
        if file.endswith('.dta') or file.endswith('.DTA'):  # Only take DTA files
            if dataType in file and concentration in file: # checks if data type (DPV, etc.) is in file name
                if '#' not in file: # can add additional identifiers as needed
                        filesList.append(file) # adds file name to list
    filesList = natsort.natsorted(filesList)
    #print(filesList)
    return filesList

# To extract list of frequencies data was measured at
def frequencies(filesList, electrodesFolder,dataType):
    if dataType == 'SWV': # For Gamry DTA files, frequency is extractable from data
        gp = parser.GamryParser()
        freq_list = []
        for i, entry in enumerate(filesList):
            # print(electrodesFolder,entry)
            entryPath = os.path.join(electrodesFolder, entry)
            # print(entryPath)
            gp.load(filename=entryPath)
            header = gp.get_header()
            freq_list.append(header['FREQUENCY'])
    else: # hard code if frequencies aren't saved into datasets
        freq_list = [25,50,75]
    return freq_list


"""
peaks functions: calculates peaks using AMA
inputs: filesList --> list of  files of interest
        electrodeFolder --> filepath of folder containing  files
        resolution --> integer used for AMA sampling resolution
outputs: peakHeights --> returns a list peak heights
         areas --> returns a list of areas
"""

# for input file type: .DTA (calculates peaks for ALL files in electrode folder)
def peaksDTA(filesList, electrodeFolder, resolution, save):
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    peakHeights = []
    areas = []
    n = 0
    while n < len(filesList):
        filename = filesList[n]
        print(filename)
        gp = parser.GamryParser()
        entryPath = os.path.join(electrodeFolder, filename)
        gp.load(filename=entryPath)
        data = gp.get_curve_data()

        yVals = data['Idif']  # units A
        xVals = data['Vstep']  # units = V
        # plt.figure(figsize=(10,7))
        # plt.title('Raw %s Data' %(filename))
        # plt.xlabel('Vfwd [V]',fontsize=16)
        # plt.ylabel('Idif [A]',fontsize=16)
        # plt.scatter(xVals,yVals)

        # Remove non significant data points
        offset = 0
        newY = yVals[offset:len(yVals)-offset]
        newX = xVals[offset:len(yVals)-offset]

        # plt.figure()
        # plt.title('% s Data Without First 5 Points' %filename)
        # plt.xlabel('Vfwd [V]')
        # plt.ylabel('Idif [A]')
        # plt.plot(newX,newY)
        # plt.xlim([xVals[0],xVals[len(xVals)-1]])

        # Add indices to numpy series to allow for indexing later
        ind = list(range(0, len(newY)))
        newY.index = [ind]
        newX.index = [ind]

        # Median Filter
        newY = scipy.ndimage.filters.median_filter(newY, size=11)

        # Smooth out raw data
        smoothY = scipy.signal.savgol_filter(newY, 17, 3)
        # plt.figure()
        # plt.title('Smoothed %s Data' % filename)
        # plt.xlabel('Vfwd [V]')
        # plt.ylabel('Idif [A]')
        # plt.plot(newX, smoothY)

        # firstDeriv = np.gradient(smoothY)
        # firstDeriv = scipy.signal.savgol_filter(firstDeriv, 25, 3)
        # plt.figure()
        # plt.title('Derivative of DPV %s Data' % filename)
        # plt.xlabel('Vfwd [V]')
        # plt.ylabel('Idif [A]')
        # plt.plot(newX, firstDeriv)

        # secondDeriv = np.gradient(firstDeriv)
        # secondDeriv = scipy.signal.savgol_filter(secondDeriv, 25, 3)
        # plt.figure()
        # plt.title('Second Derivative of DPV %s Data' %filename)
        # plt.xlabel('Vfwd [V]')
        # plt.ylabel('Idif [A]')
        # plt.plot(newX,secondDeriv)

        numPoints = len(smoothY)
        skipp_by = (int)(numPoints / resolution)
        if skipp_by ==0: skipp_by =1
        peaksDTA.area_found = defaultdict(float)  # makes a dictionary to keep track of currentArea already calculated to save time
        maxArea = None  # Maximum currentArea for voltammogram

        # Plot smoothed data
        # plt.figure(figsize=(10, 7))
        # plt.title('%s'%resolution)
        # plt.xlabel('Vfwd [V]', fontsize=16)
        # plt.ylabel('Idif [A]', fontsize=16)
        # plt.plot(newX, smoothY, '-', linewidth='2')

        # Loop through every X point in voltammogram
        for x, y in zip(newX[1::skipp_by], smoothY[1::skipp_by]):
            # Exclude first pair which is the same point repeated
            array_of_x = np.full(shape=numPoints - 1, fill_value=x)
            array_of_y = np.full(shape=numPoints - 1, fill_value=y)
            #  zip1 = an array of a given data point repeated
            zip1 = zip(array_of_x, array_of_y)  # ex: --> [ (x, y), (x,y), ... ]

            # zip2 = an array of all data point coordinates considered
            zip2 = zip(newX[1::skipp_by], smoothY[1::skipp_by])  # ex: --> [ (x0, y0), (x1, y1), ..., (xN, yN) ]

            # Potential anchor point combinations for a given point
            possibleBaselines = zip(zip1, zip2)
            currentMaxArea = None  # Maximum currentArea for a given point. Assigned to the first currentArea by default within the loop
            baselineCounter = 0

            # Plot all baselines
            # plt.figure(figsize=(10, 7))
            # plt.title('%f,%f Baselines' % (x,y))
            # plt.xlabel('Vfwd [V]', fontsize=16)
            # plt.ylabel('Idif [A]', fontsize=16)
            # plt.plot(newX, smoothY, '-')
            # plt.plot(x, y, 'o',markersize = 6, color='currentBest_rightIndex')
            # plt.ylim([4.3*(10**-8),6.6*(10**-8)])

            # Loop through the possible baselines for a given point
            for pair in possibleBaselines:
                baselineCounter += 1
                # Extract coordinates for each anchor point
                anchor1, anchor2 = pair
                x1, y1 = anchor1
                x2, y2 = anchor2

                # Find indexes of anchor points in voltammogram
                leftIndex = np.where(newX == x1)[0][0]
                rightIndex = np.where(newX == x2)[0][0]
                if leftIndex > rightIndex: continue # exclude baselines at tail ends of voltammogram

                # Generate baseline using anchor1 and anchor 2
                try:
                    # Calculate slope using high-precision math
                    m = mp.fdiv(mp.fsub(y2, y1, dps=20), mp.fsub(x2, x1, dps=20), dps=20)
                except ZeroDivisionError:
                    m = 0.0
                b = mp.fsub(y1, mp.fmul(m, x1,dps=20), dps=20)
                baseline = m * newX + b
                peaksDTA.m = m

                # Plot all baselines
                # plt.plot(newX[leftIndex:rightIndex+1], baseline[leftIndex:rightIndex+1], linestyle = 'dotted', color = 'orange')
                # plt.plot(x2, y2, '.',markersize = 10, color='r')

                # Baseline correction calculation
                correctedY = smoothY - baseline

                # Check if currentArea has already been calculated for the given anchor points
                if ((x1,y2),(x2,y2)) in peaksDTA.area_found or ((x2, y2), (x1, y1)) in peaksDTA.area_found:
                    currentArea = peaksDTA.area_found[((x1, y2), (x2, y2))]
                else: # Calculate currentArea under voltammogram if not found in dictionary
                    currentArea = np.trapz(correctedY[leftIndex: rightIndex])
                    # Save calculated currentArea in dictionary
                    peaksDTA.area_found[((x1, y2), (x2, y2))] = currentArea
                    peaksDTA.area_found[((x2, y2), (x1, y1))] = currentArea

                # Save the indexes,area,baseline of best baseline FOR A GIVEN POINT (aka. "current best baseline")
                if currentMaxArea is None or currentArea > currentMaxArea:
                    currentBest_leftIndex = leftIndex
                    currentBest_rightIndex = rightIndex
                    currentMaxArea = currentArea
                    currentBest_Baseline = baseline

            # Continues for the same given point in the voltammogram
            # Baseline Correction
            correctedY = smoothY - currentBest_Baseline

            # Check if area has already been calculated for the given anchor points
            if ((x, y), (currentBest_leftIndex, currentBest_rightIndex)) in peaksDTA.area_found or (
                    (currentBest_leftIndex, currentBest_rightIndex), (x, y)) in peaksDTA.area_found:
                area = peaksDTA[((x, y), (currentBest_leftIndex, currentBest_rightIndex))]
            else:  # Calculate area under voltammogram if not found in dictionary
                area = np.trapz(correctedY[currentBest_leftIndex:currentBest_rightIndex])
                # Save calculated area in dictionary
                peaksDTA.area_found[((x, y), (currentBest_leftIndex, currentBest_rightIndex))] = area
                peaksDTA.area_found[((currentBest_leftIndex, currentBest_rightIndex), (x, y))] = area

            # Save the indexes,area,baseline of best baseline FOR THE WHOLE VOLTAMMOGRAM
            if maxArea is None or maxArea < area:
                bestLeftIndex = currentBest_leftIndex
                bestRightIndex = currentBest_rightIndex
                bestBaseline = currentBest_Baseline
                maxArea = area
                finalCorrectedY = correctedY
                areas.append(maxArea)

        # Finished looping through voltammogram. Optimal baseline has been found at this point
        # Subtract baseline from voltammogram
        difference = [ogVals - baseVals for ogVals, baseVals in zip(smoothY, bestBaseline)]

        # Calculate peak as maximum difference between voltammogram and baseline
        subset = difference[bestLeftIndex:bestRightIndex + 1]
        newPeakHeight = np.amax(subset)
        difference = np.array(difference, dtype=float)
        newPeakHeightInd = np.argwhere(difference == newPeakHeight)[0][0]

        if save: # Save plot of each voltammogram with peak height
            plt.figure(figsize=(10,7))
            plt.title('%s Data' % filename)
            plt.xlabel('Vfwd [V]')
            plt.ylabel('Idif [A]')
            plt.plot(newX, smoothY)
            plt.plot(newX, bestBaseline)
            plt.plot(newX[newPeakHeightInd], smoothY[newPeakHeightInd], '*')
            plt.annotate(newPeakHeight, (newX[newPeakHeightInd], smoothY[newPeakHeightInd]))
            plt.tight_layout()
            plt.savefig(f'{filename} Data.png')
            # plt.show()

        if not type(newPeakHeight) is str:
            nA = round(newPeakHeight * (10 ** 9), 3)
            peakHeights.append(nA)
        else:
            nA = 0
            peakHeights.append(nA)
            maxArea = 0
        areas.append(float(maxArea))
        n += 1
    # print("number of baselines considered:", baselineCounter * numPoints * len(filesList))
    return peakHeights

# for input file type: .TXT (calculates peak for ONE file)
""" inputs: xIndex --> the column in txt sheet containing x values
            yIndex --> the column in txt sheet containing y values
"""
def peaksTXT(file, xIndex,yIndex,folder,resolution,save):
    entryPath = os.path.join(folder, file)
    data = pd.read_excel(entryPath,engine='openpyxl',sheet_name='Sheet1')
    data_df = pd.DataFrame(data[1:])

    filename = file[0:len(file)-5]


    yVals = (data_df.iloc[:,yIndex]).values # units A
    xVals = (data_df.iloc[:,xIndex]).values # units = V
    # print(type(yVals))
    # plt.figure()
    # plt.title('Raw %s Data' %(filename))
    # plt.xlabel('Vfwd [V]')
    # plt.ylabel('Idif [A]')
    # # plt.scatter(xVals,yVals)
    # plt.plot(xVals, yVals)

    # Remove non significant data points
    offset = 5
    newY = np.array(yVals[offset:len(yVals)-offset])
    newX = np.array(xVals[offset:len(yVals)-offset])

    # Replace Nan with '0'
    if len(np.argwhere(newX != newX)) > 0:
        starts_with_nan = np.argwhere(newX != newX)[0][0]
        newX = newX[:starts_with_nan]
        newY = newY[:starts_with_nan]
    # print(type(newY))
    # print(newX,newY)
    # plt.figure()
    # plt.title('% s Data Without First 5 Points' %filename)
    # plt.xlabel('Vfwd [V]')
    # plt.ylabel('Idif [A]')
    # plt.plot(newX,newY)
    # plt.xlim([xVals[0],xVals[len(xVals)-1]])

    # Add indices to numpy series to allow for indexing later
    ind = list(range(0, len(newY)))
    # newY.index = [ind]
    # newX.index = [ind]

    # Median Filter
    # newY = scipy.ndimage.filters.median_filter(newY, size=11)
    # newY = scipy.signal.medfilt(newY,kernel_size=11)
    # plt.figure()
    # plt.title('Smoothed %s 1' % filename)
    # plt.xlabel('Vfwd [V]')
    # plt.ylabel('Idif [A]')
    # plt.plot(newX, newY)

    # Smooth out raw data
    smoothY = scipy.signal.savgol_filter(newY, 17, 3)
    # plt.figure()
    # plt.title('Smoothed %s 2' % filename)
    # plt.xlabel('Vfwd [V]')
    # plt.ylabel('Idif [A]')
    # plt.plot(newX, smoothY)

    # # Find local "minimas" for baseline
    # firstDeriv = np.gradient(smoothY)
    # firstDeriv = scipy.signal.savgol_filter(firstDeriv, 25, 3)
    # plt.figure()
    # plt.title('Derivative of DPV %s Data' % filename)
    # plt.xlabel('Vfwd [V]')
    # plt.ylabel('Idif [A]')
    # plt.plot(newX, firstDeriv)

    # secondDeriv = np.gradient(firstDeriv)
    # secondDeriv = scipy.signal.savgol_filter(secondDeriv, 25, 3)
    # plt.figure()
    # plt.title('Second Derivative of DPV %s Data' %filename)
    # plt.xlabel('Vfwd [V]')
    # plt.ylabel('Idif [A]')
    # plt.plot(newX,secondDeriv)

    newX = pd.Series(newX) # Convert x values to pandas series
    numPoints = len(smoothY)
    skipp_by = (int)(numPoints / resolution)
    if skipp_by == 0: skipp_by = 1
    peaksTXT.area_found = defaultdict(float)  # makes a dictionary to keep track of already calculated areas to save time
    maxArea = None # Maximum area for voltammogram

    # Loop through every X point in voltammogram
    for x, y in zip(newX[::skipp_by], smoothY[::skipp_by]):
        array_of_x = np.full(shape=numPoints, fill_value=x)
        array_of_y = np.full(shape=numPoints, fill_value=y)
        #  zip1 = an array of a given data point repeated
        zip1 = zip(array_of_x, array_of_y)  # ex: --> [ (x, y), (x,y), ... ]
        # zip2 = an array of all data point coordinates considered
        zip2 = zip(newX[::skipp_by], smoothY[::skipp_by])  # ex: --> [ (x0, y0), (x1, y1), ..., (xN, yN) ]

        # Potential anchor point combinations for a given point
        possibleBaselines = zip(zip1, zip2)
        currentMaxArea = None # Maximum area for a given point. Assigned to the first area by default within the loop
        baselineCounter = 0
        # plt.figure()

        # Loop through the possible baselines for a given point
        for pair in possibleBaselines:
            baselineCounter += 1
            # Extract coordinates for each anchor point
            anchor1, anchor2 = pair
            x1, y1 = anchor1
            x2, y2 = anchor2

            # Generate baseline using anchor1 and anchor 2
            try:
                # Calculate slope using high-precision math
                m = mp.fdiv(mp.fsub(y2, y1, dps=20), mp.fsub(x2, x1, dps=20), dps=20)
            except ZeroDivisionError:
                m = 0.0
            b = mp.fsub(y1, mp.fmul(m, x1, dps=20), dps=20)
            baseline = m * newX + b
            peaksDTA.m = m

            # plt.title('%s Data' % filename)
            # plt.xlabel('Vfwd [V]')
            # plt.ylabel('Idif [uA]')
            # plt.plot(newX, smoothY)
            # plt.plot(newX, baseline)

            # Baseline correction calculation
            correctedY = smoothY - baseline

            # Find indexes of anchor points in voltammogram
            leftIndex = np.where(newX == x1)[0][0]
            rightIndex = np.where(newX == x2)[0][0]

            # Check if area has already been calculated for the given anchor points
            if ((x1, y2), (x2, y2)) in peaksTXT.area_found or ((x2, y2), (x1, y1)) in peaksTXT.area_found:
                currentArea = peaksTXT.area_found[((x1, y2), (x2, y2))]
            else: # Calculate area under voltammogram if not found in dictionary
                currentArea = np.trapz(correctedY[leftIndex: rightIndex])
                # Save calculated area in dictionary
                peaksTXT.area_found[((x1, y2), (x2, y2))] = currentArea
                peaksTXT.area_found[((x2, y2), (x1, y1))] = currentArea

            # Save the indexes,area,baseline of best baseline FOR A GIVEN POINT (aka. "current best baseline")
            if currentMaxArea is None or currentArea > currentMaxArea:
                currentBest_leftIndex = leftIndex
                currentBest_rightIndex = rightIndex
                currentMaxArea = currentArea
                currentBest_Baseline = baseline

        # Continues for the same given point in the voltammogram
        # Baseline Correction
        correctedY = smoothY - currentBest_Baseline

        # Check if area has already been calculated for the given anchor points
        if ((x, y), (currentBest_leftIndex, currentBest_rightIndex)) in peaksTXT.area_found or ((currentBest_leftIndex, currentBest_rightIndex), (x, y)) in peaksTXT.area_found:
            area = peaksTXT[((x, y), (currentBest_leftIndex, currentBest_rightIndex))]
        else: # Calculate area under voltammogram if not found in dictionary
            area = np.trapz(correctedY[currentBest_leftIndex:currentBest_rightIndex])
            # Save calculated area in dictionary
            peaksTXT.area_found[((x, y), (currentBest_leftIndex, currentBest_rightIndex))] = area
            peaksTXT.area_found[((currentBest_leftIndex, currentBest_rightIndex), (x, y))] = area

        # Save the indexes,area,baseline of best baseline FOR THE WHOLE VOLTAMMOGRAM
        if maxArea is None or maxArea < area:
            bestLeftIndex = currentBest_leftIndex
            bestRightIndex = currentBest_rightIndex
            bestBaseline = currentBest_Baseline
            maxArea = area
            finalCorrectedY = correctedY

    # Finished looping through voltammogram. Optimal baseline has been found at this point
    # Subtract baseline from voltammogram
    difference = [ogVals - baseVals for ogVals, baseVals in zip(smoothY, bestBaseline)]

    # Calculate peak as maximum difference between voltammogram and baseline
    subset = difference[bestLeftIndex:bestRightIndex + 1]
    newPeakHeight = np.amax(subset)
    # temp = np.argmax(subset)
    difference = np.array(difference, dtype=float)
    newPeakHeightInd = np.argwhere(difference == newPeakHeight)[0][0]

    plt.plot(newX[newPeakHeightInd], finalCorrectedY[newPeakHeightInd], '*')
    plt.annotate(newPeakHeight, (newX[newPeakHeightInd], finalCorrectedY[newPeakHeightInd]))

    if save: # # Save plot of each voltammogram with peak height
        plt.figure(figsize=(10, 7))
        plt.title('%s Data' % filename)
        plt.xlabel('Vfwd [V]')
        plt.ylabel('Idif [uA]')
        plt.plot(newX, smoothY)
        plt.plot(newX, bestBaseline)
        plt.plot(newX[newPeakHeightInd], smoothY[newPeakHeightInd], '*')
        plt.annotate(newPeakHeight, (newX[newPeakHeightInd], smoothY[newPeakHeightInd]))
        plt.tight_layout()
        plt.savefig(f'{filename} Data.png')
        # plt.show()

    # Save peak height to list.If no peak, height = 0
    if not type(newPeakHeight) is str:
        nA = round(newPeakHeight, 5)
        peak = nA
    else:
        nA = 0
        peak = nA
        maxArea=0
        # areas.append(float(maxArea))
    # print("number of baselines considered per concentration:", counter * l * len(filesList))
    print(peak)
    return peak


"""
normalize functions: Calculates the normalized signal change of peaks 
inputs: df_peaks --> dataframe of raw peaks for all electrodes
        referenceMeasurement --> index of row/column to be used in normalization calculation
outputs: stats --> dataframe of normalized signal change for all electrodes
"""

# Normalize across adjacent columns
def normalizeCol(df_peaks,referenceMeasurement):
    normal_pks = []  # Store normalized peaks for each electrode
    freq_list = (float)(df_peaks.columns)
    counter = referenceMeasurement
    while counter < len(df_peaks.columns):  # Iterate through columns of dataframe
        reference = np.array(df_peaks.iloc[:, referenceMeasurement])
        current = np.array(df_peaks.iloc[:, counter])
        normalVal = ((current - reference) / reference)*100  # Normalization Calculation
        normal_pks.append((np.around(normalVal, 3)))  # Add normalized value to array
        counter += 1
    normal_pks = np.array(normal_pks).transpose()  # Switch rows and columns for proper dataframe size
    df_normal = pd.DataFrame(normal_pks,columns=frequencies)  # Convert matrix to dataframe
    return df_normal

# Normalize across adjacent rows
def normalizeRows(df_peaks,referenceMeasurement):
    normal_pks = []  # Store normalized peaks for each electrode
    names = list(df_peaks.columns)
    freq_list = [float(i) for i in names]
    counter = referenceMeasurement
    while counter < len(df_peaks.index):  # Iterate through rows of dataframe
        reference = np.array(df_peaks.iloc[referenceMeasurement, :])
        currentRow = np.array(df_peaks.iloc[counter, :])
        normalVal = ((currentRow - reference) / reference)*100  # Normalization Calculation
        normal_pks.append((np.around(normalVal, 3)))  # Add normalized value to array
        counter += 1
    df_normal = pd.DataFrame(normal_pks,columns=freq_list)  # Convert matrix to dataframe
    return df_normal

"""
customNormalize -- normalize across non-adjacent rows
input = raw peaks where CONDITIONS are collated by rows
output = normalize change where ELECTRODES are collated by rows
"""
def customNormalize(df_peaks,conditionNum): # c = number of conditions, how many "groups" of rows
    electrodeNum = int(len(df_peaks.index) / conditionNum) # electrodeNum = number of electrodes, how many rows per "group"
    normal_pks = []  # Store normalized peaks for each electrode
    names = list(df_peaks.columns)
    freq_list = [float(i) for i in names]
    refIndex = 0 # index for reference row
    count1 =0
    while count1 < len(df_peaks.index):  # Iterate through rows of dataframe
        reference = np.array(df_peaks.iloc[refIndex, :]) # reference row
        currentIndex = refIndex # index for current row
        count2 = 0
        while count2 < conditionNum:
           currentRow = np.array(df_peaks.iloc[currentIndex, :])
           normalVal = ((currentRow - reference) / reference)*100  # Normalization Calculation
           if normalVal.sum() != 0: # exclude normalization to same row
            normal_pks.append((np.around(normalVal, 3)))  # Add normalized value to array
           currentIndex += electrodeNum
           count2 += 1
           count1 += 1
        refIndex += 1
    df_normal = pd.DataFrame(normal_pks,columns=freq_list)  # Convert matrix to dataframe
    return df_normal

"""
stats: Calculates the mean and standard deviation of peaks for each column of input dataframe
inputs: df_peaks --> dataframe of raw peaks for all electrodes
outputs: stats --> dataframe of means and standard deviations where
            mean -- row 1
            standard error -- row 2
            standard deviation - row 3
"""
def stats(df_peaks):
    avg = []
    std_error = []
    std = []
    objects = list(df_peaks.columns)  # gets column names
    for i, _ in enumerate(objects):
        column = df_peaks.iloc[:, i]
        # print(column)
        avg.append(np.average(column))
        std_error.append(np.divide(np.std(column), np.sqrt(3)))
        std.append(np.std(column))

    data = [avg, std_error,std]
    averages = pd.DataFrame(data, columns = objects)
    return averages

"""
customStats -- stats of non-adjacent columns
input - normalize change where ELECTRODES are collated by rows
output - stats of each CONDITION collated by rows
"""
def customStats(df_peaks,c): # c = number of conditions, how many "groups" of rows
    n = int(len(df_peaks.index) / c)   # n = number of electrodes, how many rows per "group"
    averages = list()
    condition = 0
    while condition <c:
        i=0
        segment =[]
        while len(segment) < n:
            if (i+c) % c ==condition:
                segment.append(np.array(df_peaks.iloc[i,:]))
            i+=1
        segment = pd.DataFrame(segment)
        currentStats = stats(segment)
        if condition ==0: averages =currentStats
        else: averages= pd.concat([averages,currentStats])
        condition+=1
    return averages

# <---------------- Plotting Functions ------------>

""" plotFreq: plot frequency map of peak heights across all frequencies"""
def plotFreq(peaksMatrix,identifiers,save):
    names = list(peaksMatrix.columns)
    freq_list = [float(i) for i in names]
    c=len(peaksMatrix.index)
    plt.figure(figsize=(12,7))
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 22
    i=0
    color=0
    colorStep=1/c
    labelIndex =0
    while i<len(peaksMatrix.index):
        plt.plot(freq_list, peaksMatrix.iloc[i,:], marker='o', linestyle='-', color=plt.cm.plasma(color), linewidth=2, markersize=6)
        # if (i+1) % n == 0:
        color+=colorStep
        labelIndex+=1
        i+=1
    plt.title('SWV Frequency Map', fontsize = '28')
    plt.xlabel('Frequencies (Hz)', fontsize = '22')
    plt.ylabel('Peak Current (nA)', fontsize = '22')
    plt.legend(['MCH 1','MCH 2','MCH 3','HSA 1','HSA 2','HSA 3'],fontsize = '14')
    plt.tight_layout()
    if save:
        plt.savefig(f'{identifiers} Peak Heights.png')

""" plotTemp: plot map of normalized signal change across temperature with each frequency overlayed"""
def plotTemp(df_values, temps,freqs,error):
    if error == 1:
        c = 1  # SE
    else:
        c = 2  # STD
    # zero = []
    # for x in len(temps):
    #     zero.append(0)
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 22
    plt.figure(figsize=(12,7))
    plt.title('SWV Frequency Map 5HP Aptamer ', fontsize='28')
    plt.xlabel('Temperature (C)', fontsize='22')
    plt.ylabel('Normalized Change (%)', fontsize='22')
    color = 0
    i=0
    while i < len(df_values.index)-2:
        print(df_values.iloc[i,:])
        plt.plot(temps, df_values.iloc[i,:], marker = 'o', linestyle = '-', linewidth=2, markersize =6,color=plt.cm.plasma(color))
        plt.errorbar(temps, df_values.iloc[i,:] , yerr= df_values.iloc[i+c,:], ecolor=plt.cm.plasma(color), linestyle='', capsize=5, linewidth=2)
        i+=3
        color += 0.12
    # plt.plot(temps, zero, linestyle='--', marker='', color='lightgray')
    plt.legend(freqs, fontsize = '18')
    plt.tight_layout()
    plt.savefig(f'Normalized {temps} Data.png')

""" plotFreqN: plot frequency map of (normalized) averaged signal change with error bars"""
def plotFreqN(averages,n,identifiers,save,error):
    names = list(averages.columns)
    freq_list = [float(i) for i in names]
    if error == 1:
        c = 1  # SE
        dev = 'SE'
    else:
        c = 2  # STD
        dev = 'STD'
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 22
    plt.figure(figsize=(12,7))
    i = 0
    color = 0
    colorStep = 1 / len(identifiers)
    # print(colorStep)
    while i < len(averages.index)-1:
        plt.plot(freq_list, averages.iloc[i, :], marker='o', linestyle='-', color=plt.cm.plasma(color), linewidth=2,
                 markersize=6)
        plt.errorbar(freq_list, averages.iloc[i, :], yerr=averages.iloc[i+c, :], ecolor=plt.cm.plasma(color), linestyle='', capsize=4, linewidth=2)
        i += 3
        color += colorStep
    plt.title('SWV Frequency Map (n=%s,'f'{dev})' %n,fontsize='28')
    plt.xlabel('Frequency (Hz)', fontsize='22')
    plt.ylabel('Normalized Change (%)', fontsize='22')
    plt.legend(['After HSA'],fontsize='14')
    plt.tight_layout()
    if save:
        plt.savefig(f'{identifiers} Averaged.png')

""" plotFreqNOverlay: plot multiple frequency plots of (normalized) averaged signal change with error bars 
                 # Adjust N by uncommenting sections (suitable for 2 frequency plots right now)
"""
def plotFreqNOverlay(n1,n2,f1,f2,legendNames,v):
    if v == 0:
        a = 1 # plot standard error
    else:
        a = 2 # plot standard deviation

    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    colors = [plt.cm.viridis(0.40),plt.cm.plasma(0.65),plt.cm.viridis(0.70),plt.cm.plasma(0.30)]

    plt.figure(figsize=(12,7))
    plt.ylim(-100, 60)

    # Plot 1
    stats1 = stats(n1)
    a1 = stats1.iloc[0,:]
    s1 = stats1.iloc[a,:]
    plt.plot(f1, a1, marker='o', linestyle='-', linewidth=2, markersize=8, color=colors[0])
    plt.errorbar(f1, a1, yerr=s1, ecolor=colors[0], linestyle='', capsize=5, linewidth=3)

    # Plot 2
    stats2 = stats(n2)
    a2 = stats2.iloc[0, :]
    s2 = stats2.iloc[a, :]
    plt.plot(f2, a2, marker='o', linestyle='-', linewidth=2, markersize=8, color=colors[1])
    plt.errorbar(f2, a2, yerr=s2, ecolor=colors[1], linestyle='', capsize=5, linewidth=3)

    # Plot 3
    # stats3 = stats(n3)
    # a3 = stats3.iloc[0, :]
    # s3 = stats3.iloc[a, :]
    # plt.plot(f3, a3, marker='o', linestyle='-', linewidth=2, markersize=8, color=colors[2])
    # plt.errorbar(f3, a3, yerr=s3, ecolor=colors[2], linestyle='', capsize=5, linewidth=3)

    # Plot 4
    # stats4 = stats(n4)
    # a4 = stats4.iloc[0, :]
    # s4 = stats4.iloc[a, :]
    # plt.plot(f4, a4, marker='o', linestyle='-', linewidth=2, markersize=8, color=colors[3])
    # plt.errorbar(f4, a4, yerr=s4, ecolor=colors[3], linestyle='', capsize=5, linewidth=3)

    # Plot dotted line at y=0
    # zero = []
    # for x in list(range(cA)):
    #     zero.append(0)
    # plt.plot(freq_listA, zero, linestyle = '--', marker = '', color = 'lightgray')

    plt.title('Change in SWV Peak Height After InacV', fontsize='24')
    plt.xlabel('Frequency (Hz)', fontsize='20')
    plt.ylabel('Normalized Signal Change (%)', fontsize='20')
    plt.legend(legendNames, fontsize = '16')
    plt.tight_layout()
    plt.tight_layout()
    # plt.savefig(f'{legendNames} {a} Data.png')


"""
plotSignalStability: Plots normalized signal change per measurement for multiple electrodes
inputs: normSignal --> dataframe of normalized signal for all electrodes
        electrodeNames --> list of electrode names
"""
def plotSignalStability(normSignal, electrodeNames):
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    row = len(normSignal.index)
    col = len(normSignal.columns)
    x = np.arange(1, col+1, 1) # Set x axis equal to total number of measurements
    plt.figure(figsize=(10,7))
    for i in range(row):
        plt.plot(x, normSignal.iloc[i,:], marker = 'o', linestyle = '-')
    # plt.title('Stability After Saliva', fontsize=28)
    # plt.ylabel('Peak Height [nA]', fontsize='22')
    plt.ylabel('Normalized Signal Change [%]', fontsize='22')
    plt.xlabel('# of Measurements', fontsize='22')
    plt.xticks(np.arange(1, col + 1, 1))
    # plt.ylim([-1,10])
    plt.legend(electrodeNames, fontsize='14')
    plt.tight_layout()
    # plt.savefig(f'Stability {electrodeNames} 10x.png')

