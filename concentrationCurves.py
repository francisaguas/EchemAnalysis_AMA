import pandas as pd
import gamry_parser as parser
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import argrelmin
import scipy
import natsort
from _collections import defaultdict
from mpmath import mp

# <---------------- Analysis Functions ------------>

"""
peaksMatrix functions: calculates peaks/areas across experiment conditions (columns) and electrodes (rows)
input: dataType --> string of measurement type (e.g. DPV, SWV, ACV) used to filter file names, case-sensitive
        electrodesFolders --> list of electrode file paths
        experimentConditions --> list of conditions used for labeling
        resolution --> integer used for AMA sampling resolution
output: df_peaks --> returns a dataframe of all electrode peaks
        df_areas --> returns a dataframe of all electrode areas
"""

# for input file type: .DTA
def peaksMatrixDTA(dataType, frequency, electrodesFolders, experimentConditions,resolution,save):
    allPeaks = []  # Store the peaks for each electrode
    allAreas = []   # Store the areas for each electrode
    for i, electrode in enumerate(electrodesFolders):  # for loop to go through all electrodes
        # Generate list of files of interest
        filesList = extractDataType(electrode, dataType, frequency)
        [p,A]= peaksDTA(filesList, electrode, resolution,save)
        allPeaks.append(p)  # Adds electrode peaks to matrix
        allAreas.append(A)   # Adds electrode areas to matrix
    # Converts matrix to dataframe where columns=conditions, rows=electrodes
    df_peaks = pd.DataFrame(allPeaks, columns=experimentConditions)
    df_areas = pd.DataFrame(allAreas,columns=experimentConditions)
    return df_areas,df_peaks

# for input file type: .TXT
def peaksMatrixTXT(dataType, frequency, electrodesFolders, experimentConditions,indexes,resolution,save):
    allPeaks = []  # Store the peaks for each electrode
    allAreas = []   # Store the areas for each electrode
    for i, electrode in enumerate(electrodesFolders):  # for loop to go through all electrodes
        filesList = extractDataType(electrode, dataType, frequency)
        [xIndex,yIndex]=indexes
        [p,A] = peaksTXT(filesList, xIndex, yIndex,electrode, resolution,save)
        allPeaks.append(p)  # Adds electrode peaks to matrix
        allAreas.append(A)  # Adds electrode areas to matrix
    # Converts matrix to dataframe where columns=conditions, rows=electrodes
    df_peaks = pd.DataFrame(allPeaks, columns=experimentConditions)
    df_areas = pd.DataFrame(allAreas,columns=experimentConditions)
    return df_areas,df_peaks

"""
extract functions: selects files within folder according to string idenitifiers in filenames
inputs: electrodeFolder --> filepath of folder containing DTA files
        dataType --> measurement type being analyzed (DPV, SWV, ACV)
output: filesList --> returns a list of files of interest
"""

# Generate file list of all concentrations at a single frequency
def extractDataType(electrodeFolder, dataType,frequency):
    allFiles = os.listdir(electrodeFolder)  # All the files in folder
    filesList = [] # Store files of interest
    for file in allFiles:  # Loop through each file in folder
        if frequency in file:
            if dataType in file:  # checks if data type (DPV, etc.) is in file name
                if ('#') not in file: # can add additional identifiers as needed
                        filesList.append(file)  # adds file name to list
    filesList = natsort.natsorted(filesList)
    print(filesList)
    return filesList

# Determines frequencies used in data acquisition for Gamry Potentiostat
def frequencies(filesList, electrodesFolder):
    gp = parser.GamryParser()
    freq_list = []
    for i, entry in enumerate(filesList):
        entryPath = os.path.join(electrodesFolder, entry)
        gp.load(filename=entryPath)
        header = gp.get_header()
        freq_list.append(header['FREQUENCY'])
    return freq_list

"""
peaks functions: calculates the peaks of all the  files in the electrode folders using AMA
inputs: filesList --> list of  files of interest
        electrodeFolder --> filepath of folder containing  files
        resolution --> integer used for AMA sampling resolution
outputs: peakHeights --> returns a list peak heights
         areas --> returns a list of areas
"""

# for input file type: .DTA
def peaksDTA(filesList, electrodeFolder,resolution,save):
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
        offset = 5
        newY = yVals[offset:len(yVals) - offset]
        newX = xVals[offset:len(yVals) - offset]
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
        # plt.ylabel('Idif [uA]')
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

        newX = pd.Series(newX)  # Convert x values to pandas series
        numPoints = len(smoothY)
        skipp_by = (int)(numPoints / resolution)
        if skipp_by == 0: skipp_by = 1
        peaksDTA.area_found = defaultdict(float)  # makes a dictionary to keep track of already calculated areas to save time
        maxArea = None  # Maximum area for voltammogram

        # Plot smoothed data
        # plt.figure(figsize=(10, 7))
        # plt.title('Initial')
        # plt.xlabel('Vfwd [V]', fontsize=16)
        # plt.ylabel('Idif [A]', fontsize=16)
        # plt.plot(newX, smoothY, '-')

        # Loop through every X point in voltammogram
        for x, y in zip(newX[1::skipp_by], smoothY[1::skipp_by]):
            # Exclude first pair which is the same point repeated
            array_of_x = np.full(shape=numPoints-1, fill_value=x)
            array_of_y = np.full(shape=numPoints-1, fill_value=y)
            #  zip1 = an array of a given data point repeated
            zip1 = zip(array_of_x, array_of_y)  # ex: --> [ (x, y), (x,y), ... ]

            # zip2 = an array of all data point coordinates considered
            zip2 = zip(newX[1::skipp_by], smoothY[1::skipp_by])  # ex: --> [ (x0, y0), (x1, y1), ..., (xN, yN) ]

            # Potential anchor point combinations for a given point
            possibleBaselines = zip(zip1, zip2)
            currentMaxArea = None  # Maximum area for a given point. Assigned to the first area by default within the loop
            baselineCounter = 0

            # Plot all baselines
            # plt.figure(figsize=(10, 7))
            # plt.title('%f,%f Baselines' % (x,y))
            # plt.xlabel('Vfwd [V]', fontsize=16)
            # plt.ylabel('Idif [A]', fontsize=16)
            # plt.plot(newX, smoothY, '-')
            # plt.plot(x, y, 'o',markersize = 6, color='r')
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
                b = mp.fsub(y1, mp.fmul(m, x1, dps=20), dps=20)
                baseline = m * newX + b
                peaksDTA.m = m

                # Plot all baselines
                # plt.plot(newX[leftIndex:rightIndex+1], baseline[leftIndex:rightIndex+1], linestyle = 'dotted', color = 'orange')
                # plt.plot(x2, y2, '.',markersize = 6, color='r')

                # Baseline correction calculation
                correctedY = smoothY - baseline

                # Check if area has already been calculated for the given anchor points
                if ((x1, y1), (x2, y2)) in peaksDTA.area_found or ((x2, y2), (x1, y1)) in peaksDTA.area_found:
                    currentArea = peaksDTA.area_found[((x1, y1), (x2, y2))]
                else:  # Calculate area under voltammogram if not found in dictionary
                    currentArea = np.trapz(correctedY[leftIndex: rightIndex])
                    # Save calculated area in dictionary
                    peaksDTA.area_found[((x1, y1), (x2, y2))] = currentArea
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

        # Finished looping through voltammogram. Optimal baseline has been found at this point

        # Plot corrected voltammogram with peak height
        # plt.figure(figsize=(10, 7))
        # plt.title('Corrected')
        # plt.xlabel('Vfwd [V]', fontsize=16)
        # plt.ylabel('Idif [A]', fontsize=16)
        # plt.plot(newX, finalCorrectedY, '-')
        # plt.plot(newX[bestLeftIndex], finalCorrectedY[bestLeftIndex], '.', markersize=8, color='r')
        # plt.plot(newX[bestRightIndex], finalCorrectedY[bestRightIndex], '.', markersize=8, color='r')
        # plt.plot(newX, [0]*numPoints, linestyle = 'solid', color = 'orange')

        # Subtract baseline from voltammogram
        difference = [ogVals - baseVals for ogVals, baseVals in zip(smoothY, bestBaseline)]

        # Calculate peak as maximum difference between voltammogram and baseline
        subset = difference[bestLeftIndex:bestRightIndex+1]
        newPeakHeight = np.amax(subset)
        # temp = np.argmax(subset)
        difference = np.array(difference, dtype=float)
        newPeakHeightInd = np.argwhere(difference==newPeakHeight)[0][0]

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
    print(peakHeights)
    # print("number of baselines considered:", counter * mid * len(filesList))
    return peakHeights,areas

# for input file type: .TXT
def peaksTXT(filesList, xIndex,yIndex,electrodeFolder,resolution, save):
    peakHeights = []
    areas = []
    n = 0
    while n < len(filesList):
        filename = filesList[n]
        # print(filename)
        entryPath = os.path.join(electrodeFolder, filename)

        data = pd.read_csv(entryPath, sep='\n', header=None) # separate lines
        # print('here',data[0].get(0))
        [xHead,yHead] = data[0].get(0).split('\t')  # get column headers
        data = pd.DataFrame(data[0].str.split('\t').tolist(),columns =[xHead,yHead]) # separate columns
        data = data.drop([0]) # delete first row of headers
        data = data.astype(float)
        data[yHead]=np.multiply(data.iloc[:,1],[-10**6]*len(data[yHead])) # convert to uA for precision and invert
        # print(data)

        filename = filename[0:len(filename) - 5]
        # Change order to increasing potential (x axis)
        # yVals = np.flipud((data.iloc[:, yIndex]).values)  # units uA
        # xVals = np.flipud((data.iloc[:, xIndex]).values)  # units = V

        yVals = ((data.iloc[:, yIndex]).values)  # units uA
        xVals = ((data.iloc[:, xIndex]).values)  # units = V
        # print(type(xVals),type(yVals))
        # plt.figure()
        # plt.title('Raw %s Data' %(filename))
        # plt.xlabel('Vfwd [V]')
        # plt.ylabel('Idif [uA]')
        # plt.scatter(xVals,yVals)

        # Remove non significant data points
        offset = 5
        newY = yVals[offset:len(yVals)-offset]
        newX = xVals[offset:len(yVals)-offset]
        # plt.figure()
        # plt.title('% s Data Without First 5 Points' %filename)
        # plt.xlabel('Vfwd [V]')
        # plt.ylabel('Idif [A]')
        # plt.plot(newX,newY)
        # plt.xlim([xVals[0],xVals[len(xVals)-1]])

        # Median Filter
        newY = scipy.ndimage.filters.median_filter(newY, size=11)

        # Smooth out raw data
        smoothY = scipy.signal.savgol_filter(newY, 17, 3)
        # plt.figure()
        # plt.title('Smoothed %s Data' % filename)
        # plt.xlabel('Vfwd [V]')
        # plt.ylabel('Idif [uA]')
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
            peakHeights.append(nA)
        else:
            nA = 0
            peakHeights.append(nA)
            maxArea=0
        areas.append(float(maxArea))
        n += 1
    # print("number of baselines considered per concentration:", counter * l * len(filesList))
    print(peakHeights)
    return peakHeights,areas

"""
normalize: Calculates the normalized signal change of peaks across columns
inputs: df_peaks --> dataframe of raw peaks for all electrodes
        referenceMeasurement --> index of column to be used in normalization calculation
outputs: stats --> dataframe of normalized signal change for all electrodes
"""
def normalizeCol(df_peaks, referenceMeasurement, experimentConditions):
    normal_pks = []  # Store normalized peaks for each electrode
    colNames = df_peaks.columns[2:len(df_peaks.columns)]  # Keeps track of column names
    counter = referenceMeasurement
    while counter < len(df_peaks.columns):  # Iterate through columns of dataframe
        reference = np.array(df_peaks.iloc[:, referenceMeasurement])
        currentRow = np.array(df_peaks.iloc[:, counter])
        normalVal = ((currentRow - reference) / reference) * 100  # Normalization Calculation
        normal_pks.append(np.abs((np.around(normalVal, 3))))  # Add normalized value to array
        counter += 1
    normal_pks = np.array(normal_pks).transpose()  # Switch rows and columns for proper dataframe size
    df_normal = pd.DataFrame(normal_pks, columns=experimentConditions)  # Convert matrix to dataframe
    return df_normal


"""
stats: Calculates the mean and standard deviation of peaks across electrodes
inputs: df_peaks --> dataframe of raw peaks for all electrodes
outputs: stats --> dataframe of means and standard deviations
"""
def stats(df_peaks):
    avg = []
    std = []
    objects = list(df_peaks.columns)  # gets column names
    for i, _ in enumerate(objects):
        column = df_peaks.iloc[:, i]
        avg.append(np.average(column))
        # std.append(np.std(column))
        std_error = np.divide(np.std(column), np.sqrt(3))
        std.append(std_error)

    data = [avg, std]
    averages = pd.DataFrame(data, columns=objects)
    return averages

# <---------------- Plotting Functions ------------>
"""
plotPeaks: plots concentration curve of peak heights across concentrations
"""
def plotPeaks(rawPeaks, concentrations, electrodeNames,save):
    plt.figure(figsize=(10, 7))
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    color = 0
    colorStep = 1 / len(rawPeaks.index)
    i = 0
    while i < len(rawPeaks.index):
        row = rawPeaks.iloc[i, :]
        plt.plot(concentrations, row, marker='o', markersize='6', linestyle='-', color=plt.cm.plasma(color),
                 markeredgecolor='k', markeredgewidth=1.0)
        i += 1
        color += colorStep
    plt.xlabel('Concentration (fgmL)', fontsize='20')
    plt.ylabel('Peak Height (nA)', fontsize='20')  # gives y-label for bar graph
    plt.title('SWV Peak Height (10Hz)', fontsize='22',fontweight='bold')  # display title
    plt.legend(electrodeNames, fontsize='18')
    plt.xscale('symlog', linthresh=1e-1, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    # plt.xlim(-0.01, 150)
    if save:
        plt.tight_layout()
        plt.savefig('Sample Peak ConCurve.png')

"""
plotAreas: plot concentration curve of peak areas across concentrations 
"""
def plotAreas(areas, concentrations, electrodeNames,save):
    plt.figure(figsize=(10, 7))
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    color = 0
    colorStep = 1 / len(areas.index)
    i = 0
    while i < len(areas.index):
        row = areas.iloc[i, :]
        plt.plot(concentrations, row, marker='o', markersize='6', linestyle='-', color=plt.cm.plasma(color),
                 markeredgecolor='k', markeredgewidth=1.0)
        i += 1
        color += colorStep
    plt.xlabel('Concentration (fgmL)', fontsize='20')
    plt.ylabel('Peak Area', fontsize='20')  # gives y-label for bar graph
    plt.title('SWV Peak Areas (10Hz)', fontsize='22', fontweight='bold')  # display title
    plt.legend(electrodeNames, fontsize='18')
    plt.xscale('symlog', linthresh=1e-1, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    # plt.xlim(-0.01, 150)
    if save:
        plt.tight_layout()
        plt.savefig('Sample Area ConCurve.png')


"""
plotCustom: customizable concentration curve plotting function analagous to plotPeaks/plotAreas 
            e.g. plot concentration curve where y values are peak/area
"""
def plotCustom(y, concentrations, electrodeNames,save):
    plt.figure(figsize=(10, 7))
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    color = 0
    colorStep = 1 / len(y.index)
    i = 0
    while i < len(y.index):
        row = y.iloc[i, :]
        plt.plot(concentrations, row, marker='o', markersize='6', linestyle='-', color=plt.cm.plasma(color),
                 markeredgecolor='k', markeredgewidth=1.0)
        i += 1
        color += colorStep
    plt.xlabel('Concentration (mM)', fontsize='20')
    plt.ylabel('Peak Height/Area', fontsize='20')  # gives y-label for bar graph
    plt.title('SWV Peak Height/Area' , fontsize='22', fontweight='bold')  # display title
    plt.legend(electrodeNames, fontsize='18')
    plt.xscale('symlog', linthresh=1e-1, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    # plt.xlim(-0.01, 150)
    if save:
        plt.tight_layout()
        plt.savefig(f'{electrodeNames}.png')

"""
plotConcentration: plots (normalized) averaged concentration curve with error bars 
"""
def plotConcentration(averages, concentrations, electrodeNames, frequency,save):
    objects = list(averages.columns)  # gets column names
    dfy = list(averages.iloc[0, :])  # access row 1 of the data frame (the peak heights)
    error = list(averages.iloc[1, :])
    plt.figure(figsize=(10, 7))  # creates figure
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['xtick.labelsize'] = 24
    plt.plot(concentrations, dfy, marker='o', markersize='10', linestyle='', color=plt.cm.viridis(0.35),
             markeredgecolor='k', markeredgewidth=1.0)  # plots graph
    plt.errorbar(concentrations, dfy, yerr=error, ecolor=plt.cm.plasma(.75), linestyle='', capsize=4)
    plt.xlabel('Concentration S1 Protein (fg/mL)', fontsize='20')
    plt.ylabel('Normalized Signal Change (%)', fontsize='20')  # gives y-label for bar graph
    plt.title('Concentration at %s (n=3)' % frequency, fontsize='24', fontweight='bold')  # display title
    plt.xscale('symlog', linthresh=1e-1, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    # plt.xlim(-0.01, 150)
    # plt.ylim(-3,40)
    if save:
        plt.tight_layout()
        plt.savefig('Sample Normalized Averaged ConCurve.png')
