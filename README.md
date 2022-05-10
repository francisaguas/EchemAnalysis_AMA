# EchemAnalysis_AMA

This repository contains the scripts for automated baseline analysis and visualization of electrochemical data. Baseline and peak analysis of voltammograms is accomplished by a novel area-maximizing algortihm (AMA) that captures peaks even in non-ideal, noisy, and sloped signals.

This open-source program was created to expedite electrochemical data analysis and support the advancement of electrochemical biosensors.

For more detailed instructions for use, refer to: https://docs.google.com/document/d/1B-2fkvCbE3QyiXFA00YbSR5AxEci5_CkFwBRkM6wl3s/edit?usp=sharing

# Script Directory

calcPeaks - Calculates peak heights/areas using AMA, given input data of file type (.DTA or .TXT) and saves results to Excel file(s) for further analysis and plotting.

calcExcelPeaks - Calculates peak heights/areas using AMA, given input data of file type (Excel) and saves results to an Excel file for further analysis and plotting.

plotConcentration - Plots concentration curves of peak heights, peak areas, and normalized signal change given Excel file(s) from calcPeaks or calcExcelPeaks.

plotFrequency - Plots frequency maps of peak heights and normalized signal change given an Excel file from calcPeaks or calcExcelPeaks.

concentrationCurves - Contains functions for file extraction, peak analysis, normalization, and plotting for concentration curves. Called upon by all other scripts.

frequencyMaps - Contains functions for file extraction, peak analysis, normalization, and plotting for frequency maps. Called upon by all other scripts.

