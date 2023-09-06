# Lifetime_fit
Multi-exponetional fitting of TCSPC histograms

fitting program

Batch proccessing, select multiple files in the GUI
reads PTU files via the PTUreader https://github.com/SumeetRohilla/readPTU_FLIM

With this script you can fit TCSPC lifetime histograms to single, double or triple order exponential fit with reconvolution method
Minimalization method are NLLS / MLE-Nelder-Mead minimalisation methods.



Input data histogream shaping options
1. select SPAD channel from PTU file (Channel)
2. Limit number of photos in the file (photons)
3. Limit/set histogram peak value (peak_lim)
4. Long lifetime/phosforence remove 2nd photon afer APD deadtime recovery (Drop_multi_AC_count)


Fitting Methods: NLLS or MLE
fit_order 'single, 'double' or 'triple'
One can fix any of the fitting parameters
Fitting boundaries can set

Output All components and intensity weigthed average lifetime.

Image output.
CSV output: time [ns], IRF, TCSPC decay, fit, residuals.

![flu4am_1_2expfit](https://github.com/RobertMolenaar-UT/Lifetime_fit/assets/74496038/0783d875-3b51-4a6b-8f92-d8714442c160)
