# Lifetime_fit
Multi-exponetional fitting of TCSPC histograms based on full reconvolution method on PicoQuant single point PTU files

## Discription
With this script you can fit TCSPC lifetime histograms to single, double or triple order exponential fit with reconvolution method
Minimalization method are NLLS / MLE-Nelder-Mead minimalisation methods. Batchwise proccessing, select 1 or multiple files via the GUI. And data is imported via the [PTUreader](https://github.com/SumeetRohilla/readPTU_FLIM)

## Dependencies

The script is developed and tested on Python 3.11, Install: 

>1. wx python 4.2.1 for the file selector app
>2. [PTU file reader:](https://github.com/RobertMolenaar-UT/readPTU_FLIM)
>3. pandas
>4. lmfit
>5. scipy 

## Input data selection tools:

1. Select SPAD channel from PTU file (Channel)
2. Limit number of photos in the file (photons)
3. Limit/set histogram peak value (peak_lim)
4. Long lifetime/phosforence remove 2nd photon afer APD deadtime recovery (Drop_multi_AC_count)

## Fitting options
1. Fitting Methods: NLLS or MLE (method)
2. Fit order 'single, 'double' or 'triple' (fit_order)
3. Fixing any of the fitting parameters. (set in lmfit params)
4. Fitting boundaries can set. (set in lmfit params)
5. IRF from experimental file or Automatic reconstruction. (irf_source)

Output shows exp fit components and intensity weighted average lifetime.

>CSV output of decays → time [ns], IRF, TCSPC decay, fit, residuals. 
>CSV output of fit values over all proccessed files →  file,t1,t2,t3,tav,a1,a2,a3



*Figure 1: File exponential fitting output summary*

![Example: 2-exponential fit](https://github.com/RobertMolenaar-UT/Lifetime_fit/assets/74496038/15d0058a-f545-4184-b22f-86f5fee39324)

*Figure 2: visual on Automatic IRF reconstruction.*
![Automatitic IRF reconstruction](https://github.com/RobertMolenaar-UT/Lifetime_fit/assets/74496038/7d0d77b3-e88a-4c96-8154-4fe1f211d94d)

---
Copyright Robert Molenaar, 2 August 2024


Keywords: PTU, Picoquant, fluorescent lifetime, reconvolution, TCSPC
