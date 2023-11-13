# Lifetime_fit
Multi-exponetional fitting of TCSPC histograms

Full reconvolution lifetime fitting

Batch proccessing, select multiple files via the GUI
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
IRF from experimental file or Automatic reconstruction

Output All components and intensity weigthed average lifetime.



Fitting output
![High1_T0s_1__2expfit](https://github.com/RobertMolenaar-UT/Lifetime_fit/assets/74496038/15d0058a-f545-4184-b22f-86f5fee39324)

visaul on Automatic IRF reconstruction
![irf_construct](https://github.com/RobertMolenaar-UT/Lifetime_fit/assets/74496038/7d0d77b3-e88a-4c96-8154-4fe1f211d94d)

CSV output of decays → time [ns], IRF, TCSPC decay, fit, residuals. 
CSV output of fit values over all proccessed files →  file,t1,t2,t3,tav,a1,a2,a3
