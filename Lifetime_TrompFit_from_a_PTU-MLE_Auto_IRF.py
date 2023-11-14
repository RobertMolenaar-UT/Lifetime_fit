# -*- coding: utf-8 -*-
"""
@author: r.molenaar@utwente.nl   20 March 2023

Lifetime fitting with picoquant hardware PTU file formats
Fitting is based on the NLLS and MLS fitting developped by MSc. Leroy Tromp.
https://lmfit.github.io/lmfit-py/index.html#

PTU files are read by read_PTU_FLIM
https://github.com/RobertMolenaar-UT/readPTU_FLIM

# seaborn-dark  https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
Models: available single, double, triple exponential fit.

20230727 - added bounds of the fit values for the MLE fits, were not implememted
20230728 - IRF is suface=1 normalization => fitted amplitudes represent tav_int 
         - NLLS fit is always done with MLE inital comes from NLLS output 'Feed_NLLSfit_as_guess_MLE'
         - in Multiple files CSV _fit_values.dat is saved with
         - options plot initial guess in figure, xlim_max value
         - Fix fitting values for NLLS,  cleanup code + variables
20231105 - Added an Automatic IRF construction 
"""

from readPTU_FLIM import PTUreader
import numpy as np
import os
from ctypes import wintypes, windll
from functools import cmp_to_key
import wx
import pandas as pd
import matplotlib.pyplot as plt
import math
import lmfit
import copy
from scipy.optimize import minimize
plt.style.use('seaborn-dark')

# %%

"""###############################################################
   #####---- ↓↓↓ - FILL in FIT settings for analysis - ↓↓↓ -----######
   ##############################################################
   Use # to comment options out """


# experimental settings
Channel            = 1       # picoquant MT200 PTU files record all counts select detector here
peak_lim           = 10000   # histogram is cycled until peak limit value is reached
photons            = 0       # select the first n- number of photons set 0 to disable the function
Drop_multi_TAC_count = True  # filter double photons after APD recovery seen at high concentrations & powers with long lifetime species recommended True

#fitting options
method             = 'leastsq'     #other is 'Nelder-mead' Nelder-mead has preference for low counts
#method            = 'Nelder-Mead'    
Feed_NLLSfit_as_guess_MLE = True   #applies for Nelder-Mead fitting, recommended TRUE MLE can use a good initial guess, so NLLS output is used as input for MLE

sample_name        =  'Sample Name'        #Fill in sample name, used in figures  
output_dname       =  'Python_Lifetime_Fit'  #Data is stored in this direcory dname
Default_Folder     = r'C:\user_folder\testsdata.sptw'                       # sympotime folder    
irf_fname          = r'C:\user_folder\testsdata.sptw\IRF_20MHz_1.ptu'       # IRF_file in the root of the *.sptw folder

GUI_MultiPick      = True
plot_ig_fig        = True       #include initial guess in the plotting, nice for initial gusee

irf_source         = 'pulse'   #Instrumnet responce from a single pulse (usefull for lifetimes >100ns)
irf_source         = 'File'     #Instrument responce read from a IRF.PTU file
irf_source         = 'auto'     #instrument responce automatically determined from the decay 
smooth_irf         = 2          #post-moving average ove the AUTO IRF
window_ma          = 4          # pre-moving average ove the AUTO IRF

save_fig           = True       #save the output figures
Save_data_files    = True       #save the corresponding histogram CVS data
Save_fit_values    = True       #save the Fit values of the fit

fit_order          = 'triple'   #exponential for order: 'triple', 'double', 'single'
fit_order          = 'double'
#fit_order          = 'single'

# initual guess parameters
#to FIT Boundaries are set ↓ keyword #SET BOUNDARIES also for Fixing parameters
ig_t1       = 0.5     #ns
ig_t2       = 2       #ns
ig_t3       = 5       #ns
ig_ampl_1   = 2500   #cnt
ig_ampl_2   = 2500   #cnt
ig_ampl_3   = 0.001  #cnt
ig_baseline = 5      #cnt
ig_offset   = 0      #time bins
start_from_ns  = 2.8 #irf start value (if no file is loaded)

xlim_max = None      #ns use 'None' to disable
# xlim_max = 25       

"""---- ↑↑↑ - FILL in settings for analysis - ↑↑↑ -----"""


""" ---- ↓ Code Functions ↓ Don' touch ---- """

def moving_average(x, w, method="same"):
    """ function for auto_irf
    x = dataarray
    w = window width
    """ 
    return np.convolve(x, np.ones(w), method) / w

def model_irf(x, tau, ampl, baseline, offset):
    """function for auto_irf"""
    y = ampl * np.exp(-x/tau)
    y += baseline
    return y[:x.size]

def residuals_irf(params, x, y, weights):
    """function for auto_irf
    Returns the array of residuals for the current parameters.
    """
    tau = params["tau"].value
    ampl = params["ampl"].value
    baseline = params["baseline"].value
    offset = params["offset"].value
    return (y - model_irf(x, tau, ampl, baseline, offset)) * weights

def auto_irf(decay_hist,timestamps, window=2, threshold=60, fast_range=0.5, plot=True, save_fig=False, smooth_irf=1):
    """
    
    Parameters
    ----------
    decay_hist : int,       Measured histogram.
    timestamps : float64,   Time axis [s]
    window     : int,       optional moving average over the decay to reduce noise. The default is 4.
    threshold  : float64.   optional Intensity threshold fraction from the peak. The default is 0.005.
    fast_range : float64.   optional time in ns to fit the fast component. The default is 0.4.
    plot       : bool,      optional plot IRF analysis. The default is True.
    save_fig   : bool,      optional save IRF reconstruction image. The default is False
    smooth_irf : int,       optional addes a adition moving average over the IRF output the supres noise, the default is 1
    
    Returns   : irf with leading & traling zeroes. 
    -------
    Source: Symphotime64 TCSPC HELP section IRF
    In an ideal system, any damped effect of the irf is absent. One can expect a linear continuation from the initial start back to the onset of the curve (were peak starts to rise)
    Just after the max decay value, we 1exp fit the fast component [+0.2:+fast_range] and extrapolate that back to the onset, which is called Convex.
       
    Reconstruction from the IRF is done point by point in the measured decay. At argument decay[n], we calculate the (max) expected value from convex[n+1] and subtract the actual change (decay n+1).
    The residual is put into IRF value for [n]
    
    From this we subtract the measured value. the difference is transvered to the IRF for that timestamp.

    """
    
    if np.diff(timestamps)[0] >33*1E-12:
        print(f"NOTE: AUTO-irf time resolution {np.diff(timestamps)[0]*1E12:.1f}ps is low: decay smoothing {window} and irf smoothing {smooth_irf} is disabled\n")
        window=1
        smooth_irf=1
    
    # first find the onset of the irf set a 0.005% from Imax.
    #IRF_argonset=np.where(decay_hist > threshold*np.max(decay_hist))[0][0]
    irf_precursor=0.2 #ns extra time BEFORE The IRF_argonset
    irf_precursor=int(irf_precursor/(np.diff(timestamps)[0]*1E9)) #swithch from nanoscends to argument index
    IRF_argonset=np.where(decay_hist > threshold)[0][0]-irf_precursor
       
    #Extracting arguments for the fast decay fit just afer the max, to fit the fastest time component
    #dynamic argument selection, start=0.2ns after stop, is the length makes it independent to timing resolution
    irf_postcursor=0.3 #ns AFTER THE MAX argument the IRF is extened with x ns.   
    start_slope=np.argmax(decay_hist)+int(irf_postcursor/(np.diff(timestamps)[0]*1E9))
    stop_slope= np.argmax(decay_hist)+int((irf_postcursor+fast_range)/(np.diff(timestamps)[0]*1E9))
    irf_postcursor=int(irf_postcursor/(np.diff(timestamps)[0]*1E9))  #swithch from nanoscends to argument index
    
    #apply a moving average over the input decay and select range 
    decay_slope=moving_average(decay_hist, window)[start_slope:stop_slope]
    times_slope=timestamps[start_slope:stop_slope]*1E9
    weights = 1/np.sqrt(decay_slope)
    
    #transfer selected fast component into 1exp fitting
    #initial guess and bounds
    params_1exp_irf = lmfit.Parameters()
    params_1exp_irf.add("tau",      value=0.5,      vary =True,  min= 0, max=5)
    params_1exp_irf.add("ampl",     value=100000,   vary =True,  min= 0       )
    params_1exp_irf.add("baseline", value=5,        vary =True,  min= 0, max=5)
    params_1exp_irf.add("offset",   value=0,        vary =False, min=-5, max=5)
    
    mini_1exp_irf = lmfit.Minimizer(
        residuals_irf,
        params_1exp_irf,
        fcn_args=(times_slope, decay_slope, weights),
        nan_policy="omit")
    #fitting is done here
    fit_nlls_1exp_irf = mini_1exp_irf.minimize(method='leastsq')
    
    lmfit.report_fit(fit_nlls_1exp_irf,show_correl=False)
    print('\nAUTO_irf ↑ fast component ↑ fit results lmfit.minimizer')
    
    
    #apply 1exp fit results to build an exponention decay from Onset to max decay value
    #the onset to max decay_hist is selected for time and decay
    
    timestamps_plot=timestamps[IRF_argonset:np.argmax(decay_hist)+irf_postcursor]
    Convex_decay_plot=moving_average(decay_hist,window, "same")[IRF_argonset:np.argmax(decay_hist)+irf_postcursor]
    #Plot extrapolation
    Convex_predict=model_irf((timestamps_plot*1E9), fit_nlls_1exp_irf.params["tau"].value,fit_nlls_1exp_irf.params["ampl"].value, fit_nlls_1exp_irf.params['baseline'].value, fit_nlls_1exp_irf.params['offset'].value)
    
    """Picoquant does not specify how decay is predicted, I guess
    Based on the Convex decay time, we can determine a fraction <0 for the next timebin n+1
    The IRF recoconstruction would be I[n]*falloff+I[n+1]
    
    """
    
    dt=np.diff(timestamps)
    #calculate fraction drop off, by using tau, but set amplitude to 1, offsets & base 0
    Fall_off=1*np.exp(-dt[0]*1E9/fit_nlls_1exp_irf.params["tau"].value)

    print(f'AUTO-irf → Fall_off | {Fall_off:.3f} \n')
    
    
    #  Convex_diff=np.diff(Convex_predict)     #predicion
    irf=np.zeros(len(Convex_predict))
    i=0
    
    for k in Convex_predict[:-1]:
        # (measured [n+1] - mesaured value [n] * fraction drop
        irf[i]=Convex_decay_plot[i+1]-(Convex_decay_plot[i]*Fall_off)
        i+=1
    
    irf=np.roll(irf,1) # the irf got shifted by the previous operation, Her its shifted back
    
    if plot:
        fig_1, axs = plt.subplots(1, 1, figsize=(8, 5))
        axs.set_title('Auto IRF reconstruction ', size=11)
        axs.set_yscale('log')
        axs.set_ylim([0.9, np.max(Convex_predict)*2])
        axs.set_xlim([1.8, 4.2])
        axs.scatter(timestamps*1e9, decay_hist, alpha =0.1,  label='Input decay')
        axs.scatter(times_slope, decay_slope, alpha =0.5,  label='Fast fit component ')
        axs.scatter(timestamps_plot*1e9, Convex_predict, alpha =0.2,  label='Extrapolated decay')
        axs.plot(timestamps_plot*1e9, irf, alpha =0.5, color='r', label='Constructed IRF')
        axs.axhline(y = threshold, color = 'y', linestyle = '--', alpha=0.5, label='int. threshold') 
        axs.axvline(x = timestamps[IRF_argonset]*1E9, color = 'y', linestyle = '--', alpha=0.5, label='Onset') 
        axs.axvline(x = timestamps[np.argmax(decay_hist)]*1E9, color = 'g', linestyle = '--', alpha=0.3, label='max intensity') 
        axs.set_ylabel(' counts [-] →', size=12)
        axs.grid(True)
        axs.grid(True, which="minor", lw=0.3)
        axs.set_xlabel('Time [ns]', size=12)
        axs.legend(loc='lower right', fontsize=11)
        plt.plot
    
    irf=np.where(irf<1,0,irf)   #delete any negative values
    irf=np.append(np.zeros(IRF_argonset+0), irf)  #add leading zeros
    irf=np.append(irf, np.zeros(len(decay_hist)-len(irf))) #add trailing zeroes to match hist_decay lenght

    
    if save_fig:
        #save the figure
        fname_save = head+output_dname
        if not os.path.exists(fname_save):
            os.makedirs(fname_save)
        fig_1.savefig(fname_save+'irf_construct.png', bbox_inches='tight', dpi=300)
    
    
    return irf


def winsort(data):
    # Python indexes files not as windows shows in File explorer
    # This definition reorganses
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype = wintypes.INT

    def cmp_fnc(psz1, psz2): return _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))


def GUI_select_Multi_file(message):
    """ Function selects one or multiple PTU filenames via a GUI,
        It starts at the current working directory of the process
    """
    wildcard = "picoquant PTU (*.ptu)| *.ptu"
    app = wx.App()
    frame = wx.Frame(None, -1, 'win.py', style=wx.STAY_ON_TOP)
    frame.SetSize(0, 0, 200, 50)
    FilePicker = wx.FileDialog(frame, "Select you PTU files | single or Multiple", defaultDir=Default_Folder,
                               wildcard=wildcard, style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)
    FilePicker.ShowModal()
    FileNames = FilePicker.GetPaths()
    app.Destroy()
    return FileNames


def GUI_select_folder(message):
    """ A function to select a PTU filename via a GUI,
        It starts at the current working directory of the process
    """
    wildcard = "picoquant PTU (*.ptu)| *.ptu"
    app = wx.App()
    path = wx.FileSelector(message='Select you the folder',
                           default_path=Default_Folder, default_extension='*.ptu', wildcard=wildcard)
    directory, filename = os.path.split(path)
    app.Destroy()
    print('Selected file folder: '+path)
    return directory


def read_ptu_irf(path, fname='IRF.ptu', irf_ch=1, plot=False):
    """
    Reads a ptu file, and converts this into a histogram, and looks for a IRF.ptu in the root.
    
    Parameters
    ----------
    path   : of the .sptw directory make sure an IRF.ptu is here
    fname  :
    irf_ch : select detectector channel 
    plot   :    plot a image of the IRF histogram
    -------
    irf_histo : dataframe timestaps, counts
    """
    fname = fname
    # read PTU file
    head, tail = os.path.split(path)
    # get pack to the root of the project folder to find irf.ptu

    base = head.split('.sptw')
    pname = base[0]+'.sptw\\'
    irf_fname = pname+fname
    irf_file = PTUreader((irf_fname), print_header_data=False)

    irf_df = pd.DataFrame(
        {'sync': irf_file.sync, 'tcspc': irf_file.tcspc, 'channel': irf_file.channel+1}, dtype=int)
    
    # select only channel
    irf_df = irf_df[(irf_df['channel'] == irf_ch)]

    # Convert TCSPC into a histogram: get TAC range and make binedges
    tac_bins = int(
        irf_file.head['MeasDesc_GlobalResolution']/irf_file.head['MeasDesc_Resolution'])
   
    print( f"IRF: {fname} is loaded: Time bins = {tac_bins} time resolution = {irf_file.head['MeasDesc_Resolution']*1e12:.1f}ps \n")

    BinEdges = np.linspace(0, tac_bins, tac_bins+1)
    TCSPC, bins1 = np.histogram(irf_df['tcspc'], bins=BinEdges)
    bins1 = np.delete(bins1, 0)
    timestamps = bins1*irf_file.head['MeasDesc_Resolution']  # make time axis

    # save IRF into a dataframe
    irf_histo = pd.DataFrame({'time': timestamps, 'counts': TCSPC, })

    if plot:
        fig, axs = plt.subplots(1, 1, figsize=(8, 5))
        axs.set_title('TCSPC IRF | '+head, size=11)
        axs.set_yscale('log')
        axs.set_ylim([0.7, np.max(TCSPC)])
        axs.set_xlim([0, np.max(timestamps*1e9)/10])
        axs.scatter(timestamps*1e9, TCSPC, alpha =0.5,  label='IRF')
        axs.set_ylabel(' counts [-] →', size=12)
        axs.grid(True)
        axs.grid(True, which="minor", lw=0.3)
        axs.set_xlabel('Time [ns]', size=12)
        #plt.figtext(0.1, 0.01, ConfigSummary, ha="left", fontsize=12)
        # plt.savefig(d_name+'\\'+"{:0>2d}".format(j)+'d_tcspc_'+f_name+'.png',dpi=300)
        axs.legend(loc='upper right', fontsize=11)
        plt.plot

    return irf_histo


def Construct_histogram(ptu_df, BinEdges, peak_limit=0):
    """
    #1. builds a histogram from tcspc data, 
    finds index of the max and finds the index of n- index. were it drops the left over tcspc data
    #2 if peak_lim is set as 0, limiter is disabled
    Parameters
    ----------
    ptu_df   : ptu_df['tcspc'] dataframe
    BinEdges : array of int bin edges
    peak_lim : float/int, optional 
                limit histogram to a max count value. The default is 0
    Returns
    -------
    decay_hist_peaklim : Array of float64 
        decay_hist
    bins1 : Array of float64
        BidEdges-1

    """

    peak_limit = int(peak_limit)

    if peak_limit == 0:
        decay_hist_peaklim, bins1 = np.histogram(
            ptu_df['tcspc'], bins=BinEdges)
        bins1 = np.delete(bins1, 0)
    else:

        try:
            decay_hist, bins1 = np.histogram(ptu_df['tcspc'], bins=BinEdges)
            max_indexes = ptu_df['tcspc'].index[ptu_df['tcspc']
                                                == np.argmax(decay_hist)].tolist()
            
            #ptu_df_peak = ptu_df.iloc[:-int(len(ptu_df)-cut)]
            ptu_df_peak = ptu_df.iloc[0:max_indexes[peak_limit-1]]
            decay_hist_peaklim, bins1 = np.histogram(
                ptu_df_peak['tcspc'], bins=BinEdges)
            bins1 = np.delete(bins1, 0)
        except:
            
            decay_hist_peaklim, bins1 = np.histogram(
                ptu_df['tcspc'], bins=BinEdges)
            bins1 = np.delete(bins1, 0)
            print(f'NOTE: construct histogram, To few photons {np.max(decay_hist_peaklim)}ph to reach histogram peak of {peak_limit}ph.')

    return decay_hist_peaklim, bins1


def get_irf_background(channels, counts):
    irf_peak = lmfit.models.GaussianModel()
    irf_background = lmfit.models.ConstantModel()
    #irf_model = irf_peak + irf_background
    irf_model = irf_background
    pars = irf_peak.guess(counts, x=channels) + \
        irf_background.guess(counts, x=channels)
    result = irf_model.fit(counts, pars, x=channels)
    return result.params["c"].value


# The background noise determined in the previous step is used to preprocess the IRF if `preprocess_irf` boolean is set to `True`.

# In[15]:

    
""" ################################################################
    ########       →  Real Code starts Here ←             ##########
    #############################################################"""


path_select = [0]
Errors      = ['']



if GUI_MultiPick == True:
    # single or multiple proccess
    print('Select a Single or Multiple files')
    path_select = GUI_select_Multi_file('Select a file')
    path_select = winsort(path_select)
else:
    # FUll Folder proccess
    print('Converting all *.ptu images in the folder')
    GUI_MultiPick = False
    path = GUI_select_folder('Select a folder')
    os.listdir(path)
    FileList = []

    i = 0
    for file in os.listdir(path):
        i = i+1
        if file.endswith(".ptu"):
            FileList.append(os.path.join(path, file))
    path_select = winsort(FileList)

# make a table of all files
output_dname = '\\'+output_dname+'\\'
Group_fit_values_df=pd.DataFrame(columns=['file','t1', 't2','t3','tav', 'a1', 'a2','a3'])        

j = 0
for path in path_select:
    # Main loop that procceses all *.PTU files (path_select) from Multiple file pick or folder
    head, tail = os.path.split(path)
    print('')
    print('File '+str(j+1)+ ' of '+ str(len(path_select))+' | opening file → '+tail)

    # open file
    ptu_file = PTUreader((path), print_header_data=False)
    # File checking if its 1D or 2d:
    if ptu_file.head["Measurement_SubMode"] != 0:
        Errors = np.append(Errors, path)
        print('NOTE: PTU-reader, File is not a Point-measurement: skip to next *.PTU file')
        j+=1
        continue
    #extract from datafram into an array
    # ptu_arr = np.zeros((len(ptu_file.sync), 3))
    # ptu_arr[:, 0] = ptu_file.sync
    # ptu_arr[:, 1] = ptu_file.tcspc
    # ptu_arr[:, 2] = ptu_file.channel
    # index = list(range(0, len(ptu_file.sync)))

    # ptu_df = pd.DataFrame(
    #     {'sync': ptu_arr[:, 0], 'tcspc': ptu_arr[:, 1], 'channel': ptu_arr[:, 2]+1}, dtype=int)
    
    ptu_df = pd.DataFrame(
        {'sync': ptu_file.sync, 'tcspc': ptu_file.tcspc, 'channel': ptu_file.channel+1}, dtype=int)
    
    # select only channel n- from the dataframe
    ptu_df = ptu_df[(ptu_df['channel'] == Channel)]

    if Drop_multi_TAC_count:
        # Option remove double counts from the same sync pulse for long TAC range >100ns used with multiharp 150
        ptu_df.drop_duplicates(
            subset=['sync'], keep='first', inplace=True, ignore_index=False)

    ptu_df.reset_index()

    if photons != 0:
        # Dataset selection: use a fixed number of photons  the first n- number of photos
        ptu_df = ptu_df.iloc[:-int(len(ptu_df)-photons)]

    # convert TCSPC to a histogram decay_hist
    tac_bins = int(ptu_file.head['MeasDesc_GlobalResolution']/ptu_file.head['MeasDesc_Resolution'])
    BinEdges = np.linspace(0, tac_bins, tac_bins+1)
    decay_hist, bins1 = Construct_histogram(ptu_df, BinEdges, peak_lim)
    timestamps = bins1*ptu_file.head['MeasDesc_Resolution']
    time_step_ns = ptu_file.head['MeasDesc_Resolution']*1e9
    time_ns = timestamps*1e9
    
    


    #%% get IRF, option is to get the read a ptu irf only at the first itteration
    
    #There are 3 methods to get a IRF: from PTU file, manual given or a pulse at a give time
    if irf_source== 'File':
        # read the IRF.PTU file in the *.SPTW folder
        irf_df = read_ptu_irf(path, irf_fname, Channel, plot=False)
        # if measured IRF has a more time bins shorten to experimental lenght
        irf_df = irf_df.head(len(timestamps))
        
        #irf histogram, noise needs to be removed from irf peak
        #1 change from dataframe to array
        irf_hist = np.zeros(len(irf_df['time']))
        irf_hist[:] = irf_df['counts']
        irf_hist = np.where(irf_hist >= get_irf_background(timestamps, irf_hist),irf_hist , 0)
        # for convolution the surface ara of the irf needs to be set to 1
        irf_hist = irf_hist/np.sum(irf_hist)
            
    elif irf_source == 'pulse':
        print(f'IRF: impulse at {start_from_ns:.3g}ns')
        startindex = int(start_from_ns/time_step_ns)
        irf_hist = np.zeros(len(timestamps))
        irf_hist[startindex] = 1
    
    elif irf_source== 'auto' and j==0:
        print('IRF: Auto construction')
        irf_hist=auto_irf(decay_hist, timestamps, window_ma, smooth_irf=smooth_irf, save_fig=True)
        #irf_hist=np.append(irf_hist, np.zeros(len(decay_hist)-len(irf_hist)))
        # for convolution the surface ara of the irf needs to be set to 1
        irf_hist = irf_hist/np.sum(irf_hist)
    
    # for plotting a peak value scalled copy is made
    irf_display = (np.max(decay_hist)/np.max(irf_hist))*irf_hist
        
    
    # Leroy his approach instead on deconvolute the measured signal and fit that to the model.
    # fitting models is deconvoluted, this is what the irf shift is doin.

    def irf_shift(prompt, shift):
        spillover, integer = math.modf(shift)
        if shift > 0:
            output = (1 - spillover) * np.roll(prompt, int(shift)) + \
                spillover * np.roll(prompt, int(shift + 1))
        if shift < 0:
            output = (1 + spillover) * np.roll(prompt, int(shift)) + \
                abs(spillover) * np.roll(prompt, int(shift - 1))
        if shift == 0:
            return prompt
        return output

    #%% #make model definitions
    if window_ma >1:
        if np.diff(timestamps)[0] >30*1E-12:
            print(f"NOTE: decay smoothing, time resolution {np.diff(timestamps)[0]*1E12:.1f}ps is low: decay smoothing {window_ma} is set back to 1.")
            window_ma=1
            
        print(f'NOTE: decay smoothing, moving average n={window_ma} over decay curve before fitting.')
        decay_hist=moving_average(decay_hist, window_ma, method="same")


    if fit_order == 'single':

        def model_1exp(x, tau, ampl, baseline, offset, irf=irf_hist):
            #y = np.zeros(x.size)
            y = ampl * np.exp(-x/tau)
            z = np.convolve(y, irf_shift(irf, offset))
            z += baseline
            return z[:x.size]
        
        def residuals_1exp(params, x, y, weights):
            """
            Returns the array of residuals for the current parameters.
            """
            tau = params["tau"].value
            ampl = params["ampl"].value
            baseline = params["baseline"].value
            offset = params["offset"].value
            return (y - model_1exp(x, tau, ampl, baseline, offset)) * weights
        
    elif fit_order == 'double':

        def model_2exp(x, tau_1, tau_2, ampl_1, ampl_2, baseline, offset, irf=irf_hist):
            y = np.zeros(x.size)
            y = ampl_1 * np.exp(-x/tau_1) + ampl_2 * np.exp(-x/tau_2)
            z = np.convolve(y, irf_shift(irf, offset))
            z += baseline
            return z[:x.size]
        
        def residuals_2exp(params, x, y, weights):
            """
            Returns the array of residuals for the current parameters.
            """
            tau_1 = params["tau_1"].value
            tau_2 = params["tau_2"].value
            ampl_1 = params["ampl_1"].value
            ampl_2 = params["ampl_2"].value
            baseline = params["baseline"].value
            offset = params["offset"].value
            return (y - model_2exp(x, tau_1, tau_2, ampl_1, ampl_2, baseline, offset)) * weights

    elif fit_order == 'triple':
    
        def model_3exp(x, tau_1, tau_2, tau_3, ampl_1, ampl_2, ampl_3, baseline, offset, irf=irf_hist):
            y = np.zeros(x.size)
            y = ampl_1 * np.exp(-x/tau_1) + ampl_2 * np.exp(-x/tau_2) + ampl_3 * np.exp(-x/tau_3)
            z = np.convolve(y, irf_shift(irf, offset))
            z += baseline
            return z[:x.size]
        
        def residuals_3exp(params, x, y, weights):
            """
            Returns the array of residuals for the current parameters.
            """
            tau_1 = params["tau_1"].value
            tau_2 = params["tau_2"].value
            tau_3 = params["tau_3"].value
            ampl_1 = params["ampl_1"].value
            ampl_2 = params["ampl_2"].value
            ampl_3 = params["ampl_3"].value
            baseline = params["baseline"].value
            offset = params["offset"].value
            return (y - model_3exp(x, tau_1, tau_2, tau_3, ampl_1, ampl_2, ampl_3, baseline, offset)) * weights

    #%%  Weights for proper scaling of the residuals according to the Poisson standard deviation.
    
    weights = 1/np.sqrt(decay_hist)
    weights[decay_hist == 0] = 1./np.sqrt(1)
    (decay_hist == 0).any()

    #%% actual fitting

        
    if fit_order == 'single':

        # Here we initialize the fit parameters with values with boundaries
        #SET BOUNDARIES 1exp  value= lifetime set vary to Fasle to fix a fit parameter min and max boundary
        params_1exp = lmfit.Parameters()
        params_1exp.add("tau",      value=ig_t1,      vary =True,  min=0.2, max=20)
        params_1exp.add("ampl",     value=ig_ampl_1,  vary =True,  min=0)
        params_1exp.add("baseline", value=ig_baseline,vary =True,  min=0)
        params_1exp.add("offset",   value=ig_offset,  vary =True,  min=-5, max=5)

        params_1exp_ig = copy.deepcopy(params_1exp)
        
        mini_1exp = lmfit.Minimizer(
            residuals_1exp,
            params_1exp,
            fcn_args=(time_ns, decay_hist, weights),
            nan_policy="omit"
        )
        fit_nlls_1exp = mini_1exp.minimize(method='leastsq')
        print('\nNLLS fit results lmfit.minimizer')
        lmfit.report_fit(fit_nlls_1exp,show_correl=False)
        
        t_nlls_1exp = fit_nlls_1exp.params["tau"].value
        a_nlls_1exp = fit_nlls_1exp.params["ampl"].value
        file_fit_values= pd.DataFrame([[tail,t_nlls_1exp,a_nlls_1exp]],   columns=['file','t1', 'a1'])

    #%% In[20]:

    if fit_order == 'double':

        #SET BOUNDARIES 2exp  value= lifetime set vary to Fasle to fix a fit parameter min and max boundary
        params_2exp = lmfit.Parameters()
        params_2exp.add("tau_1",  value=ig_t1,     vary =True, min=0.3, max=0.8)
        params_2exp.add("tau_2",  value=ig_t2,     vary =True, min=0.8, max=4.4)
        params_2exp.add("ampl_1", value=ig_ampl_1, vary =True, min=0)
        params_2exp.add("ampl_2", value=ig_ampl_2, vary =True, min=0)
        params_2exp.add("baseline", value=ig_baseline, vary =True, min= 0)
        params_2exp.add("offset", value=ig_offset,     vary =True, min=-5, max=10)
        
        params_2exp_ig = copy.deepcopy(params_2exp)
               
        mini_2exp = lmfit.Minimizer(
            residuals_2exp,
            params_2exp,
            fcn_args=(time_ns, decay_hist, weights),
            nan_policy="omit"
        )
        fit_nlls_2exp = mini_2exp.minimize(method='leastsq')
        print('\nNLLS fit results lmfit.minimizer')
        lmfit.report_fit(fit_nlls_2exp,show_correl=False)

        t1_nlls_2exp = fit_nlls_2exp.params["tau_1"].value
        t2_nlls_2exp = fit_nlls_2exp.params["tau_2"].value
        a1_nlls_2exp = fit_nlls_2exp.params["ampl_1"].value
        a2_nlls_2exp = fit_nlls_2exp.params["ampl_2"].value
        f1_nlls_2exp = a1_nlls_2exp / (a1_nlls_2exp + a2_nlls_2exp)
        f2_nlls_2exp = a2_nlls_2exp / (a1_nlls_2exp + a2_nlls_2exp)
        tav_nlls_2exp =((a1_nlls_2exp/time_step_ns)*t1_nlls_2exp**2+(a2_nlls_2exp/time_step_ns)*t2_nlls_2exp**2)/sum(decay_hist)
        file_fit_values= pd.DataFrame([[tail,t1_nlls_2exp,t2_nlls_2exp,tav_nlls_2exp,a1_nlls_2exp,a2_nlls_2exp]],   columns=['file','t1', 't2','tav', 'a1', 'a2'])
        

    # %%

    if fit_order == 'triple':
        #SET BOUNDARIES 3exp  value= lifetime set vary to Fasle to fix a fit parameter min and max boundary
        
        params_3exp = lmfit.Parameters()
        params_3exp.add("tau_1", value=ig_t1, vary =True,  min=0.2, max=10)
        params_3exp.add("tau_2", value=ig_t2, vary =True, min=0.2, max=40)
        params_3exp.add("tau_3", value=ig_t3, vary =True, min=0.2, max=180)

        params_3exp.add("ampl_1", value=ig_ampl_1, vary =True, min=0)
        params_3exp.add("ampl_2", value=ig_ampl_2, vary =True, min=0)
        params_3exp.add("ampl_3", value=ig_ampl_3, vary =True, min=0)
        params_3exp.add("baseline", value=ig_baseline, vary =True, min=0)
        params_3exp.add("offset", value=ig_offset, vary =True, min=-5, max=5)
        
        params_3exp_ig = copy.deepcopy(params_3exp)

        mini_3exp = lmfit.Minimizer(
            residuals_3exp,
            params_3exp,
            fcn_args=(time_ns, decay_hist, weights),
            nan_policy="omit"
        )
        fit_nlls_3exp = mini_3exp.minimize(method='leastsq')
        print('\nNLLS fit results lmfit.minimizer')
        lmfit.report_fit(fit_nlls_3exp,show_correl=False)
        
        t1_nlls_3exp = fit_nlls_3exp.params["tau_1"].value
        t2_nlls_3exp = fit_nlls_3exp.params["tau_2"].value
        t3_nlls_3exp = fit_nlls_3exp.params["tau_3"].value
        a1_nlls_3exp = fit_nlls_3exp.params["ampl_1"].value
        a2_nlls_3exp = fit_nlls_3exp.params["ampl_2"].value
        a3_nlls_3exp = fit_nlls_3exp.params["ampl_3"].value
        f1_nlls_3exp = a1_nlls_3exp / \
            (a1_nlls_3exp + a2_nlls_3exp + a3_nlls_3exp)
        f2_nlls_3exp = a2_nlls_3exp / \
            (a1_nlls_3exp + a2_nlls_3exp + a3_nlls_3exp)
        f3_nlls_3exp = a3_nlls_3exp / \
            (a1_nlls_3exp + a2_nlls_3exp + a3_nlls_3exp)
        tav_nlls_3exp =((a1_nlls_3exp/time_step_ns)*t1_nlls_3exp**2+(a2_nlls_3exp/time_step_ns)*t2_nlls_3exp**2+(a3_nlls_3exp/time_step_ns)*t3_nlls_3exp**2)/sum(decay_hist)
        
        file_fit_values= pd.DataFrame([[tail,t1_nlls_3exp,t2_nlls_3exp,t3_nlls_3exp,tav_nlls_3exp,a1_nlls_3exp,a2_nlls_3exp,a2_nlls_3exp]],   columns=['file','t1', 't2','t3', 'tav', 'a1', 'a2','a3'])
    

     
    # In[31]:
    # LOAD some conversion routines needed for 'Nleder-Mead'
    if method == 'Nelder-Mead':
        def loglike_1exp(params, x, ydata):
            tau, ampl, baseline, offset = params
            ymodel = model_1exp(x, tau, ampl, baseline, offset)
            return (ymodel - ydata*np.log(ymodel)).sum()
    
        def loglike_2exp(params, x, ydata):
            tau_1, tau_2, ampl_1, ampl_2, baseline, offset = params
            ymodel = model_2exp(x, tau_1, tau_2, ampl_1, ampl_2, baseline, offset)
            return (ymodel - ydata*np.log(ymodel)).sum()
    
        def loglike_3exp(params, x, ydata):
            tau_1, tau_2, tau_3, ampl_1, ampl_2, ampl_3, baseline, offset = params
            ymodel = model_3exp(x, tau_1, tau_2, tau_3, ampl_1,
                                ampl_2, ampl_3, baseline, offset)
            return (ymodel - ydata*np.log(ymodel)).sum()
    
        def from_params_1exp(params):
            return [params[k].value for k in ("tau", "ampl", "baseline", "offset")]
    
        def bounds_from_params_1exp(params) :
            bounds=[]    
            for k in ("tau", "ampl", "baseline", "offset"):
                bounds.append((params[k].min, params[k].max))
            return bounds
    
        def to_params_1exp(p, params):
            for v, k in zip(p, ("tau", "ampl", "baseline", "offset")):
                params[k].value = v
            return params
    
        def residuals_mle_1exp(params, x, y, weights):
            tau = params["tau"].value
            ampl = params["ampl"].value
            baseline = params["baseline"].value
            offset = params["offset"].value
            return (y - model_1exp(x, tau, ampl, baseline, offset)) * weights
    
        def from_params_2exp(params):
            return [params[k].value for k in ("tau_1", "tau_2", "ampl_1", "ampl_2", "baseline", "offset")]
    
        def bounds_from_params_2exp(params) :
            bounds=[]    
            for k in ("tau_1", "tau_2", "ampl_1", "ampl_2", "baseline", "offset"):
                bounds.append((params[k].min, params[k].max))
            return bounds
        
        def to_params_2exp(p, params):
            for v, k in zip(p, ("tau_1", "tau_2", "ampl_1", "ampl_2", "baseline", "offset")):
                params[k].value = v
            return params
    
        def residuals_mle_2exp(params, x, y, weights):
            tau_1 = params["tau_1"].value
            tau_2 = params["tau_2"].value
            ampl_1 = params["ampl_1"].value
            ampl_2 = params["ampl_2"].value
            baseline = params["baseline"].value
            offset = params["offset"].value
            return (y - model_2exp(x, tau_1, tau_2, ampl_1, ampl_2, baseline, offset)) * weights
    
        def from_params_3exp(params):
            return [params[k].value for k in ("tau_1", "tau_2", "tau_3", "ampl_1", "ampl_2", "ampl_3", "baseline", "offset")]
    
        def to_params_3exp(p, params):
            for v, k in zip(p, ("tau_1", "tau_2", "tau_3", "ampl_1", "ampl_2", "ampl_3", "baseline", "offset")):
                params[k].value = v
            return params
    
        def bounds_from_params_3exp(params) :
            bounds=[]    
            for k in ("tau_1", "tau_2", "tau_3", "ampl_1", "ampl_2", "ampl_3", "baseline", "offset"):
                bounds.append((params[k].min, params[k].max))
            return bounds
        
        def residuals_mle_3exp(params, x, y, weights):
            tau_1 = params["tau_1"].value
            tau_2 = params["tau_2"].value
            tau_3 = params["tau_3"].value
            ampl_1 = params["ampl_1"].value
            ampl_2 = params["ampl_2"].value
            ampl_3 = params["ampl_3"].value
            baseline = params["baseline"].value
            offset = params["offset"].value
            return (y - model_3exp(x, tau_1, tau_2, tau_3, ampl_1, ampl_2, ampl_3, baseline, offset)) * weights

    # In[33]: actual fitting is done here

        
        
    if fit_order == 'single' and method == 'Nelder-Mead':
        
        if Feed_NLLSfit_as_guess_MLE:
            MLE_ig = from_params_1exp(fit_nlls_1exp.params)
        else:
            MLE_ig = from_params_1exp(params_1exp)
            
                
        fit_mle_1exp = minimize(loglike_1exp, x0=MLE_ig, args=(time_ns, decay_hist), method='Nelder-Mead', bounds=bounds_from_params_1exp(params_1exp))
        fit_mle_1exp_params = to_params_1exp(fit_mle_1exp.x, params_1exp)
        print('\nMLE-2exp fit results scipy.optimize')
        lmfit.report_fit(fit_mle_1exp_params)

        mle_1exp_redchi = (residuals_mle_1exp(params_1exp, time_ns, decay_hist, weights)**2).sum() / (time_ns.size - fit_mle_1exp.x.size)
        a_mle_1exp = fit_mle_1exp_params["ampl"].value
        t_mle_1exp = fit_mle_1exp_params["tau"].value
        tav_mle_1exp = t_mle_1exp
        file_fit_values= pd.DataFrame([[tail,t_mle_1exp,tav_mle_1exp,a_mle_1exp]],   columns=['file','t1' ,'tav', 'a1' ])
    
    elif fit_order == 'double' and method == 'Nelder-Mead':
        
        if Feed_NLLSfit_as_guess_MLE:
            MLE_ig = from_params_2exp(fit_nlls_2exp.params)
            
        else:
            MLE_ig = from_params_2exp(params_2exp)
            
            
        fit_mle_2exp = minimize(loglike_2exp, x0=MLE_ig, args=(time_ns, decay_hist), method='Nelder-Mead', bounds=bounds_from_params_2exp(params_2exp))
        
        fit_mle_2exp_params = to_params_2exp(fit_mle_2exp.x, params_2exp)
        
        print('\nMLE-2exp fit results scipy.optimize')
        lmfit.report_fit(fit_mle_2exp_params)

        mle_2exp_redchi = (residuals_mle_2exp(params_2exp, time_ns, decay_hist,
                           weights)**2).sum() / (time_ns.size - fit_mle_2exp.x.size)
        
        t1_mle_2exp = fit_mle_2exp_params["tau_1"].value
        t2_mle_2exp = fit_mle_2exp_params["tau_2"].value
        a1_mle_2exp = fit_mle_2exp_params["ampl_1"].value
        a2_mle_2exp = fit_mle_2exp_params["ampl_2"].value
        f1_mle_2exp = a1_mle_2exp / (a1_mle_2exp + a2_mle_2exp)
        f2_mle_2exp = a2_mle_2exp / (a1_mle_2exp + a2_mle_2exp)
        tav_mle_2exp= ((a1_mle_2exp/time_step_ns)*t1_mle_2exp**2+(a2_mle_2exp/time_step_ns)*t2_mle_2exp**2)/sum(decay_hist)
        file_fit_values= pd.DataFrame([[tail,t1_mle_2exp,t2_mle_2exp,tav_mle_2exp,a1_mle_2exp,a2_mle_2exp]],   columns=['file','t1', 't2','tav', 'a1', 'a2'])
        
    elif fit_order == 'triple' and method == 'Nelder-Mead':
        
        if Feed_NLLSfit_as_guess_MLE:
            MLE_ig = from_params_3exp(fit_nlls_3exp.params)
            
        else:
            MLE_ig = from_params_3exp(params_3exp)
        
        fit_mle_3exp = minimize(loglike_3exp, x0=MLE_ig, args=(time_ns, decay_hist), method='Nelder-Mead', bounds=bounds_from_params_3exp(params_3exp))
        fit_mle_3exp_params = to_params_3exp(fit_mle_3exp.x, params_3exp)
        print('\nMLE-3exp fit results scipy.optimize')
        lmfit.report_fit(fit_mle_3exp_params)

        mle_3exp_redchi = (residuals_mle_3exp(params_3exp, time_ns, decay_hist,
                           weights)**2).sum() / (time_ns.size - fit_mle_3exp.x.size)

        t1_mle_3exp = fit_mle_3exp_params["tau_1"].value
        t2_mle_3exp = fit_mle_3exp_params["tau_2"].value
        t3_mle_3exp = fit_mle_3exp_params["tau_3"].value
        a1_mle_3exp = fit_mle_3exp_params["ampl_1"].value
        a2_mle_3exp = fit_mle_3exp_params["ampl_2"].value
        a3_mle_3exp = fit_mle_3exp_params["ampl_3"].value
        f1_mle_3exp = a1_mle_3exp / (a1_mle_3exp + a2_mle_3exp + a3_mle_3exp)
        f2_mle_3exp = a2_mle_3exp / (a1_mle_3exp + a2_mle_3exp + a3_mle_3exp)
        f3_mle_3exp = a3_mle_3exp / (a1_mle_3exp + a2_mle_3exp + a3_mle_3exp)
        tav_mle_3exp= ((a1_mle_3exp/time_step_ns)*t1_mle_3exp**2+(a2_mle_3exp/time_step_ns)*t2_mle_3exp**2+(a3_mle_3exp/time_step_ns)*t3_mle_3exp**2)/sum(decay_hist)
        file_fit_values= pd.DataFrame([[tail,t1_mle_3exp,t2_mle_3exp,t3_mle_3exp,tav_mle_3exp,a1_mle_3exp,a2_mle_3exp,a3_mle_3exp]],   columns=['file','t1', 't2','3','tav', 'a1', 'a2', 'a3'])



# %%  plotting single exp

    if fit_order == 'single':
       
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={
                               'height_ratios': [7, 2]})
        
        ax[0].semilogy(time_ns, irf_display, ".", label="IRF", alpha=0.3)
        ax[0].semilogy(time_ns, decay_hist, ".", label="Decay", alpha=0.3)
        if plot_ig_fig:
            ax[0].semilogy(time_ns, model_1exp(
                time_ns, **{k: v.value for k, v in params_1exp_ig.items()}), "steelblue", label="initial guess" , alpha =0.15)
        
        if method == 'leastsq':
            ax[0].semilogy(time_ns, model_1exp(
                time_ns, **{k: v.value for k, v in fit_nlls_1exp.params.items()}), "g", label="NLLS | leastsq | 1-exp")
            ax[1].plot(time_ns, fit_nlls_1exp.residual, "g")
            ym = max(fit_nlls_1exp.residual)
            para = f"$\\tau$ = {t_nlls_1exp:.2f} ns" + \
                "\n\n" + f"A$_1$ = {a_nlls_1exp:.2f}"
            ax[1].text(.85, 0.96, f'$\chi_v^2$ = {fit_nlls_1exp.redchi:.3f}', va="top", ha="left", transform=ax[1].transAxes, fontsize=11)

        elif method == 'Nelder-Mead':
            ax[0].semilogy(time_ns, model_1exp(
                time_ns, **{k: v.value for k, v in fit_mle_1exp_params.items()}), "g", label="MLE 1-exp")
            ax[1].plot(time_ns, residuals_mle_1exp(
                params_1exp, time_ns, decay_hist, weights), "g")
            ym = max([np.abs(residuals_mle_1exp(params_1exp, time_ns, decay_hist, weights)).max(
            ), residuals_mle_1exp(params_1exp, time_ns, decay_hist, weights).max()])
            para = f"$\\tau_1$ = {t_mle_1exp:.2f} ns"
            ax[1].text(.85, 0.96, f'$\chi_v^2$ = {mle_1exp_redchi:.3f}', va="top", ha="left", transform=ax[1].transAxes, fontsize=11)
            
        ax[0].text(.85, 0.95, para, va="top", ha="left",
                   transform=ax[0].transAxes, fontsize=12)

        ax[0].grid(True)
        ax[0].grid(True, which="minor", lw=0.3)
        ax[0].set_title(sample_name+"  |  1 exponential fit  "+tail)
        ax[0].set_ylabel("Counts [-]")
        ax[0].set_ylim(1)
        ax[0].set_xlim(0,xlim_max)
        ax[0].legend(loc='upper center', fontsize=11)

        ym = 20
        ax[1].set_title('Residuals')
        ax[1].text(.85, 0.96, f'$\chi_v^2$ = {fit_nlls_1exp.redchi:.3f}', va="top", ha="left", transform=ax[1].transAxes, fontsize=11)
        ax[1].set_ylim(-ym, ym)
        ax[1].set_xlim(0,xlim_max)
        ax[1].set_xlabel("Time [ns]")
        ax[1].set_ylabel("Residuals [-]")
        ax[1].grid(True)
        plt.show()
        
        if save_fig:
            fname_save = head+output_dname
            if not os.path.exists(fname_save):
                os.makedirs(fname_save)
            fig.savefig(fname_save+os.path.splitext(tail)
                        [0]+'_1expfit.png', bbox_inches='tight', dpi=300)

            if Save_data_files==True:
                save_df = pd.DataFrame({'time ns': time_ns, 'IRF': irf_display,'decay': decay_hist, 'fit': model_1exp(time_ns, **{k: v.value for k, v in fit_nlls_1exp.params.items()}), 'residuals':fit_nlls_1exp.residual})
                fname_save = head+output_dname
                if not os.path.exists(fname_save):
                    os.makedirs(fname_save)
                save_df.to_csv(fname_save+os.path.splitext(tail)[0]+'_1expfit.dat', index=False,  float_format='%.3f')
                
    
# %% Plotting Double EXP

    if fit_order == 'double' :

        para = f"$\\tau_1$ = {t1_nlls_2exp:.2f} ns" + "\n" + f"$\\tau_2$ = {t2_nlls_2exp:.2f} ns" + "\n" + \
            "$\\tau {int}$"+f"= {tav_nlls_2exp:.2f} ns"     +\
            "\n\n"+f"A$_1$ = {a1_nlls_2exp:.3g}"  +\
            "\n" + f"A$_2$ = {a2_nlls_2exp:.3g}" +\
            "\n" + f"f$_1$ = {f1_nlls_2exp*100:.3g}%" +\
            "\n" + f"f$_2$ = {f2_nlls_2exp*100:.3g}%"
        
        fig, ax = plt.subplots(2, 2, figsize=(12, 6),  gridspec_kw={
                               'height_ratios': [7, 2], 'width_ratios': [5, 1.5]})

        ax[0, 0].semilogy(time_ns, irf_display, ".", label="IRF", alpha=0.5)
        ax[0, 0].semilogy(time_ns, decay_hist, ".", label="Decay", alpha=0.1)
        if plot_ig_fig:
            ax[0, 0].semilogy(time_ns, model_2exp(
                time_ns, **{k: v.value for k, v in params_2exp_ig.items()}), "steelblue", label="initial guess: SET→ NLLS" , alpha =0.15)
        ax[0, 0].semilogy(time_ns, model_2exp(
            time_ns, **{k: v.value for k, v in fit_nlls_2exp.params.items()}), "g", label="NLLS | leastsq | 2-exp" , alpha =0.8)
        ym = max(fit_nlls_2exp.residual)
        ax[0, 0].text(.79, 0.97, para, va="top", ha="left",
                      transform=ax[0, 0].transAxes, fontsize=12)
        ax[0, 0].grid(True)
        ax[0, 0].grid(True, which="minor", lw=0.3)
        ax[0, 0].set_title(sample_name+"  |  2exp fit  "+ "\n"+ tail, fontsize=11)
        ax[0, 0].set_ylabel("Counts [-]")
        ax[0, 0].set_xlim(0,xlim_max)
        ax[0, 0].set_ylim(0.7)
        ax[0, 0].legend(loc='upper center', fontsize=11)

        ax[0, 1].semilogy(time_ns, irf_display, ".", label="IRF", alpha=0.5)
        ax[0, 1].semilogy(time_ns, decay_hist, ".", label="Decay", alpha=0.3)
        ax[0, 1].semilogy(time_ns, model_2exp(
            time_ns, **{k: v.value for k, v in fit_nlls_2exp.params.items()}), "g", label="fit NLLS 2-exp")
        ax[0, 1].set_title('Short time')
        ax[0, 1].set_xlim(2, 5)
        ax[0, 1].grid(True)
        ax[0, 1].grid(True, which="minor", lw=0.3)
        ax[0, 1].set_ylabel("Counts [-]")

        ax[1, 1].plot(time_ns, fit_nlls_2exp.residual, "g", alpha =0.6)
        ax[1, 1].set_xlim(2, 5)
        ax[1, 1].grid(True)
        ax[1, 1].grid(True, which="minor", lw=0.3)
        ax[1, 1].set_xlabel("Time [ns]")

        ym = 10
        ax[1, 0].plot(time_ns, fit_nlls_2exp.residual, "g", alpha =0.6)
        ax[1, 0].set_title('Residuals' )
        ax[1, 0].set_ylim(-ym, ym)
        ax[1, 0].set_xlim(0,xlim_max)
        ax[1, 0].text(.79, 0.96, f'$\chi_v^2$ = {fit_nlls_2exp.redchi:.3f}', va="top", ha="left", transform=ax[1, 0].transAxes, fontsize=11)
        ax[1, 0].set_xlabel("Time [ns]")
        ax[1, 0].set_ylabel("Residuals [-]")
        ax[1, 0].grid(True)
        plt.show()
        if save_fig:
            fname_save = head+output_dname
            if not os.path.exists(fname_save):
                os.makedirs(fname_save)
            fig.savefig(fname_save+os.path.splitext(tail)
                        [0]+'__2expfit.png', bbox_inches='tight', dpi=300)

        if Save_data_files==True:
            save_df = pd.DataFrame({'time ns': time_ns, 'IRF': irf_display,'decay': decay_hist, 'fit': model_2exp(time_ns, **{k: v.value for k, v in fit_nlls_2exp.params.items()}), 'residuals':fit_nlls_2exp.residual})
            fname_save = head+output_dname
            if not os.path.exists(fname_save):
                os.makedirs(fname_save)
            save_df.to_csv(fname_save+os.path.splitext(tail)[0]+'__2expfit.dat', index=False, float_format='%.3f')

# %%
    if fit_order == 'double' and method =='Nelder-Mead' :
           
       
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [7, 2]})

        ax[0].semilogy(time_ns, irf_display, ".", label="IRF", alpha=0.5)
        ax[0].semilogy(time_ns, decay_hist, ".", label="Decay", alpha=0.1)

        

        if method == 'leastsq':
            
            if plot_ig_fig:
                ax[0].semilogy(time_ns, model_2exp(time_ns, **{k: v.value for k, v in params_2exp_ig.items()}), "steelblue", label="initial guess" , alpha =0.15)

            ax[0].semilogy(time_ns, model_2exp(
                time_ns, **{k: v.value for k, v in fit_nlls_2exp.params.items()}), "g", label="NLLS | leastsq | 2-exp")
            ax[1].plot(time_ns, fit_nlls_2exp.residual, "g", alpha =0.6)
            ym = max(fit_nlls_2exp.residual)
            para = f"$\\tau_1$ = {t1_nlls_2exp:.2f} ns" + "\n" + f"$\\tau_2$ = {t2_nlls_2exp:.2f} ns" + "\n" + \
                "$\\tau {int}$"+f"= {tav_nlls_2exp:.2f} ns" + "\n\n" + \
                f"A$_1$ = {a1_nlls_2exp:.3g} \n"      +\
                f"A$_2$ = {a2_nlls_2exp:.3g} \n"      +\
                f"f$_2$ = {f1_nlls_2exp*100:.1f}% \n" +\
                f"f$_2$ = {f2_nlls_2exp*100:.1f}%"
                
                
            ax[1].text(.80, 0.96, f'$\chi_v^2$ = {fit_nlls_2exp.redchi:.3f}', va="top", ha="left", transform=ax[1].transAxes, fontsize=11)
            
        elif method == 'Nelder-Mead':
            
            if plot_ig_fig and Feed_NLLSfit_as_guess_MLE:
                ax[0].semilogy(time_ns, model_2exp(time_ns, **{k: v.value for k, v in params_2exp_ig.items()}), "steelblue", label="initial guess 1: SET  → NLLS" , alpha =0.25)
                ax[0].semilogy(time_ns, model_2exp(time_ns, **{k: v.value for k, v in params_2exp.items()}), "steelblue", label="initial guess 2: NLLS→ MLE" , alpha =0.15)
                
            elif plot_ig_fig:
                ax[0].semilogy(time_ns, model_2exp(time_ns, **{k: v.value for k, v in params_2exp_ig.items()}), "steelblue", label="initial guess: set→ NLLS" , alpha =0.15)
                
            ax[0].semilogy(time_ns, model_2exp(
                time_ns, **{k: v.value for k, v in fit_mle_2exp_params.items()}), "g", label="fit MLE 2-exp")
            
            
                
            ax[1].plot(time_ns, residuals_mle_2exp(
                params_2exp, time_ns, decay_hist, weights), "g", alpha=0.6)
            ym = max([np.abs(residuals_mle_2exp(params_2exp, time_ns, decay_hist, weights)).max(
            ), residuals_mle_2exp(params_2exp, time_ns, decay_hist, weights).max()])
            para = f"$\\tau_1$ = {t1_mle_2exp:.2f} ns" + "\n" + f"$\\tau_2$ = {t2_mle_2exp:.2f} ns" + \
                "\n" + "$\\tau {int}$"+ f" = {tav_mle_2exp:.2f} ns \n\n" + \
                f"A$_1$ = {a1_mle_2exp:.3g} \n"      + \
                f"A$_2$ = {a2_mle_2exp:.3g} \n"      + \
                f"f$_1$ = {f1_mle_2exp*100:.1f}% \n" + \
                f"f$_2$ = {f2_mle_2exp*100:.1f}%"
                
                
            ax[1].text(.80, 0.96, f'$\chi_v^2$ = {mle_2exp_redchi:.3f}', va="top", ha="left", transform=ax[1].transAxes, fontsize=11)    
            
        ax[0].text(.80, 0.95, para, va="top", ha="left",
                   transform=ax[0].transAxes, fontsize=11)

        ax[0].grid(True)
        ax[0].grid(True, which="minor", lw=0.3)
        ax[0].set_title(sample_name+"  |  2 exponential fit "+"\n"+tail)
        ax[0].set_ylabel("Counts [-]")
        ax[0].set_ylim(0.7)
        ax[0].set_xlim(0,xlim_max)
        ax[0].legend(loc='upper center', fontsize=11)

        ym = 10
        ax[1].set_title('Residuals')
        ax[1].set_ylim(-ym, ym)
        ax[1].set_xlim(0,xlim_max)
        ax[1].set_xlabel("Time [ns]")
        ax[1].set_ylabel("Residuals [-]")
        
        ax[1].grid(True)
        plt.show()
        if save_fig:
            fname_save = head+output_dname
            if not os.path.exists(fname_save):
                os.makedirs(fname_save)
            fig.savefig(fname_save+os.path.splitext(tail)
                        [0]+'_2expfit.png', bbox_inches='tight', dpi=300)
        
        if Save_data_files:
            save_df = pd.DataFrame({'time ns': time_ns, 'IRF': irf_display,'decay': decay_hist, 'fit': model_2exp(time_ns, **{k: v.value for k, v in fit_nlls_2exp.params.items()}), 'residuals':fit_nlls_2exp.residual})
            fname_save = head+output_dname
            if not os.path.exists(fname_save):
                os.makedirs(fname_save)
            save_df.to_csv(fname_save+os.path.splitext(tail)[0]+'_2expfit.dat',index=False, float_format='%.3f')

# %% Plotting Triple EXP

    if fit_order == 'triple':

        para = f"$\\tau_1$ = {t1_nlls_3exp:.2f} ns" + "\n" + f"$\\tau_2$ = {t2_nlls_3exp:.2f} ns" + "\n" + f"$\\tau_3$ = {t3_nlls_3exp:.2f} ns" + "\n" + \
            "$\\tau {int}$"+f"= {tav_nlls_3exp:.2f} ns" + \
            "\n\n"+f"A$_1$ = {a1_nlls_3exp:.3g}"        + \
            "\n" + f"A$_2$ = {a2_nlls_3exp:.3g} "       + \
            "\n" + f"A$_3$ = {a3_nlls_3exp:.3g}"

        fig, ax = plt.subplots(2, 2, figsize=(12, 6),  gridspec_kw={
                               'height_ratios': [7, 2], 'width_ratios': [5, 1.5]})

        ax[0, 0].semilogy(time_ns, irf_display, ".", label="IRF", alpha=0.5)
        ax[0, 0].semilogy(time_ns, decay_hist, ".", label="Decay", alpha=0.1)
        if plot_ig_fig:
            ax[0, 0].semilogy(time_ns, model_3exp(
                time_ns, **{k: v.value for k, v in params_3exp_ig.items()}), "steelblue", label="initial guess" , alpha =0.15)
        ax[0, 0].semilogy(time_ns, model_3exp(
            time_ns, **{k: v.value for k, v in fit_nlls_3exp.params.items()}), "g", label="NLLS | leastsq | 3-exp" , alpha =0.8)
        ym = max(fit_nlls_3exp.residual)
        ax[0, 0].text(.79, 0.97, para, va="top", ha="left",
                      transform=ax[0, 0].transAxes, fontsize=12)
        ax[0, 0].grid(True)
        ax[0, 0].grid(True, which="minor", lw=0.3)
        ax[0, 0].set_title(sample_name+"  |  3exp fit "+"\n"+ tail)
        ax[0, 0].set_ylabel("Counts [-]")
        ax[0, 0].set_xlim(0,xlim_max)
        ax[0, 0].set_ylim(0.7)
        ax[0, 0].legend(loc='upper center', fontsize=11)

        ax[0, 1].semilogy(time_ns, irf_display, ".", label="IRF", alpha=0.5)
        ax[0, 1].semilogy(time_ns, decay_hist, ".", label="Decay", alpha=0.3)
        ax[0, 1].semilogy(time_ns, model_3exp(
            time_ns, **{k: v.value for k, v in fit_nlls_3exp.params.items()}), "g", label="NLLS | leastsq | 3-exp")
        ax[0, 1].set_title('Short time')
        ax[0, 1].set_xlim(2, 5)
        ax[0, 1].grid(True)
        ax[0, 1].grid(True, which="minor", lw=0.3)
        ax[0, 1].set_ylabel("Counts [-]")

        ax[1, 1].plot(time_ns, fit_nlls_3exp.residual, "g", alpha =0.6)
        ax[1, 1].set_xlim(2, 5)
        ax[1, 1].grid(True)
        ax[1, 1].grid(True, which="minor", lw=0.3)
        ax[1, 1].set_xlabel("Time [ns]")

        ym = 10
        ax[1, 0].plot(time_ns, fit_nlls_3exp.residual, "g", alpha =0.6)
        ax[1, 0].set_title('Residuals' )
        ax[1, 0].set_ylim(-ym, ym)
        ax[1, 0].set_xlim(0,xlim_max)
        ax[1, 0].text(.79, 0.96, f'$\chi_v^2$ = {fit_nlls_3exp.redchi:.3f}', va="top", ha="left", transform=ax[1, 0].transAxes, fontsize=11)
        ax[1, 0].set_xlabel("Time [ns]")
        ax[1, 0].set_ylabel("Residuals [-]")
        ax[1, 0].grid(True)
        plt.show()
        if save_fig:
            fname_save = head+output_dname
            if not os.path.exists(fname_save):
                os.makedirs(fname_save)
            fig.savefig(fname_save+os.path.splitext(tail)
                        [0]+'_3expfit.png', bbox_inches='tight', dpi=300)

        if Save_data_files==True:
            save_df = pd.DataFrame({'time ns': time_ns, 'IRF': irf_display,'decay': decay_hist, 'fit': model_3exp(time_ns, **{k: v.value for k, v in fit_nlls_3exp.params.items()}), 'residuals':fit_nlls_3exp.residual})
            fname_save = head+output_dname
            if not os.path.exists(fname_save):
                os.makedirs(fname_save)
            save_df.to_csv(fname_save+os.path.splitext(tail)[0]+'_3expfit.dat', index=False, float_format='%.3f')
        

    Group_fit_values_df=pd.concat([Group_fit_values_df,file_fit_values])    
    
    j += 1


#fin max index. 




if Save_fit_values:
  
    fname_save = head+output_dname
    if not os.path.exists(fname_save):
        os.makedirs(fname_save)
    Group_fit_values_df.to_csv(fname_save+'X_'+os.path.splitext(tail)[0]+'_fit_values.dat', index=False, float_format='%.4f')





