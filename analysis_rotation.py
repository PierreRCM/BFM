# This is a collection of scripts written at the beginning of the project.
# Good stuff can be buried here, but requires some better organization.
# FP Aug 2019
#


import nptdms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mat
import progressbar 
#from scipy.optimize import curve_fit
import scipy.stats as ss
import math
import os
from sklearn.neighbors import KernelDensity
import re
import sys
import importlib

if '/home/francesco/scripts/bactMotor/py/' not in sys.path:
    sys.path.append(r'/home/francesco/scripts/bactMotor/py')
import DragRotatingBead
importlib.reload(DragRotatingBead)

if '/home/francesco/scripts/npTDMS' not in sys.path:
    sys.path.append(r'/home/francesco/scripts/npTDMS')
import openTDMS

if '/home/francesco/scripts/ellipseFit' not in sys.path:
    sys.path.append(r'/home/francesco/scripts/ellipseFit')
import fitEllipse
importlib.reload(fitEllipse)

if '/home/francesco/scripts/filters' not in sys.path:
    sys.path.append(r'/home/francesco/scripts/filters')
import filters
importlib.reload(filters)

if '/home/francesco/scripts/fitting' not in sys.path:
    sys.path.append(r'/home/francesco/scripts/fitting')
import fitGaussian
importlib.reload(fitGaussian)

# pomegranate not working in python3:
#if '/home/francesco/scripts/hmm' not in sys.path:
#    sys.path.append(r'/home/francesco/scripts/hmm')
#import hmm_FP1
#importlib.reload(hmm_FP1)


global pxsize 
# pixel size m/pixel :
pxsize = 147.5e-9



def openTdmsFile(tdms_file, plots=1):
    '''open file and remove reference bead, and plots
    return x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3, 
           x1r,y1r,x2r,y2r,x3r,y3r 
    xr yr : rotating - ref bead, compatibly with numb.of elements (warning is given)'''
    # TODO: output = dict()
    tdms_dict = openTDMS.openTdmsFile(tdms_file)
    # number of pts in ROIs traces:
    elem_rois = np.array((0,0,0,0))
    x0 = y0 = z0 = None
    x1 = y1 = z1 = None
    x2 = y2 = z2 = None
    x3 = y3 = z3 = None
    x1r = y1r = None
    
    # ROI 0, reference bead:
    if tdms_dict.has_key('/ROI0_Trk/X0') and tdms_dict.has_key('/ROI0_Trk/Y0'):
        x0 = tdms_dict['/ROI0_Trk/X0'][:-1]
        y0 = tdms_dict['/ROI0_Trk/Y0'][:-1]
        x0 = x0[1:-1]
        y0 = y0[1:-1]
        elem_rois[0] = len(x0)
    else:
        print('error ROI 0')
    if tdms_dict.has_key('/ROI0_Trk/Z0'):
        z0 = tdms_dict['/ROI0_Trk/Z0'][:-1]
        z0 = z0[1:-1]
        print('z0 ok')
    # ROI 1:
    if tdms_dict.has_key('/ROI1_Trk/X1') and tdms_dict.has_key('/ROI1_Trk/Y1'):
        x1 = tdms_dict['/ROI1_Trk/X1'][:-1]
        y1 = tdms_dict['/ROI1_Trk/Y1'][:-1]
        x1 = x1[1:-1]
        y1 = y1[1:-1]
        elem_rois[1] = len(x1)
    else:
        print('error ROI 1')
    if tdms_dict.has_key('/ROI1_Trk/Z1'):
        z1 = tdms_dict['/ROI1_Trk/Z1'][:-1]
        z1 = z1[1:-1]
        print('z1 ok')
    # ROI 2:
    if tdms_dict.has_key('/ROI2_Trk/X2') and tdms_dict.has_key('/ROI2_Trk/Y2'):
        x2 = tdms_dict['/ROI2_Trk/X2'][:-1]
        y2 = tdms_dict['/ROI2_Trk/Y2'][:-1]
        x2 = x2[1:-1]
        y2 = y2[1:-1]
        elem_rois[2] = len(x2)
    else:
        print('error ROI 2')
    if tdms_dict.has_key('/ROI2_Trk/Z2'):
        z2 = tdms_dict['/ROI2_Trk/Z2'][:-1]
        z2 = z2[1:-1]
        print('z2 ok')
    # ROI 3:
    if tdms_dict.has_key('/ROI3_Trk/X3') and tdms_dict.has_key('/ROI3_Trk/Y3'):
        x3 = tdms_dict['/ROI3_Trk/X3'][:-1]
        y3 = tdms_dict['/ROI3_Trk/Y3'][:-1]
        x3 = x3[1:-1]
        y3 = y3[1:-1]
        elem_rois[3] = len(x3)
    else:
        print('error ROI 3')
    if tdms_dict.has_key('/ROI3_Trk/Z3'):
        z3 = tdms_dict['/ROI3_Trk/Z3'][:-1]
        z3 = z3[1:-1]
        print('z3 ok')
    # remove reference:
    last_el = np.min(elem_rois[np.where(elem_rois>0)]) # last element so all len() are =
    print(' Points in ROIs 0-3 = '+str(elem_rois))
    if all(elem_rois - last_el == 0):
        print(' => Traces lengths agree each other')
    else:
        print(' => Traces with different length! trying removing ref. bead' )
    print(' Using '+str(last_el)+' pts.')
    x1r = []; y1r =[]
    x2r = []; y2r =[]
    x3r = []; y3r =[]
    if np.any(x0) and np.any(x1) :
        x1r = x1[:last_el] - x0[:last_el]
        y1r = y1[:last_el] - y0[:last_el]
        print('removing ref in ROI0 from ROI1')
    if np.any(x2) and np.any(x0):
        x2r = x2[:last_el] - x0[:last_el]
        y2r = y2[:last_el] - y0[:last_el]
        print('removing ref in ROI0 from ROI2')
    if np.any(x3) and np.any(y0):
        x3r = x3[:last_el] - x0[:last_el]
        y3r = y3[:last_el] - y0[:last_el]
        print('removing ref in ROI0 from  ROI3')
    if plots:
        plt.figure(325)
        plt.clf()
        if np.any(x0):
            plt.subplot(421)
            plt.plot(x0, y0, '.', ms=0.1)
            plt.title('ROI 0 ref')
            plt.axis('equal')
        if np.any(x1):
            plt.subplot(423)
            plt.plot(x1, y1, '.', ms=0.1)
            plt.title('ROI 1')
            plt.axis('equal')
            plt.subplot(424)
            plt.plot(x1r, y1r, '.', ms=0.1)
            plt.title('ROI 1 corr.')
            plt.axis('equal')
        if np.any(x2):
            plt.subplot(425)
            plt.plot(x2, y2, '.', ms=0.1)
            plt.title('ROI 2')
            plt.axis('equal')
            plt.subplot(426)
            plt.plot(x2r, y2r, '.', ms=0.1)
            plt.title('ROI 2 corr')
            plt.axis('equal')
        if np.any(x3):
            plt.subplot(427)
            plt.plot(x3, y3, '.', ms=0.1)
            plt.title('ROI 3')
            plt.axis('equal')
            plt.subplot(428)
            plt.plot(x3r, y3r, '.', ms=0.1)
            plt.title('ROI 3 corr')
            plt.axis('equal')
 
    return x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3, x1r,y1r,x2r,y2r,x3r,y3r 









def openTdmsFile_old(tdms_file, plots=1):
    '''open file and remove reference bead, and plots
    x1,y1,x0,y0,xr,yr = openTDMS(filename_string)
    x1 y1 : rotating bead raw data 
    x0 y0 : ref bead raw data 
    xr yr : rotating - ref bead, compatibly with numb.of elements (warning is given)'''
    output = dict()
    tdms_dict = openTDMS.openTdmsFile(tdms_file)
    # number of pts in ROIs traces:
    elem_rois = np.array((0,0,0,0))
    x0 = y0 = x1 = y1 = x2 = y2 = x3 = y3 = x1r = y1r = None
    
    # ROI 0, reference bead:
    if tdms_dict.has_key('/ROI0_Trk/X0') and tdms_dict.has_key('/ROI0_Trk/Y0'):
        x0 = tdms_dict['/ROI0_Trk/X0']
        y0 = tdms_dict['/ROI0_Trk/Y0']
        elem_rois[0] = len(x0)
    else:
        print('error ROI 0')
    # ROI 1:
    if tdms_dict.has_key('/ROI1_Trk/X1') and tdms_dict.has_key('/ROI1_Trk/Y1'):
        x1 = tdms_dict['/ROI1_Trk/X1']
        y1 = tdms_dict['/ROI1_Trk/Y1']
        elem_rois[1] = len(x1)
    else:
        print('error ROI 1')
    # ROI 2:
    if tdms_dict.has_key('/ROI2_Trk/X2') and tdms_dict.has_key('/ROI2_Trk/Y2'):
        x2 = tdms_dict['/ROI2_Trk/X2']
        y2 = tdms_dict['/ROI2_Trk/Y2']
        elem_rois[2] = len(x2)
    else:
        print('error ROI 2')
    # ROI 3:
    if tdms_dict.has_key('/ROI3_Trk/X3') and tdms_dict.has_key('/ROI3_Trk/Y3'):
        x3 = tdms_dict['/ROI3_Trk/X3']
        y3 = tdms_dict['/ROI3_Trk/Y3']
        elem_rois[3] = len(x3)
    else:
        print('error ROI 3')
    # remove reference:
    last_el = np.min(elem_rois[np.where(elem_rois>0)]) # last element so all len() are =
    print(' Points in ROIs 0-3 = '+str(elem_rois))
    if all(elem_rois - last_el == 0):
        print(' => Traces lengths agree each other')
    else:
        print(' => Traces with different length! trying removing ref. bead' )
    print(' Using '+str(last_el)+' pts.')
    x1r = []; y1r =[]
    x2r = []; y2r =[]
    x3r = []; y3r =[]
    if np.any(x0) and np.any(x1) :
        x1r = x1[:last_el] - x0[:last_el]
        y1r = y1[:last_el] - y0[:last_el]
        print('removing ref in ROI0 from ROI1')
    if np.any(x2) and np.any(x0):
        x2r = x2[:last_el] - x0[:last_el]
        y2r = y2[:last_el] - y0[:last_el]
        print('removing ref in ROI0 from ROI2')
    if np.any(x3) and np.any(y0):
        x3r = x3[:last_el] - x0[:last_el]
        y3r = y3[:last_el] - y0[:last_el]
        print('removing refin ROI0 from  ROI3')
    if plots:
        plt.figure(325)
        plt.clf()
        if np.any(x0):
            plt.subplot(421)
            plt.plot(x0, y0, '.', ms=0.1)
            plt.title('ROI 0 ref')
        if np.any(x1):
            plt.subplot(423)
            plt.plot(x1, y1, '.', ms=0.1)
            plt.title('ROI 1')
            plt.subplot(424)
            plt.plot(x1r, y1r, '.', ms=0.1)
            plt.title('ROI 1 corr.')
        if np.any(x2):
            plt.subplot(425)
            plt.plot(x2, y2, '.', ms=0.1)
            plt.title('ROI 2')
            plt.subplot(426)
            plt.plot(x2r, y2r, '.', ms=0.1)
            plt.title('ROI 2 corr')
        if np.any(x3):
            plt.subplot(427)
            plt.plot(x3, y3, '.', ms=0.1)
            plt.title('ROI 3')
            plt.subplot(428)
            plt.plot(x3r, y3r, '.', ms=0.1)
            plt.title('ROI 3 corr')
 
    return x0,y0,x1,y1,x2,y2,x3,y3, x1r,y1r,x2r,y2r,x3r,y3r 



    



def XYtoUnwrappedTheta_parallel(x,y, correct='circle_drift'):
    '''computes the angle from x,y data. parallel computation
    correct = ['circle_drift', 'circle']
    '''
    #theta = fitEllipse.XYtoThetaEllipse_parallel(x,y)
    if correct == 'circle_drift':
        theta_rot = fitEllipse.XYtoThetaEllipse_parallel_rot_drift(x,y)
    if correct == 'circle':
        theta_rot = fitEllipse.XYtoThetaEllipse_parallel_rot(x,y)
    return theta_rot



def write_to_text_file_xyphi(filename,x,y,phi,FPS,text=''):
    '''write x,y,phi to the new txt file filename'''
    f = open(filename, 'w')
    if text:
        f.write(text+'\n')
    f.write('x \t y \t phi(deg) \t FPS='+str(FPS)+'\n')
    for i in range(2,len(x)):
        f.write(str(x[i])+'\t'+str(y[i])+'\t'+str(phi[i])+'\n')
    f.close()


def write_to_text_file_torque(filename, torque, FPS=1, text=''):
    '''write a text file with torque'''
    time = np.arange(len(torque))/float(FPS)
    f = open(filename, 'w')
    if text:
        f.write(text+'\n')
    f.write('FPS = '+str(FPS)+'\n')
    f.write('time \t torque (pNnm) \n')
    for i in range(len(torque)):
        f.write(str(time[i])+'\t'+str(torque[i])+'\n')
    f.close()
    plt.figure(697498)
    plt.plot(time, torque)






def dwellTimes_1(phi, FPS, N_360, plots=False, progbar=True, prints=True):
    ''' find the dwell time of phi (only increasing, CCW), 
        linearly interpolating the point at crossing the barriers placed at 
        an interval of 360/N_360 deg.
        phi in degrees. 
        out: dwup (dwell times), dtup (time of crossing)
    '''
    #init:
    import scipy.interpolate
    dtup = [0]
    idxup = []
    phiup = []
    FPS = float(FPS)
    delta = 360./N_360
    if prints:
        print('dwellTimes_1() delta (deg): '+str(delta))
    #print('dwellTimes_1(): Dividing the circle in '+str(N_360)+' intervals, each of '+str(delta)+' deg.' )
    # time array:
    t = np.linspace(0, (len(phi)-1)/FPS, len(phi))    
    # force phi[0] = 0 :
    phi = phi - phi[0]
    # progress bar:
    if progbar:
        print("analysis_rotation.dwellTimes_1(): ...")
        pbar = progressbar.ProgressBar(maxval=phi.size)
        pbar.start()
    k = 1
    for i in range(len(phi)):
        if progbar:
            pbar.update(i)
        if phi[i] > k*delta: #and phi[i-1]<k*delta:
            # lin. interpol y=mx+q:
            m = (phi[i]-phi[i-1])/(t[i]-t[i-1])
            q = phi[i-1] - m*t[i-1]
            t_kdelta = (k*delta-q)/m
            phi_kdelta = m*t_kdelta + q
            # time interpolated exactely at crossing:
            dtup = np.append(dtup, t_kdelta)
            # angle interpolated exactely at crossing:
            phiup = np.append(phiup, phi_kdelta)
            idxup = np.append(idxup, i)
            #if phi[i]-phi[i-1] > delta:
                #print i, phi[i], phi[i-1], phi[i]-phi[i-1]
            #k = k+1
            k = k + np.ceil((phi[i]-phi[i-1])/delta)
    dwup = np.diff(dtup)
    if plots:
         plt.figure('dwellTimes_1()')
         plt.plot(t, phi, 'b-o', ms=5)
         plt.xlabel('time (sec)')
         plt.ylabel('phi (deg)')
         plt.grid(True)
         if np.any(idxup):
             plt.plot(t[idxup.astype(int)], phi[idxup.astype(int)], 'r+', ms=12, mew=1)
             plt.plot(dtup[1:], phiup, 'g.', ms=14, mfc='none',mew=2)
    if progbar:
        print("analysis_rotation.dwellTimes_1(): done")
    return dwup, dtup



def dwellTimes(phi_original, FPS, delta, interpolate=0, plots=False, progbar=True):
    ''' find the dwell time of phi, for angle interval delta. 
        interpolate = times to up-samples phi (use 10), 
        so be careful not to use a delta too small when interpolate>0
        interpolate = 0 means not interpolate
        Use:    dtup,dtdw,idxup,idxdw = dwelltimes(phi, FPS, delta, interpolate=True)
        phi and delta in degrees for plots
    '''
    #init:
    dtup = []
    dtdw = []
    idxup = []
    idxdw = []
    FPS = float(FPS)
    # time array:
    t_original = np.linspace(0, len(phi_original)/FPS, len(phi_original))    
    if interpolate:
        import scipy.interpolate
        # up-sampling (use 10 times) by lin. interpolation:
        print("DwellTimes() interpolation...")
        phi_interp = scipy.interpolate.interp1d(t_original, phi_original)
        t = np.linspace(t_original[0], t_original[-1], len(t_original)*interpolate)
        phi = phi_interp(t)
    else:
        phi = phi_original
        t = t_original
    # progress bar:
    if progbar:
        print("DwellTimes()  working...")
        pbar = progressbar.ProgressBar(maxval=phi.size)
        pbar.start()
    # find dwell times in trace:
    i = 0
    while i+1 < phi.size:
        j = 0
        while i+j+1 < phi.size:
            j = j+1
            if phi[i+j] > phi[i] + delta:
                dt = t[i+j] - t[i]
                dtup = np.append(dtup, dt)
                idxup = np.append(idxup, i)
                break
            if phi[i+j] < phi[i] - delta:
                dt = t[i+j] - t[i]
                dtdw = np.append(dtdw, dt)
                idxdw = np.append(idxdw, i)
                break
        i = i+j
        if progbar:
            pbar.update(i)
    if plots:
        plt.figure()
        plt.plot(t_original, phi_original , 'bo')
        plt.xlabel('time (sec)')
        plt.ylabel('phi (deg)')
        if np.any(idxup):
            plt.plot(t[idxup.astype(int)], phi[idxup.astype(int)], 'r+', ms=12)
        if np.any(idxdw):
            plt.plot(t[idxdw.astype(int)], phi[idxdw.astype(int)], 'g+', ms=12)

    return dtup, dtdw , idxup, idxdw








def runningDwellTime(phi, delta, win_pts, FPS, N_steps):
    ''' runs dwellTimes() for successive windows of phi
    '''
    FPS = float(FPS)
    k = 0.
    for i in np.arange(0, len(phi), win_pts):
        phi_crop = phi[i : i+win_pts]
        dtup,dtdw,idxup,idxdw = dwellTimes(phi_crop, FPS, delta, interpolate=0, plots=0, progbar=0)
        bins_norm1,hist_norm1,bins_norm0,hist_norm0 = dwellTimesHisto(dtup, 16, plots=0)
        tau, gamma = gammaPlot(N_steps, dtup, plots=0)
        phi_crop_diff = np.diff(phi_crop)*FPS/360.
        t = np.arange(i, i+len(phi_crop_diff))/FPS

        plt.figure(200)
        plt.subplot(121)
        plt.plot(phi_crop_diff, t)
        plt.xlim(60, 20)
        plt.ylabel('time (s)')
        plt.xlabel('speed (Hz)')
        plt.grid(True,which='both')
        plt.subplot(122)
        plt.semilogy(bins_norm1, hist_norm1 * 5**k, '-o', ms=2, lw=2)
        plt.semilogy(tau, gamma * 5**k, 'g--', lw=1)
        plt.ylabel('Prob. Density')
        plt.xlabel('Dwell time (s)')
        plt.axis("tight")
        plt.grid(True,which='both')
#         plt.subplot(132)
#         plt.semilogy(bins_norm0, hist_norm0*5**k , 'o-', ms=2, lw=2)
#         plt.ylabel('N events')
#         plt.xlabel('Dwell time (s)')
#         plt.axis('tight')
#         plt.grid(True,which='both')
        k = k+1
    






def dwellTimesHisto(dwt, nbin=50, plots=1, threshold=0):
    ''' makes the normalized histogram of the 
    dwell times, with logarithmic bins. 
    nbin: number of bins 
    return: histogram, bins '''
    dmin = np.min(dwt)
    dmax = np.max(dwt)
    bins = np.logspace(np.log10(dmin), np.log10(dmax), nbin)
    hist_norm0, bins_norm0 = np.histogram(dwt, bins, density=False)
    hist_norm1, bins_norm1 = np.histogram(dwt, bins, density=True)
    hist_norm2 = hist_norm0/(np.diff(bins_norm0)*np.sum(hist_norm0))
    #if normalize:
        # histogram normalization:
        # checked: the following is equivalent to np.histogram(*, density=1)
        # hi = hi/(np.diff(bi)*np.sum(hi))
    # translate bins of half a bin:
    # TODO REMOVE? artefacts ? :
    bins_norm0 = bins_norm0[:-1] + np.diff(bins_norm0)/2.
    bins_norm1 = bins_norm1[:-1] + np.diff(bins_norm1)/2.
    # remove empty bins and below 'threshold' (does not affect histogram):
    w00 = np.where(hist_norm0)
    w0 = np.where(hist_norm0>threshold)
    w1 = np.where(hist_norm1)
    hist_norm00 = hist_norm0[w00]
    hist_norm0 = hist_norm0[w0]
    hist_norm2 = hist_norm2[w0]
    hist_norm1 = hist_norm1[w1]
    bins_norm00 = bins_norm0[w00]
    bins_norm0 = bins_norm0[w0]
    bins_norm1 = bins_norm1[w1]
    if plots:
        plt.figure('dwellTimesHisto()')
        plt.subplot(211)
        plt.loglog(bins_norm00, hist_norm00, 'x')
        # threshold removed points:
        plt.loglog(bins_norm0, hist_norm0, '-o', lw=2)
        plt.ylabel('N.events')
        plt.axis("tight")
        plt.subplot(212)
        plt.loglog(bins_norm1, hist_norm1, 'x')
        # threshold removed points:
        plt.loglog(bins_norm0, hist_norm2, '-o', lw=2)
        plt.ylabel('Density')
        plt.xlabel('Time (s)')
        plt.axis("tight")
    return bins_norm1, hist_norm1, bins_norm0, hist_norm0, hist_norm2




def dwellTimes_fitGamma(phi, N_360, histobins=40, dw_range=[0,1], FPS=10000, plots=1, return_more=0):
    ''' find distribution of dwell times in 360/N_360 of angle (deg) 
    and fit a Gamma function '''
    # find dwell times from angle:
    dwup, dtup = dwellTimes_1(phi, FPS, N_360, plots=False, progbar=False)
    # tested, similar:
    #dwup,dtup,foo, foo = dwellTimes(phi, FPS, 360/N_360, interpolate=10, plots=False, progbar=True)
    # distribution of all the dwell times together:
    bn1, hn1, bn0, hn0, hn2 = dwellTimesHisto(dwup, nbin=histobins, plots=0, threshold=0)
    # crop dwup in 'dw_range' for gamma fit:
    dwup_arr = np.array(dwup)
    if dw_range:
        A = dwup_arr > dw_range[0]
        B = dwup_arr < dw_range[1]
        dwup_crop = dwup_arr[np.nonzero(A*B)[0]]
    else:
        dwup_crop = dwup_arr
    # MLE fit gamma (in our notation N=shape_fit, k=1./scale): 
    shape_fit, loc_fit, scale_fit = ss.gamma.fit(dwup_crop, floc=0)
    # define analytical gamma: 
    taus = np.linspace(np.min(bn0), np.max(bn0), 200)
    gamma_fit = ss.gamma(shape_fit, scale=scale_fit, loc=loc_fit).pdf(taus) 
    N_gamma = shape_fit
    k_gamma = 1./scale_fit
    mnspeed = (1./N_360)/(N_gamma*1./k_gamma) 
    print('dwellTimes_fitGamma(): '+str((shape_fit, loc_fit, 1./scale_fit, mnspeed)))
    if plots:
        plt.figure('Dwell times + gamma fit')
        #plt.clf()
        plt.loglog(bn0, hn2, 'o')
        plt.loglog(taus, gamma_fit, '-', lw=2)
        plt.title('All dwell times.\n N='+'{:.2f}'.format(shape_fit)+'   k='+'{:.3f}'.format(1./scale_fit))
        plt.ylim(np.min(hn2)*0.7, np.max(hn2)*1.3)
        plt.xlabel('Dwell time (s)')
    if return_more:
        return bn0, hn2, taus, gamma_fit, [N_gamma, k_gamma, mnspeed, loc_fit]
    else:
        return bn0, hn2




def dwell_sections(phi, N_360, plots=0, plots_sectors=1, x=0,y=0, FPS=10000., dw_range = [0,0.003]):
    '''angular sectioning of XY trajectory, 
    relative dwell times, 
    and Gamma distribution fit, for dwells in dw_range'''
    if plots: 
        plt.figure(123)
        plt.clf()
    d = {}
    N_gamma = []
    N_gamma_int = []
    k_gamma = []
    mnspeed = []
    dwup, dtup = dwellTimes_1(phi, FPS, N_360, plots=False, progbar=True)
    # separate dwell times dwup [from dwellTimes_1()] into N_360 sections:
    for i in range(N_360):
        d[i] = dwup[i::N_360]
        # MLE fit gamma (in our notation N=shape_fit, k=1./scale): 
        if np.any(d[i] <= 0):
            d[i] = d[i][np.where(d > 0)]
            print('Warning: d[i]<0 at points '+str([np.where(d[i] <= 0)]))
            pass
        shape_fit, loc_fit, scale_fit = ss.gamma.fit(d[i], floc=0)
        # our notations for gamma:
        N_gamma = np.append(N_gamma, shape_fit)
        N_gamma_int = np.append(N_gamma_int, np.round(shape_fit))
        k_gamma = np.append(k_gamma, 1./scale_fit)
        mnspeed = np.append(mnspeed, (1./N_360)/(N_gamma[-1]*1./k_gamma[-1]) )
        print(i, shape_fit, k_gamma[-1], loc_fit, mnspeed[-1])
        # normed histogram of dwell times in section i:
        bins_norm1, hist_norm1, bins_norm0, hist_norm0, hist_norm2 = dwellTimesHisto(d[i], nbin=20, plots=0, threshold=0)
        # define analytical gamma (N_gamma should be integer): 
        taus = np.linspace(np.min(bins_norm0), np.max(bins_norm0), 200)
        gamma_fit = ss.gamma(shape_fit, scale=scale_fit, loc=loc_fit).pdf(taus) 
        gamma_fit_round = ss.gamma(np.round(shape_fit), scale=scale_fit, loc=loc_fit).pdf(taus) 
        if plots_sectors:
            plt.figure(123)
            ax = plt.subplot(np.ceil(np.sqrt(N_360)), np.ceil(np.sqrt(N_360)), i+1)
            plt.loglog(bins_norm0, hist_norm2, 'o')
            #plt.plot(taus, gamma_fit +  (i+1)*50, 'b--', lw=2)
            plt.loglog(taus, gamma_fit_round, '-', lw=2)
            plt.xlim([0.00013,0.0017])
            #plt.ylim([-0.2,4])
            plt.grid(True)
            plt.title('sec. '+str(i))
            #plt.locator_params(nbins=3)
            if i != N_360-1:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                ax.yaxis.tick_right()
                plt.xlabel('Time (s)')
                plt.ylabel('PDF')
                ax.yaxis.set_label_position('right')
    # distribution of all the dwell times together:
    bn1, hn1, bn0, hn0, hn2 = dwellTimesHisto(dwup, nbin=40, plots=0, threshold=0)
    # crop dwup in 'dw_range' for gamma fit:
    dwup_arr = np.array(dwup)
    A = dwup_arr > dw_range[0]
    B = dwup_arr < dw_range[1]
    dwup_crop = dwup_arr[np.nonzero(A*B)[0]]
    # fit a gamma:
    shape_fit_tot, loc_fit_tot, scale_fit_tot = ss.gamma.fit(dwup_crop, floc=0)
    # define analytical gamma: 
    taus_tot = np.linspace(np.min(bn0), np.max(bn0), 200)
    gamma_fit_tot = ss.gamma(shape_fit_tot, scale=scale_fit_tot, loc=loc_fit_tot).pdf(taus_tot) 
    gamma_fit_round_tot = ss.gamma(np.round(shape_fit_tot), scale=scale_fit_tot, loc=loc_fit_tot).pdf(taus_tot) 
    N_gamma_tot = shape_fit_tot
    k_gamma_tot = 1./scale_fit_tot
    mnspeed_tot = (1./N_360)/(N_gamma_tot*1./k_gamma_tot) 
    print(shape_fit_tot, loc_fit_tot, 1./scale_fit_tot, mnspeed_tot)
    if plots:
        plt.figure(124)
        plt.clf()
        plt.subplot(311)
        plt.plot(N_gamma_int, 'rs')
        plt.plot(N_gamma, 'o')
        plt.plot(range(N_360), np.ones(N_360)*N_gamma_tot, 'r--', lw=2)
        plt.grid(True)
        plt.title('Fitted gamma distributions on angular sections')
        plt.ylabel('N')
        plt.subplot(312)
        plt.plot(k_gamma*1e-3, 'o')
        plt.plot(range(N_360), np.ones(N_360)*k_gamma_tot*1e-3, 'r--', lw=2)
        plt.grid(True)
        plt.ylabel('k (1/ms)')
        plt.subplot(313)
        plt.plot(mnspeed, 'o')
        plt.plot(range(N_360), np.ones(N_360)*mnspeed_tot, 'r--', lw=2)
        plt.ylabel('gamma mn speed (Hz)')
        plt.xlabel('section #')
        plt.grid(True)
        
        plt.figure(125)
        #plt.clf()
        plt.loglog(bn0, hn2, 'o')
        plt.loglog(taus_tot, gamma_fit_tot, '--', lw =2)
        plt.loglog(taus_tot, gamma_fit_round_tot, '-', lw =2)
        plt.title('All dwell times.\n N='+'{:.2f}'.format(shape_fit_tot)+'   k='+'{:.3f}'.format(1./scale_fit_tot))
        plt.ylim(np.min(hn2)*0.7, np.max(hn2)*1.3)
     






def gammaPlot(N,dwt,plots=1):
    ''' 
    plot gamma distribution Gamma(dwt). N:num of steps. 
    NB. if all k are the same, it must : N/k=<tau>, 
    so here forced k = N/mean(dwt)
    '''
    dwt_mn = np.mean(dwt)
    dwt_min = np.min(dwt)
    dwt_max = np.max(dwt)
    k = N/dwt_mn
    #print('gamma: k = '+str(k))
    tau = np.linspace(dwt_min, dwt_max, 100)
    #Gam = 1./math.factorial(N-1) * k**N * tau**(N-1.) * np.exp(-k*tau)
    log10Gam = np.log10(1./math.factorial(N-1)) + N*np.log10(k) + (N-1.)*np.log10(tau) + np.log10(np.exp(-k*tau))
    gam = 10**log10Gam
    if plots:
        plt.plot(tau, gam, '--',linewidth=2,label=str(N))
        plt.legend(prop={'size':7})
    return tau, gam 







def plotPhiSpeed(theta, FPS, win0=20, win1=200, x=[],y=[]):
    ''' graphs
    win0=20, win1=200 : running average windows'''
    FPS = float(FPS)
    t = np.arange(len(theta))/FPS
    theta_sm0 = filters.run_win_smooth(theta, win0, plots=0, usemode='same')
    theta_sm1 = filters.run_win_smooth(theta, win1, plots=0, usemode='same')
    dtheta_dt_Hz = np.diff(theta)*FPS/360.
    dtheta_dt_Hz_sm0 = np.diff(theta_sm0)*FPS/360.
    dtheta_dt_Hz_sm1 = np.diff(theta_sm1)*FPS/360.
    theta_hist, theta_bins = np.histogram(np.mod(theta,360.), bins=100, density=False)
    
    plt.figure(6532)    
    plt.subplot(421)
    plt.plot(t,theta/360)
    plt.axis("tight")
    plt.xlabel('Time (s)')
    plt.ylabel('Turns')
    plt.subplot(422)
    plt.plot(x,y,'.',ms=0.01)
    plt.axis("equal")
    #    
    plt.subplot(412)
    plt.plot(t[1:], dtheta_dt_Hz)#, '.',ms=1)
    plt.plot(t[1:], dtheta_dt_Hz_sm0)#, '.',ms=0.1)
    plt.plot(t[1:], dtheta_dt_Hz_sm1)#, '.',ms=0.1)
    plt.axis("tight")
    plt.xlabel('Time (s)')
    plt.ylabel('speed (Hz)')
    plt.ylim([-60,150])
    plt.title('raw, win0:'+str(win0)+'pts, win1:'+str(win1)+'pts')
    #
    plt.subplot(413)
    plt.plot(np.mod(theta,360)[1:], dtheta_dt_Hz, '.', ms=.1)
    plt.plot(np.mod(theta,360)[1:], dtheta_dt_Hz_sm0, '.', ms=.1)
    plt.plot(np.mod(theta,360)[1:], dtheta_dt_Hz_sm1, '.', ms=.1)
    plt.axis("tight")
    plt.ylabel('speed (Hz)')
    plt.grid(True)
    plt.ylim([-50,250])
    #
    plt.subplot(414)
    plt.plot(theta_bins[:-1], theta_hist)
    plt.axis("tight")
    plt.xlabel('theta (deg)')
    plt.ylabel('Theta histo')
    plt.grid(True)





def spectra(theta, x, FPS):
    '''spectrum of x(t) and d(theta)/dt'''
    import numpy.fft as fft
    FPS = float(FPS)
    # dowsample at power of 2 elements to speed up fft() :
    print('len(theta) = '+str(len(theta)))
    print('len(x) = '+str(len(x)))
    len_2use = np.min((len(theta),len(x)))
    n_pts = np.floor(np.log(len_2use)/np.log(2))
    print('considering '+str(2**n_pts)+' points (2**'+str(n_pts)+')')
    theta = theta[:2**n_pts]
    x = x[:2**n_pts]
    # spectrum of speed:
    dtheta = np.diff(theta)
    spectrum_dtheta = np.abs(fft.fft(dtheta))
    freq_dtheta = fft.fftfreq(len(dtheta), d=1./FPS)
    # spectrum of x(t):
    spectrum_x = np.abs(fft.fft(x))
    freq_x = fft.fftfreq(len(x), d=1./FPS)
    
    plt.figure()
    plt.subplot(211)
    plt.loglog(freq_x[:len(spectrum_x)/2.], spectrum_x[:len(spectrum_x)/2.])
    plt.title('spectrum(x(t))')
    plt.axis('tight')
    plt.grid(True)
    plt.subplot(212)
    plt.loglog(freq_dtheta[:len(spectrum_dtheta)/2.], spectrum_dtheta[:len(spectrum_dtheta)/2.])
    plt.title('spectrum(diff(theta(t)))')
    plt.xlabel('Frequency (Hz)')
    plt.axis('tight')
    plt.grid(True)





def spectrum(x, FPS, plots=0, new_fig=1, downsample=False):
    '''finds and plots the spectrum of x(t), 
    downsample=True : crop the first 2**N points of x (for a maximal N)
    spectrum_x[1:] = 2 * abs(fft(x))**2 / (n**2 * df) 
    spectrum_x[0]  =     abs(fft(x))**2 / (n**2 * df) 
    (as defined in labview "single sided Pw.Spectrum")
    return     freq, spectrum
    '''
    import numpy.fft as fft
    FPS = float(FPS)
    if downsample:
        # dowsample at power of 2 elements to speed up fft() :
        print('len(x) = '+str(len(x)))
        n_pts = np.floor(np.log(len(x))/np.log(2))
        print('considering '+str(2**n_pts)+' points (2**'+str(n_pts)+')')
        x_cut = x[:2**n_pts]
    else:
        x_cut = x
    # spectrum of x(t) :
    spectrum_x = np.abs(fft.fft(x_cut))**2
    freq_x = fft.fftfreq(len(x_cut), d=1./FPS)
    freq_x = freq_x[:len(spectrum_x)/2.]
    spectrum_x = spectrum_x[:len(spectrum_x)/2.]
    spectrum_x = spectrum_x/float(len(x_cut)**2 *freq_x[1])
    spectrum_x[1:] = 2*spectrum_x[1:]
    if plots:
        if new_fig:
            plt.figure()
        plt.subplot(211)
        plt.plot(np.arange(len(x))/FPS, x)
        plt.xlabel('Time (s)')
        plt.ylabel('X(t)')
        plt.grid(True)
        plt.axis('tight')
        plt.subplot(212)
        plt.loglog(freq_x, spectrum_x, '-o', ms=2)
        plt.axis('tight')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectrum(X)')
        plt.grid(True)
    return freq_x, spectrum_x




def log_avg_spectrum(freq, spectrum, npoints=500):
    ''' average a spectrum on 'npoints' logarithmic bins of frequencies
    return logwin, spt_avg = logarithmic bins used, spectrum averaged '''
    spt_avg = np.array([])
    # define bins 'logwin', is DC there? f[0]==0 ? :
    if freq[0] != 0:
        logwin = np.logspace(np.log10(freq[0]), np.log10(freq[-1]), npoints)
    else:
        logwin = np.logspace(np.log10(freq[1]), np.log10(freq[-1]), npoints)
    freq_idx = np.digitize(freq, logwin)
    # find average on every bin:
    for i in range(len(logwin)):
        mn_idx = np.where(freq_idx == i)
        spt_avg = np.append(spt_avg, np.mean(spectrum[mn_idx]))
    return logwin, spt_avg




def subctarct_phi_mean(phi_original, FPS, n_conv, plots=1):
    '''
    TODO: same idea but with high pass filter

    subtract the average speed from the angle trace.
    phi_original : in degrees, 
    n-conv: pts for running smoothing
    return: 
        time array, 
        difference phi - mean phi, filtered by n_conv'''
    FPS = float(FPS)
    n_conv = float(n_conv)
    # phi_original to turns:
    phi_original = phi_original[2:]/360.
    phi_original = phi_original - phi_original[0]
    # smoothing:
    phi_original_c = np.convolve(phi_original, np.ones(n_conv)/n_conv, mode='same')
    phi_original_c = phi_original_c[np.ceil(n_conv/2): np.floor(-n_conv/2)]
    # time array:
    t_original = np.linspace(0, len(phi_original)/FPS, len(phi_original))
    t_original_c = t_original[np.ceil(n_conv/2): np.floor(-n_conv/2)]
    # trace going perfectly at the average speed:
    phi_mean = np.linspace(0, phi_original[-1], len(phi_original))
    # difference with original:
    phi_mo = phi_original - phi_mean
    # running window smooting by convolution :
    phi_mo_c = np.convolve(phi_mo, np.ones(n_conv)/n_conv, mode='same')
    # smoothed speed:
    dphi_mo_c = np.diff(phi_mo_c)*FPS
    # original speed:
    dphi_original = np.diff(phi_original)*FPS
    dphi_original_c = np.diff(phi_original_c)*FPS
    # step size from theory:
    step = []
    step_range = range(1000, len(phi_original), 1000) 
    for i in step_range:
        step = np.append(step, 6*np.mean(phi_mo[:i]**2)/phi_original[i] )
    
    if plots:
        plt.figure(111)
        plt.clf()
        plt.plot(step_range, step, 'o')
    
        plt.figure(112)
        plt.clf()
        plt.subplot(311)
        plt.plot(t_original_c[:-1], dphi_original_c, '.', ms=0.8)
        plt.grid(True)
        plt.axis('tight')
        plt.title('<Speed> = '+str(int(np.mean(dphi_original)*10.)/10.)+' Hz')
        plt.ylabel('d/dt (phi) - <speed>')
        
        plt.subplot(312)
        plt.plot(t_original, phi_mo_c)
        plt.grid(True)
        plt.axis('tight')
        plt.xlabel('time (sec)')
        plt.ylabel('phi - phi_mean')
        
        plt.subplot(313)
        plt.plot(np.mod(phi_original_c[0:-2:5]*360.,360.), dphi_original_c[0:-1:5],'.',ms=0.8)
        plt.axis('tight')
        plt.xlabel('phi (deg)')
        plt.ylabel('speed (Hz)')
        plt.grid(True)
        
        plt.figure(113)
        plt.clf()
        plt.subplot(211)
        plt.hist(dphi_original_c, 100)
        plt.ylabel('Hist(dphi_orig_c)')
        plt.subplot(212)
        plt.hist(phi_mo_c, 100)
        plt.ylabel('Hist(phi_mo_c)')

    return t_original, phi_mo_c






def xy_speed(x,y,phi, FPS, hist_bin=100, N_360=0, pixel_tom=1):
    '''plot the speed 2D image in x,y space based on histogram2d()
    phi in deg
    x,y linear arrays
    N_360 : plot angle intervals
    TODO: pixel_tom = meters, if known'''
    # be sure x,y have same numb of points of phi:
    if len(x) != len(y):
        print('Warning x,y length !')
    if len(x) > len(phi):
        print('warning: len(x)!=len(phi). Adjusting x,y to phi.')
        x = x[:len(phi)]
        y = y[:len(phi)]
    # calculate speed wrt angle or xy displacement:
    xy_speed = np.hypot(x[1:]-x[:-1], y[1:]-y[:-1])
    xy_speed_mn = np.mean(xy_speed)
    phi_speed = np.diff(phi)*FPS/360.
    phi_speed_mn = np.mean(phi_speed)
    # tot. counts histo2d:
    count, xed, yed = np.histogram2d(x[:-1], y[:-1], \
            bins=hist_bin, normed=False)
    # phi_speed 2Dhisto:
    count_w, xed_w, yed_w = np.histogram2d(x[:-1], y[:-1], \
            bins=hist_bin, weights=phi_speed, normed=False)
    phi_2dhist = count_w/(count+1e-19)
    # xy displacement speed histo2d:
    count_xy_w, xed_w, yed_w = np.histogram2d(x[:-1], y[:-1], \
            bins=hist_bin, weights=xy_speed, normed=False)
    xy_2dhist = count_xy_w/(count+1e-19)
    # histo2D of phi mod 2pi and speed:
    #plt.hist2d(np.mod(phi[0][3:],360), phi_speed, 500)
    count_phi_speed, xed_phi,yed_phi = np.histogram2d(phi_speed, np.mod(phi[1:],360), \
            bins=hist_bin, normed=True)
    # plots:
    #plt.figure(32)
    #plt.clf()
    plt.jet()
    plt.subplot(321)
    l = np.ceil(len(x)/4.)
    plt.plot(y[:l:10], -x[:l:10], 'k.',ms=0.05)
    plt.plot(y[l:2*l:10], -x[l:2*l:10], 'r.',ms=0.05)
    plt.plot(y[2*l:3*l:10], -x[2*l:3*l:10], 'g.',ms=0.05)
    plt.plot(y[3*l:4*l:10], -x[3*l:4*l:10], 'b.',ms=0.05)
    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    # plot the angle intervals:
    if N_360:
        delta = 2*np.pi/N_360
        X = y
        Y = -x
        alpha0 = np.angle(X[0]+Y[0]*1j)
        for i in range(N_360):
            angle_i = np.exp(.9+(alpha0 + i*delta)*1j)
            plt.plot([0,np.real(angle_i)], [0,np.imag(angle_i)], '0.7', linestyle='-')
            plt.text(np.real(angle_i), np.imag(angle_i), str(i), fontsize=8)

        plt.plot([0,y[0]], [0,-x[0]], '0.2',linestyle='-')
        plt.plot(y[:10], -x[:10], '-bo')

    plt.subplot(322)
    plt.imshow(count, interpolation='none')
    cbar = plt.colorbar()
    cbar.set_label('XY histogram')
    plt.axis('image')

    plt.subplot(323)
    plt.imshow(phi_2dhist, interpolation='none', vmin=phi_speed_mn*0.5, vmax=phi_speed_mn*1.7)
    cbar = plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    # plot the intervals:
    if N_360:
        xd = np.shape(phi_2dhist)[0]/2.
        yd = np.shape(phi_2dhist)[1]/2.
        X = y + xd
        Y = -x +yd
        alpha0 = np.angle(y[0]-x[0]*1j)
        for i in range(N_360):
            angle_i = xd + yd*1j + np.exp(np.log(1.1*xd) - (alpha0 - i*delta)*1j)
            if i == 0:
                plt.plot([xd,np.real(angle_i)], [yd,np.imag(angle_i)], '0.9',lw=1, linestyle='-')
                plt.text(np.real(angle_i), np.imag(angle_i), str(i), fontsize=9)
            else:
                plt.plot([xd,np.real(angle_i)], [yd,np.imag(angle_i)], '0.6', linestyle='-')
                plt.text(np.real(angle_i), np.imag(angle_i), str(N_360-i),fontsize=9)
        plt.axis([0, 2*xd, 2*yd, 0])
    cbar.set_label('Angle speed (Hz)')
    
    plt.subplot(324)
    plt.imshow(xy_2dhist, interpolation='none', vmax=xy_speed_mn*1.5, vmin=xy_speed_mn*0.4)
    cbar = plt.colorbar()
    plt.axis('image')
    cbar.set_label('XY speed (au)')
    
    plt.subplot(313)
    plt.imshow(np.log10(count_phi_speed), interpolation='none', origin='lower', extent=(np.min(yed_phi),np.max(yed_phi),np.min(xed_phi),np.max(xed_phi)))
    plt.axis('tight')
    plt.xlabel('Angle (deg)')
    plt.ylabel('Angle speed (Hz)')
    plt.grid(True)
    




def localHisto(dphi, parts=100, phase=0, nbins=500, d2peaks=0, plots=1, prints=1):
    '''local histograms to amplify steps in dphi
    nbins = int or list'''
    maxhist = np.array([])
    dphi_min = np.min(dphi)
    dphi_max = np.max(dphi)
    Didx = np.floor(len(dphi)/parts).astype(int)
    if type(nbins) is int:
        #TODO: probably not int:
        bins = np.linspace(dphi_min, dphi_max, nbins)
    else:
        bins = nbins
    htotmax = np.zeros(len(bins)-1)
    # indexes of dphi, divided in 'parts'::
    idxi = np.floor(np.linspace(phase, len(dphi), parts, endpoint=0)).astype(int)
    #pbar = progressbar.ProgressBar(maxval=len(dphi))
    #pbar.start()
    if prints: 
        print('localHisto(): dividing in '+str(parts))
    for i in idxi:
        # histo of i-th part of dphi:
        hh,bh = np.histogram(dphi[i:i+Didx], bins, density=1)
        # bins where hh is highest:
        idxupdatemax = np.nonzero(hh>htotmax)
        # store hh when maximum:
        htotmax[idxupdatemax] = hh[idxupdatemax]
        #pbar.update(i)
    # normalize htotmax:
    htotmax = htotmax/np.sum(htotmax)
    # spectrum normalized:
    spect = np.fft.fft(htotmax)
    freq = np.fft.fftfreq(htotmax.size)
    freq = freq[1:int(freq.size/2)]
    spect = np.abs(spect[1:int(spect.size/2)])
    spect = spect/np.sum(spect)
    # period @ max of spectrum:
    periodmax = 1./freq[np.argmax(spect)]
    binsmax = bins[int(periodmax)]-bins[0]
    ### plots:
    if plots:
        plt.subplot(311)
        plt.plot(bins[:-1], htotmax, '-')
        plt.xlabel('Signal')
        # equispaced red bars by hand:
        if d2peaks:
            i = 0
            while i*d2peaks < np.max(bins):
                plt.plot([i*d2peaks,i*d2peaks],[0,np.max(htotmax)],'r',alpha=0.2,lw=5 )
                i = i+1
        plt.subplot(312)
        plt.semilogx(freq, spect, '-o')
        plt.subplot(313)
        plt.plot(1./freq, spect, '-o')
        #plt.plot(1./(bins[:len(spect)]-bins[0]), spect[:],'-o')
    return htotmax, bins[:-1]


def localHisto_generateDummyData(Nlevs, npts, noise):
    '''return artificial trace w/o noise to test localHisto()  '''
    trace = []
    for i in range(Nlevs):
        pts = npts*(1+0.8*(np.random.rand()*2-1))
        lev = np.zeros(pts)+i
        trace = np.append(trace, lev)
    tracen = trace + noise*np.random.randn(len(trace))
    return trace, tracen


def localHisto_loop(dphi, i0,i1,step, nbins=500, filterwin=3, plots=0):
    '''call many htot=localHisto() and take the max of htot
    filterwin > 0 : run avg '''
    #nbins = 500
    if type(nbins) is int:
        htotmax = np.zeros(nbins-1)
    else:
        htotmax = np.zeros(len(nbins)-1)
    for i in range(i0,i1,step):     
        htot,bins = localHisto(dphi,parts=i,phase=0,nbins=nbins,d2peaks=0,plots=0,prints=0)
        # bins where hh is highest:
        idx = np.nonzero(htot>htotmax)
        # store hh when maximum:
        htotmax[idx] = htot[idx]
    if filterwin:
        htotmax = filters.run_win_smooth(htotmax, win=filterwin, usemode='same')
    # normalize:
    htotmax = htotmax/(np.diff(bins)[0]*np.sum(htotmax))
    if plots:
        plt.figure(4369)
        plt.subplot(211)
        plt.plot(dphi)
        plt.subplot(212)
        plt.plot(bins, htotmax, lw=2)
    return htotmax, bins



def pairwiseDistance_oneArray(x):
    '''test to pairwise distance of a single trace 
    pdist() seems to have memory problems, but to check
    This is too slow...'''
    lx = len(x)
    ld = np.sum(range(lx-1,0,-1))
    d = np.zeros(ld)
    k = 0
    for i in range(lx):
        for j in range(i+1, lx):
            d[k] = x[i]-x[j]
            k = k+1
    return d


def pairwiseDistance_twoArray(x,y):
    '''pairwise distance of points x,y  
    This is slow for long arrays '''
    #print(x,y)
    zxy = list(zip(x,y))
    #print(zxy)
    d = []
    for i in range(len(zxy)):
        for j in range(i+1, len(zxy)):
            d.append( np.sqrt((zxy[i][0] - zxy[j][0])**2 + (zxy[i][1] - zxy[j][1])**2) )
            #print(i,j, zxy[i], zxy[j], d[-1])
    return d



def pairwiseDistance_shift_01(tq, i0,i1, idxstep=50, plots=0):
    '''return the pair wise distance on a subset of the distance matrix'''
    ltq = len(tq)
    print('len(tq)='+str(ltq))
    bins = np.linspace(0, np.max(tq)-np.min(tq), 300)
    dbins = np.diff(bins)[0]
    hhsum = np.zeros(len(bins)-1)
    #hhsumran = np.zeros(len(bins)-1)
    idxs = np.linspace(i0, i1, 100, endpoint=0)
    #tqran = np.copy(tq)
    #np.random.shuffle(tqran)
    for i in idxs:
        dtq = np.abs(-tq[:-i:idxstep] + tq[i::idxstep])
        #dtqran = np.abs(-tqran[:-i:idxstep] + tqran[i::idxstep])
        hh,bh = np.histogram(dtq, bins, density=1)
        #hhran,bhran = np.histogram(dtqran, bins, density=1)
        hhsum = hhsum + hh
        #hhsumran = hhsumran + hhran
        if plots:
            plt.figure(7813)
            plt.plot(bh[:-1],hh, '.', ms=1, color='0.5')
    hhsum = hhsum/(sum(hhsum)*dbins)
    #hhsumran = hhsumran/(sum(hhsumran)*dbins)
    if plots:
        plt.figure(7813)
        plt.plot(bh[:-1], hhsum,  lw=3) 
        #plt.plot(bhran[:-1], hhsumran,  lw=1) 
    return bh[:-1],hhsum#, bhran[:-1],hhsumran




def pairwiseDistance_shift(tq, i0, usebins=[], takeabs=False, idxstep=10, plots=0):
    '''return the pair wise distance on a subset of the distance matrix
    i0=[0,1], idxstep : consider only points in tq[i0*len(tq)::idxstep] '''
    ltq = len(tq)
    if usebins == []:
        bins = np.linspace(0, np.max(tq)-np.min(tq), 100)
    else:
        bins = usebins
    dbins = np.diff(bins)[0]
    hhsum = np.zeros(len(bins)-1)
    #hhmax = np.zeros(len(bins)-1)
    idxs = np.round(np.linspace(i0*ltq+1, ltq, 200, endpoint=0))
    #hhsumran = np.zeros(len(bins)-1)
    #tqran = np.copy(tq)
    #np.random.shuffle(tqran)
    for i in idxs:
        if takeabs:
            dtq = np.abs(tq[i::idxstep] - tq[:-i:idxstep])
        else:
            dtq = tq[i::idxstep] - tq[:-i:idxstep]
        #dtqran = np.abs(-tqran[:-i:idxstep] + tqran[i::idxstep])
        hh,bh = np.histogram(dtq, bins, density=1)
        #hhran,bhran = np.histogram(dtqran, bins, density=1)
        hhsum = hhsum + hh
        if np.any(np.isnan(hh)):
            print('pairwiseDistance_shiftirwiseDistance_shift(): error in idx:'+str(i))
        #hhmax[np.nonzero(hh>hhmax)] = hh[np.nonzero(hh>hhmax)]
        #hhsumran = hhsumran + hhran
        if plots:
            plt.figure(7813)
            plt.plot(bh[:-1],hh, '.', ms=1, color='0.5')
    hhsum = hhsum/(sum(hhsum)*dbins)
    #hhsumran = hhsumran/(sum(hhsumran)*dbins)
    if plots:
        plt.figure(7813)
        plt.plot(bh[:-1], hhsum,  lw=3) 
        #plt.plot(bh[:-1], hhmax, '--', lw=3) 
        #plt.plot(bhran[:-1], hhsumran,  lw=1) 
    return bh[:-1], hhsum #, bhran[:-1],hhsumran





def pairwiseDist_TimeWindows(tq, i0, usebins=[], intervals=10, deltaplot=0.001):
    '''pairwiseDistance_shift() in small time windows along signal tq
    i0 = [0,0.95]) : % of points of tq to consider (in tq[i0*len(tq):])'''
    #plt.figure(5569)
    #plt.clf()
    #plt.figure(2)
    #plt.clf()
    # divide tq in intervals, with indexes:
    idx = np.linspace(0, len(tq), intervals, endpoint=0)
    idx = np.floor(idx)
    if intervals == 1:
        didx = len(tq)-1
    else:
        didx = np.diff(idx)[0]
    # check bins not shorter than longest tq interval:
    if usebins != []:
        tq_locampl = []
        for i in idx:
            # local amplitudes of tq:
            tq_locampl = np.append(tq_locampl, np.max(tq[i:i+didx])-np.min(tq[i:i+didx]))
        if np.max(usebins) < np.max(tq_locampl):
            usebins = np.linspace(0, np.floor(np.max(tq_locampl)),200)
            print('pairwiseDist_TimeWindows():    usebins not good, forced in 0-'+str(np.floor(np.max(tq_locampl))))
        # init:
        hhtot = np.zeros(len(usebins)-1)
        hh_store = {}
    # restrict i0:
    if i0 >= 0.95: i0 = 0.95 
    if i0 < 0: i0 = 0
    print('pairwiseDist_TimeWindows():    win len = '+str(didx))
    print('pairwiseDist_TimeWindows():    considering '+str(int(i0*didx+1))+':'+str(int(didx)))
    k = 0
    gausx0s = []
    for i in idx:
        bh,hh = pairwiseDistance_shift(tq[i:i+didx], i0, usebins, plots=0)
        if usebins != []:
            hh_store[k] = hh
            hhtot = hhtot + hh
        # fit with 2 gaussians, one in 0:
        popt = fitGaussian.fit2Gaussians_1in0_optimized(bh,hh, ntry=35, plots=1)
        #if popt == []: popt = np.zeros(6)
        #gausx0s =  np.append(gausx0s, popt[4])

        plt.figure(5569)
        plt.subplot(211)
        pp, = plt.plot(np.arange(i,i+didx,10), tq[i:i+didx:10], '.', ms=1)
        plt.ylabel('signal')
        plt.subplot(212)
        plt.plot(bh, hh + deltaplot*k, lw=2, color=pp.get_color())
        plt.xlabel('signal')
        #plt.plot(bhr, hhr+0.0015*k, lw=1, color=pp.get_color())
        k = k+1
    if usebins != []:
        hhtot = hhtot/k
        plt.plot(usebins[:-1], hhtot, '--', lw=5, color='0.5')
        # fit with 2 gaussians, one in 0:
        popt = fitGaussian.fit2Gaussians_1in0_optimized(usebins[:-1],hhtot, ntry=35, plots=1)
        #plt.subplot(313)
        #for i in range(k):
        #    plt.plot(usebins[:-1], hh_store[i]-hhtot, lw=2)
    #plt.figure(64127)
    #plt.hist(gausx0s, 15)




def pairwiseDist_SlidingWindow(tq, i0, takeabs=False, winsize=10000, winstep=100, deltaplot=0.001, binfit=300):
    '''pairwiseDistance_shift() in small time window sliding along signal tq
    i0 = [0,0.95]) : % of points of tq to consider (in tq[i0*len(tq):])
    winstep = number of windows to crop tq
    winsize = pts size of each window 
    binfit : limit gaussian fit to 0:binfit (pNnm)
    '''
    plt.figure(5569)
    plt.clf()
    plt.figure(65821)
    plt.clf()
    plt.figure(2357)
    plt.clf()
    print
    # indexes for sliding window:
    idx = np.linspace(0, len(tq)-winsize, winstep, endpoint=0)
    idx = np.floor(idx)
    didx = winsize
    # def bins, = longest tq interval:
    tq_locampl = []
    for i in idx:
        # local amplitudes of tq:
        tq_locampl = np.append(tq_locampl, np.max(tq[i:i+didx])-np.min(tq[i:i+didx]))
    if takeabs:
        usebins = np.linspace(0, np.floor(np.max(tq_locampl)), 300)
    else:
        usebins = np.linspace(-np.floor(np.max(tq_locampl)), np.floor(np.max(tq_locampl)), 300)
    print('pairwiseDist_SlidingWindow(): usebins in '+str((np.min(usebins), np.max(usebins))) )
    # init:
    hhtot = np.zeros(len(usebins)-1)
    hh_store = np.zeros(len(usebins)-1)
    hhmax = np.zeros(len(usebins)-1)
    # restrict i0:
    if i0 >= 0.95: i0 = 0.95 
    if i0 < 0: i0 = 0
    print('pairwiseDist_SlidingWindow(): win len = '+str(didx))
    print('pairwiseDist_SlidingWindow(): considering '+str(int(i0*didx+1))+':'+str(int(didx)))
    pbar = progressbar.ProgressBar(maxval=winstep)
    pbar.start()
    k = 0
    # fit parameters (a0,x0,s0, a1,x1,s1, resid):
    gaus_popt = np.zeros(7)
    # slide window along trace tq:
    for i in idx:
        pbar.update(k)
        # pair wise distribution in sliding window:
        bh,hh = pairwiseDistance_shift(tq[i:i+didx], i0, usebins, takeabs=takeabs, plots=0)
        hh_store = np.vstack((hh_store, hh))
        hhtot = hhtot + hh
        hhmax[np.nonzero(hh>hhmax)] = hh[np.nonzero(hh>hhmax)]
        # cut fo fit:
        bh_fit = bh[np.nonzero(np.abs(bh)<binfit)]
        hh_fit = hh[np.nonzero(np.abs(bh)<binfit)]
        # fit with 2 gaussians, one in 0 (amp0, x0, sigma0, amp1, x1, sigma1):
        #popt, resid = fitGaussian.fit2Gaussians_1in0_optimized(bh,hh, ntry=30, plots=1, plotstep=k*deltaplot)
        # fit with 2 gaussians, free:
        popt, resid = fitGaussian.fit2Gaussians_optimized(bh_fit, hh_fit, ntry=35, plots=1, plotstep=k*deltaplot)
        if popt == []: popt = np.zeros(6)
        gaus_popt = np.vstack((gaus_popt, np.append(popt,resid)))   
        #
        plt.figure(5569)
        pp, = plt.plot(np.arange(i,i+didx,10), tq[i:i+didx:10], '.', ms=1)
        plt.ylabel('signal')
        plt.figure(65821)
        plt.plot(bh, hh + deltaplot*k, lw=2, color='k', alpha=0.5)
        plt.xlabel('signal')
        plt.ylabel('time eq.')
        k = k+1
    # remove 1st element:
    gaus_popt = gaus_popt[1:]
    if 1: 
        hhtot = hhtot/k
        plt.figure(54212)
        pp, = plt.plot(usebins[:-1], hhtot, '-', lw=2)
        plt.plot(usebins[:-1], hhmax, '-', lw=2, color=pp.get_color())
    # plot length of window used:
    plt.figure(5569)
    plt.plot([0,didx], [np.min(tq),np.min(tq)], 'k-|', lw=3, ms=20)
    # plot parwise vs time as 2D image:
    plt.figure(24723)
    plt.clf()
    plt.imshow(hh_store)
    plt.axis('tight')
    # plot popt results:
    pairwiseDist_analyse_gaus_popt(gaus_popt)
    return gaus_popt




def pairwiseDist_analyse_gaus_popt(gaus_popt):
    '''plots from pairwiseDist_SlidingWindow()'''
    xx = np.arange(len(gaus_popt[:,2]))
    x1_select = np.zeros(len(xx))
    idxmaxX = np.argmax((np.abs(gaus_popt[:,1]), np.abs(gaus_popt[:,4])), axis=0)
    # gaussian closest to 0:
    x0 = np.where(idxmaxX, gaus_popt[:,1], gaus_popt[:,4])
    s0 = np.where(idxmaxX, gaus_popt[:,2], gaus_popt[:,5])
    a0 = np.where(idxmaxX, gaus_popt[:,0], gaus_popt[:,3])
    a0 = np.abs(a0)
    # gaussian farthest from 0:
    x1 = np.where(idxmaxX, gaus_popt[:,4], gaus_popt[:,1])
    s1 = np.where(idxmaxX, gaus_popt[:,5], gaus_popt[:,2])
    a1 = np.where(idxmaxX, gaus_popt[:,3], gaus_popt[:,0])
    a1 = np.abs(a1)
    x01 = x1-x0
    # select only resolved steps:
    for i in range(len(xx)):
        if np.abs(x1[i])-np.abs(s1[i]) > np.abs(x0[i])+np.abs(s0[i]):
            x1_select[i] = x1[i]
        else:
            x1_select[i] = 0
    #
    plt.figure(64127)
    plt.clf()
    plt.subplot(411)
    plt.plot(gaus_popt[:,6],'-o')
    plt.ylabel('residues')
    #
    plt.subplot(412)
    plt.plot(xx, a0, 'o',label='a0')
    plt.plot(xx, a1, 'o',label='a1')
    plt.ylabel('amplitudes')
    plt.legend()
    #
    plt.subplot(413)
    plt.errorbar(xx, x1, yerr=s1, lw=2, label='1')
    plt.errorbar(xx, x0, yerr=s0, lw=1, label='0')
    #plt.plot(xx, x01, '-o', label='1-0')
    plt.plot(xx, np.abs(x01), '-',lw=2, label='1-0')
    plt.plot(xx, x1_select, 's')
    plt.legend(loc='best', fontsize=10)
    #
    plt.subplot(414)
    plt.hist(np.abs(x1), 30, lw=2, histtype='step')
    plt.hist(np.abs(x0), 30, lw=3, histtype='step')
    #plt.hist(np.abs(x01), 30,lw=2, histtype='step')
    plt.hist(np.abs(x1_select[np.nonzero(x1_select)]), 30,lw=4, histtype='step')
    #plt.hist(np.abs(x1_select), 30,lw=4, histtype='step')


    

def angle2torque(angle_deg, XY_diameter_m, beadradius_m, FPS, filterwin=0):
    ''' gives torque from angle (rad), diameter (m), bead radius (m), FPS
    run mean filter if filterwin!=0'''
    # calc drag_pNnms:
    drag_pNnms = DragRotatingBead.calc_drag(bead_radius=beadradius_m, axis_offset=XY_diameter_m/2., dist_beadsurf_glass=20e-09, k_hook=400.0, prints=0)
    # cal torque:
    torque_pNnm = np.diff(angle_deg/360.)*2*np.pi*FPS*drag_pNnms
    # filter torque:
    if filterwin:
        torque_pNnm = filters.run_win_smooth(torque_pNnm, win=filterwin, fs=FPS, usemode='valid',plots=0)
    return torque_pNnm




def findSteps_read_files(path='./', txtfile='./Trckd files names_WT.txt'):
    ''' open files, return a dict = (angle, diam)
    to find steps in npy files with angle '''
    angle_deg_xydiam_m_FPS = {}
    # open txt file with x,y diameter:
    t = open(path+txtfile)
    lines = t.readlines()
    print(lines)
    t.close()
    # find all .npy files:
    for f in os.listdir(path):
        print('\nfindSteps_read_files(): file = '+f)
        if f.endswith('.npy'):
            # angle (degrees):
            angle_deg = np.load(path+f)
            name,foo,foo = f.partition('.')
            print('findSteps_read_files(): analyzing '+ name)
            #search for name in lines:
            for l in range(len(lines)):
                if re.search(name+'\r\n', lines[l]):
                    print(lines[l+2][:-1])
                    patt = r'(\d+\.\d+), (\d+\.\d+)'
                    # search for maj min axis:
                    if re.search(patt, lines[l+2]) :
                        a = re.search(patt, lines[l+2]).group(1)
                        b = re.search(patt, lines[l+2]).group(2)
                        xydiam_px = np.max((float(a),float(b)))
                        xydiam_m = xydiam_px*pxsize
                        print('xydiam_px = '+str(xydiam_px)+'  xydiam_m = '+str(xydiam_m))
                    reFPS = re.search(r'FPS=(\d+)', lines[l+3])
                    if reFPS:
                        FPS = int(reFPS.group(1))
                        print('FPS = '+str(FPS))
                    angle_deg_xydiam_m_FPS[name] = (angle_deg, xydiam_m, FPS)
    print('\nfindSteps_read_files() : DONE\n')
    return angle_deg_xydiam_m_FPS






def findSteps_click_gaussians(path='/home/francesco/lavoriMiei/cbs/data/MillieData/MilliePaper/Ypet/', txtfile='Trckd files names_yPet.txt', beadradius_m=0.55e-6, filterwin=100):
    '''
    open npy files (angle) with steps in path,
    click on amplified histo to fit locally gaussians
    return all params of the chosen gaussians
    '''
    print('findSteps_read_files() ...')
    # open all .npy files in path with angle:
    angle_deg_xydiam_m_FPS = findSteps_read_files(path, txtfile)
    for key in angle_deg_xydiam_m_FPS.keys():
        print('\nfindSteps_click_gaussians(): '+path+key)
        # skip if already saved:
        if key+'_gaussians.npy' in os.listdir(path):
            print('skipping : '+key)
            continue
        print('findSteps_main(): amplifd histo on '+key)
        xydiam_m = angle_deg_xydiam_m_FPS[key][1]
        angle_deg = angle_deg_xydiam_m_FPS[key][0]
        FPS = angle_deg_xydiam_m_FPS[key][2]
        #calc torque pNnm:
        torque_pNnm = angle2torque(angle_deg, xydiam_m, beadradius_m, FPS, filterwin=filterwin)
        # plot torque filtered:
        plt.figure(98741)
        plt.clf()
        plt.plot(torque_pNnm)
        plt.title(key)
        # get amplified histogram of data:
        i0, i1, step = 10,50,1
        hh, hhbins = localHisto_loop(torque_pNnm, i0,i1,step, nbins=500, filterwin=7, plots=0)
        # click on amplifd histo to fit gaussians:
        a_x0_s_dic = hmm_FP1.click_on_gaussians(hhbins,hh, filename=path+key+'_gaussians')
        plt.figure(98742)
        plt.clf()




def findSteps_analyze(path='/home/francesco/lavoriMiei/cbs/data/MillieData/MilliePaper/Steps/WT', kdeband=25, col='r'):
    ''' 
    you run findSteps_click_gaussians() first, to save *_gaussians.npy files, 
    then this analyzes the files with gaussians for steps size, plotting the KDE of pairwise distance between peaks 
    '''
    steps_arr = []
    steps_dic = {}
    # search for files '_gaussians.npy':
    for f in os.listdir(path):
        if f.endswith(('_gaussians.npy','_gausians.npy')):
            print(f)
            a_x0_s = np.load(path+f).item()
            # mean of gaussians:
            gmean = [a_x0_s[i][1] for i in a_x0_s.keys()]
            # pairwise dist between means: 
            steps = np.abs(pairwiseDistance_oneArray(gmean))
            steps_arr = np.append(steps_arr, steps)
            steps_dic[f] = gmean
    steps_arr = steps_arr[:,np.newaxis]
    plt.figure(87427)
    #plt.clf()
    plt.subplot(211)
    # kernel density histo:
    X_plot = np.linspace(0, np.max(steps_arr)*1.1, 1000)[:, np.newaxis]
    kde = KernelDensity(kernel='tophat', bandwidth=kdeband).fit(steps_arr)
    log_dens = kde.score_samples(X_plot)
    plt.fill(X_plot[:, 0], np.exp(log_dens), fc=col, alpha=0.5)
    plt.xlabel('Pair-wise torque distance (pN nm)')
    plt.ylabel('Probability')
    # torque levels plot:
    plt.subplot(212)
    k = 0
    for i in steps_dic.values():
        plt.plot(np.ones(len(i))*k, i, '-go', ms=8)
        k = k+1
    plt.xlabel('cell #')
    plt.ylabel('Torque level (pN nm)')
    return steps_dic



def findSteps_kde_steps_plots(means_dic, steps_dic, kdeband=10):
    ''' 
    plot kernel dens. of steps in dic steps_dic, which has all the means of the gaussians 
    '''
    steps_arr = np.array([j for i in steps_dic.values() for j in i])
    steps_arr = steps_arr[:,np.newaxis]
    plt.figure(87427)
    plt.clf()
    plt.subplot(211)
    # kernel density histo:
    X = np.linspace(np.min(steps_arr), np.max(steps_arr), 500)[:, np.newaxis]
    kde = KernelDensity(kernel='tophat', bandwidth=kdeband).fit(steps_arr)
    log_dens = kde.score_samples(X)
    plt.plot(X[:, 0], np.exp(log_dens), lw=2)
    plt.xlabel('Pair-wise torque distance (pN nm)')
    plt.ylabel('Probability')
    # plot torque levels per cell :
    plt.subplot(212)
    k = 0
    for i in means_dic.values():
        print(i)
        plt.plot(np.ones(len(i))*k, i, '-go', ms=6)
        k = k+1
    plt.xlabel('cell #')
    plt.ylabel('Torque (pN nm)')
    return steps_dic






def cross_2thresholds(sig, th1, th2, plots=0, interp_pts=20):
    '''finds crossing of sig across two thresholds th1 (higher) and th2 (lower).
    Up-samples sig by lin.intepolation to avoid crossings in just one point
    output: 
            Imin, Imax = min and max of sig between subsequent crossings
            up,dw = indexes of jumps detected, [pt] float
            dt_up, dt_dw = durations in high-low state, [pt] float'''
    import scipy.interpolate
    # original signal:
    sig_orig = sig
    # up-sampling (use 10 times) by lin. interpolation:
    t_orig = np.arange(len(sig))
    sig_interp = scipy.interpolate.interp1d(t_orig, sig)
    t_oversampled = np.linspace(t_orig[0], t_orig[-1], len(t_orig)*interp_pts) 
    sig = sig_interp(t_oversampled)
    # indexes where thresholds are crossed:
    tup1 = mat.find( (sig[:-1]<th1) & (sig[1:]>th1) ) #/
    tdw1 = mat.find( (sig[:-1]>th1) & (sig[1:]<th1) ) #\
    tdw2 = mat.find( (sig[:-1]>th2) & (sig[1:]<th2) ) #\
    tup2 = mat.find( (sig[:-1]<th2) & (sig[1:]>th2) ) #/
    # define sequence of jumps:
    sth = np.zeros(len(sig))
    sth1 = np.copy(sth)
    sth[tdw1] = 1
    sth[tup1] = 2
    sth[tdw2] = -1
    sth[tup2] = -2
    sth_plot = np.copy(sth)
    sth = sth[mat.find(sth != 0)]
    # define indexes for sequence:
    sth1[tdw1] = tdw1
    sth1[tup1] = tup1
    sth1[tdw2] = tdw2
    sth1[tup2] = tup2
    sth1 = sth1[mat.find(sth1 != 0)]
    # take points where sth has the pattern (..,1,-1,..) and (..,-2,2,..):
    Up = mat.find( (sth[:-1]==-2) & (sth[1:]==2) )
    Dw = mat.find( (sth[:-1]==1) & (sth[1:]==-1) )
    # indexes for sig up and down:
    up = sth1[Up].astype(int)
    dw = sth1[Dw].astype(int)
    # find intervals dt above th1 (t_up) and below th2 (t_dw):
    if dw[0] < up[0]:
        if up[-1] < dw[-1]:
            #print('case 1')
            dt_dw = up - dw[:-1]
            dt_up = dw[1:] - up
        if dw[-1] < up[-1]:
            #print('case 2')
            dt_dw = up - dw
            dt_up = dw[1:] - up[:-1]
    if up[0] < dw[0]:
        if up[-1] < dw[-1]:
            #print('case 3')
            dt_dw = up[1:] - dw[:-1]
            dt_up = dw - up 
        if dw[-1] < up[-1]:
            #print('case 4')
            dt_dw = up[1:] - dw
            dt_up = dw - up[:-1]
    # intervals in up-dw state in real points (no upsampled)
    dt_up = dt_up/float(interp_pts)
    dt_dw = dt_dw/float(interp_pts)
    up = up/float(interp_pts)
    dw = dw/float(interp_pts)
    # period of sig: 
    periods_up = np.diff(up)
    periods_dw = np.diff(dw)
    periods = np.append(periods_up, periods_dw)
    period_mean = np.mean(periods)
    period_var = np.var(periods)
    # find max and min between two consecutive up and dw:
    if dw[0] < up[0]:
        Imin = [np.min(sig_orig[dw[i]:up[i]]) for i in range(np.min((len(up),len(dw))))]
        Imax = [np.max(sig_orig[up[i]:dw[i+1]]) for i in range(np.min((len(up),len(dw)-1)))] 
    if up[0] < dw[0]:
        Imin = [np.min(sig_orig[dw[i]:up[i+1]]) for i in range(np.min((len(up)-1,len(dw))))]
        Imax = [np.max(sig_orig[up[i]:dw[i]]) for i in range(np.min((len(up),len(dw))))]
    # plots:
    if plots:
        plt.figure(87323)
        plt.clf()
        plt.subplot(211)
        plt.plot(sig_orig,'-bo',ms=3)
        plt.plot(up, sig_orig[up.astype(int)], 'r^',ms=10)
        plt.plot(dw, sig_orig[dw.astype(int)], 'gv',ms=10)
        plt.subplot(212)
        plt.plot(sig, '-o', ms=3)
        #plt.plot(sth_plot,'-ko',ms=3)
    return periods, period_mean, period_var, Imin, Imax, up, dw, dt_up, dt_dw








def switching_analysis(phi, FPS, drag=1., filter_win=11, th_up=5, th_dw=-5, filename='', plots=1, save_fig=False):
    '''analyze the speed trace to characterize the switching CCW-CW 
    drag [=1] : drag coeff in [N m s], as obtained e.g. from tethered.main()
    filter_win (pts): window for median filter, should be odd (otherwise forced to be odd)
    th_up and th_dw (Hz): + and - thresholds to define the 2 speeds to cross to define CCW and CW respectively 
    filename: save parameters in filename, if filename !='' 
    save_fig: save .png if True
    '''
    # initialize:
    # median filter requires odd numb of points:
    if filter_win <= 0:
        print('warning: forced filter_win=1')
        filter_win = 1
    FPS = float(FPS)
    phi = phi - phi[0]
    # number of bins used in histogram of speed:
    n_bins = 100 
    # define time array:
    t = np.arange(len(phi)-1)/FPS
    # define speed and median filtered speed (Hz):
    phi_speed = np.diff(phi)*FPS/360.
    phi_speed_filtered = filters.median_filter(phi_speed, fs=FPS, win=filter_win, plots=0)
    # torque filtered (N m):
    torque_filtered = phi_speed_filtered*drag*2*np.pi
    # find moments of positive and negative speed distributions:
    phi_speed_filtered_pos = phi_speed_filtered[np.where(phi_speed_filtered > 0)]
    phi_speed_filtered_neg = phi_speed_filtered[np.where(phi_speed_filtered < 0)]
    if len(phi_speed_filtered_pos) == 0:
        phi_speed_filtered_pos = np.zeros(10)
        print('warning: len(phi_speed_filtered_pos) = 0')
    if len(phi_speed_filtered_neg) == 0:
        print('warning: len(phi_speed_filtered_neg) = 0')
        phi_speed_filtered_neg = np.zeros(10)
    speed_pos_mean = np.mean(phi_speed_filtered_pos)
    speed_pos_max = np.max(phi_speed_filtered_pos)
    speed_pos_std = np.std(phi_speed_filtered_pos)
    speed_pos_med = np.median(phi_speed_filtered_pos)
    speed_neg_mean = np.mean(phi_speed_filtered_neg)
    speed_neg_max = np.min(phi_speed_filtered_neg)
    speed_neg_std = np.std(phi_speed_filtered_neg)
    speed_neg_med = np.median(phi_speed_filtered_neg)
    # same for torque:
    torque_pos_mean = speed_pos_mean *drag*2*np.pi
    torque_pos_max = speed_pos_max *drag*2*np.pi 
    torque_pos_std = speed_pos_std *drag*2*np.pi 
    torque_pos_med = speed_pos_med *drag*2*np.pi
    torque_neg_mean = speed_neg_mean *drag*2*np.pi
    torque_neg_max = speed_neg_max *drag*2*np.pi 
    torque_neg_std = speed_neg_std *drag*2*np.pi
    torque_neg_med = speed_neg_med *drag*2*np.pi
    # find histograms for positive and negative speed:
    hist_sp, bin_sp = np.histogram(phi_speed_filtered, n_bins, density=False)    
    bin_sp = bin_sp[:-1] + np.diff(bin_sp)/2.
    bin_pos_idx = np.where(bin_sp>0)
    bin_neg_idx = np.where(bin_sp<0)
    bin_pos = bin_sp[bin_pos_idx]
    bin_neg = bin_sp[bin_neg_idx]
    hist_pos = hist_sp[bin_pos_idx]
    hist_neg = hist_sp[bin_neg_idx]
    # same for torque:
    bin_pos_torque = bin_pos *drag*2*np.pi
    bin_neg_torque = bin_neg *drag*2*np.pi
    hist_pos_torque = hist_pos
    hist_neg_torque = hist_neg
    # total time in pos and neg speed:
    sum_hist_pos = np.sum(hist_pos)/FPS
    sum_hist_neg = np.sum(hist_neg)/FPS
    # find mode (moda) of speed distributions:
    if len(bin_pos)>0:
        speed_pos_mode = bin_pos[np.argmax(hist_pos)]
        torque_pos_mode = speed_pos_mode *drag*2*np.pi
    else:
        speed_pos_mode = 0
        torque_pos_mode = 0
    if len(bin_neg)>0:
        speed_neg_mode = bin_neg[np.argmax(hist_neg)]
        torque_neg_mode = speed_neg_mode *drag*2*np.pi
    else:
        speed_neg_mode = 0
        torque_neg_mode = 0
    # find time CCW and time CW in speed trace using 2 thresholds:
    try:
        periods, period_mean, period_var, Imin, Imax, idx_up, idx_dw, dt_up, dt_dw = cross_2thresholds(phi_speed_filtered, th_up, th_dw, plots=0, interp_pts=20)
        # durations in CCW (speed>0, up) and CW (speed<0, dw)
        dtup = dt_up/FPS
        dtdw = dt_dw/FPS
    except:
        print('warning: error in switch detection')
        dtup = np.array(())
        dtdw = np.array(())
        idx_up = np.array(())
        idx_dw = np.array(())
    # def out parameters:
    tot_time = t[-1]
    tot_time_CCW = sum_hist_pos
    tot_time_CW = sum_hist_neg
    bias_CW = tot_time_CW/tot_time
    bias_CCW = tot_time_CCW/tot_time
    # a switch is defined as any change of direction (CCW <--> CW):
    numb_switches = len(idx_up)+len(idx_dw) 
    freq_switch = numb_switches/tot_time 
    # save text file:
    if filename != '':
        f = open(filename,'w')
        data_w = filename+'\t filename \n'+ \
                str(FPS)+'\t FPS \n'+ \
                str(drag)+'\t drag \n'+ \
                str(filter_win)+'\t filter_win \n'+ \
                str(th_dw)+'\t th_dw \n'+ \
                str(th_up)+'\t th_up \n'+ \
                str(tot_time)+'\t tot_time \n'+ \
                str(tot_time_CCW)+'\t tot_time_CCW \n'+ \
                str(tot_time_CW)+'\t tot_time_CW \n'+ \
                str(bias_CW)+'\t bias_CW \n'+ \
                str(bias_CCW)+'\t bias_CCW \n'+ \
                str(numb_switches)+'\t numb_switches \n'+ \
                str(freq_switch)+'\t freq_switch \n'+ \
                str(np.mean(dtup))+'\t CCW mean time sec \n'+ \
                str(np.std(dtup))+'\t CCW st.dev sec \n'+ \
                str(np.mean(dtdw))+'\t CW mean time sec \n'+ \
                str(np.std(dtdw))+'\t CW st.dev sec \n'+ \
                '- Speed distribution: \n'+ \
                '-- Positive speed: \n'+ \
                str(speed_pos_mean)+'\t pos speed mean Hz \n'+ \
                str(speed_pos_std)+'\t pos speed st.dev. Hz \n'+ \
                str(speed_pos_max)+'\t pos speed max Hz \n'+ \
                str(speed_pos_med)+'\t pos speed median Hz \n'+ \
                str(speed_pos_mode)+'\t pos speed mode Hz \n'+ \
                '-- Negative speed: \n'+ \
                str(speed_neg_mean)+'\t neg speed mean Hz \n'+ \
                str(speed_neg_std)+'\t neg speed st.dev. Hz \n'+ \
                str(speed_neg_max)+'\t neg speed max Hz \n'+ \
                str(speed_neg_med)+'\t neg speed median Hz \n'+ \
                str(speed_neg_mode)+'\t neg speed mode Hz \n' + \
                '- Torque distribution: \n'+ \
                '-- Positive torque: \n'+ \
                str(torque_pos_mean)+'\t pos torque mean Nm \n'+ \
                str(torque_pos_std)+'\t pos torque st.dev. Nm \n'+ \
                str(torque_pos_max)+'\t pos torque max Nm \n'+ \
                str(torque_pos_med)+'\t pos torque median Nm \n'+ \
                str(torque_pos_mode)+'\t pos torque mode Nm \n'+ \
                '-- Negative torque: \n'+ \
                str(torque_neg_mean)+'\t neg torque mean Nm \n'+ \
                str(torque_neg_std)+'\t neg torque st.dev. Nm \n'+ \
                str(torque_neg_max)+'\t neg torque max Nm \n'+ \
                str(torque_neg_med)+'\t neg torque median Nm \n'+ \
                str(torque_neg_mode)+'\t neg torque mode Nm \n'
        f.write(data_w)
        f.write('\n dtup \n')
        for i in dtup:
            f.write(str(i)+' \n')
        f.write(' dtup_end \n')
        f.write('\n dtdw \n')
        for i in dtdw:
            f.write(str(i)+' \n')
        f.write(' dtdw_end \n')
        f.write('\n bin_pos \n')
        for i in bin_pos:
            f.write(str(i)+' \n')
        f.write(' bin_pos_end \n')
        f.write('\n hist_pos \n')
        for i in hist_pos:
            f.write(str(i)+' \n')
        f.write(' hist_pos_end \n')
        f.write('\n bin_neg \n')
        for i in bin_neg:
            f.write(str(i)+' \n')
        f.write(' bin_neg_end \n')
        f.write('\n hist_neg \n')
        for i in hist_neg:
            f.write(str(i)+' \n')
        f.write(' hist_neg_end \n')
        f.close()
    print('----')
    print('Total time \t\t\t= '+'{:.2f}'.format(tot_time)+' sec')
    print('Total time CCW (positive speed)\t= '+'{:.2f}'.format(tot_time_CCW)+' sec')
    print('Total time CW  (negative speed)\t= '+'{:.2f}'.format(tot_time_CW)+' sec')
    print('Time CCW / time CW\t\t= '+'{:.3f}'.format(tot_time_CCW/tot_time_CW))
    print('Time CCW / tot. time\t\t= '+'{:.3f}'.format(tot_time_CCW/tot_time))
    print('Time CW / tot. time\t\t= '+'{:.3f}'.format(tot_time_CW/tot_time))
    print('Switch detection:')
    print('  Number of switches\t= '+'{:d}'.format(numb_switches))
    print('  Freq. of switch\t= {:.3f}'.format(freq_switch)+' switches/sec')
    print('  CCW mean time\t\t= '+'{:.3f}'.format(np.mean(dtup))+' sec')
    print('  CCW st.dev\t\t= '+'{:.3f}'.format(np.std(dtup))+' sec')
    print('  CW mean time\t\t= '+'{:.3f}'.format(np.mean(dtdw))+' sec')
    print('  CW st.dev\t\t= '+'{:.3f}'.format(np.std(dtdw))+' sec')
    print('Speed distribution:')
    print('  Positive speed :')
    print('\t mean\t\t= '+'{:.1f}'.format(speed_pos_mean)+' Hz')
    print('\t st.dev.\t= '+'{:.1f}'.format(speed_pos_std)+' Hz')
    print('\t max\t\t= '+'{:.1f}'.format(speed_pos_max)+' Hz')
    print('\t median\t\t= '+'{:.1f}'.format(speed_pos_med)+' Hz')
    print('\t mode \t\t= '+'{:.1f}'.format(speed_pos_mode)+' Hz')
    print('  Negative speed :')
    print('\t mean\t\t= '+'{:.1f}'.format(speed_neg_mean)+' Hz')
    print('\t st.dev.\t= '+'{:.1f}'.format(speed_neg_std)+' Hz')
    print('\t max\t\t= '+'{:.1f}'.format(speed_neg_max)+' Hz')
    print('\t median\t\t= '+'{:.1f}'.format(speed_neg_med)+' Hz')
    print('\t mode \t\t= '+'{:.1f}'.format(speed_neg_mode)+' Hz')
    print('Torque distribution:')
    print('  Positive torque :')
    print('\t mean\t\t= '+'{:.1f}'.format(torque_pos_mean*1e21)+' pNnm')
    print('\t st.dev.\t= '+'{:.1f}'.format(torque_pos_std*1e21)+' pNnm')
    print('\t max\t\t= '+'{:.1f}'.format(torque_pos_max*1e21)+' pNnm')
    print('\t median\t\t= '+'{:.1f}'.format(torque_pos_med*1e21)+' pNnm')
    print('\t mode \t\t= '+'{:.1f}'.format(torque_pos_mode*1e21)+' pNnm')
    print('  Negative torque :')
    print('\t mean\t\t= '+'{:.1f}'.format(torque_neg_mean*1e21)+' pNnm')
    print('\t st.dev.\t= '+'{:.1f}'.format(torque_neg_std*1e21)+' pNnm')
    print('\t max\t\t= '+'{:.1f}'.format(torque_neg_max*1e21)+' pNnm')
    print('\t median\t\t= '+'{:.1f}'.format(torque_neg_med*1e21)+' pNnm')
    print('\t mode \t\t= '+'{:.1f}'.format(torque_neg_mode*1e21)+' pNnm')
    print('----')
    if plots:
        plt.figure(556)
        plt.clf()
        plt.subplot(211)
        if len(dtup) > 1:
            plt.hist(dtup, 30, color='g')
        plt.xlabel('Time_CCW (s)')
        plt.ylabel('Counts')
        plt.subplot(212)
        if len(dtdw) > 1:
            plt.hist(dtdw, 30, color='r')
        plt.xlabel('Time_CW (s)')
        plt.ylabel('Counts')
        #
        plt.figure(555)
        plt.clf()
        plt.subplot(211)
        plt.plot(t, phi_speed, '.', ms=0.5)
        plt.plot(t, phi_speed_filtered, '-ko',ms=2)
        plt.plot(idx_up/FPS, phi_speed_filtered[idx_up.astype(int)], 'g^', ms=10)
        plt.plot(idx_dw/FPS, phi_speed_filtered[idx_dw.astype(int)], 'rv', ms=10)
        plt.grid(True)
        plt.axis('tight')
        plt.ylim([np.min(phi_speed_filtered),np.max(phi_speed_filtered)])
        plt.ylabel('Speed (Hz)')
        plt.xlabel('Time (s)')
        plt.subplot(212)
        plt.semilogy(bin_sp, hist_sp)
        plt.semilogy(bin_pos, hist_pos, 'ro')
        plt.semilogy(bin_neg, hist_neg, 'go')
        plt.xlabel('Speed (Hz)')
        plt.ylabel('Counts')
        plt.grid(True)
        #
        # plot overlap of switchings:                
        plt.figure(68732)
        plt.clf()
        DX1 = 10; DX2 = 50
        for uu in idx_up.astype(int):
            if uu+DX2 < len(phi_speed_filtered):
                plt.subplot(211)
                plt.plot(t[:DX1+DX2]-t[DX1], phi_speed_filtered[uu-DX1:uu+DX2],'r-',lw=2,alpha=0.2)
                plt.plot([t[0]-t[DX1],t[DX1+DX2]-t[DX1]],[0,0],'k--',lw=1)
                plt.ylabel('Speed (Hz)')
        for dd in idx_dw.astype(int):
            if dd+DX2 < len(phi_speed_filtered):
                plt.subplot(212)
                plt.plot(t[:DX1+DX2]-t[DX1], phi_speed_filtered[dd-DX1:dd+DX2],'g-',lw=2,alpha=0.2)
                plt.plot([t[0]-t[DX1],t[DX1+DX2]-t[DX1]],[0,0],'k--',lw=1)
                plt.xlabel('Time (s)')
                plt.ylabel('Speed (Hz)')
        #
        if drag != 1.:
            plt.figure(554)
            plt.clf()
            plt.subplot(211)
            plt.plot(t, 1e21*phi_speed *drag*2*np.pi, '.', ms=0.5)
            plt.plot(t, 1e21*torque_filtered, '-ko',ms=2)
            plt.grid(True)
            plt.axis('tight')
            plt.ylim([np.min(torque_filtered*1e21),np.max(torque_filtered*1e21)])
            plt.ylabel('Torque (pN nm)')
            plt.xlabel('Time (s)')
            plt.subplot(212)
            #plt.semilogy(bin_sp, hist_sp)
            plt.semilogy(bin_pos_torque*1e21, hist_pos_torque, 'ro')
            plt.semilogy(bin_neg_torque*1e21, hist_neg_torque, 'go')
            plt.xlabel('Torque (pN nm)')
            plt.ylabel('Counts')
            plt.grid(True)
            if (filename != '') and save_fig:
                plt.savefig(filename+'.png')




def switching_extract_from_file(filename):
    '''reads filename, the text file coming from switching_analysis(), and analyse '''
    import re
    f = open(filename, 'r')
    # read file as a single string:
    lines_string = f.read()
    # set file position back to zero to read file again:
    f.seek(0) 
    # read file as a list of lines:
    lines_list = f.readlines()
    # init values:
    freq_switch = np.nan
    dtup_list = dtdw_list = histpos_list = histneg_list = binpos_list = binneg_list = []
    drag = 0
    std_time_CCW = std_time_CW = mn_time_CCW = mn_time_CW = bias_CCW = bias_CW = tot_time = 0
    pos_torque_mean_Nm = pos_torque_stdev_Nm = pos_torque_max_Nm = pos_torque_mode_Nm = neg_torque_mean_Nm = neg_torque_stdev_Nm = neg_torque_max_Nm = neg_torque_mode_Nm = 0
    # search in each line:
    for l in lines_list:
        # search for 'freq_switch':
        if re.search(r'freq_switch', l) and re.search(r'^\d+\.\d+', l):
            freq_switch = float(re.search(r'^\d+\.\d+', l).group())
        # search for 'tot_time_CW' and 'tot_time_CCW' :
        if re.search(r'tot_time_CW', l) and re.search(r'^\d+\.\d+', l):
            tot_time_CW = float(re.search(r'^\d+\.\d+', l).group())
        if re.search(r'tot_time_CCW', l) and re.search(r'^\d+\.\d+', l):
            tot_time_CCW = float(re.search(r'^\d+\.\d+', l).group())
        # search for 'tot_time' :
        if re.search(r'tot_time ', l) and re.search(r'^\d+\.\d+', l):
            tot_time = float(re.search(r'^\d+\.\d+', l).group())
        # search for 'bias_CW' :
        if re.search(r'bias_CW', l) and re.search(r'^\d+\.\d+', l):
            bias_CW = float(re.search(r'^\d+\.\d+', l).group())
        # search for 'bias_CCW' :
        if re.search(r'bias_CCW', l) and re.search(r'^\d+\.\d+', l):
            bias_CCW = float(re.search(r'^\d+\.\d+', l).group())
        # search for 'CCW mean time sec' :
        if re.search(r'CCW mean time sec', l) and re.search(r'^(-|)\d+\.\d+', l):
            mn_time_CCW = float(re.search(r'^(-|)\d+\.\d+', l).group())
            if mn_time_CCW < 0:
                print('ERROR WARNING: mn_time_CCW < 0 something is wrong!')
        # search for 'CW mean time sec' :
        if re.search(r'CW mean time sec', l) and re.search(r'^(-|)\d+\.\d+', l):
            mn_time_CW = float(re.search(r'^(-|)\d+\.\d+', l).group())
            if mn_time_CW < 0:
                print('ERROR WARNING: mn_time_CW < 0  something is wrong!')
        # search for 'CW st.dev sec' :
        if re.search(r'CW st.dev sec', l) and re.search(r'^\d+\.\d+', l):
            std_time_CW = float(re.search(r'^\d+\.\d+', l).group())
        # search for 'CCW st.dev sec' :
        if re.search(r'CCW st.dev sec', l) and re.search(r'^\d+\.\d+', l):
            std_time_CCW = float(re.search(r'^\d+\.\d+', l).group())
        
        # search for 'pos speed mean Hz' :
        if re.search(r'pos speed mean Hz ', l) and re.search(r'^\d+\.\d+', l):
            pos_speed_mean_Hz = float(re.search(r'^\d+\.\d+', l).group())
        # search for 'pos speed st.dev. Hz' :
        if re.search(r'pos speed st.dev. Hz ', l) and re.search(r'^\d+\.\d+', l):
            pos_speed_stdev_Hz = float(re.search(r'^\d+\.\d+', l).group())
        # search for 'pos speed max Hz' :
        if re.search(r'pos speed max Hz ', l) and re.search(r'^\d+\.\d+', l):
            pos_speed_max_Hz = float(re.search(r'^\d+\.\d+', l).group())
        # search for 'pos speed mode Hz' also match '0':
        if re.search(r'pos speed mode Hz ', l) and re.search(r'^\d+(\.\d+|)', l):
            pos_speed_mode_Hz = float(re.search(r'^\d+(\.\d+|)', l).group())
        # search for 'neg speed mean Hz', note the '(-|)' = either - or nothing:
        if re.search(r'neg speed mean Hz ', l) and re.search(r'^(-|)\d+\.\d+', l):
            neg_speed_mean_Hz = float(re.search(r'^(-|)\d+\.\d+', l).group())
        # search for 'neg speed std.dev. Hz' :
        if re.search(r'neg speed st.dev. Hz ', l) and re.search(r'^\d+\.\d+', l):
            neg_speed_stdev_Hz = float(re.search(r'^\d+\.\d+', l).group())
        # search for 'neg speed max Hz' :
        if re.search(r'neg speed max Hz ', l) and re.search(r'^(-|)\d+\.\d+', l):
            neg_speed_max_Hz = float(re.search(r'^(-|)\d+\.\d+', l).group())
        # search for 'neg speed mode Hz' (match also '0'):
        if re.search(r'neg speed mode Hz ', l) and re.search(r'^(-|)\d+(\.\d+|)', l):
            neg_speed_mode_Hz = float(re.search(r'^(-|)\d+(\.\d+|)', l).group())
        # search for 'drag':
        if re.search(r'drag', l) and re.search(r'^\d+\.\d+e-\d+', l):
            drag = float(re.search(r'^\d+\.\d+e-\d+', l).group())
        # search for 'pos torque mean Nm' :
        if re.search(r'pos torque mean Nm ', l) and re.search(r'^\d+\.\d+e-\d+', l):
            pos_torque_mean_Nm = float(re.search(r'^\d+\.\d+e-\d+', l).group())
        # search for 'pos torque st.dev. Nm' :
        if re.search(r'pos torque st.dev. Nm ', l) and re.search(r'^\d+\.\d+e-\d+', l):
            pos_torque_stdev_Nm = float(re.search(r'^\d+\.\d+e-\d+', l).group())
        # search for 'pos torque max Nm' :
        if re.search(r'pos torque max Nm ', l) and re.search(r'^\d+\.\d+e-\d+', l):
            pos_torque_max_Nm = float(re.search(r'^\d+\.\d+e-\d+', l).group())
        # search for 'pos torque mode Nm' also match '0':
        if re.search(r'pos torque mode Nm ', l) and re.search(r'^\d+(\.\d+e-\d+|)', l):
            pos_torque_mode_Nm = float(re.search(r'^\d+(\.\d+e-\d+|)', l).group())
        # search for 'neg torque mean Nm', note the '(-|)' = either - or nothing:
        if re.search(r'neg torque mean Nm ', l) and re.search(r'^(-|)\d+\.\d+e-\d+', l):
            neg_torque_mean_Nm = float(re.search(r'^(-|)\d+\.\d+e-\d+', l).group())
        # search for 'neg torque std.dev. Nm' :
        if re.search(r'neg torque st.dev. Nm ', l) and re.search(r'^\d+\.\d+e-\d+', l):
            neg_torque_stdev_Nm = float(re.search(r'^\d+\.\d+e-\d+', l).group())
        # search for 'neg torque max Nm' :
        if re.search(r'neg torque max Nm ', l) and re.search(r'^(-|)\d+\.\d+e-\d+', l):
            neg_torque_max_Nm = float(re.search(r'^(-|)\d+\.\d+e-\d+', l).group())
        # search for 'neg torque mode Nm' (match also '0'):
        if re.search(r'neg torque mode Nm ', l) and re.search(r'^(-|)\d+(\.\d+e-\d+|)', l):
            neg_torque_mode_Nm = float(re.search(r'^(-|)\d+(\.\d+e-\d+|)', l).group())
    # search in file as single string:
    # 'dtup' search ( \d is a numeric ):
    dtup_search = re.search(r' dtup ((\n|\r\n)\d+\.\d+ )*', lines_string)
    if dtup_search: 
        dtup_str =  dtup_search.group()
        dtup_list = re.findall(r'\d+\.\d+', dtup_str)
        dtup_list = np.array([float(x) for x in dtup_list])
    # 'dtdw' search :
    dtdw_search = re.search(r' dtdw ((\n|\r\n)\d+\.\d+ )*', lines_string)
    if dtup_search: 
        dtdw_str =  dtdw_search.group()
        dtdw_list = re.findall(r'\d+\.\d+', dtdw_str)
        dtdw_list = np.array([float(x) for x in dtdw_list])
    # 'hist_pos' search :
    histpos_search = re.search(r' hist_pos ((\n|\r\n)\d+ )*', lines_string)
    if histpos_search: 
        histpos_str =  histpos_search.group()
        histpos_list = re.findall(r'\d+', histpos_str)
        histpos_list = np.array([float(x) for x in histpos_list])
        histpos_list = np.append(histpos_list, np.nan)
    # 'hist_neg' search :
    histneg_search = re.search(r' hist_neg ((\n|\r\n)\d+ )*', lines_string)
    if histneg_search: 
        histneg_str =  histneg_search.group()
        histneg_list = re.findall(r'\d+', histneg_str)
        histneg_list = np.array([float(x) for x in histneg_list])
        histneg_list = np.append(histneg_list, np.nan)
    # 'bin_neg' search :
    binneg_search = re.search(r' bin_neg ((\n|\r\n)-\d+\.\d+ )*', lines_string)
    if binneg_search: 
        binneg_str =  binneg_search.group()
        binneg_list = re.findall(r'-\d+\.\d+', binneg_str)
        binneg_list = np.array([float(x) for x in binneg_list])
        binneg_list = np.append(binneg_list, np.nan)
    # 'bin_pos' search :
    binpos_search = re.search(r' bin_pos ((\n|\r\n)\d+\.\d+ )*', lines_string)
    if binpos_search: 
        binpos_str =  binpos_search.group()
        binpos_list = re.findall(r'\d+\.\d+', binpos_str)
        binpos_list = np.array([float(x) for x in binpos_list])
        binpos_list = np.append(binpos_list, np.nan)

    return dtup_list, dtdw_list, histpos_list, histneg_list, binpos_list, binneg_list, freq_switch, tot_time_CCW, tot_time_CW, tot_time, bias_CW, bias_CCW, std_time_CW, std_time_CCW, mn_time_CW, mn_time_CCW , pos_speed_mean_Hz, pos_speed_stdev_Hz, pos_speed_max_Hz, pos_speed_mode_Hz, neg_speed_mean_Hz, neg_speed_stdev_Hz, neg_speed_max_Hz, neg_speed_mode_Hz, drag, pos_torque_mean_Nm, pos_torque_stdev_Nm, pos_torque_max_Nm, pos_torque_mode_Nm, neg_torque_mean_Nm, neg_torque_stdev_Nm, neg_torque_max_Nm, neg_torque_mode_Nm 




def histogram_of_many_arrays(x_in, y_in, numb_bins=50, plots=0):
    '''finds the average histogram of many input histograms arrays 
    which have different bins 
    x_in , y_in = dictionaries of all xs,ys '''
    # lists of arrays:
    x_list_arrays = x_in.values()
    y_list_arrays = y_in.values()
    # list of all values of all arrays:
    x_list_values_copy = [y for x in x_list_arrays for y in x]
    y_list_values_copy = [y for x in y_list_arrays for y in x]
    x_list_values = np.array(x_list_values_copy)
    y_list_values = np.array(y_list_values_copy)
    # def bins of global histo:
    bin_glob = np.linspace(np.min(x_list_values), np.max(x_list_values), numb_bins)
    bin_width = np.diff(bin_glob)[1]/2.
    # global histo:
    hist_glob = np.array([])
    for bin_now in bin_glob:
        # indexes where x_list_values is close to bin_now:
        idx_now = np.where(np.abs(x_list_values - bin_now) < bin_width)[0]
        # sum of all y_list_values in these indexes:
        hist_glob = np.append(hist_glob, np.sum(y_list_values[idx_now]))#/len(idx_now))
    # normalize distribution:
    hist_glob = hist_glob/np.sum(hist_glob)
    # remove zeros:
    hist_glob[hist_glob == 0] = np.nan

    if plots:
        plt.figure(243)
        plt.clf()
        #plt.plot(x_list_values_copy, y_list_values_copy, '-')
        plt.plot(x_list_values, y_list_values, 'o')
        plt.plot(bin_glob, hist_glob, '-', lw=2)
    return bin_glob, hist_glob, x_list_values_copy, y_list_values_copy






def switching_combine_txtFiles(path='./', fig_title='', plots=0, save_figs=False, table_filename='', save_histosspeed=False, save_dtupdw=False, save_swfreq=False):
    '''analyzes all .txt files in 'path'. 
    The files come from switching_analysis(). 
    Combine all the measurements in plots and table.
    fig_title: set the title of the plots 
        (useful to discriminate strains and conditions)
    table_filename: save everything in a table file, if 
        table_filename is not empty'''
    import os
    import numpy.ma as ma
    # init params:
    dtup_dic = {}
    dtdw_dic = {}
    dtup_dic_flag = {}
    dtdw_dic_flag = {}
    hispos_dic = {}
    hisneg_dic = {}
    binpos_dic = {}
    binneg_dic = {}
    hispos_dic_torque = {}
    hisneg_dic_torque = {}
    binpos_dic_torque = {}
    binneg_dic_torque = {}
    fr_switch_ma = ma.array([])
    fr_switch_ma_mask = ma.array([])
    dtup_list_tot = []
    dtdw_list_tot = []
    dtdw_list_onlyCW = []
    dtup_list_onlyCCW = []
    # write tablefile header:
    if table_filename:
        tablefile = open(table_filename,'w')
        tablefile.write('file\t\
            tot_time\t\
            freq_switch\t \
            tot_time_CW\t \
            tot_time_CCW\t \
            bias_CW\t \
            bias_CCW\t \
            time_mn_CW\t \
            time_mn_CCW\t \
            time_std_CW\t \
            time_std_CCW\t \
            pos_speed_mean_Hz\t\
            pos_speed_stdev_Hz\t\
            pos_speed_max_Hz\t\
            pos_speed_mode_Hz\t \
            neg_speed_mean_Hz\t\
            neg_speed_stdev_Hz\t\
            neg_speed_max_Hz\t \
            neg_speed_mode_Hz\t \
            drag_Nm\t \
            pos_torque_mean_Nm\t\
            pos_torque_stdev_Nm\t\
            pos_torque_max_Nm\t\
            pos_torque_mode_Nm\t \
            neg_torque_mean_Nm\t\
            neg_torque_stdev_Nm\t\
            neg_torque_max_Nm\t \
            neg_torque_mode_Nm\n')
    print('\nanalysis_rotation.switching_combine_txtFiles(): path now = '+path+'\n')
    for filenow in os.listdir(path):
        if filenow.endswith('txt'):
            print(filenow)
            # extract values from single txt file:
            dtup_list, dtdw_list, \
                hispos_list, hisneg_list, binpos_list, binneg_list, \
                freq_switch, tot_time_CCW, tot_time_CW, tot_time, \
                bias_CW, bias_CCW, \
                std_time_CW, std_time_CCW, mn_time_CW, mn_time_CCW, \
                pos_speed_mean_Hz, pos_speed_stdev_Hz, pos_speed_max_Hz, pos_speed_mode_Hz, neg_speed_mean_Hz, neg_speed_stdev_Hz, neg_speed_max_Hz, neg_speed_mode_Hz, \
                drag , \
                pos_torque_mean_Nm, pos_torque_stdev_Nm, pos_torque_max_Nm, pos_torque_mode_Nm, neg_torque_mean_Nm, neg_torque_stdev_Nm, neg_torque_max_Nm, neg_torque_mode_Nm \
                = switching_extract_from_file(path+filenow)
            # write table in file :
            if table_filename:
                tablefile.write(filenow+'\t')
                tablefile.write(str(tot_time)+'\t')
                tablefile.write(str(freq_switch)+'\t')
                tablefile.write(str(tot_time_CW)+'\t')
                tablefile.write(str(tot_time_CCW)+'\t')
                tablefile.write(str(bias_CW)+'\t')
                tablefile.write(str(bias_CCW)+'\t')
                tablefile.write(str(mn_time_CW)+'\t')
                tablefile.write(str(mn_time_CCW)+'\t')
                tablefile.write(str(std_time_CW)+'\t')
                tablefile.write(str(std_time_CCW)+'\t')
                tablefile.write(str(pos_speed_mean_Hz)+'\t')
                tablefile.write(str(pos_speed_stdev_Hz)+'\t')
                tablefile.write(str(pos_speed_max_Hz)+'\t')
                tablefile.write(str(pos_speed_mode_Hz)+'\t')
                tablefile.write(str(neg_speed_mean_Hz)+'\t')
                tablefile.write(str(neg_speed_stdev_Hz)+'\t')
                tablefile.write(str(neg_speed_max_Hz)+'\t')
                tablefile.write(str(neg_speed_mode_Hz)+'\t')
                tablefile.write(str(drag)+'\t')
                tablefile.write(str(pos_torque_mean_Nm)+'\t')
                tablefile.write(str(pos_torque_stdev_Nm)+'\t')
                tablefile.write(str(pos_torque_max_Nm)+'\t')
                tablefile.write(str(pos_torque_mode_Nm)+'\t')
                tablefile.write(str(neg_torque_mean_Nm)+'\t')
                tablefile.write(str(neg_torque_stdev_Nm)+'\t')
                tablefile.write(str(neg_torque_max_Nm)+'\t')
                tablefile.write(str(neg_torque_mode_Nm)+'\n')
            ### dispach values to local variables:
            ## times in CCW and CW:
            dtup_dic[filenow] = dtup_list
            dtdw_dic[filenow] = dtdw_list
            dtup_list_tot = np.append(dtup_list_tot, dtup_list)
            dtdw_list_tot = np.append(dtdw_list_tot, dtdw_list)
            ## speed distributions, normalize pos and neg histograms with tot histo:
            hispos_list = hispos_list[:-1] #remove last elem, always=nan
            binpos_list = binpos_list[:-1]
            hisneg_list = hisneg_list[:-1] 
            binneg_list = binneg_list[:-1]
            histot_list = np.append(hisneg_list, hispos_list)
            bintot_list = np.append(binneg_list, binpos_list)
            # same for torque:
            hispos_list_torque = hispos_list 
            binpos_list_torque = binpos_list * 2*np.pi*drag
            hisneg_list_torque = hisneg_list 
            binneg_list_torque = binneg_list * 2*np.pi*drag
            histot_list_torque = histot_list
            bintot_list_torque = bintot_list * 2*np.pi*drag
            if len(hispos_list)>2:
                divnorm = np.sum(histot_list) * np.diff(binpos_list)[1]  
                hispos_list_norm = hispos_list/divnorm
                hispos_list_norm_torque = hispos_list_norm
            else:
                hispos_list_norm = [0]
                binpos_list = [0]
                hispos_list_norm_torque = hispos_list_norm
                binpos_list_torque = [0]
            if len(hisneg_list)>2:
                divnorm = np.sum(histot_list) * np.diff(binneg_list)[1]
                hisneg_list_norm = hisneg_list/divnorm
                hisneg_list_norm_torque = hisneg_list_norm
            else:
                hisneg_list_norm = [0]
                binneg_list = [0]
                hisneg_list_norm_torque = hisneg_list_norm
                binneg_list_torque = [0]
            # for speed:
            hispos_dic[filenow] = hispos_list_norm
            binpos_dic[filenow] = binpos_list
            hisneg_dic[filenow] = hisneg_list_norm
            binneg_dic[filenow] = binneg_list
            # for torque:
            hispos_dic_torque[filenow] = hispos_list_norm_torque
            binpos_dic_torque[filenow] = binpos_list_torque
            hisneg_dic_torque[filenow] = hisneg_list_norm_torque
            binneg_dic_torque[filenow] = binneg_list_torque
            ## freq of switch:
            fr_switch_ma = np.append(fr_switch_ma, freq_switch)
            fr_switch_ma_mask = np.append(fr_switch_ma_mask, False)
            # by default, no flag :
            dtup_dic_flag[filenow] = False
            dtdw_dic_flag[filenow] = False
            # for traces with only CCW or only CW, with no switches:
            if freq_switch == 0 and tot_time_CW == 0 and tot_time_CCW > 0:
                # flag the dictionary:
                dtup_dic_flag[filenow] = True
                dtup_dic[filenow] = [tot_time_CCW, ]
                # max switch freq. compatible with trace is = 2 switches / tot_time:
                fr_switch_ma[-1] = 2./tot_time_CCW
                fr_switch_ma_mask[-1] = True
                #print('only CCW: ' + str(tot_time_CCW))
            if freq_switch == 0 and tot_time_CCW == 0 and tot_time_CW > 0:
                # flag the dictionary:
                dtdw_dic_flag[filenow] = True
                dtdw_dic[filenow] = [tot_time_CW,]
                # max switch freq. compatible with trace is = 2 switches / tot_time:
                fr_switch_ma[-1] = 2./tot_time_CW
                fr_switch_ma_mask[-1] = True
                #print('only CW: ' + str(tot_time_CW))
    if save_swfreq:
        print('\nanalysis_rotation.switching_combine_txtFiles(): SAVING '+path+'SpeedHistos.npy')
        np.save(path+'sw_freq' , [fr_switch_ma, fr_switch_ma_mask])
    # find total histogram from single histograms hispos hisneg:
    bin_gpos, his_gpos, bin_lpos, his_lpos = \
            histogram_of_many_arrays(binpos_dic, hispos_dic, numb_bins=25, plots=0)
    bin_gneg, his_gneg, bin_lneg, his_lneg = \
            histogram_of_many_arrays(binneg_dic, hisneg_dic, numb_bins=25, plots=0)
    # same for torque:
    bin_gpos_torque, his_gpos_torque, bin_lpos_torque, his_lpos_torque = \
            histogram_of_many_arrays(binpos_dic_torque, hispos_dic_torque, numb_bins=25, plots=0)
    bin_gneg_torque, his_gneg_torque, bin_lneg_torque, his_lneg_torque = \
            histogram_of_many_arrays(binneg_dic_torque, hisneg_dic_torque, numb_bins=25, plots=0)
    # write histogram of speeds to table file:        
    if table_filename:
        tablefile.write('\n')
        tablefile.write('speed bin_gpos\t speed his_gpos\t speed bin_gneg\t speed his_gneg\n')
        l = 0
        for l in range(len(bin_gpos)):
            tablefile.write(str(bin_gpos[l])+'\t'+str(his_gpos[l])+'\t'+ str(bin_gneg[l])+'\t'+str(his_gneg[l])+'\n')
            l = l+1
        # same for torque:
        tablefile.write('bin_gpos_torque\t his_gpos_torque\t bin_gneg_torque\t his_gneg_torque\n')
        l = 0
        for l in range(len(bin_gpos_torque)):
            tablefile.write(str(bin_gpos_torque[l])+'\t'+str(his_gpos_torque[l])+'\t'+ str(bin_gneg_torque[l])+'\t'+str(his_gneg_torque[l])+'\n')
            l = l+1
        tablefile.close()
    # save histos speed for figures?:
    if save_histosspeed:
        tosave = [bin_gpos, his_gpos, bin_gneg, his_gneg, hispos_dic, binpos_dic, hisneg_dic, binneg_dic]
        print('\nanalysis_rotation.switching_combine_txtFiles(): SAVING '+path+'SpeedHistos.npy')
        np.save(path+'SpeedHistos', tosave)
    # save dtup dtdw dict: 
    if save_dtupdw:
        tosave = [dtup_dic, dtup_dic_flag, dtdw_dic, dtdw_dic_flag]
        print('\nanalysis_rotation.switching_combine_txtFiles(): SAVING '+path+'dtupdw_dict.npy')
        np.save(path+'dtupdw_dict', tosave)
    # plots:
    if plots:
        plt.figure(345)
        plt.clf()
        plt.subplot(211)
        binlog = np.logspace(-3.1, 1, 20)
        fr_switch_ma.mask = fr_switch_ma_mask
        plt.semilogy(fr_switch_ma.data, 'ro')
        plt.plot(fr_switch_ma, 'bo')
        plt.xlabel('cell #')
        plt.ylabel('#switches / sec')
        plt.ylim([0.001, 10])
        plt.grid(True)        
        plt.subplot(212)
        # histo of masked values only:
        hfrsw_mask, bfrsw_mask = np.histogram(fr_switch_ma.compressed(), binlog)
        # histo of all data, masked and unmaked:
        hfrsw_unmask, bfrsw_unmask = np.histogram(fr_switch_ma.data, binlog)
        plt.bar(bfrsw_unmask[:-1], hfrsw_unmask, np.diff(binlog), color='r')
        plt.bar(bfrsw_mask[:-1], hfrsw_mask, np.diff(binlog))
        plt.xscale('log')
        plt.xlabel('#switches / sec')
        plt.ylabel('Occurrences')
        plt.grid(True)
        #plt.xlim([np.min(binlog),np.max(binlog)])
        plt.xlim([0.001,10])
        plt.suptitle(fig_title)
        if save_figs:
            plt.savefig(fig_title+'_switch_freq.png')
        #
        plt.figure(346, figsize=(10,7))
        plt.clf()
        jup = 0 
        jdw = 0 
        for key in dtup_dic.keys():
            #
            plt.subplot(221)
            # if flag is not True (many switches found):
            if dtup_dic_flag[key] == False:
                plt.semilogy(np.zeros(len(dtup_dic[key]))+jup, dtup_dic[key],'b-o', ms=3)
            else:
                plt.semilogy(np.zeros(len(dtup_dic[key]))+jup, dtup_dic[key],'ro',ms=3)
                dtup_list_onlyCCW = np.append(dtup_list_onlyCCW, dtup_dic[key])
            jup = jup + 1
            plt.ylabel('CCW time (s)')
            plt.xlabel('Cell number')
            plt.grid(True)
            plt.xlim([-3, jup+3])
            plt.ylim([0.001, 1000])
            #
            plt.subplot(223)
            # if flag is not True (many switches found):
            if dtdw_dic_flag[key] == False:
                plt.semilogy(np.zeros(len(dtdw_dic[key]))+jdw, dtdw_dic[key],'b-o', ms=3)
            else:
                plt.semilogy(np.zeros(len(dtdw_dic[key]))+jdw, dtdw_dic[key],'ro', ms=3)
                dtdw_list_onlyCW = np.append(dtdw_list_onlyCW, dtdw_dic[key])
            jdw = jdw + 1
            plt.ylabel('CW time (s)')
            plt.xlabel('Cell number')
            plt.xlim([-3, jdw+3])
            plt.ylim([0.001, 1000])
            plt.grid(True)
        #
        plt.subplot(222)
        binlog = np.logspace(-3, 3.1, 30)
        # init sum of all normalized distributions::
        hist_norm_up_sum = np.zeros(len(binlog)-1)
        hist_norm_dw_sum = np.zeros(len(binlog)-1)
        hist_norm_up_sum_flag = np.zeros(len(binlog)-1)
        hist_norm_dw_sum_flag = np.zeros(len(binlog)-1)
        for key in dtup_dic.keys():
            # normalized probability distributions for each cell:
            hist_norm_up, binlog_foo = np.histogram(dtup_dic[key], binlog, density=True)
            hist_norm_dw, binlog_foo = np.histogram(dtdw_dic[key], binlog, density=True)
            # change nan with 0s:
            hist_norm_up[np.where(np.isnan(hist_norm_up))] = 0
            hist_norm_dw[np.where(np.isnan(hist_norm_dw))] = 0
            # sum of all normalized distributions:
            hist_norm_up_sum = hist_norm_up_sum + hist_norm_up 
            hist_norm_dw_sum = hist_norm_dw_sum + hist_norm_dw 
            # sum of distributions with flag True:
            if dtup_dic_flag[key] == True:
                hist_norm_up_sum_flag = hist_norm_up_sum_flag + hist_norm_up
            if dtdw_dic_flag[key] == True:
                hist_norm_dw_sum_flag = hist_norm_dw_sum_flag + hist_norm_dw
        plt.bar(binlog[:-1], hist_norm_up_sum, np.diff(binlog), bottom=0.0001,color='b')
        plt.bar(binlog[:-1], hist_norm_up_sum_flag, np.diff(binlog), bottom=0.0001, color='r')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('CCW time (s)')
        plt.ylabel('Prob.Density')
        plt.title('NOT NORMALIZED ???!!!')
        plt.xlim([0.001, 1000])
        plt.grid(True)
        #
        plt.subplot(224)
        plt.bar(binlog[:-1], hist_norm_dw_sum, np.diff(binlog), bottom=0.0001,color='b')
        plt.bar(binlog[:-1], hist_norm_dw_sum_flag, np.diff(binlog), bottom=0.0001, color='r')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('CW time (s)')
        plt.ylabel('Prob.Density')
        plt.xlim([0.001, 1000])
        plt.suptitle(fig_title)
        if save_figs:
            plt.savefig(fig_title+'CWCCWtimes.png')
        #
        plt.figure(347)
        plt.clf()
        plt.suptitle(fig_title)
        plt.plot(bin_gpos, his_gpos, 'g--', lw=4)
        plt.plot(bin_gneg, his_gneg, 'g--', lw=4)
        for key in binpos_dic.keys():
            hp = np.array(hispos_dic[key])
            bp = np.array(binpos_dic[key])
            hn = np.array(hisneg_dic[key])
            bn = np.array(binneg_dic[key])
            bias_hphn = np.sum(hp)/np.sum(hn)
            if len(hp)>1:
                hp_f = hp[np.where(hp>0.000001)]
                bp_f = bp[np.where(hp>0.000001)]
                if bias_hphn >= 1:
                    plt.plot(bp_f, hp_f, 'b-', lw=1, alpha=0.3)
                else:
                    plt.plot(bp_f, hp_f, 'r-', lw=2, alpha=0.3)
                plt.yscale('log') 
            if len(hn)>1:
                hn_f = hn[np.where(hn>0.000001)]
                bn_f = bn[np.where(hn>0.000001)]
                if bias_hphn >= 1:
                    plt.plot(bn_f, hn_f, 'b-', lw=1, alpha=0.3)
                else:
                    plt.plot(bn_f, hn_f, 'r-', lw=2, alpha=0.3)
                plt.yscale('log') 
        plt.grid(True)
        plt.xlabel('Speed (Hz)')
        plt.ylabel('Probability')
        plt.xlim([-100,100])
        if save_figs:
            plt.savefig(fig_title+'Speed_Histo.png')
        #
        if drag:
            plt.figure(348)
            plt.clf()
            plt.suptitle(fig_title)
            plt.plot(bin_gpos_torque*1e21, his_gpos_torque, 'g-', lw=3)
            plt.plot(bin_gneg_torque*1e21, his_gneg_torque, 'g-', lw=3)
            for key in binpos_dic_torque.keys():
                hp = np.array(hispos_dic_torque[key])
                bp = np.array(binpos_dic_torque[key])
                hn = np.array(hisneg_dic_torque[key])
                bn = np.array(binneg_dic_torque[key])
                bias_hphn = np.sum(hp)/np.sum(hn)
                if len(hp)>1:
                    hp_f = hp[np.where(hp>0.000001)]
                    bp_f = bp[np.where(hp>0.000001)]
                    if bias_hphn >= 1:
                        plt.plot(bp_f*1e21, hp_f, 'b-', alpha=0.7)
                    else:
                        plt.plot(bp_f*1e21, hp_f, 'r-', alpha=0.7) 
                    plt.yscale('log') 
                if len(hn)>1:
                    hn_f = hn[np.where(hn>0.000001)]
                    bn_f = bn[np.where(hn>0.000001)]
                    if bias_hphn >= 1:
                        plt.plot(bn_f*1e21, hn_f, 'b-', alpha=0.7)
                    else:
                        plt.plot(bn_f*1e21, hn_f, 'r-', alpha=0.7) 
                    plt.yscale('log') 
            plt.grid(True)
            plt.xlabel('Torque (pN nm)')
            plt.ylabel('Probability')
        #plt.xlim([-100,100])
        if save_figs:
            plt.savefig(fig_title+'Torque_Histo.png')

        




def radius_xyphi(phi, x, y, FPS=1, winfilt=20, movie=True,queue=50,stepfw=5,wait_sec=0.1):
    '''analysis of the radius of the xy trajectory, 
       make a movie of x,y,theta in [idx1:idx2]'''
    from matplotlib import gridspec
    # angle to turns:
    phi = phi/360.
    FPS = float(FPS)
    time = np.arange(len(phi))/FPS
    # radius of xy trajectory:
    radius = np.sqrt(x**2 + y**2)
    radius_run = filters.run_win_smooth(radius, win=winfilt, fs=1./FPS, usemode='same')
    # angle speed:
    dphi = np.diff(phi)*FPS # !! WHY FPS HERE??!!
    dphi_run = filters.run_win_smooth(dphi, fs=1./FPS, win=winfilt, usemode='same')
    #dphi_med = filters.median_filter(dphi, fs=1./FPS, win=21)
    # x-y based speed:
    xy_speed = np.hypot(x[1:]-x[:-1], y[1:]-y[:-1])
    xy_speed_run = filters.run_win_smooth(xy_speed, win=winfilt, fs=1./FPS, usemode='same')

    plt.figure(1235)
    plt.clf()
    plt.subplot(211)
    plt.plot(phi,'k',lw=2)
    plt.subplot(212)
    plt.plot(dphi_run)
    plt.grid(True)
    # mouse click to select region to analyze further :
    print('CLICK TWICE TO SELECT REGION...')
    click1,click2 = plt.ginput(n=2, timeout=0)
    idx1 = click1[0]
    idx2 = click2[0]

    plt.figure(12351)
    gs = gridspec.GridSpec(4,1, height_ratios=[1,1,1,3])
    plt.clf()
    plt.subplot(gs[0])
    plt.plot(time[idx1:idx2], radius[idx1:idx2],color='0.6')
    plt.plot(time[idx1:idx2], radius_run[idx1:idx2],'k')
    plt.ylabel('Radius (A.U)')
    plt.grid(True)
    plt.subplot(gs[1])
    plt.plot(time[idx1:idx2], dphi[idx1:idx2],color='0.6')
    plt.plot(time[idx1:idx2], dphi_run[idx1:idx2],'k')
    plt.ylim([-100,100])
    plt.ylabel('Speed (Hz)')
    plt.grid(True)
    plt.subplot(gs[2])
    plt.plot(time[idx1:idx2], phi[idx1:idx2],'k',lw=2)
    plt.grid(True)
    plt.ylabel('Angle (turns)')
    plt.xlabel('Time (s)')
    plt.subplot(gs[3])
    plt.plot(dphi[idx1:idx2],radius[idx1:idx2], 'k.', ms=0.7)
    plt.plot(dphi_run[idx1:idx2],radius_run[idx1:idx2], 'ro', ms=2)
    plt.ylabel('Radius')
    plt.xlabel('Speed (Hz)')
    plt.xlim([-120,120])
    plt.grid(True)

    plt.figure(1233)
    plt.clf()
    x_pos = x[np.where(dphi_run > 0)[0]]
    x_neg = x[np.where(dphi_run < 0)[0]]
    y_pos = y[np.where(dphi_run > 0)[0]]
    y_neg = y[np.where(dphi_run < 0)[0]]
    if len(x_pos) > len(x_neg):
        plt.plot(x_pos, y_pos, 'b.', ms=1)
        plt.plot(x_neg, y_neg, 'r.', ms=2)
    else:
        plt.plot(x_neg, y_neg, 'r.', ms=1)
        plt.plot(x_pos, y_pos, 'b.', ms=2)
    if movie:
        plt.figure(1234)
        gs = gridspec.GridSpec(3,1, height_ratios=[2,1,2])
        plt.clf()
        plt.plot(x,y, '.', ms=1)
        for i in np.arange(idx1,idx2, stepfw):
            plt.clf()
            plt.subplot(gs[0])
            plt.plot(x[idx1:idx2], y[idx1:idx2], '.', ms=1)
            plt.plot(x[i], y[i],'ro', ms=8)
            plt.grid(True)
            plt.axis('equal')
            if i-queue > 1:
                plt.plot(x[i-queue:i], y[i-queue:i], 'go', ms=4)
            plt.subplot(gs[1])
            plt.plot(time[idx1:idx2], phi[idx1:idx2])
            if i-queue > 1:
                plt.plot(time[i-queue:i], phi[i-queue:i],'go',ms=4)
            plt.plot(time[i], phi[i], 'ro', ms=8)
            plt.ylabel('Angle (turns)')
            plt.xlabel('Time (s)')
            plt.grid(True)
            plt.subplot(gs[2])
            plt.plot(dphi[idx1:idx2],radius[idx1:idx2], 'k.', ms=0.7)
            plt.plot(dphi_run[idx1:idx2],radius_run[idx1:idx2], 'b.', ms=2)
            if i-queue > 1:
                plt.plot(dphi_run[i-queue:i], radius_run[i-queue:i],'go')
            plt.plot(dphi_run[i], radius_run[i],'ro')
            plt.ylabel('Radius')
            plt.xlabel('Speed (Hz)')
            plt.xlim([-120,120])
            plt.grid(True)
            plt.pause(wait_sec)
    


def show_movie(x,y,idx1,idx2, FPS=1,stepfw=10,queue=50,wait_sec=0.1):
    '''make a movie'''
    time = np.arange(len(x))/FPS
    plt.figure(1234)
    plt.clf()
    #plt.plot(x[1:-1],y[1:-1], '.', ms=1)
    for i in np.arange(idx1,idx2, stepfw):
        plt.clf()
        plt.plot(x[idx1:idx2], y[idx1:idx2], 'k.', ms=1)
        plt.plot(x[i], y[i],'ro', ms=8)
        plt.grid(True)
        plt.axis('equal')
        if i-queue > 1:
            plt.plot(x[i-queue:i+1], y[i-queue:i+1], 'g-o', ms=4)
        plt.pause(wait_sec)





def xyz_plot(x,y,z, theta=[], a=0,b=0,c=0, plot_type='scatter', rad_thr=0.5):
    '''plots in 3D xyz[a:b:c]'''
    from mpl_toolkits.mplot3d import Axes3D
    
    time = np.arange(len(x))
    radius = np.hypot(x[a:c]-np.mean(x[a:c]), y[a:c]-np.mean(y[a:b]))
    
    if len(theta)>1:
        plt.figure(121213)
        plt.clf()
        plt.subplot(311)
        plt.plot(time[a:b], theta[a:b], 'b')
        plt.plot(time[b:c],theta[b:c], 'r')
        plt.ylabel('Angle')
        plt.grid(True)
        plt.subplot(312)
        plt.plot(time[a:b-2], np.diff(filters.run_win_smooth(theta[a:b], win=10.0, usemode='same')[:-1]), 'b')
        plt.plot(time[b:c-2],np.diff(filters.run_win_smooth(theta[b:c], win=10.0, usemode='same')[1:]), 'r')
        plt.grid(True)
        plt.ylabel('Speed')
        plt.subplot(313)
        plt.plot(time[a:b], radius[a:b], 'b')
        plt.plot(time[b:c], radius[b:c], 'r')
        plt.ylabel('Radius')
        plt.grid(True)


    fig = plt.figure(121212)
    ax = fig.add_subplot(111, projection='3d')
    if plot_type == 'scatter':
        ax.scatter(x[a:c], y[a:c], z[a:c], c=z[a:c], marker='.')
    else:
        ax.plot(x[a:b], y[a:b], z[a:b], c='b', marker='.', lw=0.2, ms=1)
    
    fig = plt.figure(121214)
    ax = fig.add_subplot(111, projection='3d')
    if plot_type == 'scatter':
        ax.scatter(x[a:c], y[a:c], z[a:c], c=np.linspace(0,1,len(x[a:c])), marker='.')
    else:
        ax.plot(x[a:b], y[a:b], z[a:b], c='b', marker='.', lw=0.2, ms=1)
    plt.show()

