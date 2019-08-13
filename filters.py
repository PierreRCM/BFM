# all kind of filters for time traces
# Fra 2015

# TODO :  use scipy.signal.filtfilt instead of to apply the Butterworth filter. filtfilt is the forward-backward filter. It applies the filter twice, once forward and once backward, resulting in zero phase delay.

import numpy as np
from scipy.signal import butter, lfilter, freqz, bode, medfilt, filtfilt
import matplotlib.pyplot as plt
import scipy.ndimage.filters


def savgol_filter(x, win, polyorder, plots=0):
    ''' from https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way/26337730#26337730'''
    import scipy.signal as sig
    if polyorder >= win:
        polyorder = win-1
        print('savgol_filter(): bad polyorder fixed to win-1')
    if np.mod(win,2) == 0:
        win = win+1
        print('Warning savgol_filter, win must be odd: forced win = '+str(win))
    y = sig.savgol_filter(x, window_length=win, polyorder=polyorder)
    if plots:
        plt.figure('savgol_filter()')
        plt.clf()
        plt.plot(x, '.')
        plt.plot(y)
    return y



def estimated_autocorrelation(x, fs=1, check=False, plots=False):
    """
    fs : sampling freq.
    check: double check with assert analyt expression
    from:
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    if check: 
        assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))             
    result = r/(variance*(np.arange(n, 0, -1)))
    timelag = np.arange(len(result))/fs
    if plots:
        plt.figure('estimated_autocorrelation')
        plt.subplot(211)
        plt.plot(timelag, result)
        plt.xlabel('time lag (s)')
        plt.ylabel('a.correlation')
        plt.subplot(212)
        plt.plot(np.arange(len(x))/fs, x)
        plt.xlabel('time (s)')
        plt.ylabel('signal')
    return result, timelag



def median_filter(data, fs=1, win=3, usemode='valid', plots=0):
    '''median filter. fs:sampling frequency. win:window kernel size for filter, usemode=['same' | 'valid'] '''
    if np.mod(win,2) == 0:
        win = win+1
        print('Warning median_filter, win must be odd: forced win = '+str(win))
    y = medfilt(data, kernel_size=win)
    if usemode == 'valid':
        y = y[int(np.ceil(win/2.)) : int(np.floor(-win/2.))]    
    if plots:
        plt.figure('median_filter()')
        plt.plot(np.arange(len(data))/fs, data, 'bo')
        plt.plot(np.arange(len(y))/fs, y, 'r-')
        plt.xlabel('Time (s)')
        plot_orig_filtered_spectra(data, y, fs)
    return y


def run_win_smooth(data, win=10., fs=1., usemode='valid', plots=False):
    '''average running window filter, by convolution
    win = pts length of running window
    fs [= 1] sampling freq.
    usemode: same as in np.convolve()
    return: y smoothed data '''
    
    box = np.ones(int(win))/win
    y = np.convolve(data, box, mode=usemode)
    if plots:
        freq_orig, spectrum_orig = calculate_spectrum(data, fs)
        freq_filtered, spectrum_filtered = calculate_spectrum(y, fs)
        plt.figure('run_win_smooth')
        plt.subplot(311)
        plt.plot(np.arange(len(data))/fs, data, 'bo')
        plt.plot(np.arange(len(y))/fs, y, 'r-')
        plt.xlabel('Time (s)')
        plt.subplot(312)
        plt.loglog(freq_orig, spectrum_orig)
        plt.grid(True)
        plt.title('Original')
        plt.subplot(313)
        plt.loglog(freq_filtered, spectrum_filtered)
        plt.grid(True)
        plt.title('Running window average ('+str(win)+' pts, '+str(win/fs)+' sec)')
        plt.xlabel('Freq. (Hz)')
    return y


def gaussian_filter(data, gsigma=1, plots=0, gmode='reflect'):
    ''' multi dim gaussian fileter of data '''
    y = scipy.ndimage.filters.gaussian_filter(data, gsigma, mode='reflect')
    if plots:
        plt.figure()
        plt.plot(data)
        plt.plot(y)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass', analog=False)
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5, plots=True):
    ''' fs : sampling freq. (pts/s) '''
    lowcut = float(lowcut)
    highcut = float(highcut)
    fs = float(fs)
    t = np.arange(0, len(data)/fs, 1./fs)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)    
    y = lfilter(b, a, data)
    if plots:
        plot_filter(a,b,fs,y,t,data, lowcut,highcut)
    return y




def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop', analog=False)
    return b, a

def bandstop_filter(data, lowcut, highcut, fs, order=5, plots=True):
    ''' fs : sampling freq. (pts/s) '''
    lowcut = float(lowcut)
    highcut = float(highcut)
    fs = float(fs)
    t = np.arange(0, len(data)/fs, 1./fs)
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)    
    y = lfilter(b, a, data)
    if plots:
        plot_filter(a,b,fs,y,t,data, lowcut,highcut)
    return y



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5, nodelay=False, plots=True):
    ''' fs : sampling freq. (pts/s) 
    if nodelay=True : use filtfilt() to remove delay '''
    cutoff = float(cutoff)
    fs = float(fs)
    t = np.arange(0, len(data)/fs, 1./fs)
    t = t[:len(data)]
    b, a = butter_lowpass(cutoff, fs, order=order)
    if nodelay: 
        y = filtfilt(b, a, data)#, padlen=300)
    else:
        y = lfilter(b, a, data)
    w, mag, phase = bode((b,a))
    if plots:
        plot_filter(a,b,fs,y,t,data, cutoff)
    return y



def plot_filter(a,b,fs,y,t,data, f1=0, f2=0):
    ''' 
    a,b from filterfs = sampling freq.
    f1 = cutoff, f2 = 0 (butter_lowpass)
    f1 = lowcut, f2 = highcut (butter_bandpass) 
    '''
    plot_orig_filtered_spectra(data, y, fs)
    w, h = freqz(b, a, worN=8000)
    plt.figure('plot_filter')
    plt.subplot(2, 1, 1)
    plt.semilogx(0.5*fs*w/np.pi, np.abs(h), 'b')
    if f1:
        plt.plot(f1, 0.5*np.sqrt(2), 'ko')
        plt.axvline(f1, color='k')
    if f2:
        plt.plot(f2, 0.5*np.sqrt(2), 'ko')
        plt.axvline(f2, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    

 
def calculate_spectrum(data, fs):
    ''' calculates the spectra of data, downsampling data at power of 2 points 
    fs : sampling freq. (pts/s) 
    TODO: change with scripts/powerspectrum/PSpectrum.py 
    '''
    import numpy.fft as fft
    # dowsample at power of 2 points:
    n_pts = np.floor(np.log(len(data))/np.log(2)) 
    print('Spectra warning: considering '+str(2**n_pts)+' points (2**'+str(n_pts)+'), instead of len(data)='+str(len(data))) 
    data = data[:int(2**n_pts)]
    spectrum = np.abs(fft.fft(data))
    freq = fft.fftfreq(len(data), d=1./fs)
    freq = freq[0:int(len(spectrum)/2.)]
    spectrum = spectrum[0:int(len(spectrum)/2.)]
    return freq, spectrum


def plot_orig_filtered_spectra(data, data_filtered, fs):
    '''plot the filtered and original spectrum of data and of diff(data) '''
    # signal derivative : 
    data_dif = np.diff(data)
    data_filtered_dif = np.diff(data_filtered)
    freq_orig_dif, spectrum_orig_dif = calculate_spectrum(data_dif, fs)
    freq_filtered_dif, spectrum_filtered_dif = calculate_spectrum(data_filtered_dif, fs)
    # signal :
    freq_orig, spectrum_orig = calculate_spectrum(data, fs)
    freq_filtered, spectrum_filtered = calculate_spectrum(data_filtered, fs)
    #plots:
    plt.figure('plot_orig_filtered_spectra_1')
    plt.subplot(211)
    plt.loglog(freq_orig, spectrum_orig)
    plt.grid(True)
    plt.title('Signal Original')
    plt.subplot(212)
    plt.loglog(freq_filtered, spectrum_filtered)
    plt.grid(True)
    plt.title('Signal Filtered')
    plt.xlabel('Freq. (Hz)')
    plt.figure('plot_orig_filtered_spectra_2')
    plt.subplot(211)
    plt.loglog(freq_orig_dif, spectrum_orig_dif)
    plt.grid(True)
    plt.title('derivative ORIGINAL')
    plt.subplot(212)
    plt.loglog(freq_filtered_dif, spectrum_filtered_dif)
    plt.grid(True)
    plt.title('derivative FILTERED')
    plt.xlabel('Freq. (Hz)')




