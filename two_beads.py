# simulate [by Twobeads_sim()] or analyse [by Twobeads_exp()] two BFMs on the same cell 

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import sys

import filters



class Twobeads_sim():

    def __init__(self, eps=0, n=2**17, FPS=1000, cut_off=50., delay=0, xampl=1., yampl=1., 
            ext_com_type='', 
            ext_com_ampl=0, ext_com_freq=1, 
            ext_com_pulses_ampl=1, ext_com_pulses_len=100, ext_com_pulses_num=1, 
            ext_com_custom=[], 
            ext_x_type='', ext_z_type='', 
            ext_x_pulses_ampl=1, ext_x_pulses_len=100, ext_x_pulses_num=1,
            ext_z_pulses_ampl=1, ext_z_pulses_len=100, ext_z_pulses_num=1,
            ext_x_custom=[], ext_z_custom=[], 
            endfilt=1 ):
        '''
        Simulate and analyse two signals 'x' and 'z' with controlled correlation

        usage examples: 
        	import two_beads_4
        	tbs = two_beads_4.Twobeads_sim()
        	tbs = two_beads_4.Twobeads_sim(ext_com_type='pulses', ext_com_pulses_num=10, ext_com_pulses_ampl=3) 
        	
        x : random signal
        y : random signal un-correlated to x
        z : linear combination of x and y
        eps : the degree of correlation, in [0,1], controls the relative weight of x and y in z
            eps allows to pass from the two extremes:
            eps = 1 then z = x, and xcorr(x,z) = acorr(x) is calculated
            eps = 0 then z = y, and xcorr(x,y) is calculated
        delay : x can be delayed by delay (points)
        xampl, yampl : the amplitudes of x,y
        cut_off : if !=0, freq. for low pass filter the noisy x,y traces
        
        ext_com_type : external common signal to add ('sin', 'noise', 'pulses', 'custom', '')
        ext_com_ampl, ext_com_freq : controls a commn external signal added to both x and z  
        ext_com_custom : a custom external signal common
	ext_com_pulses_ampl,
	ext_com_pulses_len,
	ext_com_pulses_num : control common external pulses to add
	
        ext_x_type, ext_z_type : individual signals to add to x,z ('pulses', 'custom', '') with parameters:
        ext_x_pulses_ampl, ext_x_pulses_len, ext_x_pulses_num,
        ext_z_pulses_ampl, ext_z_pulses_len, ext_z_pulses_num,
        ext_x_custom, ext_z_custom : custom signals to add to x and z at the end

        endfilt : window for final filter of x,z (if ==1, no filter)

        notes/TODO: 
        - a low pass filter creates artifacts in the scrumbled signal, while the run_win_smooth doesn't.
        - it would be nice if eps control the xcorr (eps=0.4 => xcor(0)=0.4). it does not. why?
        - fix ext signals normalization
        '''
        self.eps = eps
        self.n = n 
        self.FPS = FPS
        self.cut_off = cut_off
        self.delay = delay
        self.xampl = xampl
        self.yampl = yampl
        self.ext_com_type = ext_com_type
        self.ext_com_ampl = ext_com_ampl
        self.ext_com_freq = ext_com_freq
        self.ext_com_pulses_ampl = ext_com_pulses_ampl
        self.ext_com_pulses_len = ext_com_pulses_len
        self.ext_com_pulses_num = ext_com_pulses_num
        self.ext_com_custom = ext_com_custom
        self.ext_x_type = ext_x_type
        self.ext_z_type = ext_z_type
        self.ext_x_pulses_ampl = ext_x_pulses_ampl
        self.ext_x_pulses_len  = ext_x_pulses_len
        self.ext_x_pulses_num  = ext_x_pulses_num
        self.ext_z_pulses_ampl = ext_z_pulses_ampl
        self.ext_z_pulses_len  = ext_z_pulses_len
        self.ext_z_pulses_num  = ext_z_pulses_num
        self.ext_x_custom = ext_x_custom
        self.ext_z_custom = ext_z_custom
        self.endfilt = endfilt

        self.make_corr_traces()
        self.find_xcorr()
        self.do_plots()
    

    def pulses(self, pulses_ampl, pulses_len, pulses_num):
        ''' return random pulses (negative)'''
        sigpulses = np.zeros(self.n)
        idx = np.random.randint(self.n-pulses_len, size=pulses_num)
        for i in idx:
            sigpulses[i : i+pulses_len] = sigpulses[i : i+pulses_len] - pulses_ampl
        return sigpulses


    def make_corr_traces(self):
        ''' creates two random signals x and z with controllable cross correlation  '''
        # define uncorrelated signals x,y: 
        x = np.random.randn(self.n)*self.xampl 
        y = np.random.randn(self.n)*self.yampl
        if self.cut_off > 1:
            x = filters.run_win_smooth(x, win=self.cut_off, usemode='same')
            y = filters.run_win_smooth(y, win=self.cut_off, usemode='same') 
            x = x/np.std(x)
            y = y/np.std(y)
        # define z from x y, the correlation with x is controlled by eps:
        z = (1-self.eps)*y + self.eps*x
        # define ext_com, an external signal common to x,z :
        if self.ext_com_type == 'sin':
            print('ext_com_type : sin')
            ext_com = self.ext_com_ampl * np.sin(np.arange(len(x))*self.ext_com_freq)
        elif self.ext_com_type == 'noise':
            print('ext_com_type : noise')
            ext_com = np.random.randn(self.n)*self.ext_com_ampl
            ext_com = filters.lowpass_filter(ext_com, self.ext_com_freq, self.FPS, order=4, plots=0)
        elif self.ext_com_type == 'pulses':
            print('ext_com_type. : pulses')
            #ext_com = np.random.randn(self.n)
            #ext_com = filters.lowpass_filter(ext_com, self.ext_com_freq, self.FPS, order=4, plots=0)*self.ext_com_ampl
            ext_com = self.pulses(self.ext_com_pulses_ampl, self.ext_com_pulses_len, self.ext_com_pulses_num)
        elif self.ext_com_type == 'custom':
            print('ext_com_type. : custom')
            ext_com = self.ext_com_custom * self.ext_com_ampl
        else:
            print('ext_com_type : none')
            ext_com = np.zeros(self.n)
        # apply ext_com:
        x = x + ext_com
        z = z + ext_com
        # add ext_x ext_z, independent external signal to x and z:
        if self.ext_x_type == 'pulses':
            print('ext_x_type : pulses')
            ext_x = self.pulses(self.ext_x_pulses_ampl, self.ext_x_pulses_len, self.ext_x_pulses_num)
        elif self.ext_x_type == 'custom':
            print('ext_x_type : custom')
            ext_x = self.ext_x_custom
        else: 
            print('ext_x_type : none')
            ext_x = np.zeros(self.n)
        if self.ext_z_type == 'pulses':
            print('ext_z_type : pulses')
            ext_z = self.pulses(self.ext_z_pulses_ampl, self.ext_z_pulses_len, self.ext_z_pulses_num)
        elif self.ext_z_type == 'custom':
            print('ext_z_type : pulses')
            ext_z = self.ext_z_custom
        else: 
            print('ext_z_type : none')
            ext_z = np.zeros(self.n)
        # apply ext_x, ext_z:
        x = x + ext_x
        z = z + ext_z
        # delay x wrt z:
        x = np.roll(x, -self.delay)
        # renormalize:
        ext_com = (ext_com - np.mean((np.mean(x), np.mean(z)))) / np.mean((np.std(x), np.std(z)))
        #TODO:
        #ext_x = (ext_x - np.mean(x)) / np.std(x)
        x = (x - np.mean(x))/np.std(x)
        z = (z - np.mean(z))/np.std(z)
        #end filter:
        if self.endfilt > 1:
            x = filters.run_win_smooth(x, win=self.endfilt, fs=1/self.FPS, usemode='same', plots=0)
            z = filters.run_win_smooth(z, win=self.endfilt, fs=1/self.FPS, usemode='same', plots=0)
        # store:
        self.x = x
        self.z = z
        self.ext_x = ext_x
        self.ext_z = ext_z
        self.ext_com = ext_com
    
    
    def find_xcorr(self): 
        '''find xcorr of x,z '''
        # normalized xcorr(x,z):
        xz_corr = scipy.signal.fftconvolve(self.x, self.z[::-1], mode='full')
        self.xz_corr = xz_corr/(len(self.x)) 
        # random permutations:
        x_perm = np.random.permutation(self.x)
        z_perm = np.random.permutation(self.z)
        x_perm = x_perm - np.mean(x_perm)
        z_perm = z_perm - np.mean(z_perm)
        if hasattr(self, 'cut_off') and self.cut_off != 0:
            x_perm = filters.run_win_smooth(x_perm, win=self.cut_off, usemode='same')
            z_perm = filters.run_win_smooth(z_perm, win=self.cut_off, usemode='same')
            x_perm = x_perm - np.mean(x_perm)
            z_perm = z_perm - np.mean(z_perm)
        x_perm = x_perm/np.std(x_perm)
        z_perm = z_perm/np.std(z_perm)
        xz_corr_perm = scipy.signal.fftconvolve(x_perm, z_perm[::-1], mode='full')
        self.xz_corr_perm = xz_corr_perm/(len(self.x)) 
        # auto correlation of x:
        acor_x = scipy.signal.fftconvolve(self.x, self.x[::-1], mode='full')
        self.acor_x = acor_x/(len(self.x))
        # autocorr of ext_com:
        if hasattr(self, 'ext_com'):
            ext_com_corr = scipy.signal.fftconvolve(self.ext_com, self.ext_com[::-1], mode='full') 
            self.ext_com_corr = ext_com_corr/(len(self.ext_com))
        #scipy.stats.pearsonr
    

    def do_plots(self, tzoom=(-1,1), clf=True):
        time = np.arange(len(self.x))/self.FPS
        timecor = np.arange(-len(self.x)+1,len(self.x))/self.FPS
        
        plt.figure('Twobeads_sim 1', clear=clf)
        plt.subplot(311)
        plt.plot(time, self.x,'-b', alpha=1, label='x')
        plt.plot(time, self.z,'-g', alpha=0.8, label='z')
        if np.any(self.ext_x):
            plt.plot(time, self.ext_x, 'c', label='ext_x')
        if np.any(self.ext_z):
            plt.plot(time, self.ext_z, 'r', label='ext_z')
        if np.any(self.ext_com):
            plt.plot(time, self.ext_com,'-y', label='ext_com')
        plt.legend()
        
        plt.subplot(312)
        plt.plot(timecor, self.xz_corr_perm, 'r.', ms=1, label='xz_perm xcorr')
        plt.plot(timecor, self.xz_corr, 'b.', ms=1, label='xz xcorr')
        plt.plot(timecor, self.ext_com_corr, 'y.', ms=1, label='ext_com acorr')
        plt.legend()
        plt.grid(True)
    
        plt.subplot(313)
        plt.plot(timecor, self.xz_corr_perm, 'r.', ms=1, label='xz_perm xcorr')
        plt.plot(timecor, self.xz_corr, 'b.', ms=1, label='xz xcorr')
        plt.plot(timecor, self.ext_com_corr, 'y.', ms=1, label='ext_com acorr')
        if len(tzoom):
            plt.xlim(tzoom)
        plt.legend()
        plt.grid(True)
        plt.xlabel('time (s)')
    
    
    



class Twobeads_exp():

    def __init__(self, angles_deg=(), speeds_Hz=(), xys_px=(), negate_0=False, negate_1=False, FPS=1, filtwin=1, filttype='run_win_smooth', ph_scramble_n=20):
        ''' 
        Analyse two experimental speed traces for correlation
        
        must provide:
        angles_deg = (theta0_deg, theta1_deg)
            or 
        speeds_Hz = (speed0_Hz, speed1_Hz)
        the speeds are filtered using 'filtwin' 'filttype'. See below for possible filttype.
        xys_px = (x0s_px, y0s_px, x1s_px, y1s_px): x-y trajectory
        take_abs : use abs of speeds
        FPS [1] : frames per seconds
        ph_scramble_n [20] : num. of iterations of phase scrambling for background in xcorr
        negate_0, negate_1 : negate speeds
        '''
        self.FPS = FPS
        self.cut_off = 0
        self.filtwin = filtwin
        self.filttype = filttype
        self.ph_scramble_n = ph_scramble_n
        self.usemode = 'valid'
        self.negate_0 = negate_0
        self.negate_1 = negate_1
        if len(speeds_Hz) == 2 and len(angles_deg) == 0:
            self.speed0_Hz = speeds_Hz[0]
            self.speed1_Hz = speeds_Hz[1]
        elif len(speeds_Hz) == 0 and len(angles_deg) == 2:
            self.speed0_Hz = np.diff(angles_deg[0])*FPS/360
            self.speed1_Hz = np.diff(angles_deg[1])*FPS/360
            self.angle0_deg = angles_deg[0]
            self.angle1_deg = angles_deg[1]
        else:
            raise Exception('Give either angles_deg or speeds_Hz as (A,B)')
        if len(xys_px) == 4:
            self.xys_px = xys_px
        if negate_0: 
            self.speed0_Hz = -self.speed0_Hz
        if negate_1:
            self.speed1_Hz = -self.speed1_Hz
        # make signals for speeds cross correlation:
        self.x = (self.speed0_Hz - np.mean(self.speed0_Hz))/np.std(self.speed0_Hz)
        self.z = (self.speed1_Hz - np.mean(self.speed1_Hz))/np.std(self.speed1_Hz)
        # make filtered signals:
        self.speed0_filt_Hz = self.do_filter(self.speed0_Hz, filttype, filtwin, self.usemode)
        self.speed1_filt_Hz = self.do_filter(self.speed1_Hz, filttype, filtwin, self.usemode)
        self.x_filt = self.do_filter(self.x, filttype, filtwin, self.usemode)
        self.z_filt = self.do_filter(self.z, filttype, filtwin, self.usemode)
        self.x_filt_copy = np.copy(self.x_filt)
        self.z_filt_copy = np.copy(self.z_filt)
        self.x_filt = (self.x_filt-np.mean(self.x_filt))/np.std(self.x_filt)
        self.z_filt = (self.z_filt-np.mean(self.z_filt))/np.std(self.z_filt)
        # do xcorrs:
        self.find_xcorr()
        self.stats()
        # do plots:
        self.do_plots()



    def do_filter(self, sig, filttype, filtwin, usemode):
        ''' apply a filter to sig
        usemode: 'same', 'valid', 'full'.
        filttype: 'run_win_smooth', 'savgol', 'median'.
        '''
        if filttype == 'run_win_smooth':
            sig_filt = filters.run_win_smooth(sig, filtwin, usemode=usemode)
        elif filttype == 'savgol':
            sig_filt = filters.savgol_filter(sig, filtwin, 8)
            print('Twobeads_exp(): savgol filter applied')
        elif filttype == 'median':
            sig_filt = filters.median_filter(sig, win=filtwin, usemode=usemode)
        else:
            sig_filt = sig
            print('Twobeads_exp(): no filter applied')
        return sig_filt



    def find_xcorr(self): 
        '''find xcorr of x,z 
        x,z should be already normalized (-mn /std)'''
        # normalized xcorr(x,z):
        self.xz_filt_corr = scipy.signal.fftconvolve(self.x_filt, self.z_filt[::-1], mode='full')/(len(self.x_filt))
        # phase scrambled:
        xz_corr_phscr_filt_stack = np.nan*np.zeros(len(self.xz_filt_corr))
        for i in range(self.ph_scramble_n):
            # make scrambled:
            x_phscr = phaseScrambleTS(self.x)
            z_phscr = phaseScrambleTS(self.z)
            # filter scrambled:
            x_phscr_filt = self.do_filter(x_phscr, self.filttype, self.filtwin, self.usemode)
            z_phscr_filt = self.do_filter(z_phscr, self.filttype, self.filtwin, self.usemode)
            # normalize scrambled_filtered:
            x_phscr_filt = (x_phscr_filt - np.mean(x_phscr_filt))/np.std(x_phscr_filt)
            z_phscr_filt = (z_phscr_filt - np.mean(z_phscr_filt))/np.std(z_phscr_filt)
            # xcorr of scrambled_filtered:
            _xz_corr_phscr_filt = scipy.signal.fftconvolve(x_phscr_filt, z_phscr_filt[::-1], mode='full')/(len(x_phscr_filt)) 
            # stack xcorrs: 
            xz_corr_phscr_filt_stack = np.vstack((xz_corr_phscr_filt_stack, _xz_corr_phscr_filt))
            print( '                                         ', end='\r') #shithack!
            print(f'phase scrumbling {i}/{self.ph_scramble_n}', end='\r', flush=1)
        print()
        # create band of scrumbled background xcorr:
        self.phscr_filt_mn  = np.nanmean(xz_corr_phscr_filt_stack, axis=0)
        self.phscr_filt_max = np.nanmax(xz_corr_phscr_filt_stack, axis=0)
        self.phscr_filt_min = np.nanmin(xz_corr_phscr_filt_stack, axis=0)
        self.phscr_filt_std = np.nanstd(xz_corr_phscr_filt_stack, axis=0)
        # auto correlation of x:
        #acor_x = scipy.signal.fftconvolve(self.x, self.x[::-1], mode='full')
        #self.acor_x = acor_x/(len(self.x))


    def stats(self):
        ''' various stats for xcorr '''
        import scipy.stats as ss 
        self.ss_pearsonr = ss.pearsonr(self.x, self.z)
        print('find_xcorr(): pearsonr: \tcorr= {:.3} \tpval= {:.3}'.format(self.ss_pearsonr[0], self.ss_pearsonr[1]))
        self.ss_spearmanr = ss.spearmanr(self.x, self.z)
        print('find_xcorr(): spearmanr: \tcorr= {:.3} \tpval= {:.3}'.format(self.ss_spearmanr.correlation, self.ss_spearmanr.pvalue))
        try:
            self.ss_kendalltau = ss.kendalltau(self.x, self.z)
            print('find_xcorr(): kendalltau: \tcorr= {:.3} \tpval= {:.3}'.format(self.ss_kendalltau.correlation, self.ss_kendalltau.pvalue))
        except:
            self.ss_kendalltau = ss.kendalltau(0,0)
            print('Warning: find_xcorr() kendalltau error. Skipping.')


    def do_plots(self):
        time = np.arange(len(self.speed0_filt_Hz))/self.FPS
        timecor = np.arange(-len(self.speed0_filt_Hz)+1, len(self.speed0_filt_Hz))/self.FPS

        fig1 = plt.figure('Twobeads_exp 1', clear=True)
        ax1 = fig1.add_subplot(611)
        ax1.plot(time, self.speed0_filt_Hz, 'b')
        ax1.set_ylabel('speed0 (Hz)')
        ax2 = fig1.add_subplot(612, sharex=ax1)
        ax2.plot(time, self.speed1_filt_Hz, 'b')
        ax2.set_ylabel('speed1 (Hz)')
        ax2.set_xlabel('Time (s)')
        ax1.set_title(self.filttype+' '+str(self.filtwin)+' pts', fontsize=10)

        plt.subplot(323)
        plt.plot(self.speed0_filt_Hz, self.speed1_filt_Hz, 'b,', ms=1, alpha=0.1, label='filt.')
        plt.xlabel('speed0 (Hz)')
        plt.ylabel('speed1(Hz)')
        plt.grid(True, linestyle='--')
        plt.axis('equal')

        if hasattr(self, 'angle0_deg'):
            plt.subplot(626)
            plt.plot(np.mod(self.angle0_deg[int(self.filtwin/2)+1 : -int(self.filtwin/2)], 360), self.speed0_filt_Hz, '.', ms=2)
            plt.ylabel('speed0 (Hz)')
            plt.subplot(628)
            plt.plot(np.mod(self.angle1_deg[int(self.filtwin/2)+1 : -int(self.filtwin/2)], 360), self.speed1_filt_Hz, '.', ms=2)
            plt.xlabel('angle (deg)')
            plt.ylabel('speed1 (Hz)')
        
        plt.subplot(313)
        plt.plot(timecor, self.xz_filt_corr, lw=3, color='w', alpha=0.7)
        plt.plot(timecor, self.xz_filt_corr, lw=1, color='k', label='xcor(filt)', alpha=0.9)
        plt.gca().fill_between(timecor, self.phscr_filt_min, self.phscr_filt_max, lw=0, color='r', alpha=0.2)
        plt.gca().fill_between(timecor, self.phscr_filt_mn + self.phscr_filt_std, self.phscr_filt_mn - self.phscr_filt_std, lw=0, color='r', alpha=0.3, label='xcorr(filt(scramb))')
        plt.legend(fontsize=8)
        plt.xlabel('Time lag (s)')
        plt.tight_layout()

        if hasattr(self, 'xys_px'):
            plt.figure('Twobeads_exp 2', clear=True)
            plt.subplot(121)
            plt.plot(self.xys_px[0][:-1], self.xys_px[1][:-1], '.', ms=0.1)
            plt.subplot(122)
            plt.plot(self.xys_px[2][:-1], self.xys_px[3][:-1], '.', ms=0.1)
            plt.tight_layout()



def phaseScrambleTS(ts):
    '''phase scramble time trace ts
    Returns a TS: original TS power is preserved; TS phase is shuffled.
    from https://stackoverflow.com/questions/39543002/returning-a-real-valued-phase-scrambled-timeseries
    '''
    fs = np.fft.fft(ts)
    pow_fs = np.abs(fs)**2
    phase_fs = np.angle(fs)
    phase_fsr = phase_fs.copy()
    if len(ts) % 2 == 0:
        phase_fsr_lh = phase_fsr[1:int(len(phase_fsr)/2)]
    else:
        phase_fsr_lh = phase_fsr[1:int(len(phase_fsr)/2 + 1)]
    np.random.shuffle(phase_fsr_lh)
    if len(ts) % 2 == 0:
        phase_fsr_rh = -phase_fsr_lh[::-1]
        phase_fsr = np.concatenate((np.array((phase_fsr[0],)), phase_fsr_lh, np.array((phase_fsr[int(len(phase_fsr)/2)],)), phase_fsr_rh))
    else:
        phase_fsr_rh = -phase_fsr_lh[::-1]
        phase_fsr = np.concatenate((np.array((phase_fsr[0],)), phase_fsr_lh, phase_fsr_rh))
    fsrp = np.sqrt(pow_fs) * (np.cos(phase_fsr) + 1j * np.sin(phase_fsr))
    tsrp = np.fft.ifft(fsrp)
    if not np.allclose(tsrp.imag, np.zeros(tsrp.shape)):
        max_imag = (np.abs(tsrp.imag)).max()
       # imag_str = '\nNOTE: a non-negligible imaginary component was discarded.\n\tMax: {}'
        print(f'NOTE: a non-negligible imaginary component was discarded.')
    return tsrp.real
