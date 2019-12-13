# collaboration with Victor Nov 2019
# IDEA: analyze and find statistics of spikes in speed (better on torque) traces
# for a given filter do
# running window on speed trace, find speed spikes locally as points < s*sigma 
# find t1 t2 of each spike
# for each spike: get t1, t2, Dt=t2-t1, amplitude, stator level start, stator level stop(=start(int)-amplitude(float)),
# extrapolate results for filter = 0

# TODO 
# needs raw data xy, angle
# run step finder, each point labeled by stator level
# spikes_analysis: in each window fit gaussian, get error, skew to negative 
# automatically set filter, extrapolate to filter = 0


import numpy as np
import matplotlib.pylab as plt
import sys
from scipy.optimize import curve_fit
import scipy.stats

paths = ['/home/francesco/scripts/repositories/BFM/', '/home/francesco/scripts/powerSpectrum']
for p in paths:
    if p not in sys.path:
        sys.path.append(p)
import PSpectrum
import filters
import break_points




class BFMnoise():

    test_filename = '/home/francesco/lavoriMiei/cbs/people/collaborations/Victor/BFM/data/D_WT_1000Res.p'

    def __init__(self, key=0, filter_name=None, filter_win=None, savgol_deg=5, filename=test_filename):
        self.umppx       = 0.1      # micron per pixel
        self.filename    = filename
        self.filter_name = filter_name
        self.savgol_deg  = savgol_deg
        self.filter_win  = filter_win
        self.key         = key
        self.get_dict(key)
        #self.make_filtered_traces()



    def get_dict(self, key=0, prints=True, plots=False):
        ''' get dict in filename and extract data in key '''
        # get dict:
        print(f'BFMnoise.get_dict(): loading {self.filename}')
        d = np.load(self.filename, allow_pickle=1, encoding='latin1')
        print(f'BFMnoise.get_dict(): d keys: {d.keys()}')
        print(f'BFMnoise.get_dict(): d[{key}] keys: {d[list(d.keys())[key]].keys()}')
        # extract from dict:
        self.FPS         = float(d[key]['FPS'])
        self.strain      = d[key]['strain']
        self.cellnum     = d[key]['cellnum']
        self.dbead_nm    = d[key]['dbead_nm']
        self.x           = d[key]['x']
        self.y           = d[key]['y']
        self.angle_turns = d[key]['theta_deg']/360
        
        

    def make_filtered_traces(self, filter_win=None, filter_name=None, savgol_deg=5, plots=False):
        ''' filter speed 
                filter_win : pts window for filer
                filter_name : 'savgol', 'run_win_smooth', None
                savgol_deg : degree of savgol filter
        '''
        # angular speed:
        self.speed_Hz = np.diff(self.angle_turns)*self.FPS
        # linear speed (um/s):
        self.xyspeed_ums = np.hypot(self.x[1:]-self.x[:-1], self.y[1:]-self.y[:-1])*self.umppx*self.FPS
        # store filter:
        if filter_name :
            self.filter_name = filter_name
        if filter_win:
            self.filter_win = filter_win
        if savgol_deg:
            self.savgol_deg = savgol_deg
        print(f'BFMnoise.make_filtered_traces(): Filtering by {self.filter_name} {self.filter_win} ...')
        # apply filter:
        if self.filter_name == 'savgol':
            self.speed_Hz_f    = filters.savgol_filter(self.speed_Hz   , self.filter_win, self.savgol_deg, plots=False)
            self.xyspeed_ums_f = filters.savgol_filter(self.xyspeed_ums, self.filter_win, self.savgol_deg, plots=False)
            self.angle_turns_f = filters.savgol_filter(self.angle_turns, self.filter_win, self.savgol_deg, plots=False)
        elif self.filter_name == 'run_win_smooth':
            self.speed_Hz_f    = filters.run_win_smooth(self.speed_Hz   , self.filter_win, plots=False)
            self.xyspeed_ums_f = filters.run_win_smooth(self.xyspeed_ums, self.filter_win, plots=False)
            self.angle_turns_f = filters.run_win_smooth(self.angle_turns, self.filter_win, plots=False)
        elif self.filter_name == None:
            self.speed_Hz_f    = self.speed_Hz
            self.xyspeed_ums_f = self.xyspeed_ums
            self.angle_turns_f = self.angle_turns
        else:
            raise Exception('BFMnoise.make_filtered_traces(): Error filter_name not valid')
        print(f'BFMnoise.make_filtered_traces(): Done.')
        if plots:
            self.speed_f_time_sec = np.arange(len(self.speed_Hz_f))/self.FPS
            plt.figure('make_speed', clear=True)
            plt.plot(self.speed_f_time_sec, self.speed_Hz_f, label=f'filter_win:{self.filter_win}')
            plt.legend()



    def speed_analysis(self, filter_win=None, filter_name=None, correct=True, use_speed='angular', polydeg=15, c0=None, c1=None, dws=None, plots=True):
        ''' various analysys of speed (xy-speed, angle-speed, ...) 
                use_speed : 'angular', 'linear'
                correct : bool, correct speed modulation
                polydeg : int, polynome degree to correct speed modulation
        '''
        self.use_speed = use_speed
        # filter speed:
        if filter_win or filter_name:
            self.make_filtered_traces(filter_win=filter_win, filter_name=filter_name, savgol_deg=5, plots=False)
        # choose idxs to crop:
        if c1:
            c1 = np.min([c1, len(self.speed_Hz), len(self.speed_Hz_f), len(self.angle_turns), len(self.x), len(self.y)])
        if not c1:
            c1 = np.min([len(self.speed_Hz), len(self.speed_Hz_f), len(self.angle_turns), len(self.x), len(self.y)])
        if not c0:
            c0 = 0
        print(f'BFMnoise.speed_analysis(): c0:{c0} c1:{c1} dws:{dws}')
        a  = self.angle_turns[c0:c1:dws] - self.angle_turns[c0]
        x  = self.x[c0:c1:dws]
        y  = self.y[c0:c1:dws]
        if use_speed == 'angular':
            sf = self.speed_Hz_f[c0:c1:dws]
        elif use_speed == 'linear':
            sf = self.xyspeed_ums_f[c0:c1:dws]
        if correct and use_speed == 'angular':
            # TODO: here angle_turns_f? or angle_turns?
            speed_corr = self.correct_speed_modulation(self.speed_Hz_f, self.angle_turns_f, polydeg=polydeg, c0=c0, c1=c1, plots=plots)    
            sfc = speed_corr[::dws]
        if correct and use_speed == 'linear':
            speed_corr = self.correct_speed_modulation(self.xyspeed_ums_f, self.angle_turns_f, polydeg=polydeg, c0=c0, c1=c1, plots=plots)    
            sfc = speed_corr[::dws]
        if not correct:
            sfc = None
        # store
        self.speed_f_corr = sfc
        nbins = 30
        # histo2D:
        h,bx,by = plt.histogram2d(x,y,nbins)
        # histo2D + mean speed:
        ss = scipy.stats.binned_statistic_2d(x, y, sf, statistic='mean', bins=nbins)
        # histo2D + mean speed corrected:
        ssc = scipy.stats.binned_statistic_2d(x, y, sfc, statistic='mean', bins=nbins)
        # spectrum:
        #freq, spectrum = PSpectrum.spectrum(self.speed_Hz_f, self.FPS, plots=plots)
        if plots:
            plt.figure('speed_analysis', clear=True)
            plt.subplot(231)
            plt.plot(np.arange(c0,c1,dws), a, label='raw')
            plt.plot(np.arange(c0,c1,dws), self.angle_turns_f[c0:c1:dws]-self.angle_turns_f[c0], label='filt')
            plt.legend()
            plt.xlabel('index')
            plt.ylabel('Turns')

            plt.subplot(432)
            plt.plot(np.arange(c0,c1,dws), sf, label='filt')
            plt.ylabel(f'{use_speed} speed')
            plt.legend()
            plt.subplot(435)
            if correct:
                plt.plot(np.arange(c0,c1,dws), sfc, 'g', label='filt corr')
                plt.legend()
            plt.ylabel(f'{use_speed} speed')
            plt.xlabel('index')
            
            plt.subplot(433)
            plt.plot(np.mod(a,1), sf, '.', ms=1, alpha=0.8, label='filt')
            plt.legend()
            plt.ylabel(f'{use_speed} speed')
            plt.subplot(436)
            if correct:
                plt.plot(np.mod(a,1), sfc, 'g.', ms=1, alpha=0.8, label='filt corr')
                plt.legend()
            plt.ylabel(f'{use_speed} speed')
            plt.xlabel('Turns (mod 1)')

            plt.subplot(245)
            plt.plot(y, -x, 'k.', mec='none', ms=3, alpha=0.3)
            plt.axis('image')

            plt.subplot(246)
            plt.imshow(h)
            plt.title('histo2d')
            plt.axis('image')

            plt.subplot(247)
            plt.imshow(ss.statistic)
            plt.title('mn speed filt')
            plt.axis('image')
            plt.colorbar()
            
            plt.subplot(248)
            plt.imshow(ssc.statistic)
            plt.title('mn speed filt corr')
            plt.axis('image')
            plt.colorbar()
            #plt.tight_layout()



    def correct_speed_modulation(self, speed_trace, angle_turns, polydeg=10, c0=None, c1=None, plots=False):
        ''' correct the modulation in mod(angle_turns,1) Vs speed_Hz_f 
        speed_trace, angle_turns: speed trace, angular or linear, and relative angle_turns trace
        polydeg : polyn degree to fit
        '''
        if c1:
            c1 = np.min([c1, len(speed_trace), len(self.angle_turns), len(self.x), len(self.y)])
        if not c1:
            c1 = np.min([len(self.speed_Hz), len(speed_trace), len(self.angle_turns), len(self.x), len(self.y)])
        if not c0:
            c0 = 0
        sf = speed_trace[c0:c1]
        am  = np.mod(angle_turns[c0:c1] - angle_turns[c0], 1)
        # polyn fit:
        pf = np.polyfit(am, sf, polydeg)
        po = np.poly1d(pf)
        # speed corrected:
        speed_corr = sf - po(am) + np.mean(sf)
        if plots:
            # get spectrum filtered and filtered_corrected:
            freq_corr, sp_corr = PSpectrum.spectrum(speed_corr - np.mean(speed_corr), self.FPS, plots=False, downsample=True)
            freq_full, sp_full = PSpectrum.spectrum(sf-np.mean(sf), self.FPS, plots=False, downsample=True)
            idx = int(len(freq_full)/2)
            freq_full = freq_full[1:idx]
            sp_full = sp_full[1:idx]
            freq_corr = freq_corr[1:idx]
            sp_corr = sp_corr[1:idx]
            plt.figure('correct_speed_modulation()', clear=True)
            _x = np.linspace(0,1,100)
            plt.subplot(311)
            plt.plot(am, sf, '.', alpha=0.1)
            plt.plot(_x, po(_x))
            plt.subplot(312)
            plt.loglog(freq_full, sp_full)
            plt.loglog(freq_corr, sp_corr, alpha=0.7)
            plt.axis('tight')
            plt.subplot(313)
            plt.plot(speed_corr)
            plt.plot(sf)
            plt.plot(po(am))
        return speed_corr



    def spikes_analysis(self, speed, angle_turns, correct=True, nwin=10, std_fact=3, cond_thr=10, plots_lev=0):
        ''' locally find and analyse spikes in speed    
        std_fact : find locally points at std(speed)*std_fact
        plots_lev: 0,1,2 levels
        cond_thr : threshold to separate points contigous in spikes
        '''
        if len(speed) != len(angle_turns):
            raise Exception(f'len(speed) {len(speed)} must be = len(angle_turns) {len(angle_turns)}')
        if plots_lev >= 1:
            fig1 = plt.figure('spikes_analysis 1', clear=True)
            ax1 = fig1.add_subplot(311)
            ax2 = fig1.add_subplot(312, sharex=ax1, sharey=ax1)
            ax3 = fig1.add_subplot(313)
        if plots_lev >= 2:
            fig2 = plt.figure('spikes_analysis 2', clear=True)
            ax21 = fig2.add_subplot(311)
            ax22 = fig2.add_subplot(312, sharex=ax21)
            ax23 = fig2.add_subplot(313)
        # idx to cut speed in windows:
        idxs = np.linspace(0, len(speed), nwin+1, endpoint=True).astype(int)
        didxs = np.diff(idxs)
        idx_spikes = []
        speed_corr = []
        k = 1
        for i,di in zip(idxs[:-1], didxs):
            print(f'BFMnoise.spikes_analysis():{k}/{len(didxs)} i:{i} di:{di}\r', end='')
            # speed in window:
            speedw = speed[i:i+di]
            angle_turns_w = angle_turns[i:i+di]
            # correct speed modulation:
            if correct:
                # TODO? rm lin fit from speedw, to remove trend, then correct_speed_modulation
                speedw = self.correct_speed_modulation(speedw, angle_turns_w, polydeg=10, plots=plots_lev==3)
            speed_corr = np.append(speed_corr, speedw)
            # mean and std in window:
            mn, std = np.mean(speedw), np.std(speedw)
            # find spikes:
            _idx_spikes = i + np.nonzero(speedw < mn - std*std_fact)[0]
            idx_spikes = np.append(idx_spikes, _idx_spikes).astype(int)
            k += 1
            if plots_lev >= 2:
                ax21.plot(range(i,i+di), speed[i:i+di])
                ax21.plot(_idx_spikes, speed[_idx_spikes], 'o')
                ax22.plot(range(i,i+di), speedw)
                ax22.plot(_idx_spikes, speedw[_idx_spikes-i], 'o')
                ax23.hist(speedw - np.mean(speedw), 50, log=True, histtype='step')
                plt.pause(.1)
        # filter out spikes:
        didx_spikes = np.diff(idx_spikes)
        cond_0 = (didx_spikes <= cond_thr)
        cond_1 = (didx_spikes > cond_thr)
        # initial (0) and final (1) idx of each spike: 
        spikes_idx0s = np.append(idx_spikes[0], idx_spikes[1:][cond_1])
        spikes_idx1s = np.append(idx_spikes[:-1][cond_1], idx_spikes[-1])
        # TODO? correct if trace starts or ends in the middle of a spike:
        if spikes_idx1s[0] < spikes_idx0s[0]:
            print('WARNING! spikes_idx1s[0] < spikes_idx0s[0]')
        if spikes_idx1s[-1] < spikes_idx0s[-1]:
            print('WARNING! spikes_idx1s[-1] < spikes_idx0s[-1]')
        # remove spikes of 1 pt duration:
        rm0pt = np.nonzero(spikes_idx1s-spikes_idx0s)[0]
        spikes_idx0s = spikes_idx0s[rm0pt]
        spikes_idx1s = spikes_idx1s[rm0pt]
        # initial (0) and final (1) time of each spike: 
        spikes_t0s = spikes_idx0s/self.FPS
        spikes_t1s = spikes_idx1s/self.FPS
        # spikes durations 1 (counts n.pts below cond_thr):
        spikes_durations1_s = np.diff(np.append(0, np.nonzero(cond_1)[0]))/self.FPS
        # spikes durations 2 (from t0 to t1):
        spikes_durations2_s = spikes_t1s - spikes_t0s
        # time between spikes:
        spikes_timebtw_s = spikes_t0s[1:] - spikes_t1s[:-1]
        # number of spikes in entire trace:
        spikes_numb = np.sum(cond_1)
        
        # find amplitude of spikes:
        speed_amp_atspike = []
        for i0,i1 in zip(spikes_idx0s, spikes_idx1s):
            meansp0 = np.mean(speed_corr[np.max([i0 - int(didxs[0]/2), 0]): i0])
            meansp1 = np.mean(speed_corr[i1: np.min([i1 + int(didxs[0]/2), len(speed_corr)])])
            speed_avg_atspike = np.mean([meansp0, meansp1])
            speed_min_atspike = np.min(speed_corr[i0:i1])
            speed_amp_atspike = np.append(speed_amp_atspike, speed_avg_atspike - speed_min_atspike)

        if plots_lev:
            ax1.plot(speed)
            ax1.plot(idx_spikes[:-1][cond_0], speed[idx_spikes[:-1][cond_0]], 'ro', alpha=0.2, mec='none')
            ax1.plot(idx_spikes[:-1][cond_1], speed[idx_spikes[:-1][cond_1]], 'bo', mfc='none')
            ax1.plot(idx_spikes, speed[idx_spikes], 'k.', ms=2)
            ax1.plot(idxs[:-1], speed[idxs[:-1]], 'k+')
            ax1.plot(spikes_idx0s, speed[spikes_idx0s], 's', mfc='none', ms=3)
            ax1.plot(spikes_idx1s, speed[spikes_idx1s], 'd', mfc='none', ms=3)
            ax1.set_ylabel('Speed (Hz)')
            ax1.set_xlabel('idx')
            ax2.plot(speed_corr)
            ax2.plot(idx_spikes, speed_corr[idx_spikes], 'r.', mec='none', alpha=0.1)
            ax2.set_ylabel('Speed corrected')
            ax2.set_xlabel('idx')
            ax3.semilogy(np.arange(len(idx_spikes)-1)[cond_0], didx_spikes[cond_0], 'ro', alpha=0.2, mec='none')
            ax3.semilogy(np.arange(len(idx_spikes)-1)[cond_1], didx_spikes[cond_1], 'bo', mfc='none')
            ax3.semilogy(didx_spikes, 'k.', ms=2)
            ax3.set_ylabel('diff(idx_spikes)')
            ax3.set_xlabel('idx_spikes')
            plt.tight_layout()

            fig3 = plt.figure('spikes_analysis 3', clear=True)
            bbins = 50
            ax31 = fig3.add_subplot(311)
            ax31.hist(spikes_durations1_s, bbins, label='method1')
            ax31.hist(spikes_durations2_s, bbins, alpha=0.7, label='method2')
            ax31.set_xlabel('spikes duration (s)')
            ax31.legend()
            ax32 = fig3.add_subplot(312)
            ax32.hist(spikes_timebtw_s, bbins)
            ax32.set_xlabel('Time between spikes(s)')
            ax33 = fig3.add_subplot(313)
            ax33.hist(speed_amp_atspike, bbins)
            ax33.set_xlabel('Spikes amplitude (Hz)')
            plt.tight_layout()



    def find_stoichiometry(self, c0=None, c1=None, dws=1000, penalty=40, stoich_thr=None, plots=True):
        ''' step finder based on ruptures plus basic stoichimoetry algorithm '''
        import break_points
        bkpts = break_points.BreakPoints(self.speed_Hz_f[c0:c1])
        bkpts.rpt_find_brkpts(segmentation_fn='BottomUp', cost_fn='l1', dws=dws, pen=penalty, plots=plots, clear=True)
        # all jumps in signal: TODO use mean or meidian for stoich_thr?
        sig_jumps = bkpts.sig_jumps
        # define threshold if not given:
        if not stoich_thr:
            stoich_thr = np.median(sig_jumps)
        print(f'BFMnoise.find_stoichiometry(): using stoich_thr:{stoich_thr:.3f}')
        # pice-wise linear signal:
        sig_bkpts = bkpts.sig_bkpts
        # round jumps wrt to threshold:
        jumps = np.round(np.diff(sig_bkpts)/stoich_thr)
        # idxs of large enough jumps:
        idxs = np.nonzero(jumps)[0]
        # generate stoichiometry trace:
        stoich = np.zeros(len(sig_bkpts))
        for i in idxs:
            stoich[i:] = stoich[i:] + jumps[i]
        stoich_offset = np.mean(stoich - sig_bkpts/stoich_thr)
        stoich_cor = stoich + np.round(stoich_offset)
        # store: 
        self.stoich_cor = stoich_cor
        if plots:
            plt.figure('find_stoichiometry', clear=True)
            plt.subplot(311)
            plt.plot(self.speed_Hz_f[c0:c1:dws])
            plt.plot(bkpts.sig_dws_bkpts, '-k', lw=2)
            plt.ylabel('relative stoich jump')
            plt.subplot(312)
            plt.plot(stoich)
            plt.plot(stoich_cor, label='+offset')
            plt.plot(jumps, label='jumps')
            plt.legend()
            plt.ylabel('absolute stoich')
            plt.xlabel('idx sig')
            plt.subplot(313)
            plt.plot(stoich - sig_bkpts/stoich_thr)
            plt.title(f'stoich_offset={stoich_offset:.3}', fontsize=10)
            plt.tight_layout()





    def gauss(self, x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    def gauss_fit(self, x, y):
        ''' '''
        popt, pcov = curve_fit(self.gauss, x, y, p0=[np.max(y), np.mean(x), np.std(x)])
        return popt



    def thr_noise_filt(self, key=0, c0=None, c1=None, filtwins=(51, 201, 1001), plots=True, clear=True, dws=10):
        ''' OLD, to fix.  fit histogram of speed cropped in c0:c1, for different filtwin '''
        fig = plt.figure('thr_noise_filt', clear=clear)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for i,fw in enumerate(filtwins):
            print(fw)
            # TODO fix this:
            self.make_speed(key, filtwin=fw, plots=False)
            if c1 == None: c1 = len(self.speed_Hz_f)
            sig = self.speed_Hz_f[c0:c1]
            # find probab. density of speed:
            h, b = np.histogram(sig, bins=300, density=True)
            idx = np.nonzero(h[:-1])
            b = b[idx]
            h = h[idx]
            # gaussian fit of all the histo:
            popt = self.gauss_fit(b, h)
            idx0 = np.argmin(np.abs(b - popt[1] + popt[2]))
            # gaussian fit of right side of histo only:
            #popt1 = self.gauss_fit(b[idx0:], h[idx0:])
            # parabola fit to log, only on right side:
            parab_fit1 = np.poly1d(np.polyfit(b[idx0:], np.log(h[idx0:]) ,2))
            # plot each:
            p, = ax2.semilogy(b, h, lw=4, alpha=.4, label='filt='+str(fw))
            ax2.semilogy(b, self.gauss(b, *popt), '--', lw=1, color=p.get_color())
            #ax2.semilogy(b, self.gauss(b, *popt1), '-', lw=1, color=p.get_color())
            ax2.semilogy(b, np.exp(parab_fit1(b)), color=p.get_color())
            ax2.semilogy(b[idx0], h[idx0], 'o', color=p.get_color())
            if i == 0 or i == len(filtwins)-1:
                ax1.plot(sig[::dws], color=p.get_color())
        ax2.set_xlabel('Speed (Hz)')
        ax2.set_ylabel('Prob. density')
        ax2.legend()
        ax2.set_ylim(bottom=np.min(h))
        ax1.set_ylabel('Speed (Hz)')



        







