import numpy as np
import matplotlib.pyplot as plt
import itertools

from LangmuirSimple import LangmuirSimple
from TwoStateCatchTrad import TwoStateCatchTrad



Nmax = 12


def make_traces_LangmuirSimple(n_traces=20, kin=0.5, kout=0.5, N0=0, njumps=20, saves=False):
    ''' return n_traces LangmuirSimple traces, sampled at FPS, in a dict '''
    d = {}
    FPS = 1000
    for i in range(n_traces):
        print(f'{i}/{n_traces}', end='\r')
        l = LangmuirSimple()
        l.make_trace(kin, kout, N0, njumps, plots=False)
        t_smpl , N_smpl = l.make_sampled_trace(l.time, l.N, FPS=FPS, plots=False)
        d[i] = {'time':t_smpl, 'N':N_smpl, 'FPS':FPS}
        if saves:
            np.savetxt(f'LangmuirSimple_{i}.txt', [l.time,l.N])
    if saves:
        np.savetxt(f'LangmuirSimple_True.txt', [kin,kout,N0,njumps], header='kin, kout, N0, njumps')
    return d



def langmuir_exp(t, kin, kout, N0):
    KD = kout/kin
    Neq = Nmax/(KD + 1)
    tc = 1./(kout+kin)
    with np.errstate(over='ignore'):
        N = Neq + (N0-Neq)*np.exp(-t/tc)
    return N



def make_traces_TwoStateCatchTrad(n_traces=20, kuw=1, kus=0, kwu=1, kws=1, ksu=0, ksw=1, Ns0=0, Nw0=0, njumps=1e3):
    ''' return n_traces TwoStateCatchTrad traces, sampled at FPS, in a dict '''
    d = {}
    FPS = 1000
    for i in range(n_traces):
        print(f'{i}/{n_traces}', end='\r')
        l = TwoStateCatchTrad()
        l.make_trace(kuw=kuw, kus=kus, kwu=kwu, kws=kws, ksu=ksu, ksw=ksw, Ns0=Ns0, Nw0=Nw0, njumps=njumps, plots=False)
        t_smpl , N_smpl = l.make_sampled_trace(l.time, l.Ns+l.Nw, FPS=FPS, plots=False)
        d[i] = {'time':t_smpl, 'N':N_smpl, 'FPS':FPS}
    return d



def find_dwelltimes(t_smpl, N_smpl):
    ''' from the sampled trace N_smpl(t_smpl) find the times of the jumps and dwell times '''
    dN = np.diff(N_smpl)
    i = np.where(np.abs(dN)>0)
    ti = t_smpl[i]
    dwells = np.diff(ti)
    return ti, dwells



def find_jumps_intime(d, n_win=20, last='max', kin=None, kout=None, N0=None, plots=False):
    ''' running window, find all jumps in all traces 
    contained in dict d {time_sampled,N_sampled}, as coming from make_sampled_trace().
    return windows, and the num.jumps/num.traces in each window
        n_win: number of windows to use
        last ['max','min',(0,1)]: consider time up to the shortest or longest trace, or intermediate
        kin, kout, N0 : true LangmuirSimple parameters
    ''' 
    jumps_in_win      = np.zeros(n_win)
    traces_in_win     = np.zeros(n_win)
    jumps_over_traces = np.zeros(n_win)
    Nmn_in_win        = np.zeros(n_win)
    # consider time up to the shortest or up longest trace:
    if last == 'min':
        t_max = np.min([d[i]['time'][-1] for i in d])
    elif last == 'max':
        t_max = np.max([d[i]['time'][-1] for i in d])
    elif type(last) == float:
        t_max = last*np.max([d[i]['time'][-1] for i in d])
    print(f'find_jumps_intime(): t_max:{t_max}')
    # durations of traces :
    traces_Ts = np.array([d[i]['time'][-1] for i in d])
    # define time windows:
    w_ts = np.linspace(0, t_max, n_win, endpoint=0)
    w_dt = w_ts[1]
    print(f'find_jumps_intime(): win dt:{w_dt:.3} s')
    # find times of jumps in traces: 
    jump_times = [find_dwelltimes(d[k]['time'], d[k]['N'])[0] for k in d]
    jump_times = np.array([j for i in jump_times for j in i])
    # find mean trace:
    t_mn, N_mn = find_mean_trace(d, last=last, plots=False)
    # find jumps of all traces and n.of traces in windows:
    for l, wi in enumerate(w_ts):
        print(f'find_jumps_intime(): {l+1}/{n_win}', end='\r')
        jumps_in_win[l] = np.sum(len(np.nonzero(wi < jump_times*(jump_times <= wi+w_dt))[0]))
        traces_in_win[l] = len(np.nonzero(wi+w_dt <= traces_Ts)[0])
        jumps_over_traces[l] = jumps_in_win[l]/traces_in_win[l]
        # find N_nm in window:
        tmn_widx = np.nonzero(wi < t_mn*(t_mn <= wi+w_dt))[0]
        Nmn_in_win[l] = np.mean(N_mn[tmn_widx]) if len(N_mn[tmn_widx]) else None
    
    # PLOTS:
    if plots:
        
        def njumps_theo(kin, kout, Nmn, w_dt):
            '''theoretical n.jumps in win from LangmuirSimple'''
            return (kin*(Nmax-Nmn) + kout*Nmn)*w_dt
        
        def maxmin_LangTheo(kin0, dkin, kout0, dkout, N, plots=False):
            ''' return max and min of njumps_theo() and langmuir_exp()
            given rates (kin0,kout0) and rate error (dkin,dkout) '''
            if plots:
                fig = plt.figure('maxmin_LangTheo', clear=True)
                ax1 = fig.add_subplot(311)
                ax2 = fig.add_subplot(312)
                ax3 = fig.add_subplot(313)
            for i,(kin,kout) in enumerate(itertools.product((kin0-dkin, kin0+dkin), (kout0-dkout, kout0+dkout))):
                if i==0:
                    njumps_theo_max = njumps_theo(kin, kout, N, w_dt)
                    njumps_theo_min = njumps_theo(kin, kout, N, w_dt)
                    N_theo_min = langmuir_exp(t_mn, kin, kout, N0)
                    N_theo_max = langmuir_exp(t_mn, kin, kout, N0)
                njumps_theo_max = np.max([njumps_theo_max, njumps_theo(kin, kout, N, w_dt)], axis=0)
                njumps_theo_min = np.min([njumps_theo_min, njumps_theo(kin, kout, N, w_dt)], axis=0)
                N_theo_max      = np.max([N_theo_max, langmuir_exp(t_mn, kin, kout, N0)], axis=0)
                N_theo_min      = np.min([N_theo_min, langmuir_exp(t_mn, kin, kout, N0)], axis=0)
                if plots:
                    ax1.plot(njumps_theo(kin, kout, N, w_dt), label=f'{kin:.2f} {kout:.2f}')
                    ax2.plot(njumps_theo_max)
                    ax3.plot(njumps_theo_min)
                    ax1.legend(fontsize=9)
            return njumps_theo_max, njumps_theo_min, N_theo_max, N_theo_min
        
        fig = plt.figure('find_jumps_intime', clear=1)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222, sharex=ax1)
        ax3 = fig.add_subplot(223, sharex=ax1)
        ax4 = fig.add_subplot(224)
        for k in d:
            # all traces:
            ax1.plot(d[k]['time'][::10], d[k]['N'][::10], alpha=0.4)
        # dots for jumps in traces:
        ax1.plot(jump_times, np.zeros(len(jump_times)), 'o', alpha=0.2)
        # mean trace:
        ax1.plot(t_mn, N_mn, 'k', lw=2, label='mean')
        for w in w_ts:
            # plot windows:
            ax1.axvline(w, alpha=0.2)
            ax2.axvline(w, alpha=0.2)
            ax3.axvline(w, alpha=0.2)
        ax2.plot(w_ts+w_dt/2, jumps_in_win, 'o', label='jumps in win')
        ax2.plot(w_ts+w_dt/2, traces_in_win, 's', alpha=0.6, label='traces in win')
        ax3.plot(w_ts+w_dt/2, jumps_over_traces, 'ko', label='data') #TODO error on points?
        ax4.plot(Nmn_in_win, jumps_over_traces, '-ko', alpha=0.5, label='data')

        # if kin +- dkin provided:
        if kin != None and kout != None and N0 != None:
            if type(kin)==tuple and type(kout)==tuple:
                kin0, dkin  = kin
                kout0, dkout = kout
                njmax, njmin, Nthmax, Nthmin = maxmin_LangTheo(kin0, dkin, kout0, dkout, Nmn_in_win, plots=0)
                # LangmuirSimple error band:
                ax3.fill_between(w_ts+w_dt/2, njmax, njmin, color='r', alpha=0.7, linewidth=0) 
            elif type(kin)==float and type(kout)==float:
                kin0, kout0 = kin, kout
            # theoretical numb. of jumps in win for LangmuirSimple:
            ax3.plot(w_ts+w_dt/2, njumps_theo(kin0, kout0, Nmn_in_win, w_dt), 'r--', label='Lang.Theo')
            ax4.plot(np.arange(Nmax+1), (kin0*(Nmax-np.arange(Nmax+1)) + kout0*np.arange(Nmax+1))*w_dt, 'r--', label='Lang.Theo')
            ax4.fill_between(Nmn_in_win, njmin, njmax, linewidth=0, color='r', alpha=0.4)
            # LangmuirSimple error band:
            lab = f'Lang.Theo N0:{N0}\nkin:{kin0}$\pm{dkin}$\nkout:{kout0}$\pm${dkout}'
            ax1.fill_between(t_mn, Nthmin, Nthmax, color='r', alpha=0.9, linewidth=0)
            ax1.plot(t_mn, langmuir_exp(t_mn, kin0, kout0, N0), 'r--', label=lab)

        ax1.set_ylabel('N(t)')
        ax3.set_ylabel('<jumps> in win.')
        ax1.grid(False)
        ax2.grid(False)
        ax3.grid(False)
        ax1.legend(fontsize=7)
        ax2.legend(fontsize=7)
        ax3.legend(fontsize=7)
        ax4.legend(fontsize=7)
        ax1.set_xlabel('time (s)')
        ax2.set_xlabel('time (s)')
        ax3.set_xlabel('time (s)')
        ax4.set_ylabel('<jumps> in win')
        ax4.set_xlabel('N(t)')
        fig.tight_layout()

    return w_ts, jumps_over_traces



def find_mean_trace(d, last='min', plots=False):
    ''' mean trace from traces in dict d {time_smpl, N_smpl}
    last: ['min', 'max', num. in (0,1)] to make mean trace minimally, maximally (pad with nan), or fractional long
    '''
    import copy
    dd = copy.deepcopy(d) 
    if last=='min':
        # min length of traces:
        idx = np.min([len(i['time']) for i in list(dd.values())])
    elif last=='max' or type(last)==float:
        # increase length, expand traces with nan:
        idx = np.max([len(i['time']) for i in list(dd.values())])
        m = last if type(last)==float else 1
        idx = int(idx*m)
        for i in dd:
            dif = idx-len(dd[i]['N'])
            if dif > 0:
                dd[i]['N'] = np.append(dd[i]['N'], np.repeat(np.nan, idx-len(dd[i]['N'])))
            else:
                dd[i]['N'] = dd[i]['N'][:idx]

    # find avg trace:
    N_mn = np.nanmean(np.vstack([i['N'][:idx] for i in list(dd.values())]), axis=0)  
    t_mn = np.arange(0, idx)*np.diff(dd[0]['time'])[0]
    if plots:
        fig = plt.figure('find_mean_trace', clear=True)
        ax = fig.add_subplot(111)
        for i in dd:
            if last=='min':
                ax.plot(dd[i]['time'], dd[i]['N'])
            if last=='max' or type(last)==float:
                ax.plot(t_mn, dd[i]['N'])
        ax.plot(t_mn, N_mn, 'k', lw=2)
    return t_mn, N_mn


