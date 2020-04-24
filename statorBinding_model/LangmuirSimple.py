# kin. montecarlo simulation of stator binding and unbinding in simple Langmuir
# trying to fit individiual traces and see the distribution of the fit parameters (John)


import sys
import matplotlib.pyplot as plt
import numpy as np
import progressbar
from scipy.optimize import curve_fit



class LangmuirSimple():

    def __init__(self):
        ''' ex:
                l = statorBinding_model_4.LangmuirSimple() 
                l.auto_make(kin=1, kout=0.2, N0=11, njumps=20, plots_each=1, ntraces=10)
        '''
        self.Nmax = 12 # max numb. of stators:
        self.S = 1 # stator number in membrane:



    def make_trace(self, kin, kout, N0, njumps, plots=False):
        ''' Kinetic Montecarlo 
        dN/dt = kin*S*(Nmax-N) - kout*N
        '''
        kin = float(kin)
        kout = float(kout)
        # bound stators:
        N = np.zeros(njumps)
        time = np.zeros(njumps)
        N[0] = N0
        # for next state:
        ra1 = np.random.rand(njumps)
        # for time step:
        ra2 = np.random.rand(njumps)
        for t in range(njumps-1):
            rup = kin*self.S*(self.Nmax-N[t])
            rdw = kout*N[t]
            rtot = rup + rdw
            # cumulative rate array:
            rvec = np.array([rdw, rup+rdw])/rtot
            N_next = [-1,1]
            # dice for next state (idx of rvec_next):
            choose = np.where(rvec > ra1[t])[0][0]
            # update state N:
            N[t+1] = N[t] + N_next[choose]
            # update time:
            time[t+1] = time[t] + np.log(1./ra2[t])/rtot
            #print(t, N[t], rvec, rtot, choose, N_next[choose])
        # store:
        self.time = time
        self.N = N
        # get theoretical values:
        self.make_theory(kin, kout, N0, prints=False)
        if plots: 
            self.make_plots()



    def make_sampled_trace(self, t_kmc, N_kmc, FPS=1000, plots=True):
        ''' from KMC trace (t,N) makes a sampled trace with FPS points per second '''
        t_smpl = np.arange(0, t_kmc[-1], 1/FPS)
        N_smpl = np.zeros(len(t_smpl))
        for i,t in enumerate(t_smpl):
            idx = len(np.nonzero(t_kmc < t)[0])
            N_smpl[i] = N_kmc[np.clip(idx-1, 0, None)]
        if plots:
            fig = plt.figure('make_sampled_trace', clear=True)
            plt.plot(t_kmc, N_kmc, 'o')
            plt.plot(t_smpl, N_smpl, '-x' )
        return t_smpl, N_smpl


    
    def make_theory(self, kin, kout, N0, prints=True):
        # dissociation constant:
        KD = kout/kin
        # equilibrium N:
        self.Neq = self.Nmax*self.S/(KD+self.S)
        # N(t) theoretical.
        #self.time_Nt = np.logspace(np.log10(self.time[1]), np.log10(self.time[-1]), 2000)
        self.time_Nt = np.linspace(self.time[0], self.time[-1], 2000)
        tc = 1./(kout+kin*self.S)
        self.Nt = self.Neq + (N0-self.Neq)*np.exp(-self.time_Nt/tc)
        if prints:
            print('\nKD = kout/kin = '+str(KD))
            print('self.Neq = '+'{:.2f}'.format(self.Neq))
            print('t_c = 1/(kout+kin*self.S) = '+str(tc))
            print('1/t_c = (kout+kin*self.S) = '+str(1./tc))
        


    def langmuir_exp(self, t, kin, kout, N0):
        KD = kout/kin
        Neq = self.Nmax*self.S/(KD + self.S)
        tc = 1./(kout+kin*self.S)
        with np.errstate(over='ignore'):
            N = Neq + (N0-Neq)*np.exp(-t/tc)
        return N



    def fit_trace(self, prints=True):
        p0 = [0.5, 0.5, self.Nmax/2]
        self.popt, _ = curve_fit(self.langmuir_exp, self.time, self.N, p0=p0)
        if prints: print(f'fit_trace(): {self.popt}')
        if prints: print(f'fit_trace(): {self.time}')
        if prints: print(f'fit_trace(): {self.N}')



    def auto_make(self, kin=0.5, kout=0.5, N0=0, njumps=20, ntraces=10, plots_each=False):
        ''' automatically make trace and N stats  '''
        fig = plt.figure('auto_make', clear=True)
        ax1 = fig.add_subplot(121)        
        ax2 = fig.add_subplot(122)        
        N0 = int(N0)
        N_cumul = []
        for i in range(ntraces):
            print(f'auto_make: {i+1}/{ntraces}', end='\r')
            self.make_trace(kin, kout, N0, njumps, plots=False)
            # to make hist of all N:        
            dwt = (np.diff(self.time)*1000).astype(int)
            N_cumul = np.append(N_cumul, np.repeat(self.N[:-1], dwt))
            if plots_each:
                ax1.plot(self.time, self.N, '-o',ms=3, drawstyle='steps-post')
                ax1.set_ylim(0, self.Nmax+1)
        # plots:
        ax2.axvline(self.Neq, linestyle='--', lw=2, color='k', label=f'true({self.Neq:.1f})')
        ax2.hist(N_cumul, range(self.Nmax+1), density=1, align='left')
        ax2.set_xlabel('N(t) simul')
        ax2.set_ylabel('PDF')
        ax2.legend()
        ax2.set_title(f'<N> ={np.mean(N_cumul):.1f} $\pm$ {np.std(N_cumul):.1f} std')
        ax1.set_ylabel('N(t)')
        ax1.set_title(f'kin={kin}, kout={kout}, N0={N0}', fontsize=9)
        fig.tight_layout()



    def auto_make_fit(self, kin=0.5, kout=0.5, N0=0, njumps=20, ntraces=10, ntraces_mean=10, plots_each=False):
        ''' automatically make trace and fit it 
        make distribution of fit parameters 
        '''
        fig = plt.figure('auto_make_fit', clear=True)
        ax0 = fig.add_subplot(321)
        ax00 = fig.add_subplot(322)
        ax1 = fig.add_subplot(323)
        ax2 = fig.add_subplot(324)
        ax3 = fig.add_subplot(325)
        ax4 = fig.add_subplot(326)
        # init:
        N0 = int(N0)
        kin_fit = []
        kout_fit = []
        N0_fit = []
        Neq_fit = []
        N_all = {}
        fitok = 0
        fitok1 = 0
        for i in range(ntraces):
            print(f'auto_make_fit(): {i+1}/{ntraces}')
            self.make_trace(kin, kout, N0, njumps, plots=False)
            # fit single trace:
            try:
                self.fit_trace(prints=0)
                # keep only kin kout>0:
                if self.popt[0]>0 and self.popt[1]>0:
                    # append and store all fit params:
                    kin_fit  = np.append(kin_fit , self.popt[0])
                    kout_fit = np.append(kout_fit, self.popt[1])
                    N0_fit   = np.append(N0_fit  , self.popt[2])
                    Neq_fit  = np.append(Neq_fit , self.Nmax*self.S/((self.popt[1]/self.popt[0])+self.S))
                    fitok += 1
                    if i < ntraces_mean:
                        # make sampled trace:
                        N_all[i] = self.make_sampled_trace(self.time, self.N, FPS=1000, plots=False)
                        p0, = ax0.plot(self.time, self.N, '-', drawstyle='steps-post', alpha=0.2)
                        ax0.plot(self.time_Nt, self.langmuir_exp(self.time_Nt, *self.popt), '-', color=p0.get_color(), lw=2)
                        fitok1 += 1
                else:
                    print(f'auto_make_fit(): {i} Error in fit. Skipping.')
            except:
                print(f'auto_make_fit(): {i} Error in fit. Skipping.')
            if plots_each:
                self.make_plots(clear=False)
                plt.pause(0.1)

        ax0.axhline(self.Neq, color='k', linestyle='--', lw=2)
        ax0.plot(self.time_Nt, self.Nt, 'k--', ms=3, lw=2, label='theo.')
        ax0.legend(fontsize=9)
        ax0.set_ylim((-.2, self.Nmax+0.9))
        ax0.set_ylabel('N(t)')
        ax0.set_title(f'Simul.param:\nkin={kin}, kout={kout}, N0={N0}, Neq={self.Neq:.1f}, n.jumps={njumps}', fontsize=9)
        
        # crop all N traces to same min length:
        idx_min = np.min([len(i[0]) for i in list(N_all.values())])
        # find avg trace:
        N_all_mn = np.mean(np.vstack([i[1][:idx_min] for i in list(N_all.values())]), axis=0)
        for i in list(N_all.values()):
            ax00.plot(i[0], i[1], alpha=0.2)
        ax00.plot(self.time_Nt, self.Nt, 'k--', ms=3, lw=4, label='theo.')
        ax00.plot(i[0][:idx_min], N_all_mn, 'r', lw=2, label='mean')
        ax00.legend(fontsize=9)
        ax00.set_title(f'mean of {fitok1} sim.traces', fontsize=9)

        ax1.hist(kin_fit,  np.logspace(np.log10(np.min(kin_fit)),  np.log10(np.max(kin_fit)),  100), label='kin_fit', density=0)
        ax2.hist(kout_fit, np.logspace(np.log10(np.min(kout_fit)), np.log10(np.max(kout_fit)), 100), label='kout_fit', density=0)
        ax3.hist(N0_fit , np.linspace(0, self.Nmax+1, 30), label='N0_fit', density=1, align='right')
        ax4.hist(Neq_fit, np.linspace(0, self.Nmax+1, 30), label='Neq_fit', density=1, align='right')
        ax1.axvline(kin, linestyle='--', lw=2, color='k', label=f'true({kin})')
        ax2.axvline(kout, linestyle='--', lw=2, color='k', label=f'true({kout})')
        ax3.axvline(N0, linestyle='--', lw=2, color='k', label=f'true({N0})')
        ax4.axvline(self.Neq, linestyle='--', lw=2, color='k', label=f'true({self.Neq:.1f})')
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax3.set_xticks(np.arange(self.Nmax+1))
        ax4.set_xticks(np.arange(self.Nmax+1))
        ax1.legend(fontsize=9)
        ax2.legend(fontsize=9)
        ax3.legend(fontsize=9)
        ax4.legend(fontsize=9)
        ax1.set_title(f'Fit param. of {fitok} sim.traces:\n<kin_fit>={10**(np.mean(np.log10(kin_fit))):.1f} $\pm$ {10**(np.std(np.log10(kin_fit))):.1f}', fontsize=9)
        ax2.set_title(f'<kout_fit>={10**(np.mean(np.log10(kout_fit))):.1f} $\pm$ {10**(np.std(np.log10(kout_fit))):.1f}', fontsize=9)
        ax3.set_title(f'<N0_fit>={np.mean(N0_fit):.1f} $\pm$ {np.std(N0_fit):.1f}', fontsize=9)
        ax4.set_title(f'<Neq_fit>={np.mean(Neq_fit):.1f} $\pm$ {np.std(Neq_fit):.1f}', fontsize=9)
        fig.tight_layout()
        #return kin_fit, kout_fit, N0_fit



    def make_plots(self, clear=False): 
        fig1 = plt.figure('make_plots 1', clear=clear)     
        ax11 = fig1.add_subplot(111)
        
        if hasattr(self, 'N'):
            N = self.N + np.random.randn(len(self.N))*0.1
            time = self.time
            ax11.plot(time, N, '-o',ms=3, drawstyle='steps-post')
            
            if hasattr(self, 'popt'):
                ax11.plot(time, self.langmuir_exp(time, *self.popt), 'r--', lw=2)

        if hasattr(self, 'Nt'):
            Nt = self.Nt
            time_Nt = self.time_Nt
            ax11.plot([0,time[-1]], [self.Neq,self.Neq], 'k--', lw =2)
            ax11.plot(time_Nt, Nt, 'k-', ms=3, lw=2)
        
        ax11.set_ylim((-.2, self.Nmax+0.9))
        ax11.set_ylabel('N(t)')



# NOT USED ##################################

def KMCresidencetimes(N, time, levs=(), kin=0, kout=0, debug=0):
    ''' residence times from KMCmaketrace() of levels. 
    eg levs = (1,2,..)
    kin, kout : the ones used in KMCmaketrace() '''
    global Nmax,S
    eps = 0.5
    dts = np.diff(time)
    #maxbin = 1./kin
    plt.figure(8722)
    for lev in levs:
        # indexes where N in lev:
        idxlev = np.nonzero(np.abs(N[:-1]-lev)<eps)[0]
        # signed residence times (>0 leaves up, <0 leaves dw):
        dwell = dts[idxlev] * np.sign(N[idxlev+1]-N[idxlev])
        dwellup = dwell[np.where(dwell>0)]
        dwelldw = -dwell[np.where(dwell<0)]
        dwelltot = np.abs(dwell)
        dwellnum = len(dwell)
        # histogram:
        binpos = np.linspace(0, np.max(dwelltot),100)
        binlen = np.diff(binpos)[0]
        hhup,bhup = np.histogram(dwellup, binpos, density=0)
        hhdw,bhdw = np.histogram(dwelldw, binpos, density=0)
        hhtot,bhtot = np.histogram(dwelltot, binpos, density=0)
        #binsign = np.linspace(-binmax,binmax,200)
        #hhsign,bhsign = np.histogram(dwell, binsign, density=0)
        # plots:
        plt.semilogy(bhdw[:-1], hhdw,'-v', label=str(lev)+'dw')
        plt.semilogy(bhup[:-1], hhup,'-^', label=str(lev)+'up')
        plt.semilogy(bhtot[:-1], hhtot,'-x', ms=9,label=str(lev)+'tot')
        plt.xlabel('Time')
        plt.ylabel('N.events')
        #plt.semilogy(bhsign[:-1], hhsign,'-', label=str(lev)+'tot')
        plt.legend(fontsize=8)
        if kin and kout:
            Kup = kin*S*(Nmax-lev)
            Kdw = kout*lev
            Ktot = Kup+Kdw
            print('\nKup = kin*S*(Nmax-N) = '+'{:.4f}'.format(Kup))
            print('Kdw = kout*N = '+'{:.4f}'.format(Kdw))
            print('Ktot = Kup+Kdw = '+'{:.4f}'.format(Ktot))
            plt.plot(binpos, dwellnum*binlen*Kup*np.exp(-Ktot*binpos), 'r-', lw=2)
            plt.plot(binpos, dwellnum*binlen*Kdw*np.exp(-Ktot*binpos), 'g-', lw=2)

        if debug:
            plt.figure(123)
            plt.subplot(221)
            plt.plot(N,'-o', drawstyle='steps-post')
            plt.subplot(222)
            plt.plot(t)

            plt.subplot(223)
            plt.plot(time[:-1], dwellup, 'rs')
            plt.plot(time[:-1], dts, 'bs')
            plt.plot(time[:-1], dwellup, 'r+', lw=3, ms=10)
            plt.plot(time[:-1], dwelldw, 'r_', lw=3, ms=10)



def timeDistributions(N,levs,plots=0):
    ''' find residence times in different levels'''
    global Nmax
    eps = 0.5
    #levs = (6,) #np.arange(0,Nmax)
    for lev in levs:
        # distance of N from lev:
        dist = 1*(np.abs(N-lev) < eps)
        difdist = np.diff(dist)
        # indexes where N enters and exits level lev:
        inlev = np.where(difdist == 1)[0]
        outlev = np.where(difdist == -1)[0]
        # correct indexes:
        if outlev[0] < inlev[0]:
            outlev = outlev[1:]
        inlev = inlev[:len(outlev)]
        # Dt spent in lev, >0 leaves for up, <0 leaves for dw::
        levDts = (outlev - inlev) * np.sign(N[outlev+1]-lev)
        # distribution of Dts spent in lev::
        hh,bh = np.histogram(levDts, np.linspace(-600,600,200), density=1)
        plt.plot(bh[:-1], hh,'-o')

        if plots:
            plt.plot(N-lev, '-o',ms=1)
            plt.plot(dist)
            #plt.plot(difdist)
            plt.plot(inlev, N[inlev]-lev, 'go')
            plt.plot(outlev, N[outlev]-lev, 'rs')
    return levDts

