

import numpy as np
import matplotlib.pylab as plt
import time
import ruptures as rpt



class BreakPoints():
    ''' find brake points in trace using ruptures '''


    def __init__(self, sig, plots=False):
        ''' sig: np.array of signal '''
        #self.sig = np.load('test_sig0.npy')
        #self.sig = np.load('test_sig1.npy')
        self.sig = sig
        print(f'BreakPoints.__init__(): found len(sig) = {len(self.sig)}')
        if plots:
            plt.figure('BreakPoints.__init__',clear=True)
            plt.plot(self.sig)



    def rpt_find_brkpts(self, segmentation_fn='BottomUp', cost_fn='l1', dws=None, pen=None, plots=True, clear=False):
        ''' find brake points
        segmentation_fn:         cost_fn:
        Pelt                     l1, l2, rbf 
        Binseg                   l1, l2, rbf, linear, normal, ar
        BottomUp                 l1, l2, rbf, linear, normal, ar
        '''
        self.cost_fn = cost_fn
        self.sig_dws = self.sig[::dws]
        if not pen:
            pen = np.log(len(self.sig_dws)*np.std(self.sig_dws)**2)
        # store:
        self.pen = pen
        self.dws = dws
        print(f'BreakPoints.rpt_find_brkpts(): pen : {pen:.2f}')
        print(f'BreakPoints.rpt_find_brkpts(): len(sig dws) : {len(self.sig_dws)}')
        # find break points:
        if segmentation_fn == 'Pelt':
            self.rpt_segm = rpt.Pelt
        elif segmentation_fn == 'Binseg':
            self.rpt_segm = rpt.Binseg
        elif segmentation_fn == 'BottomUp':
            self.rpt_segm = rpt.BottomUp
        else:
            raise Exception('segmentation_fn not defined')
        print(f'BreakPoints.rpt_find_brkpts(): Using {self.rpt_segm} with cost {self.cost_fn}. Working...')
        t0 = time.time()
        self.bkpts = self.rpt_segm(model=self.cost_fn, jump=1).fit_predict(self.sig_dws, pen=pen)
        # store break points:
        self.bkpts = np.append(0, self.bkpts)
        self.dt = time.time()-t0
        print(f'BreakPoints.rpt_find_brkpts(): Done. Time: {self.dt:.2f} sec')
        # store signal of discete levels between bkpts, downsampled:
        self.sig_dws_bkpts = []
        for i0,i1 in zip(self.bkpts[:-1], self.bkpts[1:]):
            self.sig_dws_bkpts = np.append(self.sig_dws_bkpts, np.repeat(np.median(self.sig_dws[i0:i1]), i1-i0))
        # store signal of discete levels between bkpts, original sampling:
        self.sig_bkpts = np.array([np.repeat(i,self.dws) for i in self.sig_dws_bkpts]).ravel()
        dsig = np.diff(self.sig_dws_bkpts) 
        # store jumps in sig:
        self.sig_jumps = dsig[dsig > 0]
        # plots:
        if plots:
            self.rpt_find_brkpts_plots(clear)



    def rpt_find_brkpts_plots(self, clear=False):
        ''' plot after rpt_find_brkpts() '''
        plt.figure('rpt_find_brkpts_plots', clear=clear)
        plt.vlines(self.bkpts, np.min(self.sig), np.max(self.sig), 'y', lw=1)
        plt.plot(np.arange(len(self.sig))[::10]/self.dws, self.sig[::10], label='orig')
        plt.plot(self.sig_dws, '.', ms=2, label='dwsampled')
        plt.plot(self.sig_dws_bkpts, '-k', lw=2)
        plt.title(f'segm_type:{self.rpt_segm}    cost_fn:{self.cost_fn}    pen:{self.pen:.1f}    npts:{len(self.sig_dws)}    dt:{self.dt:.1f} sec', fontsize=10)
        plt.legend()
        plt.xlabel('dws_idx')

    

    def auto_penalty(self, segmentation_fn='BottomUp', cost_fn='l1', dws=1500, pens=[None,]):
        ''' check results changing penalty automatically '''
        n_bkpts = []
        d_bkpts = {}
        for p in pens:
            self.rpt_find_brkpts(segmentation_fn=segmentation_fn, cost_fn=cost_fn, dws=dws, pen=p, clear=True)
            n_bkpts = np.append(n_bkpts, len(self.bkpts))
            d_bkpts[p] = self.bkpts
        # plots:
        fig = plt.figure('auto_penalty', clear=True)
        plt.subplot(211)
        for p in d_bkpts:
            plt.plot(np.repeat(p, len(d_bkpts[p])) , d_bkpts[p], 'k.', ms=2)
        plt.subplot(212)
        plt.plot(pens, n_bkpts, '-o', ms=2)
        plt.xlabel('penalty')




