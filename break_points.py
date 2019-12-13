

import numpy as np
import matplotlib.pylab as plt
import time
import ruptures as rpt



class BreakPoints():
    ''' find brake points in trace using ruptures '''


    def __init__(self):
        self.sig = np.load('test_sig0.npy')
        #self.sig = np.load('test_sig1.npy')
        print(f'len(sig) : {len(self.sig)}')
        plt.figure('rpt_test0',clear=True)
        plt.plot(self.sig)



    def rpt_find_brkpts(self, segmentation_fn='Pelt', cost_fn='l1', dws=None, pen=None, clear=False):
        ''' find brake points
        segmentation_fn : Pelt, Binseg, BottomUp
        cost_fn : l1, l2, ...        
        '''
        self.cost_fn = cost_fn
        self.sig_dws = self.sig[::dws]
        if not pen:
            pen = np.log(len(self.sig_dws)*np.std(self.sig_dws)**2)
        # store:
        self.pen = pen
        self.dws = dws
        print(f'rpt_find_brkpts(): pen : {pen:.2f}')
        print(f'rpt_find_brkpts(): len(sig dws) : {len(self.sig_dws)}')
        # find break points:
        print(f'rpt_find_brkpts(): Working..')
        if segmentation_fn == 'Pelt':
            self.rpt_segm = rpt.Pelt
        elif segmentation_fn == 'Binseg':
            self.rpt_segm = rpt.Binseg
        elif segmentation_fn == 'BottomUp':
            self.rpt_segm = rpt.BottomUp
        else:
            raise Exception('segmentation_fn not defined')
        t0 = time.time()
        self.bkpts = self.rpt_segm(model=self.cost_fn, jump=1).fit_predict(self.sig_dws, pen=pen)
        self.bkpts = np.append(0, self.bkpts)
        self.dt = time.time()-t0
        print(f'rpt_find_brkpts(): Done. Time: {self.dt:.2f} sec')
        # signal of discete levels between bkpts, downsampled:
        self.sig_dws_bkpts = []
        for i0,i1 in zip(self.bkpts[:-1], self.bkpts[1:]):
            self.sig_dws_bkpts = np.append(self.sig_dws_bkpts, np.repeat(np.median(self.sig_dws[i0:i1]), i1-i0))
        # signal of discete levels between bkpts, original sampling:
        self.sig_bkpts = np.array([np.repeat(i,self.dws) for i in self.sig_dws_bkpts]).ravel()
        # plots:
        self.rpt_find_brkpts_plots(clear)



    def rpt_find_brkpts_plots(self, clear=False):
        ''' plot after rpt_find_brkpts() '''
        plt.figure('rpt_test', clear=clear)
        plt.vlines(self.bkpts, np.min(self.sig), np.max(self.sig), 'y', lw=2)
        plt.plot(np.arange(len(self.sig))/self.dws, self.sig, label='orig')
        plt.plot(self.sig_dws, '.', ms=2, label='dwsampled')
        plt.plot(self.sig_dws_bkpts, '-k')
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




