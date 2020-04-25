# fitEllipse.py

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv
import multiprocessing as mulpr
import time


#TODO
def XYtoThetaSimple(x,y, plots=0):
    ''' move xy elliptical trajectory to (0,0), rescale to be circular, find angle of each x,y simply by atan'''
    # fit ellipse on xy :
    xx,yy,center0,a,b,phi = makeBestEllipse(x,y, nel=50)
    # rotate xy so major axis is vertical: (check if this is always the case :NO!)
    x_rot, y_rot = rotateArray(np.array((x,y)), -phi)
    # fit again an ellipse on the rotated xy:
    xx,yy,center,a,b,phi = makeBestEllipse(x_rot, y_rot, nel=50) 
    # translate to 0,0 and scale x to make a circle from the ellipse:
    if a>b:
        x_rot = (x_rot - center[0])
        y_rot = (y_rot - center[1])*a/b
    else:
        x_rot = (x_rot - center[0])*b/a
        y_rot = (y_rot - center[1])
    # fit again on the scaled circular x_rot y_rot data:
    xx,yy,center,a,b,phi = makeBestEllipse(x_rot, y_rot, nel=50) 
    # find angle of x,y on the circle:
    theta = np.unwrap(np.arctan2(y_rot, x_rot))
    if plots:
        plt.figure('XYtoThetaSimple')
        #plt.clf()
        plt.subplot(211)
        plt.plot(x,y,'.',ms=1)
        plt.plot(x_rot,y_rot, '.', ms=1)
        plt.plot(xx, yy, '-')
        plt.subplot(212)
        plt.plot(theta, '-o')
    return theta


def XYtoThetaSimple_DriftCorr(x,y, driftcorrPts=10, plots=0):
    ''' as XYtoThetaSimple plus drift correction by sliding window every driftcorrPts '''
    #for i in 



def XYtoThetaEllipse_parallel_rot_drift(x,y, N=100):
    ''' same as XYtoThetaEllipse_parallel, 
    + correction to make the xy ellipse a cirle (this reduces the periodc variations of speed in a cycle, wrt to the uncorrected script (checked 160315))
    + online drift correction, using N intervals along x,y
    return
        angle, x_ and y_rotated_drift-corrected 
        angle in DEG'''
    x_orig = x
    y_orig = y
    # correct drift: 
    x, y = correct_drift(x,y, nwin=N, plots=1)
    # fit ellipse on xy :
    xx,yy,center0,a,b,phi = makeBestEllipse(x,y, nel=50)
    # rotate xy so major axis is vertical: (check if this is always the case :NO!)
    x_rot, y_rot = rotateArray(np.array((x,y)), -phi)
    # fit again an ellipse on the rotated xy:
    xx,yy,center,a,b,phi = makeBestEllipse(x_rot, y_rot, nel=50) 
    # translate to 0,0 and scale x to make a circle from the ellipse:
    if a>b:
        x_rot = (x_rot - center[0])
        y_rot = (y_rot - center[1])*a/b
    else:
        x_rot = (x_rot - center[0])*b/a
        y_rot = (y_rot - center[1])
    # fit again on the scaled xy data:
    xx,yy,center,a,b,phi = makeBestEllipse(x_rot, y_rot, nel=50) 
    # find theta from the scaled xy data:
    theta_rot = minDistFromBestEllipse_parallel_manager(x_rot,y_rot, center,a,b,phi)
    # unwrap theta:
    theta_rot = np.unwrap(theta_rot)*180./np.pi
    # plots:
    plt.figure(68731)
    plt.clf()
    plt.subplot(221)
    plt.plot(x_orig, y_orig, '.', ms=0.1, label='orig.')
    plt.legend()
    plt.subplot(222)
    plt.plot(x-center0[0], y-center0[1],'g.',ms=0.5, label='drift corr.')
    plt.legend()
    plt.subplot(223)
    plt.plot(x_rot,y_rot,'b.',ms=0.5,label='scaled')
    plt.plot(xx,yy,'ro',ms=4, label='fit')
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.subplot(426)
    plt.plot(theta_rot/360.)
    plt.ylabel('Turns')
    plt.subplot(428)
    dth_f = run_win_smooth(np.diff(theta_rot), win=100)
    plt.plot(dth_f)
    plt.ylabel('Speed AU')
    return theta_rot, x_rot,y_rot 



def ellipse_FindRotateMoveto0Scale(x,y, idx1,idx2):
    ''' on x[idx1:idx2] and y[idx1:id2] find ellipse, 
    based on that, rotate all data vertical, move all data to origin, 
    and scale all data to a circle'''
    # fit ellipse on part of xy :
    xx,yy,center0,a,b,phi = makeBestEllipse(x[idx1:idx2],y[idx1:idx2], nel=50)
    #plt.figure(13527)
    #plt.clf()
    #plt.plot(x[idx1:idx2],y[idx1:idx2],'.',ms=1)
    # rotate xy so major axis is vertical: (check if this is always the case :NO!)
    x_rot, y_rot = rotateArray(np.array((x,y)), -phi)
    # fit again an ellipse on the rotated xy:
    xx,yy,center,a,b,phi = makeBestEllipse(x_rot[idx1:idx2], y_rot[idx1:idx2], nel=50) 
    # translate to 0,0 and scale x to make a circle from the ellipse:
    if a>b:
        x_rot = (x_rot - center[0])
        y_rot = (y_rot - center[1])*a/b
    else:
        x_rot = (x_rot - center[0])*b/a
        y_rot = (y_rot - center[1])
    # fit again on the scaled xy data:
    xx,yy,center,a,b,phi = makeBestEllipse(x_rot[idx1:idx2], y_rot[idx1:idx2], nel=50) 
    return x_rot,y_rot, xx,yy,center,a,b,phi




def XYtoThetaEllipse_parallel_rot(x,y):
    ''' same as XYtoThetaEllipse_parallel, with correction to make the xy ellipse a cirle,
    this reduces the periodc variations of speed in a cycle, 
    wrt to the uncorrected script (checked 160315)'''
    # fit ellipse on original xy data:
    xx,yy,center0,a,b,phi = makeBestEllipse(x,y, nel=50)
    # rotate xy so major axis is vertical: (check if this is always the case :NO!)
    x_rot, y_rot = rotateArray(np.array((x,y)), -phi)
    # fit again an ellipse on the rotated xy:
    xx,yy,center,a,b,phi = makeBestEllipse(x_rot, y_rot, nel=50) 
    # translate to 0,0 and scale x to make a circle from the ellipse:
    if a>b:
        x_rot = (x_rot - center[0])
        y_rot = (y_rot - center[1])*a/b
    else:
        x_rot = (x_rot - center[0])*b/a
        y_rot = (y_rot - center[1])
    # fit again on the scaled xy data:
    xx,yy,center,a,b,phi = makeBestEllipse(x_rot, y_rot, nel=50) 
    # plots:
    plt.figure(68731)
    plt.clf()
    plt.plot(x-center0[0], y-center0[1],'g.',ms=0.2, label='original')
    plt.plot(x_rot,y_rot,'.',ms=0.2)
    plt.plot(xx,yy,'ro',ms=4, label='final')
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    # find theta from the scaled xy data:
    theta_rot = minDistFromBestEllipse_parallel_manager(x_rot,y_rot, center,a,b,phi)
    # unwrap theta:
    theta_rot = np.unwrap(theta_rot)*180./np.pi
    
    return theta_rot, x_rot,y_rot 




def XYtoThetaEllipse_parallel(x,y):
    ''' Main function **PARALLEL computing**  
        theta = XYtoPhiEllipse_parallel(x,y)
    From input positions (x,y) gives the unwrapped angle phi(x,y)  
    in degrees, obtained from the best ellipse fit of input (x,y).
    return: 
         theta = unwrapped angle from ellipse fit (degrees)
    '''
    print(" fitEllipse.XYtoThetaEllipse_parallel(x,y)  working...")
    xx,yy,center,a,b,phi = makeBestEllipse(x,y, nel=50)
    theta = minDistFromBestEllipse_parallel_manager(x,y, center,a,b,phi)
    theta = np.unwrap(theta)*180/np.pi
    return theta 



def XYtoThetaEllipse(x,y):
    ''' Main function 
        xe,ye,theta,xr,yr = XYtoThetaEllipse(x,y)
    From input positions (x,y) gives the angle theta(x,y)  
    obtained from the best ellipse fit of input (x,y).
    return: 
         xe,ye = x- and y-ellipse pts, 
         theta = wrapped angle from ellipse fit 
         xr,yr = input pts translated and rotated'''
    print(" fitEllipse.XYtoThetaEllipse(x,y)  working...")
    xx,yy,center,a,b,phi = makeBestEllipse(x,y, nel=50)
    xe,ye,theta,xr,yr = minDistFromBestEllipse(x,y, center,a,b,phi)
    return xe,ye,theta,xr,yr 



def correct_drift(x,y, nwin=100, plots=0):
    ''' remove drift by fitting nwin ellipses on nwin segments of points x,y
    return x_corrected, y_corrected '''
    if plots:
        plt.figure(6871), plt.clf()
        plt.figure(646), plt.clf()
    lx = len(x)
    x_corr = np.zeros(lx)+2
    y_corr = np.zeros(lx)+2
    if lx != len(y):
        print('antidrift(): error len(x)!=len(y)')
        return 0, 0
    # windows to split x,y to fit ellipse
    win = np.floor(np.linspace(0,lx, num=nwin, endpoint=False))
    win = win.astype(int)
    dwin = np.diff(win)[1]
    for i in win:
        xx,yy,center0,a,b,phi = makeBestEllipse(x[i:i+dwin],y[i:i+dwin], nel=50)
        x_corr[i:i+dwin] = x[i:i+dwin] - center0[0]
        y_corr[i:i+dwin] = y[i:i+dwin] - center0[1]
        if plots:
            plt.figure(646)
            plt.subplot(211)
            pp, = plt.plot(range(i,i+dwin), x_corr[i:i+dwin], '-')
            plt.subplot(212)
            plt.plot(range(i,i+dwin), y_corr[i:i+dwin], '-',color=pp.get_color())
            plt.figure(6871)
            plt.subplot(211)
            pp, = plt.plot(xx,yy,'-')
            plt.plot(x[i:i+dwin], y[i:i+dwin], '.',ms=.2, color=pp.get_color())
            plt.axis('equal')
            plt.subplot(212)
            pp, = plt.plot(xx-center0[0],yy-center0[1],'-')
            plt.plot(x[i:i+dwin]-center0[0], y[i:i+dwin]-center0[1], '.',ms=.2, color=pp.get_color())
            plt.axis('equal')
    if plots:
        plt.figure(12121)
        plt.clf()
        plt.plot(x_corr, y_corr, 'o', ms=1)
        plt.axis('equal')
    return x_corr, y_corr




def fitEllipse(x,y):
    """Algorithm from Fitzgibbon et al 1996, Direct Least Squares Fitting of Ellipsees.  
    Formulated in terms of Langrangian multipliers, rewritten as a generalized eigenvalue problem. """
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a
    
def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))

def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(abs(up/down1))
    res2=np.sqrt(abs(up/down2))
    return np.array([res1, res2])

def makeBestEllipse(x,y, nel=100):
    ''' x,y : input position data 
    nel: number of pts in best ellipse
    returns: xx,yy,center,a,b,phi parameters of best ellipse'''
    a = fitEllipse(x,y)
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    a,b = ellipse_axis_length(a)
    epts = np.arange(0, 2*np.pi, 2*np.pi/nel)
    xx = center[0] + a*np.cos(epts)*np.cos(phi) - b*np.sin(epts)*np.sin(phi)
    yy = center[1] + a*np.cos(epts)*np.sin(phi) + b*np.sin(epts)*np.cos(phi)
    return xx,yy,center,a,b,phi


def toyMakeEllipse(xc,yc, a,b, phi ,noise ,N=100):
    ''' creates a dummy ellipse for testing '''
    t = np.linspace(0,2*np.pi, N)
    rax = np.random.rand(N) -0.5
    ray = np.random.rand(N) -0.5
    x = xc + a*np.cos(t)*np.cos(phi) - b*np.sin(t)*np.sin(phi)
    y = yc + a*np.cos(t)*np.sin(phi) + b*np.sin(t)*np.cos(phi)
    xn = noise*rax + x
    yn = noise*ray + y
    return x,y,xn,yn


def rotateArray(a,th):
    ''' rotates the array a of the angle theta (rad)'''
    R = np.array(([np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]))
    rota = np.dot(R,a)
    return rota


def f1(xe, xo,yo,a,b):
    ''' defines analitically diff(dist((xe,ye), (xo,yo))), where
    xe,ye: ellipse, xo,yo: one experimental point 
    to be used by optimize.brentq()
    Upper branch '''
    f1 = 2*(xe-xo)-(2*b*xe*(b*np.sqrt(1-(xe/a)**2)-yo))/(a**2*np.sqrt(1-(xe/a)**2))
    return f1

def f2(xe, xo,yo,a,b):
    ''' defines analitically diff(dist((xe,ye), (xo,yo))), where
    xe,ye: ellipse, xo,yo: one experimental point 
    to be used by optimize.brentq()
    Lower branch '''
    f2 = 2*(xe-xo)+(2*b*xe*(-b*np.sqrt(1-(xe/a)**2)-yo))/(a**2*np.sqrt(1-(xe/a)**2))
    return f2


def minDistFromBestEllipse(x,y, center,a,b,phi):
    ''' find zeros of derivative of distance pt-ellipse '''
    from scipy import optimize
    import progressbar 
    # translate exp pts to 0,0:
    x = x-center[0]
    y = y-center[1]
    # rotate x,y because ellipse is on axis:
    xr,yr = rotateArray(np.array((x,y)), -phi)
    # to plot ellipse on axis:
    #Xec,Yec = toyMakeEllipse(0,0, a,b, 0, 0)
    xd = []
    yd = []
    # progress bar:
    print(" fitEllipse.minDistFromBestEllipse() working...")
    widgets = [' ', progressbar.Percentage(), ' ', progressbar.Bar(marker='#',left='[',right=']'),' ', progressbar.ETA()] 
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=np.size(np.array((xr,yr))))
    pbar.start()
    pbarK = 0
    for xo,yo in np.nditer(np.array((xr,yr)), flags=['external_loop'], order='F'):
        # zero of f1 in interval (-a,a):
        fx1 = optimize.brentq(f1, -a,a, args=(xo,yo,a,b))  
        fy1 = b*np.sqrt(1-(fx1/a)**2)
        # dist exp.pt and point on ellipse:
        fd1 = np.sqrt((fx1-xo)**2 + (fy1-yo)**2)
        # zero of f2 in interval (-a,a):
        fx2 = optimize.brentq(f2, -a,a, args=(xo,yo,a,b)) 
        fy2 = -b*np.sqrt(1-(fx2/a)**2)
        # dist exp.pt and point on ellipse:
        fd2 = np.sqrt((fx2-xo)**2 + (fy2-yo)**2)
        if fd1 <= fd2:
            xd = np.append(xd, fx1)
            yd = np.append(yd, fy1)
        else:
            xd = np.append(xd, fx2)
            yd = np.append(yd, fy2)
        # angle:
        th = np.arctan2(yd, xd)
        if 0:
            plt.clf()
            plt.plot(Xec,Yec,'r-')
            plt.plot(xo,yo, 'ko',ms=10)
            plt.plot(xd[-1], yd[-1], 'rs')
            plt.axis('equal')
            plt.pause(0.01)
        pbarK = pbarK + 2
        pbar.update(pbarK) #this adds a little symbol at each iteration
    pbar.finish()
    return xd,yd,th,xr,yr


def minDistFromBestEllipse_parallel_manager(x,y,center,a,b,phi):
    ''' dispaches jobs in parallel to find angle '''
    time_start = time.time()
    # number of cores to use for processes 
    # (fastest with all cores on my office pc):
    cores_to_use = mulpr.cpu_count()
    # init queue, to be fitted with data:
    queue_out = mulpr.Queue()
    # parameter to split x,y in parts:
    split = len(x)/cores_to_use
    # define parallel processes:
    jobs = []
    for i in range(cores_to_use):
        # split x,y in parts:
        idx0, idx1 = int(i*split), int((i+1)*split)
        x_split = x[idx0: idx1]
        y_split = y[idx0: idx1]
        # def processes:
        proc = mulpr.Process(target = minDistFromBestEllipse_parallel, 
                args = ((i, x_split,y_split, idx0,idx1, queue_out, center,a,b,phi)))
        jobs.append(proc)
    # start every job:
    for jo in jobs:
        jo.start()
    # wait for queue_out to be populated:
    while queue_out.empty():
        time.sleep(0.1)
    # read the queue_out until all processes are ENDed:
    q_output = []
    q_finished = 0
    while q_finished != cores_to_use:
        qo = queue_out.get()
        # check if any process ENDed and take it into account:
        if qo[1] == 'END':
            q_finished = q_finished + 1
            #print('q_finished: '+str(q_finished))
        else:
            q_output.append(qo)
    # sort q_output and take only theta out:
    q_output.sort()
    theta = np.hstack([q_output[i][2] for i in range(cores_to_use)])
    print('Elapsed time: {0:.3f}'.format(time.time()-time_start)+' s')
    return theta



def minDistFromBestEllipse_parallel(proc_id, x,y,idx0,idx1, queue_out, center,a,b,phi):
    ''' find zeros of derivative of distance pt-ellipse 
    x,y: cropped input positions in [idx0:idx1]
    center,a,b,phi: ellipse parameters
    puts data in queue_out
    '''
    from scipy import optimize
    import progressbar
    # ignore divide by zero in the following brentq:
    np.seterr(divide='ignore')
    # translate exp pts to 0,0:
    x = x-center[0]
    y = y-center[1]
    # rotate x,y because ellipse is on axis:
    xr,yr = rotateArray(np.array((x,y)), -phi)
    # progress bar [######] :
    widgets = [' ', progressbar.Percentage(), ' ', \
            progressbar.Bar(marker='#',left='[',right=']'),' ', progressbar.ETA()] 
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=np.size(np.array((xr,yr))))
    pbar.start()
    # initialize points on fitted ellipse:
    xd = []
    yd = []
    idxfor = 0
    for xo,yo in np.nditer(np.array((xr,yr)), flags=['external_loop'], order='F'):
        # zero of f1 in interval (-a,a):
        fx1 = optimize.brentq(f1, -a,a, args=(xo,yo,a,b), xtol=1e-4, rtol=1e-5) #TEST added xtol, rtol  
        fy1 = b*np.sqrt(1-(fx1/a)**2)
        # dist exp.pt and point on ellipse:
        fd1 = np.sqrt((fx1-xo)**2 + (fy1-yo)**2)
        # zero of f2 in interval (-a,a):
        fx2 = optimize.brentq(f2, -a,a, args=(xo,yo,a,b), xtol=1e-4, rtol=1e-5) #TEST added xtol, rtol
        fy2 = -b*np.sqrt(1-(fx2/a)**2)
        # dist exp.pt and point on ellipse:
        fd2 = np.sqrt((fx2-xo)**2 + (fy2-yo)**2)
        if fd1 <= fd2:
            xd = np.append(xd, fx1)
            yd = np.append(yd, fy1)
        else:
            xd = np.append(xd, fx2)
            yd = np.append(yd, fy2)
        # angle:
        th = np.arctan2(yd, xd)
        idxfor = idxfor + 1; 
        # update progressbar:
        pbar.update(2*idxfor) 
    # put data in multiprocessing.queue:
    queue_out.put((idx0, idx1, th))
    # stop progressbar:
    pbar.finish()
    queue_out.put((proc_id, 'END'))



def run_win_smooth(data, win=10., fs=1., usemode='valid', plots=False):
    '''average running window filter, by convolution
    win = pts length of running window
    fs [= 1] sampling freq.
    usemode: same as in np.convolve()
    return: y smoothed data '''

    box = np.ones(win)/win
    y = np.convolve(data, box, mode=usemode)
    if plots:
        freq_orig, spectrum_orig = calculate_spectrum(data, fs)
        freq_filtered, spectrum_filtered = calculate_spectrum(y, fs)
        plt.figure()
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
