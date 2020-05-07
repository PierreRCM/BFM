# tethered cell analysis, video from tdms file
# francesco Nov 2015
# TODO make it Class


import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import skimage
from scipy import ndimage
from skimage import filters, measure
from scipy.interpolate import interp1d
import scipy
import progressbar
import multiprocessing as mulpr
import time

import openTDMS



px2m = 147e-9 # meter/pixel
mouse_x = []
mouse_y = []
mouse_button = []



def main(filename, saves=False, Nmeas=500, plotMeas=False, negateImgs=False, returnImgs=False):
    ''' 
    Main function for tracking tethered cells.
    
    Returns
        angle (array) [degrees], drag_coefficient [N m s]
        or
        angle (array), drag_coefficient, all_images (for debug)

    Arguments:
        - filename = string of .tdms file to track
        - saves [0|1] = saves a .npy file with [{'angle[deg]':angle, 'drag[Nms]':gamma}]
        - Nmeas = n. of images used to measure cell size
        - plotMeasured [0|1] : plot individual images used to measure the cell size 
        - negateImgs [0|1] : to take the negative of the images 
        - returnImgs [0|1] : to return all the images of the movie
    
    Examples: 
    1) normal call to have angle and drag, and to save them:
        angle, drag = tethered.main('CL_151116_180257.tdms', saves=1, Nmeas=1000, plotMeas=0, returnImgs=0)

    then you can use 'angle' in 
        analysis_rotation.switching_analysis(angle, FPS, ...)
    
    2) special call to return also all frames (imgs) to see the movie:
        angle, drag, imgs = tethered.main('CL_151116_180257.tdms', saves=0, Nmeas=100, plotMeas=0, returnImgs=1)

    followed by the movie in index 3000-3100 :
        tethered_1.checkTrackedAngle(imgs, 3000, 3100, theta=a)

    '''
    print('\nmain : Opening file '+filename+' ...')
    imgs = openTdmsOneROI(filename)
    if negateImgs: imgs = -imgs
    print('main : Crop the wanted region ...')
    imgs = cropRoi(imgs)
    print('main : removing average image ...')
    imgs = rmMovieMean(imgs)
    #plt.figure(5399)
    #plt.clf()
    #plt.plot(np.sum(np.sum(imgs),1),0)
    #print('main : check max and min images ...')
    #happy = showImMaxMin_signal(imgs)
    #if not happy: 
    #    print('Bye')
    #    return 0,0 
    print('main : removing mean from each frame ...')
    imgs = rmFrameMean(imgs) 
    print('main : find center of rotation ...')
    xc,yc = pickStdCenter(imgs)
    print('main : measuring cells ...')
    cellWidth, dcellWidth, axLong, daxLong, axShort, daxShort = measureNCells(imgs, xc,yc, Nimgs=Nmeas, plots=plotMeas)
    # find angle [deg]:
    print('\nmain : tracking the angle ...')
    angle = findAngle_parallel(imgs)
    # calculate drag coeff. gamma [N m s]:
    print('\nmain : calculating drag coefficient...')
    gamma = calcGamma(cellWidth, dcellWidth, axLong, daxLong, axShort,daxShort, xc,yc)
    if saves:
        filesave = filename.rsplit('.tdms')[0]
        extention = raw_input('\nAdd if you want an extension EXT (e.g roi1) to file to save ('+filesave+'_EXT_tether-trkd.npy) ... :')
        np.save(filesave +'_'+ extention + '_tether-trkd', [{'angle[deg]':angle, 'drag[Nms]':gamma}] )
    # return different things (for debugging):
    if returnImgs:
        return angle, gamma, imgs
    else:
        return angle, gamma




def openTdmsOneROI(filename):
    ''' open a movie in tdms file "filename", 
    using the config in "/CL-config/#X" (which must be in the .tdms file)
    returns one single image(t) with all the ROIs 
    '''
    d = openTDMS.openTdmsFile(filename, print_found=False)
    # find camera configuration string:
    if d.has_key('/CL-config/#X'):
        CLconfig = d['/CL-config/#X'][0]
    else:
        print('        openTdmsOneROI : ERROR, configuration not found')
    # find frame size (tuple) :  
    re_framesize = re.search(r'Frame Size : (\d+),(\d+)', CLconfig)
    if re_framesize: 
        framesize = (int(re_framesize.groups()[0]), int(re_framesize.groups()[1]))
        print('        Frame size = '+str(framesize))
    else: 
        print('        openTdmsOneROI : ERROR framesize')
    # find number of ROIs (int):
    re_nrois = re.search(r'Number of ROI : ([1-4])', CLconfig)
    nrois = int(re_nrois.groups()[0])
    print('        Num. ROIs = ' + str(nrois))
    # find all images and reshape to 2D:
    if d.has_key('/CLImg/ROIs'):
        imgs = np.array(d['/CLImg/ROIs'])
        imgs = np.reshape(imgs, (len(imgs)/(framesize[0]*framesize[1]*nrois), framesize[1]*nrois, framesize[0]))
        print('        Num. of frames = '+str(imgs.shape[0]))
    else:
        print('        openTdmsOneROI : ERROR no images found')
    return imgs



def pickPx(print_out=0): 
    ''' pick up one pixel with the mouse '''
    coord = plt.ginput(1)
    # floating:
    xflo = coord[0][0]
    yflo = coord[0][1]
    # integers:
    xint = int(round(coord[0][0]))
    yint = int(round(coord[0][1]))
    if print_out == 1:
        print('chosen: x={0} y={1}'.format(xx,yy))
        sys.stdout.flush()
    return xint,yint, xflo,yflo



def rmMovieMean(imgs):
    ''' remove from the movie roi[t,x,y] the mean[x,y] image of the movie '''
    return imgs - np.mean(imgs, 0)



def rmFrameMean(imgs):
    ''' from the movie roi[t,x,y] removes from each frame[x,y] its mean '''
    imgs_c = np.zeros(imgs.shape)
    # mean of each [x,y] frame:
    frames_avg = np.mean(np.mean(imgs,1),1)
    # remove from each frame its mean:
    for i in range(imgs.shape[0]):
        imgs_c[i,:,:] = imgs[i,:,:] - frames_avg[i]
    return imgs_c



def showImMaxMin_signal(imgs):
    ''' show the image max of the frames 
    ask to continue or not , and return signal'''
    global mouse_button
    immax = np.max(imgs[::10,:,:],0)
    immin = np.min(imgs[::10,:,:],0)
    fig = plt.gcf()
    plt.subplot(121)
    plt.imshow(immax)
    plt.title('max')
    plt.subplot(122)
    plt.imshow(immin)
    plt.title('min')
    plt.suptitle('Accept?    Yes:Left   No:Right', fontsize=16)
    # start listening to mouse events:
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # wait for mouse click:
    plt.waitforbuttonpress(timeout=-1)
    if mouse_button == 1: 
        happy = True
    else: 
        happy = False;
    # stop listening to mouse events:
    fig.canvas.mpl_disconnect(cid)
    return happy


    
def onclick(event):
    ''' stores button, x, y of the mouse event to global variable '''
    global mouse_button, mouse_x, mouse_y
    mouse_button = event.button
    mouse_x = event.xdata
    mouse_y = event.ydata



def normalize(a):
    ''' normalize a in [0,1]. Return normalized a '''
    maxa = np.max(a)
    mina = np.min(a)
    an = (a-mina)/(maxa-mina)
    return an



def cropRoi(imgs):
    ''' plot the movie stdev, click on center and edge, crop movie accordingly '''
    global mouse_button
    # stop warning from matplotlib about ginput:
    import warnings
    warnings.filterwarnings("ignore")
    # to control while loop:
    happy = False
    # movie st.dev (part of):
    imgs_std = np.std(imgs[::30, :,:], 0)
    imgs_max = np.max(imgs[::30, :,:], 0)
    imgs_min = np.min(imgs[::30, :,:], 0)
    while not happy:
        # plot stdev:
        fig = plt.figure(555, figsize=(10,5))
        plt.clf()
        plt.subplot(131)
        plt.imshow(imgs_std, interpolation='none')
        plt.title('st.dev.')
        plt.subplot(132)
        plt.imshow(imgs_max, interpolation='none')
        plt.title('max.')
        plt.subplot(133)
        plt.imshow(imgs_min, interpolation='none')
        plt.title('min.')
        # pick up by mouse the center of rotation and cell edge :
        plt.suptitle('Click on the center of rotation..', fontsize=16)
        xcnt, ycnt, foo, foo = pickPx(print_out=0)
        plt.suptitle('Click on the cell edge..', fontsize=16)
        sys.stdout.flush() # force print output to be displayed
        xedg, yedg, foo, foo = pickPx(print_out=0)
        # approx cell length to use for cropping:
        cnt_edg = np.ceil(np.sqrt((xcnt-xedg)**2+(ycnt-yedg)**2)) 
        # crop imgs_std and imgs:
        imgs_std_crop = imgs_std[ycnt-cnt_edg:ycnt+cnt_edg, xcnt-cnt_edg:xcnt+cnt_edg]
        imgs_crop = imgs[:, ycnt-cnt_edg:ycnt+cnt_edg, xcnt-cnt_edg:xcnt+cnt_edg]
        # plot cropped stdev image:        
        plt.clf()
        plt.imshow(imgs_std_crop, interpolation='none')
        plt.suptitle('Accept?   YES:left    NO:right', fontsize=16)
        # start listening to mouse events:
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        # wait for mouse click:
        plt.waitforbuttonpress(timeout=-1)
        if mouse_button == 1: happy = True
        else: happy = False; print('    again..')        
    plt.clf()
    # stop listening to mouse events:
    fig.canvas.mpl_disconnect(cid)
    
    return imgs_crop



def pickStdCenter(imgs):
    '''pick the center of rotation by std(img,0) 
    return xcenter, ycenter, center-edge distance '''
    global mouse_button
    mouse_button = 0
    #imgs_std = np.std(imgs, 0) 
    imgs_std = np.mean(abs(imgs),0) 
    # threshold the std image:
    #thr = (np.mean(imgs_std) + np.max(imgs_std))/2.
    thr = np.max(imgs_std)*0.8
    # fit a circle to find the center:
    xc0, yc0 = findCircleImg(imgs_std, filt=thr)
    fig = plt.gcf()
    # start listening to mouse events:
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    while mouse_button != 1:
        plt.clf()
        plt.imshow(imgs_std)
        plt.plot(yc0,xc0, 'w+', ms=256,mew=3)
        plt.plot(yc0,xc0, 'k+', ms=252,mew=1)
        plt.axis('image')
        plt.title('Accept center?  YES:left   NO:right')
        # wait for mouse click:
        plt.waitforbuttonpress(timeout = -1)
        # if not accepted:
        if mouse_button != 1: 
            plt.title('Click on center of rotation..')
            foo, foo, yc0, xc0 = pickPx()
            mouse_button = 0        
    # stop listening to mouse events:
    fig.canvas.mpl_disconnect(cid)
    # click to get the edge of the cell:
    #plt.title('Click on edge of cell..')
    #foo,foo, xedge, yedge = pickPx()
    # rotation center - edge distance:
    #ce_dist = np.ceil(np.sqrt((xc0-xedge)**2+(yc0-yedge)**2))
    return xc0, yc0



def findCircleImg(img, filt=10, plots=False):
    ''' fit a circle (xc,yc,radius) in img, 
    filt = threshold to img
    return xc,yc '''
    img_filtered = img>filt
    #img_canny = skimage.filter.canny(img, sigma=canny_sigma)
    coords = np.column_stack(np.nonzero(img_filtered))
    model, inliers = skimage.measure.ransac(coords, measure.CircleModel, min_samples=3, residual_threshold=1, max_trials=500)
    xcenter, ycenter, radius = model.params[0], model.params[1], model.params[2]
    if plots:
        plt.figure(2154)
        plt.subplot(121)
        plt.imshow(img)
        plt.plot(yc, xc, 'o')
        plt.subplot(122)
        plt.imshow(img_filtered)
        plt.plot(yc, xc, 'wo')
    return xcenter, ycenter



def modeOfFloat(arr, nbins=50):
    ''' finds the mode of an array arr (forced to be flat 1D) of floating pts, using its histogram'''
    histo, bins = np.histogram(arr, nbins)
    return bins[np.argmax(histo)]



def interpolateArray(arr, interpPts=500, interpMode='linear'):
    ''' interpolate array '''    
    from scipy.interpolate import interp1d
    X = np.arange(0, len(arr))
    Y = interp1d(X, arr, kind=interpMode)
    x_interp = np.linspace(0, len(X)-1, interpPts)    # x of interpolated
    y_interp = Y(x_interp)                            # y of interpolated
    return x_interp, y_interp



def autoCellMeasurements(img, xc, yc, plots=1):
    '''find cell length and cell width in input image img
    with center of rotation xc,yc
    retun cellLength, cellWidth, cellAxLong, cellAxShort '''
    img_n = normalize(img)
    # gaussian filter:
    img_n = normalize(skimage.filters.gaussian_filter(img_n, sigma=1))
    # find mode:
    imgMode = modeOfFloat(img_n)
    # threshold normalized image:
    img_nt = 1.*(img_n < imgMode*0.8)
    # find angle to rotate horizontally:
    angleCell = findAngleByMoments(img_n)
    # rotate image:
    img_nr  = ndimage.rotate(img_n, angleCell*180/np.pi, reshape=False)
    img_ntr = ndimage.rotate(img_nt, angleCell*180/np.pi, reshape=False)
    # x y profiles:
    sum0_img = np.sum(img_ntr, 0)
    sum1_img = np.sum(img_ntr, 1)
    sum0_img = normalize(sum0_img)
    sum1_img = normalize(sum1_img)
    # interpolate profiles:
    sum0_x, sum0_y = interpolateArray(sum0_img, interpPts=500, interpMode='cubic')
    sum1_x, sum1_y = interpolateArray(sum1_img, interpPts=500, interpMode='cubic')
    # find where center of rotation xc,yc is rotated to, cz = delta-image:
    cz = np.zeros(np.shape(img))
    cz[xc,yc] = 1
    # rotated delta-image of center of rot. :
    rcz = ndimage.rotate(cz, angleCell*180/np.pi, reshape=False)
    xcr, ycr = np.unravel_index(np.argmax(rcz), np.shape(rcz))
    # cell length and width from x,y profiles, threshold "thl":
    thl = 0.5
    up0 = np.argwhere(sum0_y > thl)
    up1 = np.argwhere(sum1_y > thl)
    cellLength = np.max(sum0_x[up0]) - np.min(sum0_x[up0])
    cellWidth  = np.max(sum1_x[up1]) - np.min(sum1_x[up1])
    # Try to orrect systematic over measure of length and width due to thres:l
    cellLength = cellLength - 2
    cellWidth = cellWidth - 2
    # find long and short axis from center of rotation:
    cellAxis0 = np.max(sum0_x[up0]) - ycr
    cellAxis1 = ycr - np.min(sum0_x[up0])
    cellAxLo = np.max((cellAxis0, cellAxis1))
    cellAxSh = np.min((cellAxis0, cellAxis1))
    # send error if long short axis are negative:
    if cellAxLo <0 or cellAxSh < 0 or cellLength < cellWidth:
        #cellLength = cellWidth = cellAxLo = cellAxSh = 0
        error = True
    else: 
        error = False
    if plots:
        plt.figure(8546)
        plt.clf() 
        plt.subplot(221)
        if error:
            plt.title('!!--ERROR-- !!', fontsize=22)
            plt.pause(0.5)
        plt.imshow(img_nr)
        plt.plot(ycr, xcr, 'go')
        plt.axis('image')
        plt.subplot(222)
        plt.imshow(img)
        plt.plot(yc, xc, 'gs')
        plt.axis('image')
        plt.subplot(223)
        plt.plot(sum0_img, '-', lw=2)
        plt.plot(sum0_x[up0], sum0_y[up0],'o')
        plt.subplot(224)
        plt.plot(sum1_img, '-', lw=2)
        plt.plot(sum1_x[up1], sum1_y[up1],'o')
        plt.pause(0.01)
    return cellLength, cellWidth, cellAxLo, cellAxSh, error




def measureNCells(imgs, xc,yc, Nimgs=10, plots=False):
    '''open Nimgs random images, measure their cells '''
    idx = np.random.random_integers(low=0, high=np.shape(imgs)[0]-1, size=Nimgs)
    cle = np.array(())
    cwi = np.array(())
    loax = np.array(())
    shax = np.array(())
    pbar = progressbar.ProgressBar(maxval=len(idx))
    pbar.start()
    k = 0
    for i in idx:
        pbar.update(k)
        a = imgs[i] 
        # automatically finds the cell dimensions:
        Len,Wid,AxLo,AxSh, error = autoCellMeasurements(a, xc, yc, plots=plots)
        if error == False:
            cle = np.append(cle, Len)
            cwi = np.append(cwi, Wid)
            loax = np.append(loax, AxLo)
            shax = np.append(shax, AxSh)
        else:
            print('  measureNCells() ERROR, image n.'+str(i)+' skipped')
        k = k+1
    pbar.finish()
    cellLength = np.mean(cle)		# cell length
    dcellLength = np.std(cle)		# cell length st.dev.
    cellWidth = np.mean(cwi)		# cell width
    dcellWidth = np.std(cwi)		# cell width st.dev.
    cellLongAx = np.mean(loax)		# cell long axis
    dcellLongAx = np.std(loax)		# cell long axis st.dev.
    cellShortAx = np.mean(shax)		# cell short axis
    dcellShortAx = np.std(shax)		# cell short axis st.dev.
    # plots:
    if 1:
        plt.figure(2, figsize=(9,5))
        plt.clf()
        plt.subplot(121)
        plt.hist(cle)
        plt.xlabel('cell length (Pixels)')
        plt.ylabel('Occurrences')
        plt.title('Cell measurements')
        plt.subplot(122)
        plt.hist(cwi)
        plt.xlabel('cell width (Pixels)')
        plt.ylabel('Occurrences')
        plt.show()
        plt.pause(0.2)
    return cellWidth, dcellWidth, cellLongAx, dcellLongAx, cellShortAx, dcellShortAx


def findAngleByMoments(a):
    '''find orientation (in rad) in one image "a" '''
    a = normalize(a)
    # threshold image:
    mode = modeOfFloat(a)
    th = mode + 0.5*(np.max(a) - mode) 
    a = 1.*(a>th)
    # find image moments and orientation: 
    mom = skimage.measure.moments(a)
    momcen = skimage.measure.moments_central(a, mom[0,1]/mom[0,0], mom[1,0]/mom[0,0])
    momcen = momcen/momcen[0,0]
    orientation = 0.5*np.arctan2(2*momcen[1,1], (momcen[2,0]-momcen[0,2]))
    return orientation


def findAngleAllFrames(imgs):
    ''' serial computing 
    find the orientations (in deg) of the cell in all frames of imgs''' 
    # to debug, consider first Nimgs images in imgs:
    Nimgs = imgs.shape[0]
    # init:
    orientation = np.zeros(Nimgs)
    # progress bar:
    pbar = progressbar.ProgressBar(maxval=Nimgs)
    pbar.start()
    # open each image:
    for i in range(Nimgs):
        pbar.update(i)
        orientation[i] = findAngleByMoments(imgs[i,:,:])
    # unwrap jumps of pi (hacked by factor 2) and to deg:
    orientations = np.unwrap(2*orientation)/2. *180/np.pi
    return orientations


def findAngle_parallel(imgs):
    ''' parallel computing
    find the orientations (in deg) of the cell in all frames of imgs''' 
    # number of cores to use for processes:
    cores_to_use = mulpr.cpu_count()
    # open pool:
    pool = mulpr.Pool(processes=cores_to_use)
    # to debug, consider first Nimgs images in imgs:
    Nimgs = float(imgs.shape[0])
    # parameter to split x,y in parts:
    split = Nimgs/cores_to_use
    results = []
    # send splitted imgs to different cores:
    for i in range(cores_to_use):
        # split imgs in parts:
        idx0, idx1 = i*split, (i+1)*split
        imgs_split = imgs[idx0:idx1, :,:]
        results.append(pool.apply_async(findAngleAllFrames, args=(imgs_split,)))
    # recollect all the results from cores:
    output = np.array([p.get() for p in results])
    # stich together the results: 
    for i in range(1, len(output)):
        output[i] = output[i] + output[i-1][-1]
    angle = np.array([item for sublist in output for item in sublist])
    return angle

    

def checkTrackedAngle(imgs, idx0, idx1, theta=[]):
    ''' show a movie with the tracked angle in the frames of imgs[idx0:idx1] 
    theta: angle already tracked (if any) '''
    xc,yc = pickStdCenter(imgs)
    anglearr = np.array([])
    if np.any(theta):
        plt.plot(range(idx0,idx1), theta[idx0:idx1]-theta[idx0], '-o', ms=3)
        plt.title('angle')
    for i in range(idx0, idx1+1):
        angle = findAngleByMoments(imgs[i])
        anglearr = np.append(anglearr, angle)
        plt.figure(3284)
        plt.subplot(121)
        plt.plot(i, (anglearr[-1]-anglearr[0])*180/np.pi, 'ro')
        #plt.xlim(0, idx1-idx0)
        plt.subplot(122)
        plt.cla()
        plt.imshow(imgs[i])
        plt.plot(yc,xc, 'wo', ms=10)
        plt.plot([yc-10.*np.cos(angle),yc+10.*np.cos(angle)], [xc-10.*np.sin(angle), xc+10.*np.sin(angle)],'w-',lw=5)
        plt.plot([yc-10.*np.cos(angle),yc+10.*np.cos(angle)], [xc-10.*np.sin(angle), xc+10.*np.sin(angle)],'k-',lw=2)
        plt.title(i)
        plt.axis('image')
        plt.pause(0.1)




def calcGamma(cellWidth, dcellWidth, axLong, daxLong, axShort, daxShort, xc, yc):
    ''' calculates the analytical expressions for gamma and its error [n m s]
    from: 'Suppressor Analysis of the MotB' 2008 '''
    global px2m
    # viscosity [Ns/m**2]:
    eta = 9.6e-4 
    # long L, short S axis and errors [m]:
    aL = axLong/2.*px2m
    aS = axShort/2.*px2m
    daL = daxLong/2.*px2m
    daS = daxShort/2.*px2m
    # half cell width and error [m]:
    b = cellWidth/2.*px2m 
    db = dcellWidth/2.*px2m
    # calculates the drag coeff [Nms] (with 2 ellipsoids, 
    # from: 'Suppressor Analysis of the MotB' 2008)
    # bigger ellipsoid :
    fl = (8*np.pi*eta*aL**3/3.)/(2*(np.log(2.*aL/b)-0.5))
    # smaller ellipsoid (problem when 2*aS/b < exp(0.5) ):
    if np.log(2.*aS/b)-0.5 > 0.001: 
        fs = (8*np.pi*eta*aS**3/3.)/(2*(np.log(2.*aS/b)-0.5)) # ok
    else:
        # radius of a sphere equivalent to small ellipsoid:
        eq_bead_rad = (b + aS)/2.
        # drag of a sphere instead of an ellipsoid:
        fs = 14.*np.pi*eta*(eq_bead_rad)**3
#    print '\n\n'
#    print('aL = '+str(aL))
#    print('aS = '+str(aS))
#    print('fl = '+str(fl))
#    print('fs = '+str(fs))
#    print('cellWidth = '+str(cellWidth))
#    print('b = '+str(b))
#    print np.log(2*aS/b)-0.5
    # drag [N m s]:
    gamma = fl+fs
    # partial derivatives:
    Dfl_aL = (4*np.pi*aL**2*eta)/(np.log(2*aL/b)-0.5) - (4*np.pi*aL**2*eta)/(3*(np.log(2*aL/b)-0.5)**2) 
    Dfs_aS = (4*np.pi*aS**2*eta)/(np.log(2*aS/b)-0.5) - (4*np.pi*aS**2*eta)/(3*(np.log(2*aS/b)-0.5)**2) 
    Dfl_b = (4*np.pi*aL**3*eta)/(3*b*(np.log(2*aL/b)-0.5)**2)
    Dfs_b = (4*np.pi*aS**3*eta)/(3*b*(np.log(2*aS/b)-0.5)**2)
    # propagated errors (st.dev.):
    dfl	= np.sqrt((Dfl_aL)**2*daL**2 + (Dfl_b)**2*db**2)
    dfs	= np.sqrt((Dfs_aS)**2*daS**2 + (Dfs_b)**2*db**2)
    dgamma = np.sqrt(dfl**2 + dfs**2)
    # cell width in um:
    W_um = b*2.*1e6
    dW_um = db*2.*1e6
    # cell length in um:
    L_um = (aL + aS)*2*1e6
    dL_um = (daL + daS)*2.*1e6
    # drag in pN nm s:
    gamma_pNnms = gamma*1e21
    dgamma_pNnms = dgamma*1e21
    print('\n    Cell Length (+- st.dev.) = {0:.1f}'.format(L_um)+' +- {0:.1f}'.format(dL_um)+' (um)')
    print('    Cell Width (+- st.dev.) = {0:.1f}'.format(W_um)+' +- {0:.1f}'.format(dW_um)+' (um)')
    print('    Long/Short cell axis = {0:.1f} / {1:.1f}'.format(aL*2.*1e6, aS*2.*1e6)+' (um)')
    #print(' > Cell rotation frequency = {0:.1f}'.format(peakFreq)+' Hz')
    print('    Drag = {0:.2f}'.format(gamma_pNnms) + ' +- {0:.2f}'.format(dgamma_pNnms) + '(pN nm s)')
    #print(' > Torque = {0:.2f}'.format(torque*1e21) + ' +- {0:.2f}'.format(dtorque*1e21) + ' pN*nm')
    return gamma		





##################################################################

def openTdmsROIs(filename):
    ''' open a movie in tdms file "filename", 
    using the config in "/CL-config/#X" (which must be in the .tdms file)
    returns the four ROIs, empty if not existing
    
    tip: 
    for example
        imshow(sum(abs(roi0),0), interpolation='none') 
    shows the center of rotation
    '''
    d = openTDMS.openTdmsFile(filename, print_found=True)
    # find camera configuration string:
    if d.has_key('/CL-config/#X'):
        CLconfig = d['/CL-config/#X'][0]
    else:
        print('openTdmsROIs : ERROR, configuration not found')
    # find frame size (tuple) :  
    re_framesize = re.search(r'Frame Size : (\d+),(\d+)', CLconfig)
    if re_framesize: 
        framesize = (int(re_framesize.groups()[0]), int(re_framesize.groups()[1]))
        print('Frame size = '+str(framesize))
    else: 
        print('openTdmsROIs : ERROR framesize')
    # find number of ROIs (int):
    re_nrois = re.search(r'Number of ROI : ([1-4])', CLconfig)
    nrois = int(re_nrois.groups()[0])
    print('Num. ROIs = ' + str(nrois))
    # find all mages:
    if d.has_key('/CLImg/ROIs'):
        imgs = np.array(d['/CLImg/ROIs'])
        imgs = np.reshape(imgs, (len(imgs)/(framesize[0]*framesize[1]),framesize[1],framesize[0]))
    else:
        print('openTdmsROIs : ERROR no images found')
    # dispatch ROIs:
    if nrois == 1:
        roi0 = imgs
        roi1 = np.array([0])
        roi2 = np.array([0])
        roi3 = np.array([0])
    if nrois == 2:
        roi0 = imgs[0::2,:,:]
        roi1 = imgs[1::2,:,:]
        roi2 = np.array([0])
        roi3 = np.array([0])
    if nrois == 3:
        roi0 = imgs[0::3,:,:]
        roi1 = imgs[1::3,:,:]
        roi2 = imgs[2::3,:,:]
        roi3 = np.array([0])
    if nrois == 4:
        roi0 = imgs[0::4,:,:]
        roi1 = imgs[1::4,:,:]
        roi2 = imgs[2::4,:,:]
        roi3 = imgs[3::4,:,:]
    return roi0, roi1, roi2, roi3



#main('CL_151116_180257.tdms', Nmeas=300, plotMeas=0, negateImgs=0)
