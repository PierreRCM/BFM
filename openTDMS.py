import matplotlib.pyplot as plt
import numpy as np
import re
import nptdms


def openTdmsFile(filename, print_found=True):
    ''' open a tdms file and put all its structure into a dictionary "d".
    to access the data of one channel, e.g.:
    plot(d[d.keys()[3]], '.')
    
    example of .tdms file porganization:
     -TDMS file
     --group1
     ---group1/channel1
     ---group1/channel2
     --group2
     ---group2/channel1
     ---group2/channel2
     ---group2/channel3
    '''

    # open tdms file:
    f = nptdms.TdmsFile(filename)
    # all the groups in file:
    fg = f.groups()
    # empty dict, will contain all data
    d = {}
    # for each group:
    for grp in fg:
        # channels in grp:
        fc = f.group_channels(grp)
        # for each channel:
        for chn in fc:
            if chn.has_data:
                chname = chn.path.replace('\'','')
                d[chname] = chn.data
                if print_found:
                    print('Found: '+str(chname) )
    return d


def openTdmsOneROI(filename, prints=False):
    ''' open a movie in tdms file "filename", 
    using the config in "/CL-config/#X" (which must be in the .tdms file)
    returns one single image(t) with all the ROIs 
    '''
    d = openTdmsFile(filename, print_found=prints)
    # find camera configuration string:
    if '/CL-config/#X' in d:
        CLconfig = d['/CL-config/#X'][0]
    else:
        print('        openTdmsOneROI : ERROR, configuration not found')
    if prints:
        print(CLconfig)
    # find frame size (tuple) :  
    re_framesize = re.search(r'Frame Size : (\d+),(\d+)', CLconfig)
    if re_framesize:
        framesize = (int(re_framesize.groups()[0]), int(re_framesize.groups()[1]))
        if prints: print('        Frame size = '+str(framesize))
    else:
        print('        openTdmsOneROI : ERROR framesize')
    # find number of ROIs (int):
    re_nrois = re.search(r'Number of ROI : ([1-4])', CLconfig)
    nrois = int(re_nrois.groups()[0])
    if prints: print('        Num. ROIs = ' + str(nrois))
    # find all images and reshape to 2D:
    if '/CLImg/ROIs' in d:
        imgs = np.array(d['/CLImg/ROIs'])
        imgs = np.reshape(imgs, (int(len(imgs)/(framesize[0]*framesize[1]*nrois)), framesize[1]*nrois, framesize[0]))
        if prints: print('        Num. of frames = '+str(imgs.shape[0]))
    else:
        print('        openTdmsOneROI : ERROR no images found')
    return imgs



def show_movie(imgs,n=1):
    ''' show a simple movie of imgs (imgs from openTdmsOneROI() )
    downsample with n>1'''
    i = 0
    try:
        while True:
            plt.figure(68769)
            plt.clf()
            plt.imshow(imgs[i,::n,::n])
            plt.title(str(i)+'/'+str(len(imgs)))
            i = np.mod(i+1, len(imgs))
            plt.pause(0.001)
    except KeyboardInterrupt:
        print('Stopped')
    


