import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig, inv
import pandas as pd
import datetime as dt
from scipy.optimize import curve_fit


def fitEllipse(x,y):
    """Algorithm from Fitzgibbon et al 1996, Direct Least Squares Fitting of Ellipsees.
    Formulated in terms of Langrangian multipliers, rewritten as a generalized eigenvalue problem. """
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a


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


def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation(a):

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


def rotateArray(a,th):
    ''' rotates the array a of the angle theta (rad)'''
    R = np.array(([np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]))
    rota = np.dot(R,a)
    return rota


def XYtoThetaSimple(x, y, plots=0):
    ''' move xy elliptical trajectory to (0,0), rescale to be circular, find angle of each x,y simply by atan'''

    # fit ellipse on xy :
    xx, yy, center0, a, b, phi = makeBestEllipse(x, y, nel=50)
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


def model(f, a, gamma, kl):

    global vmean
    Kb = 1.380649E-23
    T = 300
    tc = gamma/kl

    return ((a*vmean)/((2*np.pi*f*tc)**2 + 1)) + ((2*Kb*T)/gamma)*((2*np.pi*f*tc)**2/((2*np.pi*f*tc)**2 + 1))


def fit_data(x, y):

    parameters, pcov = curve_fit(model, x[:len(vitesse) // 4], y[:len(vitesse) // 4],
                                        p0=[1.44044095E00, 5.78308321E-05, 5.76981313E-01], # average fitting parameters value
                                        bounds=(0, [np.inf, np.inf, np.inf]))
    return parameters


data = np.load("../data/D_WT_1000SS.p")

data.pop(2)
data.pop(43)
exps = [x for x in data.keys()]

beads_diameter = [data[x]["dbead_nm"] for x in exps]
list_parameter = []


for exp in exps:

    posx_1 = data[exp]["x"]
    posy_1 = data[exp]["y"]
    FPS = data[exp]["FPS"]  # Frame per sec
    # nm_per_pix = data[exp]["nm_per_pix"]  # number of nanometers by pix

    df_xy = pd.DataFrame(np.array([posx_1, posy_1]).T)
    df_xy.columns = ["x", "y"]
    df_xy_1 = df_xy.iloc[:500000]  # reduce size of data
    df_xy_1.index = df_xy_1.index / FPS

    angle_theta = XYtoThetaSimple(df_xy_1["x"], df_xy_1["y"])
    angle_degree = (angle_theta*180/np.pi)
    vitesse = np.gradient(angle_degree)*FPS
    vmean = vitesse.mean()
    vitesse -= vmean
    # print(np.sum(vitesse**2)/FPS)

    n = max(df_xy_1.index)
    N = len(df_xy_1.index)
    step = n/N

    fft_vitesse = abs(np.fft.fft(vitesse))/N
    PSD_speed = abs(np.fft.fft(vitesse))**2/(N*FPS)
    freq = np.linspace(0.0, 1/step, N)

    parameters_fitted = fit_data(freq, PSD_speed)
    list_parameter.append(parameters_fitted)

a_list = []
gamma_list = []
k_list = []
list_parameter.pop(24)
for parameters in list_parameter:

    a_list.append(parameters[0])
    gamma_list.append(parameters[1])
    k_list.append(parameters[2])

fig, (ax1, ax2) = plt.subplots(2, 2)
a_mean = [np.array([a_list for parameters in list_parameter]).mean() for x in range(len(exps)-1)]
gamma_mean = [np.array([gamma_list for parameters in list_parameter]).mean() for x in range(len(exps)-1)]
k_mean = [np.array([k_list for parameters in list_parameter]).mean() for x in range(len(exps)-1)]

a_std = [np.std(a_list) for x in range(len(list_parameter))]
gamma_std = [np.std(gamma_list) for x in range(len(list_parameter))]
k_std = [np.std(k_list) for x in range(len(list_parameter))]

plt.suptitle("Fitted parameters for each experiments")
ax1[0].set_title("Step size")
ax1[0].set_xlabel("Experiments")
ax1[0].set_ylabel("Degree")
ax1[0].errorbar(np.arange(0, len(exps)-1), a_list, label="Step size", yerr=a_std, fmt="o")
ax1[0].plot(np.arange(0, len(exps)-1), a_mean,  label="Mean", color="r")
ax1[0].legend()

ax1[1].set_title("Drag coefficient")
ax1[1].set_xlabel("Experiments")
ax1[1].set_ylabel("Kg.m².s^-1*°^-2 ????")
ax1[1].errorbar(np.arange(0, len(exps)-1), gamma_list, label="Drag coefficient", yerr=gamma_std, fmt="o")
ax1[1].plot(np.arange(0, len(exps)-1), gamma_mean,  label="Mean", color="r")
ax1[1].legend()

ax2[0].set_title("Stiffness coefficient")
ax2[0].set_xlabel("Experiments")
ax2[0].set_ylabel("N * m . °-1 ?????")

ax2[0].errorbar(np.arange(0, len(exps)-1), k_list, label="Stiffness", yerr=k_std, fmt="o")
ax2[0].plot(np.arange(0, len(exps)-1), k_mean, label="Mean", color="r")
ax2[0].legend()
plt.show()


# for k in [1, 1.5]:
#     for gamma in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
#         plt.plot(freq[:len(vitesse)//2], model(freq[:len(vitesse)//2], 1, gamma, k), label=str(k) + " " + str(gamma))


# fig, (ax1, ax2) = plt.subplots(2, 2)
#
# ax1[0].set_title("Angle of the bead")
# ax1[1].set_title("Speed of the bead")
# ax2[0].set_title("FFT: Speed of the bead")
# ax2[1].set_title("PSD: Speed of the bead")
#
# ax1[0].set_xlabel("Time : milliseconds")
# ax1[0].set_ylabel("Degree")
#
# ax1[1].set_xlabel("Time : milliseconds")
# ax1[1].set_ylabel("Speed: degree per millisecond")
#
# ax2[0].set_xlabel("Frequencies : Hertz")
# ax2[0].set_ylabel("Amplitude : Degree per milliseconds?")
#
# ax2[1].set_xlabel("Frequencies : Hertz")
# ax2[1].set_ylabel("Amplitude : Degree per milliseconds ^ 2 ?")
#
# ax1[0].scatter(df_xy_1.index, angle_degree, s=1)
#
# ax1[1].scatter(df_xy_1.index, vitesse*10, s=1, label="gradient theta")  # Multiply by 10, to have degree per millisecond
# ax1[1].scatter(df_xy_1.index, [vitesse.mean()*10 for x in range(len(df_xy_1.index))], s=1, label="mean speed")
# ax1[1].legend()
#
# ax2[0].plot(freq[:len(vitesse)//2], fft_vitesse[:len(vitesse)//2])
# ax2[1].loglog(freq[:len(vitesse)//2], PSD_speed[:len(vitesse)//2])
# plt.show()