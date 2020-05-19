
# !/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Tue May 12 19:01:41 2020

​

@author: ALNord

"""

import numpy as np
import matplotlib.pyplot as plt
from filters import run_win_smooth
Kb = 1.38e-23  # Boltzmann const. N m / K

T = 295.                # temperature

eta = 0.954e-3          # [Pa*s]=[N*s/m^2]] water viscosity


def SimulateTrace(bead_radius, axis_offset, speed_hz, numsteps, trace_length_s, Nstates, FPS, k_hook = 400 *1E-9 *1e-12 ,plots=0):
    '''Simulated trace of stepping motor, using KMC, taking into account filtering

    affect of the hook and thermal noise. Returned values of angles are in rad.

    bead_radius, axis_offset in m

    speed_hz in turns/s

    numsteps = machanical steps/turn

    trace_length_s : in s

    Nstates : n. of internal states in 1 mechano-chem cycle, one mechanical step out per cycle

    k_hook : N m /rad

    '''

    FPS = float(FPS)
    drag_bead = BeadDrag(bead_radius, axis_offset, k_hook)  # N m s/rad
    bead_relaxation = drag_bead / k_hook
    upsample = np.int(np.ceil(400 / bead_relaxation / FPS))
    print('upsample factor = ', upsample)
    working_FPS = upsample * FPS
    npts_trace = int(trace_length_s * working_FPS)

    time = np.linspace(0, npts_trace, npts_trace) / working_FPS
    dt = 1. / working_FPS
    # Poisson stepper simulation, angle in rad:
    motor_angle = KMC_linearpath(Nstates=Nstates, k=1, nevents=speed_hz * numsteps * np.rint(npts_trace / working_FPS),
                                 npts_trace=npts_trace, Dangle=2 * np.pi / numsteps)
    bead_angle = np.zeros(len(motor_angle))
    d_bead_angle = np.zeros(len(motor_angle))
    # calc thermal noise. In rad.:
    noise = BrownianNoise(drag_bead, dt, len(bead_angle))
    # Calculate the bead angle with noise, according to Eq 5.8-5.10

    for i in range(1, len(motor_angle)):
        d_bead_angle[i] = (1 / bead_relaxation) * dt * (motor_angle[i] - bead_angle[i - 1]) + noise[i]
        bead_angle[i] = bead_angle[i - 1] + d_bead_angle[i]

    print('SimulateTrace(): k_hook = {:.4f} pN nm/rad'.format(k_hook * 1e21))
    print('SimulateTrace(): bead relax. = {:.6f} s'.format(bead_relaxation))
    print('SimulateTrace(): dt = {:.6f} s'.format(dt))
    print('SimulateTrace(): Mean noise = %s rad' % (np.mean(abs(noise))))

    if plots:
        theta = bead_angle / (2 * np.pi)  # In revs
        dtheta = np.diff(theta)
        speed = dtheta * working_FPS
        speed_filt = run_win_smooth(speed, win=np.int(0.5 * working_FPS), usemode='valid')
        plt.figure()
        plt.subplot(121)
        plt.plot(time[:-1], motor_angle / (2 * np.pi))
        plt.plot(time[:-1], bead_angle / (2 * np.pi), 'k.')
        plt.legend(['motor angle', 'bead angle'])
        plt.xlabel('Time (s)')
        plt.ylabel('Theta (Revs)')
        plt.axis('tight')
        plt.subplot(122)
        plt.plot(time[:len(speed_filt)], speed_filt)
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (Hz)')
        plt.axis('tight')

    motor_angle, bead_angle = motor_angle[::upsample], bead_angle[::upsample]
    return motor_angle, bead_angle


def KMC_linearpath(Nstates=6, k=1., nevents=10000, npts_trace=50000, Dangle=2 * np.pi / 26, verbose=1):
    '''

    ex. usage:

        angle = KMC_linearpath(Nstates=3, k=.5, winfilter=5, noiseampl=2, nevents=5000, npts_trace=100000, Dangle=(2*np.pi)/26, N360=26.)

    '''

    # make dwell times:
    T = makedwelltimes_linearpath(Nstates=Nstates, k=k, nevents=nevents, fitsgamma=0)
    # make angle trace:
    angle_rad, fps = dwelltimes2angle(T, npts_ttrace=npts_trace, Dangle=Dangle)

    if verbose:
        print('KMC_linearpath():' + str(Nstates) + ' states/mechanical step')
        print('KMC_linearpath():' + str((2 * np.pi) / Dangle) + ' mechanical steps/turn, each of ' + '{:.2f}'.format(
            Dangle) + ' rad')
        print('KMC_linearpath(): => {:.2f}'.format((2 * np.pi) / Dangle * Nstates) + ' states/turn,  k = ' + str(
            k) + ' 1/s')
        print('exiting KMC_linearpath')

    return angle_rad


def makedwelltimes_linearpath(Nstates=6, k=1., knoise=0, nevents=100000, fitsgamma=0):
    '''

    Code stolen from FP's kin3.py

    KMC simulate a linear path

        A_1  --k-->  A_2  --k--> ... --k-->  A_N  --k-->  A_1

    returns series of 'nevents' times between Nstates '''

    nevents = int(nevents)
    Nstates = int(Nstates)
    k = float(k)
    krand = k + knoise * np.random.randn(nevents)
    krand = np.maximum(1e-6, krand)
    T = np.zeros(nevents)

    # generate times between states

    for i in range(Nstates):
        ra = np.random.rand(nevents)
        t = np.log(1. / ra) / krand
        # time between Nstates states:
        T = T + t

    return T


def dwelltimes2angle(T, npts_ttrace=1000, Dangle=2 * np.pi / 26):
    ''' Code stolen from FP's kin3.py.  Updated 29/11/16.

    generate angle time trace, sampling the serie of dwell times T

    with 'npts_ttrace' points. Dangle = angle moved every mechanical step, [rad] '''

    sT = np.cumsum(T)
    t = np.linspace(0, sT[-1], npts_ttrace)
    stepsperframe, _ = np.histogram(sT, t)
    cumstepsperframe = np.cumsum(stepsperframe)
    angle_resampled = cumstepsperframe * Dangle
    fps = float(npts_ttrace) / sT[-1]

    return angle_resampled, fps


def BeadDrag(bead_radius, axis_offset, k_hook, dist_beadsurf=5000e-9, verbose=1):
    ''' Calculates the drag of the bead. Code taken from BeadDrag.py.

    bead_radius, axis_offset, dist_beadsurf in m

    Returned bead drag in N m s/rad

    '''

    faxen_1 = 1 - (1 / 8.) * (bead_radius / dist_beadsurf) ** 3
    faxen_2 = 1 - (9 / 16.) * (bead_radius / dist_beadsurf) + (1. / 8) * (bead_radius / dist_beadsurf) ** 3 - (
                45 / 256.) * (bead_radius / dist_beadsurf) ** 4 - (1. / 16) * (bead_radius / dist_beadsurf) ** 5
    drag_bead = (8 * np.pi * eta * bead_radius ** 3) / faxen_1 + (
                6 * np.pi * eta * bead_radius * axis_offset ** 2) / faxen_2

    if verbose:
        print('BeadDrag(): bead radius :' + str(bead_radius) + 'm')
        print("BeadDrag(): bead drag : " + str(drag_bead) + " N m s/rad")
        print("BeadDrag(): charact.time on hook: " + str(1000 * drag_bead / k_hook) + " ms")
    return drag_bead


def BrownianNoise(drag_bead, dt, samples):
    '''Returns the thermal noise of the bead'''

    D = Kb * T / drag_bead  # in rad/s or rad^2/s
    rand_num = np.random.randn(samples)
    noise = rand_num * np.sqrt(2 * D * dt)  # in rad
    return noise

