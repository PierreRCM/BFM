import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from simulated_motor_lib import SimulateTrace
from fitEllipse import XYtoThetaSimple
from average_psd_maker import average_psd_maker as avg
from average_psd_maker import log_bin_average as lba


def fit_data(x, y):

    parameters, pcov = curve_fit(model, x, y,
                                     # average fitting parameters value
                                        bounds=(0, [np.inf, np.inf, np.inf]))
    return parameters


def model(f, a, gamma, kl):

    global vmean
    Kb = 1.380649E-23
    T = 295.
    tc = gamma/kl
    return ((a*vmean)/((2*np.pi*f*tc)**2 + 1)) + ((2*Kb*T)/gamma)*((2*np.pi*f*tc)**2/((2*np.pi*f*tc)**2 + 1))


def model2(f, a, gamma, kl, vmean):

    Kb = 1.380649E-23
    T = 295.
    tc = gamma/kl
    return ((a*vmean)/((2*np.pi*f*tc)**2 + 1)) + ((2*Kb*T)/gamma)*((2*np.pi*f*tc)**2/((2*np.pi*f*tc)**2 + 1))


FPS = 10000
psd_bead_angle_list = []

data = np.load("../data/D_WT_1000SS.p", allow_pickle=True)
psd_data, freq, vmean = avg([data[0]["x"], data[0]["y"]], seg_number=10)

parameters = fit_data(freq, psd_data)

motor_angle, bead_angle = SimulateTrace(bead_radius=1100E-9, axis_offset=200E-9, speed_hz=55, numsteps=400, trace_length_s=120, Nstates=1, FPS=10000, k_hook= 400*1e-9*1e-12)
psd, freqs, vmean1 = avg(bead_angle, seg_number=10, data_type="angle")
# log_freq, psd_binned = lba(freqs, psd)
# parameters_sim = fit_data(log_freq, psd_binned)

print(parameters)

plt.loglog(freq, psd_data, marker="o", linewidth=0, markersize=0.4, color="red", label="PSD data")
plt.plot(freq, model(freq, *parameters), label="fit: a={:.2f} rad, ".format(parameters[0])+ "Drag coef={:.2f} N.m.s/rad, "
         .format(parameters[1]) + "stiffness={:.2f} N.m/rad".format(parameters[2]))
# plt.loglog(freqs, model(freqs, *parameters_sim))

para = [0.02, 3.285749261471205e-20, 400 *1E-9 *1e-12]

plt.loglog(freqs, model2(freqs, *para, vmean1), label="Simulation parameters: a={:.2E} rad, ".format(para[0])+ "Drag coef={:.2E} N.m.s/rad, "
         .format(para[1]) + "stiffness={:.2E} N.m/rad".format(para[2]))
plt.plot(freqs, psd, marker="o", linewidth=0, markersize=0.4, color="b", label="PSD simulation")
plt.xlabel("Freq")
plt.ylabel("(rad/s)^2")
plt.title("PSD speed (bead angle) simulation vs PSD speed data (10 equal length array averaged)")
plt.legend()
plt.show()