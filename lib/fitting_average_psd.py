import numpy as np
import matplotlib.pyplot as plt
from fitEllipse import XYtoThetaSimple
from scipy.optimize import curve_fit
import pandas as pd


def reorganize_data(list):

    matrix = np.zeros([len(list), len(list[0])])

    for i, array in enumerate(list):
        for j, value in enumerate(array):

            matrix[i, j] = value

    return matrix


def model(f, a, gamma, kl):

    global vmean

    Kb = 1.380649E-23
    T = 300
    tc = gamma/kl

    return ((a*vmean)/((2*np.pi*f*tc)**2 + 1)) + ((2*Kb*T)/gamma)*((2*np.pi*f*tc)**2/((2*np.pi*f*tc)**2 + 1))


def fit_data(x, y):

    parameters, pcov = curve_fit(model, x, y,
                                        p0=[1.44044095E00, 5.78308321E-05, 5.76981313E-01], # average fitting parameters value
                                        bounds=(0, [np.inf, np.inf, np.inf]))
    return parameters


def parameters_to_list(list_para):

    a_list = []
    gamma_list = []
    k_list = []

    for parameters in list_para:
        a_list.append(parameters[0])
        gamma_list.append(parameters[1])
        k_list.append(parameters[2])

    return np.array(a_list), np.array(gamma_list), np.array(k_list)


def moments(list_para):

    a_list, gamma_list, k_list = parameters_to_list(list_para)

    a_mean = [a_list.mean() for x in range(len(exps))]
    gamma_mean = [gamma_list.mean() for x in range(len(exps))]
    k_mean = [k_list.mean() for x in range(len(exps))]

    a_std = [np.std(a_list) for x in range(len(list_para))]
    gamma_std = [np.std(gamma_list) for x in range(len(list_para))]
    k_std = [np.std(k_list) for x in range(len(list_para))]

    return a_mean, gamma_mean, k_mean, a_std, gamma_std, k_std, a_list, gamma_list, k_list


def average_psd_maker(data, exp, seg_number=100):

    posx_1 = data[exp]["x"]
    posy_1 = data[exp]["y"]
    FPS = data[exp]["FPS"]
    df_xy = pd.DataFrame(np.array([posx_1, posy_1]).T)
    new_index_len = int(len(df_xy.index)/100)*100
    df_xy = df_xy.iloc[:new_index_len]
    df_xy.columns = ["x", "y"]
    nposx = df_xy["x"]
    nposy = df_xy["y"]
    seg_number = seg_number

    equal_lenght_df = [pd.DataFrame(np.array([nposx[int(x * len(nposx) / seg_number):int((x + 1) * len(nposx) / seg_number)],
                                              nposy[
                                              int(x * len(nposx) / seg_number):int((x + 1) * len(nposy) / seg_number)]]).T,
                                    columns=["x", "y"]) for x in range(seg_number)]
    psd_list = []

    list_vmean = []

    for df in equal_lenght_df:

        df.index = df.index / FPS

        angle_theta = XYtoThetaSimple(df["x"], df["y"])
        angle_degree = (angle_theta * 180 / np.pi)
        vitesse = np.gradient(angle_degree) * FPS

        vmean = vitesse.mean()
        list_vmean.append(vmean)
        vitesse -= vmean
        n = max(df.index)
        N = len(df.index)
        step = n / N
        PSD_speed = abs(np.fft.fft(vitesse)) ** 2 / (N * FPS)
        freq = np.linspace(0.0, 1 / step, N)

        psd_list.append(PSD_speed)

    f_vmean = np.array(list_vmean).mean()

    psd_mat = reorganize_data(psd_list)
    average_psd = [psd_mat[:, x].mean() for x in range(psd_mat.shape[1])]

    return average_psd, freq, f_vmean


def plot_parameters(a_mean, gamma_mean, k_mean, a_std, gamma_std, k_std, a_list, gamma_list, k_list):

    fig, (ax1, ax2) = plt.subplots(2, 2)

    plt.suptitle("Fitted parameters for each experiments")
    ax1[0].set_title("Step size")
    ax1[0].set_xlabel("Experiments")
    ax1[0].set_ylabel("Degree")

    ax1[0].errorbar(np.arange(0, len(exps)), a_list, label="Step size", yerr=a_std, fmt="o")
    ax1[0].plot(np.arange(0, len(exps)), a_mean, label="Mean", color="r")
    ax1[0].legend()

    ax1[1].set_title("Drag coefficient")
    ax1[1].set_xlabel("Experiments")
    ax1[1].set_ylabel("Kg.m².s^-1*°^-2 ????")
    ax1[1].errorbar(np.arange(0, len(exps)), gamma_list, label="Drag coefficient", yerr=gamma_std, fmt="o")
    ax1[1].plot(np.arange(0, len(exps)), gamma_mean, label="Mean", color="r")
    ax1[1].legend()

    ax2[0].set_title("Stiffness coefficient")
    ax2[0].set_xlabel("Experiments")
    ax2[0].set_ylabel("N * m . °-1 ?????")

    ax2[0].errorbar(np.arange(0, len(exps)), k_list, label="Stiffness", yerr=k_std, fmt="o")
    ax2[0].plot(np.arange(0, len(exps)), k_mean, label="Mean", color="r")
    ax2[0].legend()
    plt.show()


data = np.load("../data/D_WT_1000SS.p")
data.pop(2)
data.pop(43)
exps = [x for x in data.keys()]
print(len(exps))
list_parameters = []

for i, exp in enumerate(exps):

    print("Average psd:")
    average_psd, freq, vmean = average_psd_maker(data, exp)
    plt.title("data n°: " + str(exp) + " : Average PSD, sampled 100 times.")
    plt.xlabel("freq")
    plt.loglog(freq[:len(average_psd)//2], average_psd[:len(average_psd)//2], label="data")

    print("Done")
    print("\n")
    print("Parameters:")
    parameters = fit_data(freq[:len(average_psd)//2], average_psd[:len(average_psd)//2])
    plt.loglog(freq[:len(average_psd)//2], model(freq[:len(average_psd)//2], parameters[0], parameters[1], parameters[2]), label="fitted model")
    plt.legend()
    plt.show()
    print("Done")
    print("\n")
    list_parameters.append(parameters)
    print(len(list_parameters))
a_mean, gamma_mean, k_mean, a_std, gamma_std, k_std, a_list, gamma_list, k_list = moments(list_parameters)


plot_parameters(a_mean, gamma_mean, k_mean, a_std, gamma_std, k_std, a_list, gamma_list, k_list)
