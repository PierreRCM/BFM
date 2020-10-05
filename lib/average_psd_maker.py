import numpy as np
import pandas as pd
from fitEllipse import XYtoThetaSimple
import warnings
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.special import huber
Kb = 1.380649E-23
T = 295.


def huber_loss(y_pred, y, delta=1.0):

    huber_mse = 0.5*(y-y_pred)**2
    huber_mae = delta * (np.abs(y - y_pred) - 0.5 * delta)

    return np.where(np.abs(y - y_pred) <= delta, huber_mse, huber_mae)


def correct_sig_modulation(sig, angle_turns, plots=False, polydeg=10):
    ''' generalize correct_speed_modulation()
        correct the modulation in mod(angle_turns,1) Vs sig
      sig, angle_turns: any signal and relative angle_turns trace
      polydeg : polyn degree to fit
    '''
    # angle turns in 0,1:
    am = np.mod(angle_turns - angle_turns[0], 1)
    # polyn fit:
    pf = np.polyfit(am, sig, polydeg)
    po = np.poly1d(pf)
    # sig corrected:
    sig_corr = sig - po(am) + np.mean(sig)
    if plots:
        plt.figure('correct_sig_modulation', clear=True)
        plt.subplot(321)
        plt.plot(sig, '.', label='sig.raw')
        plt.legend()
        plt.subplot(322)
        plt.plot(angle_turns, '.', label='angle_turns')
        plt.legend()
        plt.subplot(312)
        plt.plot(am, sig, '.', ms=2, alpha=0.3, label='sig.raw')
        plt.plot(am, po(am), '.', ms=2, label='poly.fit')
        plt.xlabel('angle_turns mod 1')
        plt.legend()
        plt.subplot(313)
        plt.plot(am,sig_corr, '.', ms=2, alpha=0.3, label='sig.corr.')
        plt.xlabel('angle_turns mod 1')
        plt.legend()
        plt.tight_layout()
    return sig_corr


class HandleData:
    """Not really useful to create, only helpful for changing the variable vmean in the function model, i did not find
        other way to do it, even using global
        This class store all the function useful to transform and fit the experimental data, functions are dependant of
         each other"""
    def __init__(self, data, d_type="x_y"):

        self.vmean = float()
        self.d_type = d_type
        self.data = data
        self.f = None
        self.psd= None
        self.log_f = None
        self.log_psd = None
        # self._full_psd()

    def model(self, f, a, gamma, kl):

        tc = gamma/kl

        return 2*(((a*self.vmean)/((2*np.pi*f*tc)**2 + 1)) + ((2*Kb*T)/gamma)*((2*np.pi*f*tc)**2/((2*np.pi*f*tc)**2 + 1)))

    def loss_func(self, *w):

        y_pred = self.model(self.log_f, w[0][0], w[0][1], w[0][2])
        return sum(abs(self.log_psd - y_pred)) / len(self.log_psd)
        # return sum(huber(2, y_pred))

    @staticmethod
    def reorganize_data(a_list):
        # doublecheck but i think this func is useless, might be similar as np.reshape()
        matrix = np.zeros([len(a_list), len(a_list[0])])

        for i, array in enumerate(a_list):
            for j, value in enumerate(array):

                matrix[i, j] = value

        return matrix

    def average_psd_maker(self, data, FPS=10000, seg_number=100, data_type="x_y", corr=False):
        """Input: data: if data_type="x_y" data is list of posx and posy (2 elements)
                        else if datatype ="angle" data= array of angle
                        Seg_number: number of equal length array to average
           Perform PSD for each equal length array, then compute the average
           Output: len(df)/seg_number array-like PSD and freq, and speed mean"""

        equal_lenght_df = []

        if data_type == "x_y":

            posx = data[0]
            posy = data[1]
            m_df = pd.DataFrame(np.array([posx, posy]).T)
            new_index_len = int(len(m_df.index)/seg_number)*seg_number
            m_df = m_df.iloc[:new_index_len]
            seg_number = seg_number
            m_df.columns = ["x", "y"]
            posx = m_df["x"]
            posy = m_df["y"]

            equal_lenght_df = [
                pd.DataFrame(np.array([posx[int(x * len(posx) / seg_number):int((x + 1) * len(posx) / seg_number)],
                                       posy[
                                       int(x * len(posx) / seg_number):int((x + 1) * len(posy) / seg_number)]]).T,
                             columns=["x", "y"]) for x in range(seg_number)]

        elif data_type == "angle":
            angle = data
            m_df = pd.DataFrame(np.array([angle]).T)
            new_index_len = int(len(m_df.index) / seg_number) * seg_number
            m_df = m_df.iloc[:new_index_len]
            seg_number = seg_number
            m_df.columns = ["angle"]
            angle = m_df["angle"]

            equal_lenght_df = [
                pd.DataFrame(np.array([angle[int(x * len(angle) / seg_number):int((x + 1) * len(angle) / seg_number)]]).T,
                             columns=["angle"]) for x in range(seg_number)]  # split the dataframe into -seg_number- data
                                                                             # frame


        psd_list = []
        list_vmean = []

        for a_df in equal_lenght_df:

            angle_theta = []
            a_df.index = a_df.index / FPS

            if data_type == "x_y":
                angle_theta = XYtoThetaSimple(a_df["x"], a_df["y"])
            elif data_type == "angle":
                angle_theta = a_df["angle"].to_numpy()

            vitesse = np.diff(angle_theta) * FPS

            if corr:
                angle_rev = angle_theta / (2*np.pi)
                vitesse_rev = np.diff(angle_rev) * FPS

                vitesse_corr = correct_sig_modulation(vitesse_rev, angle_rev[1:])
                vitesse = vitesse_corr * 2*np.pi
            vmean = vitesse.mean()
            list_vmean.append(vmean)
            # vitesse -= vmean
            dft = np.abs(np.fft.fft(vitesse))
            psd = (dft ** 2)
            freqs = np.fft.fftfreq(vitesse.size, 1 / FPS)
            PSD_speed, freqs = np.fft.fftshift(psd), np.fft.fftshift(freqs)
            PSD_speed = PSD_speed / (len(vitesse) * FPS)
            # Make it one sided:

            freqs, PSD_speed = freqs[int(len(freqs) / 2) + 1:], PSD_speed[int(len(freqs) / 2) + 1:]
            PSD_speed *= 2

            psd_list.append(PSD_speed)

        vmean = np.array(list_vmean).mean()
        psd_mat = self.reorganize_data(psd_list)
        average_psd = [psd_mat[:, x].mean() for x in range(psd_mat.shape[1])]

        return average_psd, freqs, vmean

    @staticmethod
    def log_bin_average(F, PS, bin_number=20):
        '''
        Log-bins a power spectrum
        '''
        ls_indices = np.unique(np.logspace(0, np.log10(len(F)), bin_number, dtype=int) - 1)
        logbinned_F = F[ls_indices]
        logbinned_F = logbinned_F[:-1] + np.diff(logbinned_F) / 2.
        logbinned_PS = np.empty(0)

        for i in range(len(ls_indices) - 1):
            logbinned_PS = np.append(logbinned_PS, np.mean(PS[ls_indices[i]:ls_indices[i+1]+1]))

        return logbinned_F, logbinned_PS

    def segment_data(self, data, bounds, sim, segment_length=10000, sub_seg_number=5, slide_coef=2, data_type="angle", corr=True):
        """Input, data trace, segment split numbers and sub_segment number,
            select a piece of the trace: newdata = data[:len(data)*(1/segment)], split the new data into sub_seg new segments
            perform PSD on each sub segment, then average the PSD of the segment. Repeat on each segment
            Output: Parameters fitted on each sub segment"""

        if data_type == "x_y":

            data = XYtoThetaSimple(data[0], data[1])
            segment = data[:segment_length]
        else:
            segment = data[:segment_length]

        a_list = []
        k_list = []
        gamma_list = []
        vmean_list = []
        position_in_trace = 0

        while (position_in_trace + segment_length) <= len(data):  # cancel the loop when there is not enough points to continue the algorithm
            sub_list_psd = []
            sub_list_vmean = []
            for y in range(sub_seg_number):

                sub_seg = segment[int(len(segment)*y/sub_seg_number):int(len(segment)*(y+1)/sub_seg_number)]  # split segment

                # Compute PSD, using average PSD maker with 1 seg is just the PSD of the data seg
                psd_sub_seg, freq_sub_seg, vmean = self.average_psd_maker(sub_seg, seg_number=1, data_type="angle", corr=corr)
                sub_list_psd.append(psd_sub_seg)
                sub_list_vmean.append(vmean)

            psd_mat = self.reorganize_data(sub_list_psd)
            average_psd = [psd_mat[:, x].mean() for x in range(psd_mat.shape[1])]
            self.f = freq_sub_seg
            self.psd = average_psd
            self.vmean = np.array(sub_list_vmean).mean()

            self.log_f, self.log_psd = self.log_bin_average(self.f, self.psd, bin_number=15)
            result = differential_evolution(self.loss_func, bounds, disp=True, tol=1E-3, popsize=40, mutation=1)
            plt.loglog(self.f, self.psd, markersize=1, marker="o", linewidth=0, label="full-psd")
            plt.loglog(self.log_f, self.log_psd, label="log-bin-average")
            plt.plot(self.f, self.model(self.f, *result.x), label="DE pred")
            plt.plot(self.f, self.model(self.f, sim[0], sim[1], sim[2]), label="model with expected drag")
            print("DE :{} loss : {}".format(result.x, result.fun))
            # print("sim loss: {}".format(self.sim_loss(sim)))
            # r_step = (sim[0]- result.x[0])/result.x[0]
            # print("r_step = {}".format(r_step))
            plt.legend()
            plt.show()
            vmean_list.append(self.vmean)
            a_list.append(result.x[0])
            gamma_list.append(result.x[1])
            k_list.append(result.x[2])

            new_position = int(segment_length/slide_coef) + position_in_trace
            segment = data[new_position:new_position + segment_length]
            position_in_trace = new_position

        return vmean_list, np.array(a_list), np.array(gamma_list), np.array(k_list)

    def _full_psd(self):

        self.psd, self.f, self.vmean = self.average_psd_maker(self.data, data_type=self.d_type, seg_number=1)
        self.log_f, self.log_psd = self.log_bin_average(self.f, self.psd)











