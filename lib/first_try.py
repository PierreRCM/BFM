import numpy as np
import matplotlib.pyplot as plt


def fct(x, f):

    return np.sin(2*np.pi*x*f) + np.sin(2*np.pi*x*f*20) + np.sin(2*np.pi*x*f*2000) + np.random.uniform(-5, 5, x.size)


def fct2(x):

    return np.cos(2*np.pi*50*x)


def triangle(x):

    y = []

    for i in x:

        if 3 <= i < 4:

            y.append(1-(i-3))
        else:

            y.append(0)

    return y


def porte(x):

    y = []

    for i in x:

        if 3 <= i < 4:

            y.append(1)

        else:
            y.append(0)
    return y


n = 1
N = 300000
f = 10
a = np.linspace(0, n, N, endpoint=True)
plt.plot(a, fct(a, f))
plt.show()
h = n/N
yf = 4*abs(np.fft.fft(fct(a, f), N))**2/N**2

# get amplitude spectrum
freq = np.linspace(0.0, 1/h, N)  # get freq axis

# plot the amp spectrum

plt.plot(freq, yf)
plt.show()