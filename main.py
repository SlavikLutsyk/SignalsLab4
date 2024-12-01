import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

n = 8
T_values = [4, 8, 16, 32, 64, 128]
L = 10

# Function f(t) for case (a): f(t) = t^(2 * n)
def f_a(t):
    return t ** (2 * n)

# Fourier Integral function for k-th term
def fourier_integral(k, T):
    w_k = (2 * np.pi * k) / T

    # Real part of F(w_k): integrate f(t) * cos(-w_k * t) over [-L, L]
    real_part, _ = quad(lambda t: f_a(t) * np.cos(-w_k * t), -L, L, limit=100, epsabs=1e-6, epsrel=1e-6, weight='cos',
                        wvar=-w_k)

    # Imaginary part of F(w_k): integrate f(t) * sin(-w_k * t) over [-L, L]
    imag_part, _ = quad(lambda t: f_a(t) * np.sin(-w_k * t), -L, L, limit=100, epsabs=1e-6, epsrel=1e-6, weight='sin',
                        wvar=-w_k)

    return real_part, imag_part

# Amplitude spectrum function
def amplitude_spectrum(real_part, imag_part):
    return np.sqrt(real_part ** 2 + imag_part ** 2)

# Plotting Re(F(w_k)) and |F(w_k)| on separate figures for each T
for T in T_values:
    real_values = []
    amplitude_values = []

    # Compute Fourier integral for k = 0 to 10
    for k in range(11):
        real_part, imag_part = fourier_integral(k, T)
        real_values.append(real_part)
        amplitude_values.append(amplitude_spectrum(real_part, imag_part))

    # Creating subplots for Re(F(w_k)) and |F(w_k)| in a single figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plotting Re(F(w_k)) for each T separately
    ax1.plot(range(11), real_values, marker='o', label=f'T={T}')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Re(F(w_k))')
    ax1.set_title(f'Real part of Fourier Integral (Re(F(w_k))) for T={T}')
    ax1.legend()
    ax1.grid()

    # Plotting |F(w_k)| for each T separately
    ax2.plot(range(11), amplitude_values, marker='o', label=f'T={T}')
    ax2.set_xlabel('k')
    ax2.set_ylabel('|F(w_k)|')
    ax2.set_title(f'Amplitude Spectrum |F(w_k)| for T={T}')
    ax2.legend()
    ax2.grid()

    plt.suptitle(f'Fourier Integral Components for T={T}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Combined plot for Re(F(w_k)) and |F(w_k)| for all T values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for T in T_values:
    real_values = []
    amplitude_values = []

    # Compute Fourier integral for k = 0 to 10
    for k in range(11):
        real_part, imag_part = fourier_integral(k, T)
        real_values.append(real_part)
        amplitude_values.append(amplitude_spectrum(real_part, imag_part))

    # Plotting Re(F(w_k)) for all T on the same plot
    ax1.plot(range(11), real_values, marker='o', label=f'T={T}')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Re(F(w_k))')
    ax1.set_title('Real part of Fourier Integral (Re(F(w_k))) for all T')
    ax1.legend()
    ax1.grid()

    # Plotting |F(w_k)| for all T on the same plot
    ax2.plot(range(11), amplitude_values, marker='o', label=f'T={T}')
    ax2.set_xlabel('k')
    ax2.set_ylabel('|F(w_k)|')
    ax2.set_title('Amplitude Spectrum |F(w_k)| for all T')
    ax2.legend()
    ax2.grid()

plt.suptitle('Fourier Integral Components for all T values')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()