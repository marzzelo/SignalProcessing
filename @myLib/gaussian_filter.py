import numpy as np
from scipy.signal import convolve

def gaussian_filter(signal, sample_rate, fwhm_s, window_s, return_kernel=False):
    """
    Aplica un filtro gaussiano a una señal 1D (voltaje vs tiempo, por ejemplo).

    Parámetros:
    - signal: array 1D con la señal a filtrar
    - sample_rate: frecuencia de muestreo en Hz
    - fwhm_s: ancho a mitad de altura (FWHM) del kernel gaussiano en segundos
    - window_s: ventana total del kernel en segundos (ej. 0.1 → [-0.05 s, +0.05 s])

    Retorna:
    - signal_filtrada: array 1D con la señal filtrada
    """
    # Conversión de FWHM a sigma
    sigma_s = fwhm_s / 2.355
    sigma_samples = sigma_s * sample_rate

    # Construcción del kernel
    k = int((window_s * sample_rate) / 2)
    x = np.arange(-k, k + 1)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    kernel /= np.sum(kernel)  # normalizar

    # Aplicar convolución
    filtered = convolve(signal, kernel, mode='same')
    
    if return_kernel: 
        return filtered, (x/sample_rate, kernel)

    return filtered
