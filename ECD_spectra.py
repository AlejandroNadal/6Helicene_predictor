import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Fixed parameters
conv = 4.32e-3      # conversion factor R → Δε
iFWHM_nm = 25.0     # gaussian width band(nm)
concentration_M = 4e-5
cell_length_cm = 1.0
slit_width_mm = 1.0
instrument_scale = 1.0 / (concentration_M * cell_length_cm)

# —————————————————————
# Input parameters
# —————————————————————
R_values = [845.44, 938.79]     # nm, molecules peaks
lambdas = [343.96, 351.08]    # Rmax in 10⁻⁴⁰ cgs

# Creates the wavelenght net
lam_grid = np.linspace(min(lambdas) - 50,
                       max(lambdas) + 50,
                       2000)

# Gaussian parameters
sigma = iFWHM_nm / (2 * np.sqrt(2 * np.log(2)))

plt.figure(figsize=(6, 4))

# For each molecule, calculate and graphs it's individual spectra
for idx, (l0, R0) in enumerate(zip(lambdas, R_values), start=1):
    # Calculation of discrete Δε
    delta_eps = (conv * R0) / l0
    # Theoretic spectra
    spectrum = delta_eps * np.exp(-((lam_grid - l0)**2) / (2 * sigma**2))
    # Smoothness and scale
    exp_spectrum = gaussian_filter1d(spectrum * instrument_scale,
                                     slit_width_mm * 0.5)
    # Graph
    plt.plot(lam_grid, exp_spectrum, label=f"Molécula {idx} (λ₀={l0} nm)")


plt.axhline(0, color='k', lw=0.5)
plt.xlabel("λ/nm)")
plt.ylabel("Δε/M⁻¹·cm⁻¹")

x_min, x_max = lam_grid.min(), lam_grid.max()
y_min, y_max = plt.ylim()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tight_layout()
plt.show()
