import os, sys
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from pyLIMA import event, telescopes
from pyLIMA.simulations import simulator
from astropy import units as u

# from pyLIMA.fits import TRF_fit
from pyLIMA.models import FSPLarge_model,USBL_model
# FSPL_model,USBL_model,PSPL_model,
# from ipywidgets import interactive, HBox, VBox, Layout
# from ipywidgets import (FloatSlider, FloatLogSlider, interactive_output, HBox, VBox, GridBox, Layout, Label)
# from IPython.display import display
current_path = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_path, os.pardir))
print("Parent Directory:", parent_directory)
sys.path.append(parent_directory)
sys.path.append("/home/anibal/microlensing/simulation_Rubin/roman_rubin/")
from ulens_params import microlensing_params
#from astropy import constants as C
# from pyLIMA.xallarap.xallarap import xallarap_shifts, compute_xallarap_curvature
# import scipy.optimize as so

simulated_event = event.Event()
simulated_event.name = 'Simulated'
simulated_event.ra = 170
simulated_event.dec = -70

path_ephemerides = "/home/anibal/microlensing/simulation_Rubin/roman_rubin/ephemerides/Roman_positions.npy"
ephemerides = np.load(path_ephemerides)
t0 = np.mean(ephemerides[:,0])
tE = 25
t1 = np.arange(t0-1.5*tE, t0+1.5*tE, 12/(24*60))
t2 = np.arange(t0-1.5*tE, t0+1.5*tE, 20/(24*60*60))
lightcurve_simroman = np.c_[t1, np.full_like(t1, 19.0), np.full_like(t1, 0.000000001)]
lightcurve_simodi = np.c_[t2, np.full_like(t2, 19.0), np.full_like(t2, 0.000000001)]

tel_earth = telescopes.Telescope(
    name='ODI',
    camera_filter='G',
    lightcurve=lightcurve_simodi.astype(float),
    lightcurve_names=['time','mag','err_mag'],
    lightcurve_units=['JD','mag','mag'],
    location='Earth'
)
simulated_event.telescopes.append(tel_earth)

tel_space= telescopes.Telescope(
    name='Roman',
    camera_filter='G',
    lightcurve=lightcurve_simroman.astype(float),
    lightcurve_names=['time','mag','err_mag'],
    lightcurve_units=['JD','mag','mag'],
    location='Space'
)

tel_space.spacecraft_name = 'Roman'
tel_space.spacecraft_positions = {'astrometry': [], 'photometry': ephemerides}
simulated_event.telescopes.append(tel_space)
# model_FSPL = FSPLarge_model.FSPLargemodel(simulated_event, parallax=['Full', t0]) 
model_FSPL = USBL_model.USBLmodel(simulated_event,origin=["central_caustic", [0, 0]], parallax=['Full', t0])


model_FSPL.define_model_parameters()
# %%


M_planet=float(1*u.M_earth.to("M_jup")+1*u.M_sun.to("M_jup"))
print(M_planet)
DS = 8000
DL = 4000
params = microlensing_params("a", 0, 0,DL,0, M_planet, DS,5, 0, 0)
thetaE = params.theta_E()

# plt.figure()
# plt.plot(M_planet,thetaE)
# plt.show()

print(thetaE)
u0=0.00005

piE = params.piE()
angle=np.pi/2
piEE=np.cos(angle)*piE
piEN=np.sin(angle)*piE
theta_star = (1*u.R_sun/(DS*u.pc))*u.rad.to("mas")*u.mas
rho=(theta_star/thetaE).decompose()
print("rho",rho)
fs1=200
fb1=0
fs2=200
fb2=0
# params_list = [
#     float(t0), float(u0), float(tE), float(rho),
#     float(piEE), float(piEN),
#     float(fs1), float(fb1), float(fs2), float(fb2)
# ]
q = (1*u.M_earth/(1*u.M_sun)).decompose()
s = 1
alpha=np.pi/2

params_list = [
    float(t0), float(u0), float(tE), float(rho),
    float(s),float(q), float(alpha),
    float(piEE), float(piEN),
    float(fs1), float(fb1), float(fs2), float(fb2)
]


pyLIMA_parameters= model_FSPL.compute_pyLIMA_parameters(params_list)

simulator.simulate_lightcurve(model_FSPL, pyLIMA_parameters)
 
# def mag(zp, Flux):
#     return zp - 2.5 * np.log10(np.abs(Flux))
# for k in range(len(simulated_event.telescopes)):
#     model_flux = model_FSPL.compute_the_microlensing_model(simulated_event.telescopes[k],                                                              pyLIMA_parameters)['photometry']
#     simulated_event.telescopes[k].lightcurve['flux'] = model_flux #con esto el flujo es el teorico sin errores
#     simulated_event.telescopes[k].lightcurve['mag'] = mag(27,model_flux)
 
A_1= model_FSPL.model_magnification(
    model_FSPL.event.telescopes[0], pyLIMA_parameters
)
A_2= model_FSPL.model_magnification(
    model_FSPL.event.telescopes[1], pyLIMA_parameters
)
# %%
plt.axvline(pyLIMA_parameters["t_center"])
plt.plot(t2,A_1,label= "Earth")
plt.plot(t1,A_2,label= "Space")
plt.legend()
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy import units as u

# ============================================================
# Grid de masas
# ============================================================
m_list = np.logspace(-2, 1, 30)  # masas en M_earth

# ============================================================
# Distancias DL
# ============================================================
DL_values = np.linspace(100, 7000, 50)

# ============================================================
# Colormap y normalización
# ============================================================
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(
    vmin=DL_values.min(),
    vmax=DL_values.max()
)

# ============================================================
# Figura
# ============================================================
fig, ax = plt.subplots(figsize=(8,6))

# ============================================================
# Loop sobre DL
# ============================================================
for DL in DL_values:
    theta_list = []
    for m in m_list:
        M_planet = float(m * u.M_earth.to("M_jup"))
        params = microlensing_params(
            "a",
            0,
            0,
            DL,
            0,
            M_planet,
            DS,
            5,
            0,
            0
        )
        thetaE = params.theta_E()
        theta_list.append(thetaE.value)
    # Color según DL
    color = cmap(norm(DL))

    ax.plot(
        m_list,
        theta_list,
        color=color,
        alpha=0.9,
        lw=1.5
    )

# ============================================================
# Escalas
# ============================================================
ax.set_xscale("log")
ax.set_yscale("log")

# ============================================================
# Labels
# ============================================================
ax.set_xlabel(r"$M_{\rm planet}\ [M_\oplus]$", fontsize=14)
ax.set_ylabel(r"$\theta_E$", fontsize=14)

# ============================================================
# Colorbar
# ============================================================
sm = mpl.cm.ScalarMappable(
    cmap=cmap,
    norm=norm
)

sm.set_array([])

cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r"$D_L\ [{\rm pc}]$", fontsize=13)

# ============================================================
# Grid
# ============================================================
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
# %%
import astropy.units as u
print((2*4e-2*u.day.to("minute"))/15)

