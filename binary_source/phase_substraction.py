import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
t0 = 0.0
u0 = 0.15
tE = 50.0

xiE = 5.0
omega = 2*np.pi/30.0
phi = 1.0
t_ref = 0.0
sin_lambda = 1.0

t = np.linspace(-100, 100, 50000)

# -----------------------------
# Trajectories
# -----------------------------
def rect_trajectory(t, t0, u0, tE):
    tau = (t - t0)/tE
    beta = np.full_like(tau, u0, dtype=float)
    return tau, beta

def xallarap_shifts_no_sub(t):
    Omega = omega*(t - t_ref) + phi
    return xiE*np.cos(Omega), xiE*sin_lambda*np.sin(Omega)

def xallarap_shifts_with_sub(t):
    Omega = omega*(t - t_ref) + phi
    return xiE*(np.cos(Omega)-np.cos(phi)), \
           xiE*sin_lambda*(np.sin(Omega)-np.sin(phi))

tau_rect, beta_rect = rect_trajectory(t, t0, u0, tE)

d_tau_ns, d_beta_ns = xallarap_shifts_no_sub(t)
tau_ns = tau_rect + d_tau_ns
beta_ns = beta_rect + d_beta_ns

d_tau_s, d_beta_s = xallarap_shifts_with_sub(t)
tau_s = tau_rect + d_tau_s
beta_s = beta_rect + d_beta_s

u_rect = np.sqrt(tau_rect**2 + beta_rect**2)
u_ns   = np.sqrt(tau_ns**2   + beta_ns**2)
u_s    = np.sqrt(tau_s**2    + beta_s**2)

# -----------------------------
# Plot grid 1x2
# -----------------------------
#%matplotlib inline
plt.close("all")
fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=130)

# ---- Left: trajectory in (tau,beta)
axs[0].plot(tau_rect, beta_rect, label="PSPL")
axs[0].plot(tau_ns, beta_ns, label="No phase subtraction")
axs[0].plot(tau_s, beta_s, label="With phase subtraction")
axs[0].scatter([0], [u0], marker="x", s=70, label=r"Nominal $(0,u_0)$")
axs[0].set_xlabel(r"$\tau$")
axs[0].set_ylabel(r"$\beta$")
axs[0].set_title("Trajectory in $(\\tau,\\beta)$")
axs[0].grid(True)
axs[0].legend(fontsize=8)

# ---- Right: u(t)
axs[1].plot(t, u_rect, label="PSPL")
axs[1].plot(t, u_ns, label="No phase subtraction")
axs[1].plot(t, u_s, label="With phase subtraction")
axs[1].axvline(t0, linestyle="--", label=r"$t_0$")
axs[1].axvline(t_ref, linestyle=":", label=r"$t_{\rm ref}$")
axs[1].set_xlabel("time")
axs[1].set_ylabel(r"$u(t)$")
axs[1].set_title(r"$u(t)=\sqrt{\tau^2+\beta^2}$")
axs[1].grid(True)
axs[1].legend(fontsize=8)


#

def value_at_time(tgrid, arr, t_query):
    idx = np.argmin(np.abs(tgrid - t_query))
    return arr[idx], idx

# Values at t0
u_rect_t0, _ = value_at_time(t, u_rect, t0)
u_ns_t0, _   = value_at_time(t, u_ns, t0)
u_s_t0, _    = value_at_time(t, u_s, t0)

# Values at t_ref
u_rect_tref, _ = value_at_time(t, u_rect, t_ref)
u_ns_tref, _   = value_at_time(t, u_ns, t_ref)
u_s_tref, _    = value_at_time(t, u_s, t_ref)

# Mark points
axs[1].scatter([t0], [u_rect_t0])
axs[1].scatter([t0], [u_ns_t0])
axs[1].scatter([t0], [u_s_t0])

axs[1].scatter([t_ref], [u_rect_tref])
axs[1].scatter([t_ref], [u_ns_tref])
axs[1].scatter([t_ref], [u_s_tref])

# Annotate values
axs[1].annotate(rf"$u_{{\rm PSPL}}(t_0)={u_rect_t0:.3f}$",
                (t0, u_rect_t0),
                textcoords="offset points",
                xytext=(5,10),
                fontsize=8)

axs[1].annotate(rf"$u_{{\rm no-sub}}(t_0)={u_ns_t0:.3f}$",
                (t0, u_ns_t0),
                textcoords="offset points",
                xytext=(5,-15),
                fontsize=8)

# axs[1].annotate(rf"$u_{{\rm sub}}(t_0)={u_s_t0:.3f}$",
#                 (t0, u_s_t0),
#                 textcoords="offset points",
#                 xytext=(5,-35),
#                 fontsize=8)

axs[1].annotate(rf"$u_{{\rm sub}}(t_{{ref}}=t_0)={u_s_tref:.3f}$",
                (t_ref, u_s_tref),
                textcoords="offset points",
                xytext=(5,20),
                fontsize=8)
# -----------------------------
# Global title with parameters
# -----------------------------
fig.suptitle(
    rf"$t_0={t0},\ u_0={u0},\ t_E={tE},\ \xi_E={xiE},\ "
    rf"\omega={omega:.3f},\ \phi={phi},\ t_{{ref}}={t_ref},\ "
    rf"\sin\lambda={sin_lambda}$",
    fontsize=11
)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
