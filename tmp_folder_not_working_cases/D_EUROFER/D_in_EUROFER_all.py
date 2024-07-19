import festim as F
import fenics as f
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import warnings
from scipy import special

warnings.filterwarnings("ignore")

################### PARAMETERS ###################
N_A_const = 6.022e23  # Avogadro, mol^-1
e = 1.602e-19
M_D2 = 4.028e-3 / N_A_const  # the D2 mass, kg mol^-1

# Exposure conditions
P_D2 = 1  # Pa
T_load = 370  # D loading temperature, K
T_storage = 290  # temperature after cooling phase, K
ramp = 3 / 60  # TDS heating rate, K/s
# t_load = 143 * 3600  # exposure duration, s
t_cool = 1000  # colling duration, s
t_storage = 24 * 3600  # storage time, s
t_TDS = (800 - T_storage) / ramp  # TDS duration (up to 800 K), s
cooling_rate = (T_load - T_storage) / t_cool  # cooling rate, K/s

# Sample
L = 0.8e-3  # half thickness, m
A = 12e-3 * 15e-3  # surface area, m^2

# EUROFER properties
rho_EFe = 8.59e28  # EUROFER atomic concentration, m^-3
n_IS = 6 * rho_EFe  # concentration of interstitial sites, m^-3
n_surf = rho_EFe ** (2 / 3)  # concentration of adsorption sites, m^-2
lambda_lat = rho_EFe ** (-1 / 3)  # Typical lattice spacing, m

D0 = 1.5e-7  # diffusivity pre-factor, m^2 s^-1
E_diff = 0.15  # diffusion activation energy, eV

# Energy landscape
E_bs = E_diff  # energy barrier from bulk to surface transition, eV
nu_bs = D0 / lambda_lat**2  # attempt frequency for b-to-s transition, s^-1
E_diss = 0.4  # energy barrier for D2 dissociation, eV
E_rec = 0.7  # energy barrier for D2 recombination, eV
E_sol = 0.238  # heat of solution, eV
S0 = 1.5e-6
Xi0 = 1e-4
chi0 = 1e-6
E_sb = (
    E_rec / 2 - E_diss / 2 + 0.238 + E_diff
)  # energy barrier from bulk to surface transition, eV

# Trap properties
nu_tr = D0 / lambda_lat**2  # trapping attempt frequency, s^-1
nu_dt = 2.0e13  # detrapping attempt frequency, s^-1
E_tr = E_diff
E_dt_intr = 0.9  # detrapping energy for intrinsic traps, eV
E_dt_dpa = 1.1  # detrapping energy for DPA traps, eV

# Implantation parameters
Gamma = 9e19  # irradiation flux, m^-2 s^-1
R = 4.0e-10  # implantation range, m
# sigma = 2.35e-10 # standart deviation, m
sigma = 1 / np.sqrt(2 * 1.77778e6) * 1e-6

r = 0.612  # reflection coefficient


################### FUNCTIONS ###################
def Xi(T):
    # unitless
    return Xi0 * f.exp(-E_diss / F.k_B / T)


def chi(T):
    # in m^2 s^-1
    return chi0 * f.exp(-E_rec / F.k_B / T)


def S(T):
    # solubility
    return S0 * f.exp(-E_sol / F.k_B / T)


def Psi(T):
    return 1 / f.sqrt(2 * np.pi * M_D2 * F.k_B * T * e)


def k_bs(T, surf_conc, t):
    return nu_bs * f.exp(-E_bs / F.k_B / T)


def k_sb(T, surf_conc, t):
    # see eqs. (13-14) in K. Schmid and M. Zibrov 2021 Nucl. Fusion 61 086008
    return k_bs(T, surf_conc, t) * S(T) * n_surf * f.sqrt(chi(T) / Psi(T) / Xi(T))


def norm_flux(X, sigma):
    return 2 / (
        special.erf((0.8e-3 - X) / np.sqrt(2) / sigma)
        + special.erf((X) / np.sqrt(2) / sigma)
    )


################### MODEL ###################
def run_simulation(t_load, is_dpa):

    def J_vs(T, surf_conc, t):

        tt = 0.002 * (t - t_load)
        cond = 0.5 - 0.5 * (f.exp(2 * tt) - 1) / (f.exp(2 * tt) + 1)

        J_diss = (
            2 * P_D2 * Xi(T) * (1 - surf_conc / n_surf) ** 2 * Psi(T)
        )  # dissociation flux

        J_rec = 2 * chi(T) * surf_conc**2  # recombination flux

        Omega_loss = 8e4
        J_loss = (
            (surf_conc / n_surf) * Omega_loss * Gamma * (1 - r)
        )  # ad hoc flux for fit

        J_net = (J_diss - J_loss) * cond - J_rec
        return J_net

    EFe_model = F.Simulation(log_level=40)

    # Mesh
    vertices = np.concatenate(
        [
            np.linspace(0, 1e-8, num=100),
            np.linspace(1e-8, 4e-6, num=200),
            np.linspace(4e-6, L, num=200),
        ]
    )

    EFe_model.mesh = F.MeshFromVertices(np.sort(vertices))

    EFe_model.materials = [F.Material(id=1, D_0=D0, E_D=E_diff)]

    surf_conc1 = F.SurfaceKinetics(
        k_sb=k_sb,
        k_bs=k_bs,
        lambda_IS=lambda_lat,
        n_surf=n_surf,
        n_IS=4 * rho_EFe,
        J_vs=J_vs,
        surfaces=[1, 2],
        initial_condition=0,
        t=F.t,
    )

    EFe_model.boundary_conditions = [surf_conc1]

    trap_intr = F.Trap(
        k_0=nu_tr / n_IS,
        E_k=E_tr,
        p_0=nu_dt,
        E_p=E_dt_intr,
        density=1e-5 * rho_EFe,
        materials=EFe_model.materials[0],
    )
    trap_dpa = F.Trap(
        k_0=nu_tr / n_IS,
        E_k=E_tr,
        p_0=nu_dt,
        E_p=E_dt_dpa,
        # density=sp.Piecewise((0.25e-3 * rho_EFe, F.x <= 3.3e-6), (0, True)),
        # density=0.25e-3 * rho_EFe * (1/(1 + sp.exp((F.x-3e-6)*5e6))),
        density=(
            2.5e-4 + (1e-12 - 2.5e-4) * (0.5 + 0.5 * sp.tanh(100e6 * (F.x - 3.3e-6)))
        )
        * rho_EFe,
        materials=EFe_model.materials[0],
    )

    EFe_model.traps = [trap_intr]
    if is_dpa:
        EFe_model.traps = [trap_intr, trap_dpa]

    # EFe_model.initial_conditions = [F.InitialCondition(field="1", value=1e-5*rho_EFe)]

    EFe_model.sources = [
        F.ImplantationFlux(
            flux=Gamma
            * (1 - r)
            * norm_flux(R, sigma)
            * (0.5 - 0.5 * sp.tanh(0.002 * (F.t - t_load))),
            imp_depth=R,
            width=sigma,
            volume=1,
        )
    ]

    log_bis = sp.Function("std::log")

    a1 = sp.cosh(3011 - 0.005 * (F.t + (143 * 3600 - t_load)))
    a2 = sp.cosh(3016 - 0.005 * (F.t + (143 * 3600 - t_load)))
    a3 = sp.cosh(3021.5 - 0.005 * (F.t + (143 * 3600 - t_load)))
    a4 = sp.cosh(3036.5 - 0.005 * (F.t + (143 * 3600 - t_load)))
    a5 = sp.cosh(3063.5 - 0.005 * (F.t + (143 * 3600 - t_load)))
    fun = (
        543.708
        + 0.339015 * log_bis(a1)
        + 3.66105 * log_bis(a2)
        + 1.99939 * log_bis(a3)
        - 0.806 * log_bis(a4)
        - 5.194 * log_bis(a5)
    )

    a1 = log_bis(sp.cosh(0.005 * (-612700 + F.t)))
    a2 = log_bis(sp.cosh(0.005 * (-607300 + F.t)))
    a3 = log_bis(sp.cosh(0.005 * (-603200 + F.t)))
    a4 = log_bis(sp.cosh(0.005 * (-603200 + F.t)))
    a5 = log_bis(sp.cosh(0.005 * (-602200 + F.t)))

    fun = (
        293.55
        + 50
        * (
            0
            - 0.05194 * a1
            + 0.05194
            * (
                -3035.806852819440
                - (3035.806852819440 + a1)
                + 2 * (3062.806852819440 + a2)
            )
        )
        + 50
        * (
            0
            - 0.06 * a2
            + 0.06
            * (
                -3020.806852819440
                - (3020.806852819440 + a2)
                + 2 * (3035.806852819440 + a3)
            )
        )
        + 50
        * (
            0
            - 0.04 * a3
            + 0.04
            * (
                -3015.306852819440
                - (3015.306852819440 + a3)
                + 2.00003 * (3020.806852819440 + a4)
            )
        )
        + 50
        * (
            0
            - 0.00339 * a4
            + 0.00339
            * (
                -3010.30685
                - (3010.306852819440 + a4)
                + 2.00009 * (3015.306852819440 + a5)
            )
        )
        + 76.45 * 0.5 * (1 - sp.tanh(0.002 * (F.t - 515800)))
    )

    EFe_model.T = F.Temperature(
        value=sp.Piecewise(
            (T_load, F.t <= t_load),
            (T_load - cooling_rate * (F.t - t_load), F.t <= t_load + t_cool),
            (T_storage, F.t <= t_load + t_cool + t_storage),
            (fun, True),
        )
    )

    def step_size(t):
        if t <= t_load:
            return 500
        elif t > t_load and t <= t_load + t_cool + t_storage:
            return 1000
        else:
            return 40

    EFe_model.dt = F.Stepsize(
        initial_value=1e-4,
        stepsize_change_ratio=1.1,
        max_stepsize=step_size,
        dt_min=1e-5,
        milestones=[
            t_load,
            t_load + t_cool,
            t_load + t_cool + t_storage,
        ],
    )

    EFe_model.settings = F.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-9,
        maximum_iterations=50,
        final_time=t_load + t_cool + t_storage + t_TDS,
    )

    derived_quantities = F.DerivedQuantities(
        [
            F.AdsorbedHydrogen(surface=1),
            F.AdsorbedHydrogen(surface=2),
            F.TotalSurface(field="T", surface=1),
            F.TotalVolume(field="retention", volume=1),
        ],
    )

    EFe_model.exports = [derived_quantities]

    EFe_model.initialise()
    EFe_model.run()

    return derived_quantities


params = {
    0: {"t_load": 143 * 3600, "is_dpa": False, "exp_data": "143hplasma"},
    1: {"t_load": 48 * 3600, "is_dpa": True, "exp_data": "DPA+48hplasma"},
    2: {"t_load": 143 * 3600, "is_dpa": True, "exp_data": "DPA+143hplasma"},
}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

for i in range(3):

    results = run_simulation(t_load=params[i]["t_load"], is_dpa=params[i]["is_dpa"])

    surf_conc1 = np.array(results[0].data)
    surf_conc2 = np.array(results[1].data)
    T = np.array(results[2].data)
    retention = np.array(results[3].data)
    t = np.array(results.t)

    t_load = params[i]["t_load"]
    exp_label = params[i]["exp_data"]

    Flux_left = surf_conc1**2 * chi0 * np.exp(-E_rec / F.k_B / T)
    Flux_right = surf_conc2**2 * chi0 * np.exp(-E_rec / F.k_B / T)

    exp = pd.read_csv(f"./exp_data/{exp_label}.csv", header=None, skiprows=1, sep=",")

    a1 = np.cosh(3011 - 0.005 * (t + (143 * 3600 - t_load)))
    a2 = np.cosh(3016 - 0.005 * (t + (143 * 3600 - t_load)))
    a3 = np.cosh(3021.5 - 0.005 * (t + (143 * 3600 - t_load)))
    a4 = np.cosh(3036.5 - 0.005 * (t + (143 * 3600 - t_load)))
    a5 = np.cosh(3063.5 - 0.005 * (t + (143 * 3600 - t_load)))
    T_exp = (
        543.708
        + 0.339015 * np.log(a1)
        + 3.66105 * np.log(a2)
        + 1.99939 * np.log(a3)
        - 0.806 * np.log(a4)
        - 5.194 * np.log(a5)
    )

    ax[i].plot(T_exp, Flux_left / 1e17, label="Flux_left")
    ax[i].plot(T_exp, Flux_right / 1e17, label="Flux_right")
    ax[i].plot(T_exp, (Flux_left + Flux_right) / 1e17, label="Total flux")

    ax[i].scatter(
        exp[0],
        exp[1] / 1e5,
        marker="x",
        s=75,
        linewidths=1.2,
        label=f"exp.: {exp_label}",
    )

    ax[i].set_xlim(300, 800)
    ax[i].set_ylim(0, 1.5)
    ax[i].set_xlabel("T, K")
    ax[i].set_ylabel(r"Flux, $10^5$ $\mu$m$^{-2}$s$^{-1}$")
    ax[i].legend()

plt.tight_layout()
plt.savefig("./Figure_1.png")
plt.show()
