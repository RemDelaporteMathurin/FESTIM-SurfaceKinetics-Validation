import festim as F
import sympy as sp
import fenics as f
import matplotlib.pyplot as plt
import numpy as np

# Create and mark the mesh


# Create the FESTIM model
my_model = F.Simulation()

my_model.mesh = F.MeshFromVertices(np.linspace(0, 1, 1000))

# Variational formulation
n_IS = 20
n_surf = 5
D = 5
lambda_IS = 2
k_bs = n_IS / n_surf
k_sb = 2 * n_IS / n_surf

solute_source = 2 * (1 - 2 * D)

exact_solution_cm = lambda x, t: 1 + 2 * x**2 + x + 2 * t
exact_solution_cs = lambda t: n_surf * (1 + 2 * t + 2 * lambda_IS - D) / (2 * n_IS - 1 - 2 * t)

solute_source = 2 * (1 - 2 * D)

my_model.sources = [F.Source(solute_source, volume=1, field="solute")]

def J_vs(T, surf_conc, t):
    return 2 * n_surf * (2 * n_IS + 2 * lambda_IS - D) / (2 * n_IS - 1 - 2 * t)**2 + 2 * lambda_IS - D

my_model.boundary_conditions = [
    F.DirichletBC(surfaces=[2], value=exact_solution_cm(x=F.x, t=F.t), field="solute"),
    F.SurfaceKinetics(
        k_sb=k_sb,
        k_bs=k_bs,
        lambda_IS=lambda_IS,
        n_surf=n_surf,
        n_IS=n_IS,
        J_vs=J_vs,
        surfaces=1,
        initial_condition=exact_solution_cs(t=0),
        t=F.t,
    ),
]

my_model.initial_conditions = [
    F.InitialCondition(field="solute", value=exact_solution_cm(x=F.x, t=F.t))
]

my_model.materials = F.Material(id=1, D_0=D, E_D=0)

my_model.T = 300

my_model.settings = F.Settings(
    absolute_tolerance=1e-10, relative_tolerance=1e-10, transient=True, final_time=5
)

my_model.dt = F.Stepsize(initial_value=1e-2)

export_times = [1, 3, 5]
my_model.exports = [
    F.TXTExport("solute", filename="./mobile_conc.txt", times=export_times),
    F.DerivedQuantities([F.AdsorbedHydrogen(surface=1)]),
]

my_model.initialise()
my_model.run()

def norm(x, c_comp, c_ex):
    return np.sqrt(np.trapz(y=(c_comp-c_ex)**2, x=x))

data = np.genfromtxt("mobile_conc.txt", names=True, delimiter=",")
for t in export_times:
    x = data["x"]
    y = data[f"t{t:.2e}s".replace(".", "").replace("+", "")]
    # order y by x
    x, y = zip(*sorted(zip(x, y)))

    (l1,) = plt.plot(
        x,
        exact_solution_cm(np.array(x), t),
        "--",
        label=f"exact t={t}",
    )
    plt.scatter(
        x[::100],
        y[::100],
        label=f"t={t}",
        color=l1.get_color(),
    )

    print(f"L2 error for cm at t={t}: {norm(np.array(x), np.array(y), exact_solution_cm(x=np.array(x), t=t))}")

plt.legend(reverse=True)
plt.ylabel("$c_m$")
plt.xlabel("x")
plt.savefig("cm.png")

plt.figure()

c_s_computed = my_model.exports[1][0].data
t = my_model.exports[1][0].t

plt.plot(t, c_s_computed, label="computed")
plt.plot(t, exact_solution_cs(np.array(t)), label="exact")
plt.ylabel("$c_s$")
plt.xlabel("t")
plt.legend()
plt.savefig("cs.png")
plt.show()

print(f"L2 error for cs: {norm(t, c_s_computed, exact_solution_cs(t=np.array(t)))}")