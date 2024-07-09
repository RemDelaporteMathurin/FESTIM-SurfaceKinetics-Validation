import festim as F
import sympy as sp
import fenics as f
import matplotlib.pyplot as plt
import numpy as np

# Create and mark the mesh


# Create the FESTIM model
my_model = F.Simulation()

my_model.mesh = F.MeshFromVertices(np.linspace(0,1,100))

# Variational formulation
n_IS = 20
n_surf = 5
D = 2
lambda_IS = 10


exact_solution_cm = lambda x, t: 1 + 2 * x**2 + x + 2 * t
exact_solution_cs = lambda t: n_surf * (1 + 2 * t + 2 * lambda_IS - D) / n_IS

k_bs = n_IS / n_surf
k_sb = n_IS / n_surf

solute_source = 2 * (1 - 2 * D)

my_model.sources = [F.Source(solute_source, volume=1, field="solute")]

my_model.boundary_conditions = [
    F.DirichletBC(surfaces=[2], value=exact_solution_cm(x=F.x, t=F.t), field="solute"),
    F.SurfaceKinetics(
        k_sb=k_sb,
        k_bs=k_bs,
        lambda_IS=lambda_IS,
        n_surf=n_surf,
        n_IS=n_IS,
        J_vs=n_surf * 2 / n_IS + 2 * lambda_IS - D,
        surfaces=1,
        initial_condition=exact_solution_cs(t=0),
        t = F.t
    ),
]

my_model.initial_conditions = [F.InitialCondition(field="solute", value=exact_solution_cm(x=F.x, t=0))]

my_model.materials = F.Material(id=1, D_0=D, E_D=0)

my_model.T = 300

my_model.settings = F.Settings(
    absolute_tolerance=1e-10,
    relative_tolerance=1e-10,
    transient=True,
    final_time=5
)

my_model.dt = F.Stepsize(0.01)

export_times = [1,3,5]
my_model.exports = [
    F.TXTExport("solute", filename="./mobile_conc.txt", times=export_times),
    F.DerivedQuantities([F.AdsorbedHydrogen(surface=1)]),
]

my_model.initialise()
my_model.run()

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
        x[::10],
        y[::10],
        label=f"t={t}",
        color=l1.get_color(),
    )

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
