import numpy as np
import matplotlib.pyplot as plt

en = np.genfromtxt("dat")
it = np.arange(1, 100001, 1)

conved_en = np.mean(en[-1000:] / it[-1000:])
delta = conved_en - en[1:] / it

fig, ax = plt.subplots()

ax.plot(it, delta, "o")
ax.set_yscale("log")
ax.set_xlabel(r"$N_{MC}$")
ax.set_ylabel(r"$\log \Delta E$")

ax.set_title("Convergence de l'énergie potentielle en chaine de Markov")
plt.show()
