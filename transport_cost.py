import cvxpy as cp
import numpy as np

# Coûts de transport
costs = np.array([
    [4, 6, 9, 5],
    [7, 3, 4, 8],
    [5, 8, 6, 7]
], dtype=np.float32)

# Capacités des entrepôts
capacities = np.array([100, 150, 200], dtype=np.float32)

# Demandes des magasins
demands = np.array([80, 120, 150, 100], dtype=np.float32)

# Variables de décision
x = cp.Variable((3, 4), nonneg=True)

# Fonction de coût
total_cost = cp.sum(cp.multiply(costs, x))

# Contraintes
constraints = [
    cp.sum(x, axis=1) <= capacities,  # Contraintes des capacités des entrepôts
    cp.sum(x, axis=0) >= demands      # Contraintes des demandes des magasins
]

# Problème d'optimisation
problem = cp.Problem(cp.Minimize(total_cost), constraints)

# Résoudre le problème
problem.solve()

# Affichage du plan de transport optimisé
print("Optimized transport plan:")
print(x.value)
