import numpy as np

# Transition matrix P
P = np.array([
    [0, 0, 0, 0, 0.5, 0, 0, 0.5, 0, 0],     # A
    [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0],     # B
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],         # C
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],         # D
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],         # E
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],         # F
    [0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0],     # G
    [0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0],     # H
    [0, 1/6, 0, 0, 1/6, 1/6, 1/6, 1/6, 0, 1/6], # I
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]          # J
])

def compute_limiting_distribution_class(P, indices):
    """Compute limiting distribution for a recurrent class"""
    n = len(indices)
    if n == 1:
        return np.array([1.0])
    
    # Extract submatrix
    P_sub = P[np.ix_(indices, indices)]
    
    # Solve πP = π and Σπ = 1
    A = (P_sub.T - np.eye(n))
    A[-1] = np.ones(n)  # Replace last equation with Σπ = 1
    b = np.zeros(n)
    b[-1] = 1
    
    return np.linalg.solve(A, b)

def compute_absorption_probabilities(P, transient_states, recurrent_classes):
    """Compute absorption probabilities for transient states into each recurrent class"""
    n_transient = len(transient_states)
    n_classes = len(recurrent_classes)
    
    # For each recurrent class
    probs = np.zeros((n_transient, n_classes))
    
    # Extract Q (transitions between transient states)
    Q = P[np.ix_(transient_states, transient_states)]
    
    # For each recurrent class
    for i, rec_class in enumerate(recurrent_classes):
        # Extract R (transitions from transient to this recurrent class)
        R = P[np.ix_(transient_states, rec_class)]
        R_sum = R.sum(axis=1).reshape(-1, 1)
        
        # Solve (I-Q)x = R1
        N = np.linalg.inv(np.eye(n_transient) - Q)
        probs[:, i] = N @ R_sum.flatten()
    
    return probs

# Define the recurrent classes and transient states
class1 = [0,3,4,7]  # A,D,E,H
class2 = [2,5,6]    # C,F,G
class3 = [9]        # J
transient = [1,8]   # B,I

# Compute limiting distributions for each recurrent class
pi1 = compute_limiting_distribution_class(P, class1)
pi2 = compute_limiting_distribution_class(P, class2)
pi3 = compute_limiting_distribution_class(P, class3)

# Compute absorption probabilities
abs_probs = compute_absorption_probabilities(P, transient, [class1, class2, class3])

# Build complete limiting matrix
P_limit = np.zeros((10, 10))

# Fill in the limiting distributions for each recurrent class
print("\nLimiting distributions within recurrent classes:")
print("\nClass {A,D,E,H}:")
for i, idx in enumerate(class1):
    for j, jdx in enumerate(class1):
        P_limit[idx, jdx] = pi1[j]
    print(f"π_{chr(65+idx)} = {pi1[i]:.6f}")

print("\nClass {C,F,G}:")
for i, idx in enumerate(class2):
    for j, jdx in enumerate(class2):
        P_limit[idx, jdx] = pi2[j]
    print(f"π_{chr(65+idx)} = {pi2[i]:.6f}")

print("\nClass {J}:")
P_limit[9,9] = 1
print("π_J = 1.000000")

print("\nAbsorption probabilities for transient states:")
states = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
for i, t_state in enumerate(transient):
    print(f"\nState {states[t_state]}:")
    print(f"To class {{A,D,E,H}}: {abs_probs[i,0]:.6f}")
    print(f"To class {{C,F,G}}: {abs_probs[i,1]:.6f}")
    print(f"To class {{J}}: {abs_probs[i,2]:.6f}")

print("\nComplete limiting matrix P^∞:")
print("     A        B        C        D        E        F        G        H        I        J")
for i in range(10):
    print(f"{chr(65+i)}: ", end="")
    for j in range(10):
        print(f"{P_limit[i,j]:.6f} ", end="")
    print()
