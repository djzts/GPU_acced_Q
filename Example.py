import torch
import simulated_bifurcation as sb

# Define a simple 2D polynomial
Q = torch.tensor([[1.0, -2.0],
                  [0.0,  3.0]], dtype=torch.float32)

# Minimize the polynomial in a binary domain:
best_vector, best_val = sb.minimize(Q, input_type="binary",device="cuda")
print("Best vector (binary domain):", best_vector)
print("Best value:", best_val)


best_vector, best_val = sb.maximize(Q, input_type="spin")
print("Best vector (spin domain):", best_vector)
print("Best value:", best_val)

# Define the numerical matrix Q, vector l, and constant c
Q = torch.tensor([
    [1.0,  0.5,  0.0],
    [0.5, -1.0, -1.0],
    [0.0, -1.0,  2.0]
], dtype=torch.float32)

l = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32)
c = 2.0

# Build the polynomial model for the spin domain (values in {-1, 1})
poly_model = sb.build_model(Q, vector=l, constant=c, input_type="binary")

# Run the optimization (minimize in this case)
all_vectors, all_values = poly_model.minimize(agents=64, best_only=False)

print("All solutions shape:", all_vectors.shape)
print("All evaluations shape:", all_values.shape)