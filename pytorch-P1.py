import torch

# Create a 4x4 tensor filled with ones
tensor = torch.ones(4, 4)

# Print the first row and column
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

# Set the second column to zero
tensor[:, 1] = 0
print(tensor)

# Matrix multiplication between tensor and its transpose
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# Element-wise multiplication
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Printing the results
print("Matrix multiplication results:")
print(f"y1: \n{y1}")
print(f"y2: \n{y2}")
print(f"y3 (with out parameter): \n{y3}")

print("\nElement-wise multiplication results:")
print(f"z1: \n{z1}")
print(f"z2: \n{z2}")
print(f"z3 (with out parameter): \n{z3}")
