import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- Data Generation ---
num_rows = 100
num_columns = 50 
latent_dim = 20  # Hidden dimensionality of the data
rank_approx = 10  # Desired rank for matrix approximation

# Simulated low-rank matrices (for illustration)
true_matrix_A = np.random.rand(num_rows, latent_dim)
true_matrix_B = np.random.rand(latent_dim, num_columns)
noisy_matrix = true_matrix_A @ true_matrix_B + 0.1 * np.random.randn(num_rows, num_columns) 
print(f'Shape of the noisy matrix: {noisy_matrix.shape}')

# The input is an identity matrix, which simplifies the matrix decomposition task
input_data = torch.from_numpy(np.eye(num_rows)).float()

# Original noisy matrix serves as the target output (label)
target_matrix = torch.from_numpy(noisy_matrix).float()

# --- Defining the Model ---
class MatrixFactorizationModel(nn.Module):
    def __init__(self, rows, cols, rank):
        super().__init__()
        self.A = nn.Linear(rows, rank, bias=False)
        self.B = nn.Linear(rank, cols, bias=False)

    def forward(self, x):
        return self.B(self.A(x))

model = MatrixFactorizationModel(num_rows, num_columns, rank_approx)
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

# --- Model Training ---
num_epochs = 500
for epoch in range(num_epochs):
    predicted_output = model(input_data)
    loss_value = loss_function(predicted_output, target_matrix)

    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Loss = {loss_value.item():.4f}')

# --- Storing Factor Matrices ---
A_matrix = model.A.weight.detach().numpy().transpose()
B_matrix = model.B.weight.detach().numpy().transpose()
print(f'Shape of A_matrix: {A_matrix.shape}')
print(f'Shape of B_matrix: {B_matrix.shape}')
np.save('A_matrix.npy', A_matrix)
np.save('B_matrix.npy', B_matrix)

# --- Loading Factor Matrices ---
A_matrix = np.load('A_matrix.npy')
B_matrix = np.load('B_matrix.npy')
print(f'Loaded A_matrix shape: {A_matrix.shape}')
print(f'Loaded B_matrix shape: {B_matrix.shape}')
reconstructed_matrix = A_matrix @ B_matrix

# --- Performance Evaluation ---
compression_rate = (num_rows * num_columns)  / (num_rows * rank_approx + rank_approx * num_columns)
print(f'Compression Rate: {compression_rate:.2f}')

matrix_error = np.linalg.norm(noisy_matrix - reconstructed_matrix)
print(f'Matrix Approximation Error (Frobenius Norm): {matrix_error:.4f}')

absolute_error = np.abs(noisy_matrix - reconstructed_matrix)
relative_error = absolute_error / (np.abs(noisy_matrix) + 1e-8)
percentage_error = 100 * np.mean(relative_error)
print(f'Approximation Error (Average Percentage): {percentage_error:.2f}%')

# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(noisy_matrix)
plt.title(f'Noisy Matrix: {num_rows}x{num_columns}')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_matrix)
plt.title(f'Reconstructed: {num_rows}x{rank_approx}+{rank_approx}x{num_columns}')
plt.show()
