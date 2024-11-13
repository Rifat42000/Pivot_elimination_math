import numpy as np

# Define the augmented matrix [A|b]
matrix = np.array([[8, 3, -5],
                         [10, 7, 2],
                         [6, 4, 7]], dtype=float)
#A = XB

constant = np.array([3, 4, 8], dtype=float)

# Perform Gaussian elimination with partial pivoting
n = len(constant)
for i in range(n):
    # Partial pivoting: find the row with the maximum element in the current column
    max_row = np.argmax(np.abs(matrix[i:, i])) + i
    if i != max_row:
        # Swap rows in both A and b
        matrix[[i, max_row]] = matrix[[max_row, i]]
        constant[[i, max_row]] = constant[[max_row, i]]

    # Eliminate entries below the pivot
    for j in range(i+1, n):
        factor = matrix[j, i] / matrix[i, i]
        matrix[j, i:] -= factor * matrix[i, i:]
        constant[j] -= factor * constant[i]

# Back substitution to find the solution vector x
x = np.zeros(n)
for i in range(n-1, -1, -1):
    x[i] = (constant[i] - np.dot(matrix[i, i + 1:], x[i + 1:])) / matrix[i, i]

# Round the solution to 2 decimal places
x_rounded = np.round(x, 2)
print("Solution to the system:")
print("x =", x_rounded[0])
print("y =", x_rounded[1])
print("z =", x_rounded[2])
