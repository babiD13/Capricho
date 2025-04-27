import numpy as np

# I. Creating Matrices
print("Initials: BEC -> [2, 5, 3] \nSecond letters: ORA -> [15, 19, 1]")

# Name : Bob Dylan Ereno Capricho
# Student Number: 2023-04-0067

# Initials BEC: B=2, E=5, C=3
# Second letters of names ORA: O=15, R=19, A=1
matrix1 = np.array([[2, 5, 3], [15, 19, 1]])
matrix2 = np.array([[2, 0, 2], [0, 6, 7]])

# Print both matrices
print("\nI. Printing both matrices:")
print("First Matrix:")
print(matrix1)
print("\nSecond Matrix:")
print(matrix2)

# II. Matrix Addition
print("\nII. Matrix Addition")

# 3rd matrix
matrix3 = matrix1 + matrix2

# Print the 3rd matrix
print("\nThird Matrix (Matrix1 plus Matrix2):")
print(matrix3)

# III. Scalar Multiplication
print("\nIII. Scalar Multiplication")

# Multiply 1st matrix by a scalar value of 2
matrix4 = matrix1 * 2

# Print the 4th matrix
print("\nFourth (First Matrix multiplied by 2):")
print(matrix4)

# IV. Transpose of a Matrix
print("\nIV. Transpose of a Matrix")

# Create a transposed 5th matrix from the 2nd matrix
matrix5 = np.transpose(matrix2)

# Print the 5th matrix
print("\nFifth Matrix (Transpose of Second Matrix):")
print(matrix5)

# V. Matrix Multiplication
print("\nV. Matrix Multiplication")

# 5.a. Multiply the 3rd matrix and 5th matrix to create a 6th matrix
matrix6 = np.matmul(matrix3, matrix5)

# 5.b. Print the 6th Matrix
print("\nSixth Matrix (Product of Third and Fifth Matrix):")
print(matrix6)

# VI. Sum of All Elements
print("\nVI. Sum of All Elements")

# 6.a. Sum of all elements in the 3rd matrix
sum_elements = np.sum(matrix3)

# 6.b. Print the sum
print("\nSum of all elements in Third Matrix:")
print(sum_elements)

# VII. Zero Matrix
print("\nVII. Zero Matrix")

# 7.a. Create a 2x3 zero matrix
matrix7 = np.zeros((2, 3))

# 7.b. Print the 7th matrix
print("\nSeventh Matrix (2x3 Zero Matrix):")
print(matrix7)