import numpy as np

# arrays
a = np.array([4, 3, 2, 1])
print("Array:")
print(a)
print()
m = np.array([[1,2,3],[10,20,30],[100,200,300],[1000,2000,3000]])
print("2D array:")
print(m)
print()

r = np.arange(5)
print("Range array arange(5):")
print(r)
print()

# identity matrix
e = np.eye(4, 3)
print("Identity matrix 4x3:")
print(e)
print()

# row concatenation, c_ - column concatenation
r_conc = np.r_[r, a]
print("Concatenated range and array:")
print(r_conc)
print()

r_conc = np.r_[e, m]
print("Concatenated eye and matrix rows:")
print(r_conc)
print()


# getting information about the arrays
print("Matrix has ", np.ndim(m), "dimensions")
print("Matrix has ", np.size(m), "elements")
print("Matrix has ", np.shape(m), "elements in each dimension")

reshaped_a = np.reshape(a,(2,2))
print("Reshaped a:")
print(reshaped_a)
print()

reshaped_e = np.reshape(e,(2,-1)) #-1 means as many as required
print("Reshaped eye:")
print(reshaped_e)
print()

ravel_m = np.ravel(m)
print("Ravel makes matrix m 1-dimensional:")
print(ravel_m)
print()

transp_m = np.transpose(m)
print("Transposed matrix m:")
print(transp_m)
print()

rev_m = m[::-1]
print("Reversed rows in matrix m:")
print(rev_m)
print()

m_plus_e = m + e
print("Sum of matrices m+e:")
print(m_plus_e)
print()

m_mult_e = m*e
print("Matrix multiplication m*e:")
print(m_mult_e)
print()

b = [1,2,3,4]
dot_product = np.dot(a,b)
print("Dot product [4,3,2,1]*[1,2,3,4]:")
print(dot_product)
print()

logic = np.where(a>2, 1, 0)
print("Replaced values >2 by True/False")
print(logic)
print()

