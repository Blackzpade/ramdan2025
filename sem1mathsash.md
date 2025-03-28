```python
Write a python program to create a matrix using numpy.

```


```python
import numpy as np
n = int(input("Enter the number of rows: "));
matrix = []
for i in range(n):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix.append(row)
a = np.array(matrix)
print("The matrix you have entered is:")
print(a)

```

    Enter the number of rows:  2
    Enter the row 1 elements separated by spaces:  2 3
    Enter the row 2 elements separated by spaces:  5 6
    

    The matrix you have entered is:
    [[2 3]
     [5 6]]
    


```python
Write a python program to show matrix addition and subtraction.

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of first matrix: "))
n2 = int(input("Enter the number of rows of second matrix: "))
matrix1 = []
matrix2 = []
print("Enter the elements for the first matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print("Enter the elements for the second matrix: ")
for i in range(n2):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix2.append(row)
b = np.array(matrix2)
print(f"The matrices you have entered are: \n {a} \n\n {b}")
sum = a + b
diff = a - b
print(f"The sum of the two matrices is: \n {sum}")
print(f"The difference of the two matrices is: \n {diff}")

```

    Enter the number of rows of first matrix:  2
    Enter the number of rows of second matrix:  2
    

    Enter the elements for the first matrix: 
    

    Enter the row 1 elements separated by spaces:  2 5
    Enter the row 2 elements separated by spaces:  4 5
    

    Enter the elements for the second matrix: 
    

    Enter the row 1 elements separated by spaces:  9 6
    Enter the row 2 elements separated by spaces:  3 7
    

    The matrices you have entered are: 
     [[2 5]
     [4 5]] 
    
     [[9 6]
     [3 7]]
    The sum of the two matrices is: 
     [[11 11]
     [ 7 12]]
    The difference of the two matrices is: 
     [[-7 -1]
     [ 1 -2]]
    


```python
Write a python program to find the Hadamard product of two matrices.

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of first matrix: "))
n2 = int(input("Enter the number of rows of second matrix: "))
matrix1 = []
matrix2 = []
print("Enter the elements for the first matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print("Enter the elements for the second matrix: ")
for i in range(n2):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix2.append(row)
b = np.array(matrix2)
print(f"The matrices you have entered are: \n {a} \n\n {b}")
print(f"The Hadamard product of the two matrices is: \n {np.multiply(a, b)}")




```

    Enter the number of rows of first matrix:  2
    Enter the number of rows of second matrix:  2
    

    Enter the elements for the first matrix: 
    

    Enter the row 1 elements separated by spaces:  3 5
    Enter the row 2 elements separated by spaces:  6 7
    

    Enter the elements for the second matrix: 
    

    Enter the row 1 elements separated by spaces:  2 6
    Enter the row 2 elements separated by spaces:  9 3
    

    The matrices you have entered are: 
     [[3 5]
     [6 7]] 
    
     [[2 6]
     [9 3]]
    The Hadamard product of the two matrices is: 
     [[ 6 30]
     [54 21]]
    


```python
Write a python program to perform matrix division

```


```python
import numpy as np
np.set_printoptions(precision=3)
n1 = int(input("Enter the number of rows of first matrix: "))
n2 = int(input("Enter the number of rows of second matrix: "))
matrix1 = []
matrix2 = []
print("Enter the elements for the first matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print("Enter the elements for the second matrix: ")
for i in range(n2):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix2.append(row)
b = np.array(matrix2)
print(f"The matrices you have entered are: \n {a} \n\n {b}")
print(f"Matrices after division: \n {a / b}")

```

    Enter the number of rows of first matrix:  2
    Enter the number of rows of second matrix:  2
    

    Enter the elements for the first matrix: 
    

    Enter the row 1 elements separated by spaces:  6 4
    Enter the row 2 elements separated by spaces:  7 4
    

    Enter the elements for the second matrix: 
    

    Enter the row 1 elements separated by spaces:  1 5
    Enter the row 2 elements separated by spaces:  9 8
    

    The matrices you have entered are: 
     [[6 4]
     [7 4]] 
    
     [[1 5]
     [9 8]]
    Matrices after division: 
     [[6.    0.8  ]
     [0.778 0.5  ]]
    


```python
Write a python program to perform multiplication on two matrices.

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of first matrix: "))
n2 = int(input("Enter the number of rows of second matrix: "))
matrix1 = []
matrix2 = []
print("Enter the elements for the first matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print("Enter the elements for the second matrix: ")
for i in range(n2):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix2.append(row)
b = np.array(matrix2)
print(f"The matrices you have entered are: \n {a} \n\n {b}")
print(f"Matrices after multiplication: \n {np.dot(a, b)}")


```

    Enter the number of rows of first matrix:  2
    Enter the number of rows of second matrix:  2
    

    Enter the elements for the first matrix: 
    

    Enter the row 1 elements separated by spaces:  8 5
    Enter the row 2 elements separated by spaces:  2 3
    

    Enter the elements for the second matrix: 
    

    Enter the row 1 elements separated by spaces:  6 4
    Enter the row 2 elements separated by spaces:  7 8
    

    The matrices you have entered are: 
     [[8 5]
     [2 3]] 
    
     [[6 4]
     [7 8]]
    Matrices after multiplication: 
     [[83 72]
     [33 32]]
    


```python
Write a python program to perform scalar multiplication on two matrices.

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of the matrix: "))
matrix1 = []
print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print(f"The matrix you have entered is: \n {a}")
s = eval(input("Enter a value to multiply the matrix with: "))
print(f"The matrix after multiplication is: \n {a * s}")

```

    Enter the number of rows of the matrix:  2
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  8 6
    Enter the row 2 elements separated by spaces:  3 5
    

    The matrix you have entered is: 
     [[8 6]
     [3 5]]
    

    Enter a value to multiply the matrix with:  4
    

    The matrix after multiplication is: 
     [[32 24]
     [12 20]]
    


```python
Write a python program to generate the upper triangular matrix and lower triangular matrix from a regular matrix

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of the matrix: "))
matrix1 = []
print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print(f"The matrix you have entered is: \n {a}")
upper_triangular = np.triu(a)
lower_triangular = np.tril(a)
print(f"The upper triangular matrix is: \n {upper_triangular}")
print(f"The lower triangular matrix is: \n {lower_triangular}")


```

    Enter the number of rows of the matrix:  2
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  5 6
    Enter the row 2 elements separated by spaces:  8 4
    

    The matrix you have entered is: 
     [[5 6]
     [8 4]]
    The upper triangular matrix is: 
     [[5 6]
     [0 4]]
    The lower triangular matrix is: 
     [[5 0]
     [8 4]]
    


```python
Write a python program to extract the diagonal vector from a matrix and create a diagonal matrix from a vector

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of the matrix: "))
matrix1 = []
print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print(f"The matrix you have entered is: \n {a}")
diagonal = np.diag(a)
print(f"The diagonal elements are: {diagonal}")
diag_vector = [int(item) for item in input("Enter the diagonal elements: ").split()]
diagonal_matrix = np.diagflat(diag_vector)
print(f"The diagonal matrix is: \n {diagonal_matrix}")



```

    Enter the number of rows of the matrix:  2
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  3 6
    Enter the row 2 elements separated by spaces:  8 5
    

    The matrix you have entered is: 
     [[3 6]
     [8 5]]
    The diagonal elements are: [3 5]
    

    Enter the diagonal elements:  4 5
    

    The diagonal matrix is: 
     [[4 0]
     [0 5]]
    


```python
Write a python program to create an identity matrix

```


```python
import numpy as np
rank = int(input("Enter the rank of the identity matrix you want: "))
identity_matrix = np.identity(rank, dtype = int)
print(f"The identity matrix is: \n {identity_matrix}")


```

    Enter the rank of the identity matrix you want:  6
    

    The identity matrix is: 
     [[1 0 0 0 0 0]
     [0 1 0 0 0 0]
     [0 0 1 0 0 0]
     [0 0 0 1 0 0]
     [0 0 0 0 1 0]
     [0 0 0 0 0 1]]
    


```python
Write a python program to generate a matrix of 1s using dimensions input.

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of the matrix: "))
m1 = int(input("Enter the number of columns of the matrix: "))
ones_matrix = np.ones((n1, m1), dtype = int)
print(f"The ones matrix with {n1} rows and {m1} columns is:\n{ones_matrix}")


```

    Enter the number of rows of the matrix:  4
    Enter the number of columns of the matrix:  4
    

    The ones matrix with 4 rows and 4 columns is:
    [[1 1 1 1]
     [1 1 1 1]
     [1 1 1 1]
     [1 1 1 1]]
    


```python
Write a python program to return an array of 1s with the same shape and type as input array.

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of the matrix: "))
matrix1 = []
print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print(f"The matrix you have entered is: \n {a}")
ones_matrix = np.ones_like(matrix1, dtype = int, order = 'K')
print(f"The 1s matrix equivalent to the matrix you entered is: \n{ones_matrix}")


```

    Enter the number of rows of the matrix:  2
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  4 5
    Enter the row 2 elements separated by spaces:  6 9
    

    The matrix you have entered is: 
     [[4 5]
     [6 9]]
    The 1s matrix equivalent to the matrix you entered is: 
    [[1 1]
     [1 1]]
    


```python
Write a python program to return an array of 0s with the same shape and type as input array.

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of the matrix: "))
matrix1 = []
print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print(f"The matrix you have entered is: \n {a}")
zeroes_matrix = np.zeros_like(matrix1, dtype = int, order = 'K')
print(f"The 0s matrix equivalent to the matrix you entered is: \n{zeroes_matrix}")


```

    Enter the number of rows of the matrix:  2
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  3 9
    Enter the row 2 elements separated by spaces:  6 4
    

    The matrix you have entered is: 
     [[3 9]
     [6 4]]
    The 0s matrix equivalent to the matrix you entered is: 
    [[0 0]
     [0 0]]
    


```python
Write a python program to find the transpose of a matrix.

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of the matrix: "))
matrix1 = []
print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print(f"The matrix you have entered is: \n {a}")
t_matrix = np.transpose(a)
print(f"The transpose of the matrix you have entered is: \n{t_matrix}")


```

    Enter the number of rows of the matrix:  2
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  3 5
    Enter the row 2 elements separated by spaces:  8 5
    

    The matrix you have entered is: 
     [[3 5]
     [8 5]]
    The transpose of the matrix you have entered is: 
    [[3 8]
     [5 5]]
    


```python
Write a python program to find the inverse of a matrix.
```


```python
import numpy as np
n1 = int(input("Enter the number of rows of the matrix: "))
matrix1 = []
print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print(f"The matrix you have entered is: \n {a}")
determinant = np.linalg.det(a)
if(determinant == 0):
    print(f"The matrix you have entered is a singular matrix.\nThe inverse of this matrix cannot be calculated.")
else:
    inverse_matrix = np.linalg.inv(a)
    print(f"The inverse of the matrix you have entered is: \n{inverse_matrix}")

```

    Enter the number of rows of the matrix:  2
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  3 8
    Enter the row 2 elements separated by spaces:  6 4
    

    The matrix you have entered is: 
     [[3 8]
     [6 4]]
    The inverse of the matrix you have entered is: 
    [[-0.11111111  0.22222222]
     [ 0.16666667 -0.08333333]]
    


```python
Write a python program to find the trace of a matrix.

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of the matrix: "))
matrix1 = []
print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print(f"The matrix you have entered is: \n {a}")
trace = np.trace(a)
print(f"The trace of the matrix you have entered is: {trace}")


```

    Enter the number of rows of the matrix:  2
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  3 5
    Enter the row 2 elements separated by spaces:  7 5
    

    The matrix you have entered is: 
     [[3 5]
     [7 5]]
    The trace of the matrix you have entered is: 8
    


```python
Write a python program to find the determinant of a matrix.

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of the matrix: "))
matrix1 = []
print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print(f"The matrix you have entered is: \n{a}")
determinant = np.linalg.det(a)
print(f"The determinant of the matrix you have entered is: {determinant}")


```

    Enter the number of rows of the matrix:  2
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  3 5
    Enter the row 2 elements separated by spaces:  7 4
    

    The matrix you have entered is: 
    [[3 5]
     [7 4]]
    The determinant of the matrix you have entered is: -23.0
    


```python
Write a python program to find the rank of a matrix.

```


```python
import numpy as np
n1 = int(input("Enter the number of rows of the matrix: "))
matrix1 = []
print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print(f"The matrix you have entered is: \n{a}")
rank = np.linalg.matrix_rank(a)
print(f"The rank of the matrix you have entered is: {rank}")


```

    Enter the number of rows of the matrix:  2
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  5 3
    Enter the row 2 elements separated by spaces:  6 7
    

    The matrix you have entered is: 
    [[5 3]
     [6 7]]
    The rank of the matrix you have entered is: 2
    


```python
Write a python program to find the sparsity of a matrix.

```


```python
import numpy as np

n1 = int(input("Enter the number of rows of the matrix: "))
m1 = int(input("Enter the number of columns of the matrix: "))  # Add this line to take number of columns as input
matrix1 = []

print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)

a = np.array(matrix1)
print(f"The matrix you have entered is: \n{a}")

sparsity = (n1 * m1) - np.count_nonzero(a)
print(f"The sparsity of the matrix you have entered is: {sparsity}")

```

    Enter the number of rows of the matrix:  2
    Enter the number of columns of the matrix:  3
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  4 5
    Enter the row 2 elements separated by spaces:  6 7
    

    The matrix you have entered is: 
    [[4 5]
     [6 7]]
    The sparsity of the matrix you have entered is: 2
    


```python
Write a python program to create two numpy arrays of random integers between 0 and 20 of shape (3, 3), perform matrix addition, 
multiplication and find the transpose of the product matrix.

```


```python
import numpy as np
print(f"valuvating...")
print(f"Creating two 3x3 matrices with random elements between 0 and 20...")
a = np.random.randint(0, 20, (3, 3))
b = np.random.randint(0, 20, (3, 3))

print(f"The matrices are created successfully \n{a}\n\n{b}")
print("Matrics after addition")
sum = a + b
print(f"The sum of the two matrices is: \n{sum}")
print("Matrics after multiplication")
product = np.dot(a, b)
print(f"The product of the two matrices is: \n{product}\n")
print(f"The transpose of the product matrix is: \n{np.transpose(product)}")

```

    valuvating...
    Creating two 3x3 matrices with random elements between 0 and 20...
    The matrices are created successfully 
    [[11  5 11]
     [17  7  3]
     [18  0  6]]
    
    [[ 9  7 19]
     [ 3  5 18]
     [ 8  8 13]]
    Matrics after addition
    The sum of the two matrices is: 
    [[20 12 30]
     [20 12 21]
     [26  8 19]]
    Matrics after multiplication
    The product of the two matrices is: 
    [[202 190 442]
     [198 178 488]
     [210 174 420]]
    
    The transpose of the product matrix is: 
    [[202 198 210]
     [190 178 174]
     [442 488 420]]
    


```python
Write a python program to find the echelon form of a matrix and find its rank.

```


```python
import numpy as np
import sympy as sp
n1 = int(input("Enter the number of rows of the matrix: "))
matrix1 = []
print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
print(f"The matrix you have entered is: \n{a}")
print("Creating an echelon form of the matrix...")
echelon = sp.Matrix(a).echelon_form()
print(f"The echelon form of the matrix you have entered is: \n{np.array(echelon)}")
print(f"The rank of the matrix you have entered is: {echelon.rank()}")


```

    Enter the number of rows of the matrix:  2
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  3 4
    Enter the row 2 elements separated by spaces:  5 6
    

    The matrix you have entered is: 
    [[3 4]
     [5 6]]
    Creating an echelon form of the matrix...
    The echelon form of the matrix you have entered is: 
    [[3 4]
     [0 -2]]
    The rank of the matrix you have entered is: 2
    


```python
Write a python program to find the eigen values and eigen vectors

```


```python
import numpy as np
np.set_printoptions(precision=2)
n1 = int(input("Enter the number of rows of the matrix: "))
matrix1 = []
print("Enter the elements for the matrix: ")
for i in range(n1):
    row = list(map(int, input(f"Enter the row {i + 1} elements separated by spaces: ").split()))
    matrix1.append(row)
a = np.array(matrix1)
eigen_values, eigen_vectors = np.linalg.eig(a)
print(f"The eigen values of the matrix you have entered are: \n{eigen_values}")
print(f"The eigen vectors of the matrix you have entered are: \n{eigen_vectors}")

```

    Enter the number of rows of the matrix:  2
    

    Enter the elements for the matrix: 
    

    Enter the row 1 elements separated by spaces:  3 4
    Enter the row 2 elements separated by spaces:  5 6
    

    The eigen values of the matrix you have entered are: 
    [-0.22  9.22]
    The eigen vectors of the matrix you have entered are: 
    [[-0.78 -0.54]
     [ 0.63 -0.84]]
    


```python
Write a python program to find the cosine similarity between two vectors.

```


```python
import numpy as np
v1 = list(map(int, input("Enter the first vector, each value must be separated by a space: ").split()))
v2 = list(map(int, input("Enter the first vector, each value must be separated by a space: ").split()))

dp = np.dot(v1, v2)
norm_v1 = np.linalg.norm(v1)
norm_v2 = np.linalg.norm(v2)

sim = dp / ( norm_v1 * norm_v2 )

print(f"The cosine similarity of the two vectors v1<{v1[0]}, {v1[1]}, {v1[2]}> and v2<{v2[0]}, {v2[1]}, {v2[2]}> is {sim}")

```

    Enter the first vector, each value must be separated by a space:   3 4 5
    Enter the first vector, each value must be separated by a space:  4 5 7
    

    The cosine similarity of the two vectors v1<3, 4, 5> and v2<4, 5, 7> is 0.9987770299499059
    


```python
Write a python program to find whether two vectors are orthogonal or not.

```


```python
import numpy as np
import math
v1 = list(map(int, input("Enter the first vector, each value must be separated by a space: ").split()))
v2 = list(map(int, input("Enter the first vector, each value must be separated by a space: ").split()))
dp = np.dot(v1, v2)
if dp == 0:
    print(f"The two vectors v1<{v1[0]}, {v1[1]}, {v1[2]}> and v2<{v2[0]}, {v2[1]}, {v2[2]}> are orthogonal.")
else:
    a = math.degrees(math.acos(dp/(np.linalg.norm(v1) * np.linalg.norm(v2))))
    print(f"The two vectors v1<{v1[0]}, {v1[1]}, {v1[2]}> and v2<{v2[0]}, {v2[1]}, {v2[2]}> intersect at an angle of {a} and are not orthogonal.")


```

    Enter the first vector, each value must be separated by a space:  2 4 6
    Enter the first vector, each value must be separated by a space:  7 4 3
    

    The two vectors v1<2, 4, 6> and v2<7, 4, 3> intersect at an angle of 41.785580637555334 and are not orthogonal.
    


```python
Write a python program to find the norms of a given vector.

```


```python
import numpy as np
v1 = list(map(int, input("Enter the vector, each value must be separated by a space: ").split()))
l1 = np.linalg.norm(v1, 1)
l2 = np.linalg.norm(v1, 2)
sqrd_l2 = np.linalg.norm(v1, 2) ** 2
max_l = np.linalg.norm(v1, np.inf)
print(f"The L1 norm of the vector v {v1} is {l1}")
print(f"The L2 norm of the vector v {v1} is {l2}")
print(f"The squared L2 norm of the vector v {v1} is {sqrd_l2}")
print(f"The max norm of the vector v {v1} is {max_l}")


```

    Enter the vector, each value must be separated by a space:  5 4
    

    The L1 norm of the vector v [5, 4] is 9.0
    The L2 norm of the vector v [5, 4] is 6.4031242374328485
    The squared L2 norm of the vector v [5, 4] is 41.0
    The max norm of the vector v [5, 4] is 5.0
    


```python
Write a python program to show triangle inequality of two vectors.

```


```python
import numpy as np
v1 = list(map(int, input("Enter the first vector, each value must be separated by a space: ").split()))
v2 = list(map(int, input("Enter the second vector, each value must be separated by a space: ").split()))
norm_sum = np.linalg.norm(np.add(v1, v2))
sum_norm = np.add(np.linalg.norm(v1), np.linalg.norm(v2))
print(f"||{v1} + {v2}|| is {norm_sum}\n")
print(f"||{v1}|| + ||{v2}||is {sum_norm}\n")
if norm_sum <= sum_norm:
    print("Therefore, ||v1 + v2|| <= ||v1|| + ||v2||")


```

    Enter the first vector, each value must be separated by a space:  3 4 5
    Enter the second vector, each value must be separated by a space:  8 7 5
    

    ||[3, 4, 5] + [8, 7, 5]|| is 18.49324200890693
    
    ||[3, 4, 5]|| + ||[8, 7, 5]||is 18.818407936336207
    
    Therefore, ||v1 + v2|| <= ||v1|| + ||v2||
    


```python
Write a python program to show linearity of a vector.

```


```python
import numpy as np
v1 = list(map(int, input("Enter the vector, each value must be separated by a space: ").split()))
sc = eval(input("Enter a scalar quantity: "))
norm_sc_v1 = np.linalg.norm(np.multiply(v1, sc))
sc_norm_v1 = np.multiply(np.linalg.norm(v1), sc)
print(f"||{sc} . {v1}|| is {norm_sc_v1}")
print(f"{sc} . ||{v1}|| is {sc_norm_v1}")
if norm_sc_v1 == sc_norm_v1:
    print(f"Therefore, ||{sc} . {v1}|| = {sc} . ||{v1}||")


```

    Enter the vector, each value must be separated by a space:  6 7 8
    Enter a scalar quantity:  3
    

    ||3 . [6, 7, 8]|| is 36.61966684720111
    3 . ||[6, 7, 8]|| is 36.6196668472011
    


```python
Write a python program to show the axioms associated with a vector space.

```


```python
import numpy as np
v1 = list(map(int, input("Enter the first vector, each value must be separated by a space: ").split()))
v2 = list(map(int, input("Enter the second vector, each value must be separated by a space: ").split()))
v3 = list(map(int, input("Enter the third vector, each value must be separated by a space: ").split()))
sc1 = eval(input("Enter first scalar quantity: "))
sc2 = eval(input("Enter second scalar quantity: "))

print("Associativity of addition")
print(f"( {v1} + {v2} ) + {v3} = {np.add(np.add(v1, v2), v3)}")
print(f"{v1} + ( {v2} + {v3} ) = {np.add(v1, np.add(v2, v3))}\n")

print("Commutativity of addition")
print(f"{v1} + {v2} = {np.add(v1, v2)}")
print(f"{v2} + {v1} = {np.add(v2, v1)}\n")

print("Identity element of addition")
print(f"{v1} + 0 = {np.add(v1, 0)}")
print(f"Which is equal to {v1}\n")

print("Inverse element of addition")
print(f"{v1} + {np.negative(v1)} = {np.add(v1, np.negative(v1))}\n")

print("Distributivity of scalar mutliplication over vector addition")
print(f"{sc1} . ({v1} + {v2}) = {np.multiply(sc1, np.add(v1, v2))}")
print(f"{sc1} . {v1} + {sc1} . {v2}) = {np.add(np.multiply(v1, sc1), np.multiply(v2, sc1))}")
print(f"{sc1} . ({v1} + {v2}) = {sc1} . {v1} + {sc1} . {v2}\n")

print("Distributivity of scalar mutliplication over field addition")
print(f"({sc1} + {sc2}) . {v1} = {np.multiply(v1, sc1 + sc2)}")
print(f"{sc1} . {v1} + {sc2} . {v2} = {np.add(np.multiply(v1, sc1), np.multiply(v2, sc2))}\n")

print("Compatability of scalar mutliplication over field mutliplication")
print(f"{sc1} . ({sc2} . {v1}) = {np.multiply(sc1, np.multiply(sc2, v1))}")
print(f"{sc1} . {sc2} . ({v1}) = {np.multiply(sc1 * sc2, v1)}")
print(f"{sc1} . ({sc2} . {v1}) = {sc1} . {sc2} . ({v1})\n")

print("Identity element of scalar mutliplication")
print(f"1 . {v1} = {np.multiply(v1, 1)}")
print(f"Which is equal to {v1}\n")

```

    Enter the first vector, each value must be separated by a space:  3 5
    Enter the second vector, each value must be separated by a space:  8 6
    Enter the third vector, each value must be separated by a space:  5 4
    Enter first scalar quantity:  4
    Enter second scalar quantity:  7
    

    Associativity of addition
    ( [3, 5] + [8, 6] ) + [5, 4] = [16 15]
    [3, 5] + ( [8, 6] + [5, 4] ) = [16 15]
    
    Commutativity of addition
    [3, 5] + [8, 6] = [11 11]
    [8, 6] + [3, 5] = [11 11]
    
    Identity element of addition
    [3, 5] + 0 = [3 5]
    Which is equal to [3, 5]
    
    Inverse element of addition
    [3, 5] + [-3 -5] = [0 0]
    
    Distributivity of scalar mutliplication over vector addition
    4 . ([3, 5] + [8, 6]) = [44 44]
    4 . [3, 5] + 4 . [8, 6]) = [44 44]
    4 . ([3, 5] + [8, 6]) = 4 . [3, 5] + 4 . [8, 6]
    
    Distributivity of scalar mutliplication over field addition
    (4 + 7) . [3, 5] = [33 55]
    4 . [3, 5] + 7 . [8, 6] = [68 62]
    
    Compatability of scalar mutliplication over field mutliplication
    4 . (7 . [3, 5]) = [ 84 140]
    4 . 7 . ([3, 5]) = [ 84 140]
    4 . (7 . [3, 5]) = 4 . 7 . ([3, 5])
    
    Identity element of scalar mutliplication
    1 . [3, 5] = [3 5]
    Which is equal to [3, 5]
    
    


```python

```
