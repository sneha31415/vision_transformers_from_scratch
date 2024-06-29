# Linear Algebra

from 3Blue1Brown

# 1. Vectors :-

### a. *Various perspectives regarding vectors :-*

- Physics - A vector is an arrow that has a length and a specified direction
- Computer Science - An ordered list of numbers
- Mathematics - an entity on which operations of vector addition and vector multipliaction can be conducted

### b. *With respect to Linear Algebra :-*

- Vector is any arrow, originating from the origin (in most of the cases).
1. **For a 2-D space :**
- It is represented by a 2*1 column matrix, where the upper and lower elements represent the x and y co-ordinates of the head of the vector (tip of the vector).
1. **For a 3-D space :**
- Every vector can be represented by a unique 3*1 matrix, where the top, middle and bottom elements of the matrix represent the x, y and z co-ordinates of the vector.
1. **Basic Vector Operations :**
- *Vector Addition*

If we have 2 vectors, and we move second vector such that its tail coincides with head of first one, while remaining parallel to its own direction, then the new vector, from the tail of first vector to head of second vector, represents the sum of the 2 vectors.

This is because, comsider, for (**v** + **w**), we will first move along vector **v**, then, starting from head of **v**, we will move in the direction of **w**, according to the length of **w**. This would be equivalent to the addition of the 2 vectors.

![Screenshot from 2024-06-29 12-25-15.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-29_12-25-15.png)

![Screenshot from 2024-06-29 12-24-44.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-29_12-24-44.png)

![Screenshot from 2024-06-29 12-28-22.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-29_12-28-22.png)

- *Vector Multiplication (Scalar Multiplication)*

When a vector is multiplied by a scalar, each of its components (x component, y comonent, z component etc.) gets multiplied by that number and thus, the whole vector gets ‘scaled’ i.e. changes its length. 

Such numbers are termed ‘scalars’. If scalar is negative, the vector gets scaled, but in opposite direction.

![Screenshot from 2024-06-29 12-34-32.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-29_12-34-32.png)

# 2.  Basis vectors, Linear Combination of Vectors and Span :-

### a. *Basis :-*

- In x-y plane, there are 2 unit vectors along positive directions of x and y axes, which are known as ‘i hat’ or ‘ i cap’ and ‘j hat’ or ‘j cap’ respectively. (Fig. 1)
- Technically, we can say that ‘*the basis of a vector space ois a set of linearaly independent vectors that span the full space*’.

### b. *Linear Combination of Vectors :-*

- A ‘Linear Combination of vectors’ is the sum of the scaled versions of vectors.
- It is called ‘linear’ because, if one of those scalars is kept constant and the scalar of tother vector is varied, the resultant vector traces a straight line, as shown in Fig. 3.

![Fig. 3](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-29_12-49-42.png)

Fig. 3

### c. *Span of vectors :-*

- The set of all vectors that can be reached by changing linear combination of a pair of vectors is termed as the ‘Span’ of those 2 vectors.

### d. *Vectors v/s Points :-*

- A point in a vector space is the position of the head of the coreresponding vector in that space.
- When dealing with collection of vectors, it is convinient to dela with them as points; but when individual vectors are considered, considering them as vectors is convinient.

### e. *Linearly Dependent Vectors :-*

- If, in a Linear Combination of vectros, the non-consideration of one of the vectors doesn’t affect the span of those vectors, then that vector is termed to be ‘Linearly Dependent’ w.r.t. other vectors.
- This basically means that one of the vectors can be expresssed as a linear combination of other vectors, as that  vector is included in the span of the remaining vectors.

# 3. Linear Transformations and Matrices :-

- A ‘Linear Transformation’ is like a function, which takes vector(s) as input and give another vectror(s) as output. It kind of moves the input vector to the position of the output vector; that’s why it is called as ‘transformation’ rather than a function.
- For a transformation to be linear, there are 2 requirements :
1. All lines must remain lines (hence grid lines are even and parallely spaced).
2. Origin must be fixed at its place before and after transformation.
- Depending on where the i hat and j hat vectors land after transformation, we can find transformation of any vector in the space as :

Transformed **v** = (scalar_1)(Transformed **i hat**) + (scalar_2)(Transformed **j hat**) 

- The straight and diagonal lines in Fig. 4 respectively represent the untransformed and transformed 2-D space. It also represents tranformation of a vector **v**.

![Screenshot from 2024-06-29 13-41-48.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-29_13-41-48.png)

- The transformed **i hat** and **j hat** vectros can be written in the form of a 2*2 matrix. then, if we have the co-ordinates of any point in the untransformed space, we can find its co-ordinates in the transformed space by mulilpying the columns of this 2*2 matrix with the matrix of original co-ordinates of the point and add them.
- this ida can be extended to define matrix vector multiplication as :

![Screenshot from 2024-06-29 14-14-33.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-29_14-14-33.png)

- Thus, *Matrices are Transformations of spaces*.