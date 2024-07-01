# Linear Algebra

from 3Blue1Brown

- Any way of translating between vectors and sets of numbers is termed as ‘Co-ordinate System’.

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
- Unless mentioned, in these notes, we consider linear transformations only.
- For a transformation to be linear, there are 2 requirements :
1. All lines must remain lines (hence grid lines are even and parallely spaced).
2. Origin must be fixed at its place before and after transformation.
- Depending on where the i hat and j hat vectors land after transformation, we can find transformation of any vector in the space as :

Transformed **v** = (scalar_1)(Transformed **i hat**) + (scalar_2)(Transformed **j hat**) 

- The straight and diagonal lines in Fig. 4 respectively represent the untransformed and transformed 2-D space. It also represents tranformation of a vector **v**.

![Fig. 4](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-29_13-41-48.png)

Fig. 4

- The transformed **i hat** and **j hat** vectros can be written in the form of a 2*2 matrix. then, if we have the co-ordinates of any point in the untransformed space, we can find its co-ordinates in the transformed space by mulilpying the columns of this 2*2 matrix with the matrix of original co-ordinates of the point and add them.
- this ida can be extended to define matrix vector multiplication as :

![Fig. 5](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-29_14-14-33.png)

Fig. 5

![Fig. 6](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_11-03-07.png)

Fig. 6

- Thus, *Matrices are Transformations of spaces*.
- Multiplying a matrix to a vector is equivalent to transforming the vector.

# 4. Matrix multiplication as a Composition :-

## a. *Intuitive approach :-*

- A ‘composition’ of a transformation refers to the action of two or more transformations, either one after the other or all of them acting simultaneously.
- Consider a composition of 2 transformations - a 90 degrees anticlockwise rotation and a shear transformation. We can form individual matrices for the individual transformations and matrix for composition of the transformations as :

![Fig. 7](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_11-09-09.png)

Fig. 7

- This idea can be continued to define matrix multiplication of two 2*2 matrices as :

![Fig. 8](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_11-09-23.png)

Fig. 8

- These matrices on the L.H.S. have right to left associativity, like the function of a function, or like the sequence in whch we apply the transformations.

![Fig. 9](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_11-09-55.png)

Fig. 9

## *b. Mathematical Method Derivation :-*

- Consider the same 2 matrices. In each of them, first and second columns respectively represent the transformations in **i hat** and **j hat** vectors.
- We take the matrix M2 and apply its transformation to the first and second column of M1, i.e. to **i hat** and **j hat** vcetors of teransformation by M1, so as to get transformations in **i hat** and **j hat** due to the composition of transformation of M1 and M2.

![Fig. 10](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_11-34-24.png)

Fig. 10

![Fig. 11](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_11-35-59.png)

Fig. 11

![Fig. 12](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_11-35-12.png)

Fig. 12

![Fig. 13](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_11-35-35.png)

Fig. 13

# 5. Three-dimensional Linear Transformations :-

![Fig. 14](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_11-53-40.png)

Fig. 14

![Fig. 15](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_11-53-51.png)

Fig. 15

- Linear transformations in 3 dimensions is very similar to that of 2 dimensions, just that now we also consider three co-ordinates of any point in space and the unit vector along z axis -  **k hat**.
- Fig. 14 shows how applying a linear transformation affect the basis vectors of 3 dimensional space; new positions of those basis vectors are along the solid yellow linea and faint yellow lines depict the original positions of the 3 axes.
- These 3 vectors of transformations of basis vectors can be combined into one 3*3 matrix, representing the trabnsfromation in matrix form. (Fig. 15)
- The Matrix multiplication operations can be extended in the same way as those of 2 dimensions i.e. associativity form right to left, and same manner of multiplcation of terms.

# 6. The Determinant :-

## a. *Intuitive approach :-*

- The ‘Determinant’ of a matrix is the factor by which the area of a unit square, whose edges rest on the basis vectors, in the original 2-D space changes after applying transformation specified by the matrix.

![Scaling transformation](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_12-16-16.png)

Scaling transformation

![Shear tramsformation](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_12-13-29.png)

Shear tramsformation

- Whatever change of area affects one region of the space, after the transformation, affects every other part of the region in the same way.

![Fig. 16](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_12-14-13.png)

Fig. 16

![Fig. 17](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_12-14-00.png)

Fig. 17

- For an irregular shaped object, its area can be approximated to the sum of areas of a good number of small squares.

![Fig. 18](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_12-12-43.png)

Fig. 18

- If the determinant of a matrix (transformation) is zero, it means that the area in the original space is reduced to a line or even a point.
- If the determinant of a matrix is zero, it means that the orientation of the 2-D space is, in some aspect, flipped over, like flipped from right to left or from top to bottom (w.r.t the positions of the basis vecrtors).
- In case of 3-D space, the determinant gives information of the change in volume of a unit cube, having its edges resting on the basis vectors, after it gets warped into some kind of ‘parallelepiped’.

![Fig. 19](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_12-51-05.png)

Fig. 19

- In this case, if determinant is zero, then the transformation transforms the space to a plane, line or a point, all of which have zero volume. In this case, it is evident that the column vectors in the matrix are linearly dependent.
- In 3-D space, if the determinant is negative, it means that the orientation of the 3-D space is flipped in some respect. The positive orientation of the 3-D space can be represented by the Right Hand Rule, as shown below. If this rule doesn’t hold after the transformation, the determinant comes out to be negative.

![Fig. 20](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_12-31-31.png)

Fig. 20

## b. *Mathematical Formulae :-*

![Fig. 21](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_12-49-24.png)

Fig. 21

![Fig. 22](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_12-50-37.png)

Fig. 22

# 7. Inverse Matrices, Column space, Rank and Null Space :-

- Linear System of Equations :

A set of equations having variables raised only to power 1 and sum of their scaled versions being equated to a constant. They can be represented in Matrix form.

- This is equivalent to applying a transformation ‘A’ to vector ‘**x**’ in 3-D space, so that it aligns with ‘**v**’.

![Fig. 23](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_14-42-22.png)

Fig. 23

![Fig. 24](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_14-43-22.png)

Fig. 24

- We consider 2 cases while solving such a problem :-
1. **det(A)≠0 :-**
- We try to trace the vector **v** from the transformed space to the original space, so that it will now coincide with x and thus, value of **x** acn be computed. This type of transformation, where we go from space transformed by A to the original, untransformed space, is termed as ‘Inverse Transformation’ or ‘Inverse Matrix’, denoted as ‘A^-1’.
- This implies that the operation of multiplying A^-1 with A doesn’t affect the original space; hence, such transformation is termed ‘Identity Transformation’ and is characterized by a zero matrix, having 1  as the diagonal elements.
- Then, we pre-multiply this inverse on both sides of the equation in Fig. 24, and calculate value of **x** as shown in Fig. 25 and 26.
- likewise, if the number of unknowns and number of equations is the same, then they, most probably have one unique solution.

![Fig. 25](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_14-55-19.png)

Fig. 25

![Fig. 26](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_14-55-49.png)

Fig. 26

1. **det(A)=0 :-**
- In such cases, the transformation transforms the space into a lower dimensional space, for which, inverse transformation doesn’t exist.
- Here, we can define ‘Rank’ :- The number of dimensions in the output of a transformation (Column Space)
- ‘Column Space’ of a Matrix : The set of all possible vectors in the output of a transformation.
- A matrix is termed as ‘Full Rank’ when its rank equals number of columns in the matrix.
- ‘Null Space’ / ‘Kernel’ of a matrix : Set of vectors that land on the zero vector. There are more than one vector in the kernel if the matrix doesn’t have a full rank.
1. If a plane is transformed to a line, the kernel is a line.
2. If a volume is transformed to a plane. the kernel is a line.
3. If a volume is transformed to a line. the kernel is a plane.
- Thus, if **v** is a zero vector, the null space is the set of solutions.

# 8. Nonsquare matrices as transformations between dimensions :-

- Consider a nonsquare matrix with 3 rows and 2 columns, i.e. 3*2 matrix.
- The 2 columns indicate that the input is from a 2-D space while the 3 rows indicate that the landing location of those 2 basis bectors is described with 3 co-ordinates i.e. a 3-D space. However, as we have ony 2 basis vectors as input, so the output will be mapped on a plane in the 3-D space.
- The matrix is still having full rank, as the number of dimensions in column space (which is a plane, though in 3-D space) is same as the number of dimensions in the input space.

![Fig. 27](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_15-24-40.png)

Fig. 27

- Example of 2*3 matrix :

![Fig. 28](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_15-31-12.png)

Fig. 28

- Example of 1*2 matrix (2-D space to 1-D space) :-

Here, though it may seem confusing how all grid lines remain parallel and equidistant, we can verify it by placing equidistant dots, along a line, in the 2-D space, and verifying that they are equidistant even in the 1-D space.

![Fig. 29](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_15-33-12.png)

Fig. 29

# 9. Dot Product and Duality :-

## a. *Dot Product :-*

- Dot Prodcut is a commutative operation.
- For 2 vectors of same dimensions, dot product is equivalent to the sum of the product of corresponding elements of the 2 vectors.
- Technically, ‘Dot Product’ can be defined as the product of length of one vector with the length of he projection of the other vector on the previous vector.
- Consider a vector **v**=4**i**+3**j** . If there is a transformation, which transforms the 2-D space to a 1-D space (Number Line), then we would have a 182 matrix for that transformation. Multiplying this transfromation with the column vector of **v** in 2-D space, we get its co-ordinate in 1-D space. This looks very analogous to the operation we perfrom during dot product.
- This also highlights the connection between 1*2 transformation matrices and vectors in 2-D space, as we can go from one entity to othher by rotating the vector.

![Fig. 30](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_16-09-04.png)

Fig. 30

- Example :-

Consider a 1-D space placed diagonally on the 2-D space. Let **u** be a unit vector of the 1-D space. We also braw the basis vectors of the 2-D space. On the basis of symmetry, we can infer that the transformation would be ; [ux uy], where ux and uy are respectively the x and y components of the co-ordinates of **u** w.r.t the 2-D space.

![Fig. 31](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_16-17-03.png)

Fig. 31

![Fig. 32](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_16-17-11.png)

Fig. 32

![Fig. 33](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_16-21-49.png)

Fig. 33

As the matrices can be turned over as described over, so this matrix multiplication is same as the dot product of 2 vectors according to the definition.

This highlights the similarity in the calculation of dot-product and matrix multiplication, atleast where 1 unit vector is involved (here, **u**). It is applicable in a similar way, for non-unit vectors, which are just scaled versions of the unit vectors. (Fig. 34)

![Fig. 34](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_16-23-52.png)

Fig. 34

Thus, dot product of a vector with other is same as applying transformation of that evctor to the second vector by flipping the first vector horizontally.

## b. *Duality :-*

- ‘Duality’ refers to the similarity/correspondence between 2 types of  mathematical things.
- In this case, we can say that ‘Dual of a vector (in 2-D space) is the linear transforamtion (2-D space to 1-D space) and vice versa.

# 10. Cross Products :-

- In magnitude, cross product of 2 vectors **v** and **w** is the area of the parallelogram formed by those 2 vectors.

![Fig. 35](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_16-44-33.png)

Fig. 35

- If **v** lies to the right of **w**, then the cross product is positve; if **v** lies on left of **w**, corss product is negative. This is with reference to the relative position of the basis vectors.
- Hence, we have the result :

![Fig. 36](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_16-47-12.png)

Fig. 36

- The cross product is just like a linear transformation on the basis vectors : **i hat** moving to the position of first vector in the cross product and **j hat**, to that of the second vector.
- The transformation is formed by writing the co-ordinates of the first vector in cross product as first column and those of the second vector as the second column.
- The determinant of this transformation is the required area of parallelogram.

![Fig. 37](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_16-49-54.png)

Fig. 37

![Fig. 38](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_16-55-34.png)

Fig. 38

- The result of a cross product is a vector whose magnitude is equal to rea of the parallelogram formed by the 2 vectors and its direction is given by the right hand thumb rule, with index finger and the middle finger pointing in the directions of the first and second vector respectively and thumb giving the direction of the resultant vector.
- Methods of calculating cross product :

![Fig. 39](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-06-30_22-11-40.png)

Fig. 39

# 11. Cross Products in the light of Linear Transformations :-

## a. *Computational Method :-*

- The vector which describes some kind of linear transformation to a number line is termed as ‘Dual Vector’.
- We consider 2 vectors, **v** and **w**, and a variable vector, defined by its x, y and z co-ordinates. We then consider a function, which is linear, which takes in the variable input from 3-D space and maps it to a 1-D Number line.

![Fig. 40](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_18-34-44.png)

Fig. 40

- This basically suggests that we multiply a 1*3 matrix with the column vector of variable 3-D co-ordinates to get a single number, as is observed on the R.H.S.

![Fig. 41](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_18-36-07.png)

Fig. 41

- By the principle of Duality, this can be expressed as a dot product as :

![Fig. 42](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_18-39-14.png)

Fig. 42

- Let this unknown vector be ‘**p**’. This can thus be further solved as :

![Fig. 43](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_18-40-14.png)

Fig. 43

![Fig. 44](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_18-40-23.png)

Fig. 44

## b. *Geometric Interpretetion :-*

- The volume of parallelpiped formed by vectors **p**, **v** and **w** is given on the R.H.S. of the equation in Fig. 42. and it is equal to the area of the base of the parallelpiped (**v** x **w**) times the height of parallelpiped.
- The height of parallelpiped is equal to the component of p in the direction perpendicular to (**v** x **w**) which is equal to the dot product of **p** with vector (x**i**+y**j**+z**k**) ****

# 12. Cramer’s Rule - Geometric Interpretation :-

- Consider a system of linear equations :

![Fig. 45](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_19-19-52.png)

Fig. 45

- We assume that the determinant of the matrix of coefficients is not zero.
- Hence, applying the transformation specified by matrix of coefficients, we aim to find the values of vector [x y] in the untransformed space such that it equates to the matrix of coefficients in the transformed space.

![Fig. 46](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_19-19-25.png)

Fig. 46

- The dot product before and after transformation can be very different; those dot products where they are preserved, are termed as ‘orthonomal’; in such transformatuions, the basis vectors are still having unit length and are perpendiclar to each other i.e. they correspond to rigid motion - no stretching or squishing.
- Different dot products brfore and after transformations :-

![Fig. 47](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_21-07-57.png)

Fig. 47

![Fig. 48](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_21-09-01.png)

Fig. 48

- For orthonomal transformations :-

![Fig. 49](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_21-11-56.png)

Fig. 49

## a. *For 2-D space (system of 2 linear equations) :-*

![Fig. 50](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_21-13-56.png)

Fig. 50

![Fig. 51](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_21-13-44.png)

Fig. 51

- We consider the ‘mystery vector’ or let it be **s**. In the untransformed space, referring to Fig. 51, we consider a parallelogram formed by vectors **j hat** and **s**. Its area would thus be ‘x’ (x co-ordinate of **s**).
- Likewise, we can consider a parallelogram formed with **i hat** and **s**, whose area would be y (y co-ordinate of **s**).
- The areas are signed, in the sense that they can also be negative, and are considered with their sign.
- After applying the transformation, the vectors change their positions. Consider untransformed space of Fig. 50. After transformation, Fig. 50 would look like :

![Fig. 52](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_21-23-00.png)

Fig. 52

- The y co-ordinate changes tfrom [0 1] to [-1 1]; likewise, the vector **s** reaches position [4 2]. Due to transformation, the areas get scaled by a factor equal to the determinant of the transformation. Hence,
    
    Area=det(A)*x
    
    x=Area/det(A)
    

Now, for the parallelogram whose area is to be considered, the y co-ordinate changes to the value of first column of the transformation (as initially it was considered along **j hat** so unit length). But, x co-ordinate of **s** was related to its original p[osition, which now changed to [4 2]. Hence, in the area matrix we substitute [4 2] as its first column; else, the matrix is same as that of A.

- Likewise y co-ordinate can be calculated.

## b. *For 3-D space (system of 3 linear equations) :-*

![Fig. 53](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_21-15-48.png)

Fig. 53

![Fig. 54](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_21-15-42.png)

Fig. 54

![Fig. 55](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_21-15-54.png)

Fig. 55

- A simular process can be done for 3-D space, just that now, we will be dealing with a parallelpiped. Any of the above 3 cases can be considered as starting point.
- For instance, consider the following example :-

![Fig. 56](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_21-46-28.png)

Fig. 56

- Using similar procedure as in 2-D space we get the following results :-

![Fig. 57](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_21-46-15.png)

Fig. 57

# 13. Change of Basis :-

## a. *Translation of basis vectors among co-ordinate systems:-*

- The current co-ordinate system is based on implicit assumptions, like :
1. The length of a unit distance
2. How first co-ordinate is x co-ordinate and indicates sideward motion
3. How second co-ordinate is y co-ordinate and indicates vertical motion
- Space itself has no grid so anyone can set any kind of co-ordinate system according to their convenience.
- Consider a new co-ordinate system, where the basis vectors i hat and j hat are changed to [2 1] and [-1 1] respectively. Arranging them respectively column-wise, we get the matrix that corresponds to the transformation of current space to that of new co-ordinate system.
- Hence, if we multiply any co-ordinates of new system with this transformation, we will get corresponding co-ordinates of the same vector in our co-ordinate system. this transformation/matrix will now be termed as ‘Change of Basis Matrix’.
- Likewise, if we multiply co-ordinates from original system with the inverse matrix of this transformation, we will get the co-ordinates according to the new system.

## b. *Translation of matrices among co-ordinate systems :-*

- Firstly, we will convert the vector from new system of co-ordinates to our system of co-ordinates. Here, we consider vector [-1 2] in new system of co-ordinates.

![Fig. 58](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_22-39-59.png)

Fig. 58

- Next, we will apply the concerned transformation.

![Fig. 59](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_22-40-14.png)

Fig. 59

- Lastly, we will use inverse of change of basis matrix to convert the co-ordinates back to new system.

![Fig. 60](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/136e57b2-5a88-4b50-abeb-c18de4d9d2ae.png)

Fig. 60

- Hence, we get the transformation matrix for new co-ordinate system as : (**v** is any vector in new co-ordinate system)

![Fig. 61](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/aab2bd1c-eec3-4c9d-a08e-2c6d904d342c.png)

Fig. 61

- Whenever an equation of a form as shown below is observed, we can ingfer that middle matrix denotes transformation of some kind and outer matrices, translation of some kind 9like from one co-ordinate system to another).

![Fig. 62](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_22-45-53.png)

Fig. 62

# 14. Eigen Vectors and Eigen Values :-

- Generally, after a transformation, vectors get deviated fom their own original spans. However, sometimes the vectors do maintain their original span even after ttansformation. Such vectors are termed as ‘Eigenvectors’.
- The factor by which an eigenvector or eigenvalue is stretched or squished during its transformation, is termd as its ‘eigenvalue’. A negative eigenvalue suggests that the vector gets flipped after the transformation.
- For example, if we find an eigenvector for somekind of 3-D rotation, then wecan infer that the eigenvector is the axis of rotation and its eigenvalue is 1 (as rotation doesn’t stretch or squish vectors).
- Hence, we get the relation :

![Fig. 63](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_22-58-40.png)

Fig. 63

- Sooving further, we get the following result, where we aim to find the non-zero solution :

![Fig. 64](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_23-01-06.png)

Fig. 64

- Hence now for a non-zero solution, i.e. if v≠0, we need :

![Fig. 65](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_23-02-52.png)

Fig. 65

- Consider another example, where the eigenvalues are 2 and 3. We can compute the eigenvectors by substituting the values of eigenvalues one by one in equation given in Fig. 64.

![Screenshot from 2024-07-01 23-06-35.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_23-06-35.png)

- Non-real solution for lambda indicate absence of eigenvalues and thus, eigenvectors. Also, eigenvalues, whichare complex numbers, correspond to some kind of rotation.
- We can also have just one eigenvalue but multiple eigenvectors e.g. transformation in which co-ordiante system is scaled up/down.
- A ‘Diagnal Matrix’ is a special kind of matrix, whose digonal elemnts are non-zero and are eigen values for corresponding basis vectors.

# 15. A quick trick for computing Eigenvalues :-

- The equation, whose roots are the eigenvalues, is termed as the ‘characteristic polynomial’.

![Screenshot from 2024-07-01 23-21-47.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_23-21-47.png)

- Important results :-

![Screenshot from 2024-07-01 23-25-23.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_23-25-23.png)

# 16. Abstract Vector Spaces :-

- Functions are like vectors, in the sense that they can be added at their corresponding points and their output can also be scaled.
- Functions can also have transformations, which transform the function from one form to another form. Example - derivative; generally, such transoprmations are termed as ‘operators’.
- Properties of a linear transformation :- (Linear transformations preserve addition and scalar multiplication).

![Screenshot from 2024-07-01 23-40-09.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_23-40-09.png)

- It can be observed that the above properties are equally valid for operators like the derivative.
- Consider a space of a function, for instance, polynomial functions. there can be infinitely many polynomials in this space, but each polynomial will have finite number of terms. We choose increasing powers of x, as the basis vectors of this space, just like **i hat**,  **j hat** and **k hat** in 3-D space. This space is infinitely dimensional.

![Screenshot from 2024-07-01 23-44-40.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_23-44-40.png)

- We take the derivative of each basis vector term and substitute them, colum-wise, in the derivative matrix. Multiplying this matrix witha any vector, we can get derivative of the polynomial represented by that vector.
- Thus, there are several similarities between functions and linear algebra concepts.

![Screenshot from 2024-07-01 23-51-41.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_23-51-41.png)

- Hence, several kinds of vector spaces can be defined, like one with arows, one with matrices, one with functions, etc. Linear Algebra concepts are applicable in all of them in a similar manner.

![Screenshot from 2024-07-01 23-53-18.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_23-53-18.png)

- Axioms for various vector spaces (They are like interface between discovered theories and theories yet to be discovered):-

![Screenshot from 2024-07-01 23-54-49.png](Linear%20Algebra%207b2ecec57ba24509bdd0f099c8067e7b/Screenshot_from_2024-07-01_23-54-49.png)