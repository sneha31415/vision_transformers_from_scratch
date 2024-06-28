# Linear Algebra Notes

---

## Chapter 1: Vectors

### Vectors: List of numbers.

each number in the list signifies a direction is some space.

e.g. [2 3] is a point/vector in a 2-dimensional space → 2 units in x-direction and 3 units in the y-direction. Arrow pointing from origin to point (2,3).

---

### Vector Addition:

Adding two vectors: [2 3] + [1 -2] = [(2+1) (3-2)] = [3 1]. → adding the movement in each direction.

---

### Vector Multiplication:

Scaling a vector: 

e.g k[2 5] = [2k 5k]

---

## Chapter 2: Linear Combinations, span, and basis vectors

### Linear Combination:

Let v and w be two vectors and a and b be two scalars,

then a*v + b*w is a linear combination for v and w.

---

### Span:

Set of all possible linear combinations of v and w.

e.g. If v = i and w = j, in 2D-space, then the span of v and w is the whole 2D-space.

---

### Basis Vectors:

For our standard 2D-space, the basis vectors are i and j, which are [1 0] and [0 1] respectively.

if we take two basis vectors v and w for our coordinate system, and if they are parallel to each other then their span would be just a straight line.

***Linearly independent vectors***: For two or more vectors,if no vector among them can be obtained by linear combination of others.

Same ideas apply to any n-dimensional space.

---

## Chapter 3: Linear Transformations and matrices

### Linear Transformation

Talking about 2D-Space:

A *linear transformation* can be considered like a function where we give a point and we get where the point lands after a transformation.

What does "**Linear**" mean? : Origin remains fixed and Lines remains lines(straight) OR Grid lines remain evenly spaced. OR

if the transformation satisfies these properties:

**L(v + w) = L(v) (where v & w are two vectors)**

**L(cv) = cL(v) (where c is a scalar)**

if we just find where the original basis vector go after the transformation we can find any point after the transformation.

![Untitled](Linear%20Algebra%20Notes%20f58400cb6eec4aadbad204f8c5e40a17/Untitled.png)

![https://file+.vscode-resource.vscode-cdn.net/home/shinymack/vision_transformers_from_scratch/notes/images/image.png](https://file+.vscode-resource.vscode-cdn.net/home/shinymack/vision_transformers_from_scratch/notes/images/image.png)

For 2D-space, we can arrange the coordinates where i and j lands in a 2x2 matrix

![Untitled](Linear%20Algebra%20Notes%20f58400cb6eec4aadbad204f8c5e40a17/Untitled%201.png)

![https://file+.vscode-resource.vscode-cdn.net/home/shinymack/vision_transformers_from_scratch/notes/images/image1.png](https://file+.vscode-resource.vscode-cdn.net/home/shinymack/vision_transformers_from_scratch/notes/images/image1.png)

Here, (a,c) is where i lands and (b,d) is where j lands.

(x,y) is an arbitrary point whose coordinates we want to find after the transformation.

There are many transformations that can be applied here like rotation, shear, etc.

---

## Chapter 4: Matrix Multiplication as Composition

### Composition:

When you apply one linear transformation and then another, it is called composition of transformations.

### Matrix Multiplication:

- If A and B are matrices representing two linear transformations, then the matrix AB represents the composition of those transformations.
- If B is applied first then A, then the final composition matrix will be AB.

i.e. AB != BA

Let v be a vector, Transformed v = A(Bv).

---

## Chapter 5: Three-dimensional linear transformations

Similar to two dimensional but with an extra dimension.

So, Transforming matrix would be a 3x3 matrix.

---

## Chapter 6: The determinant

After a linear transformation, Ares may get bigger or smaller.

For the area of the unit square, if we know the factor by which area of that unit square changes we can find the area of any block after the transformation.

Geometrically, the determinant of a transforming matrix represents the factor by which the area of the unit square changes after applying that transformation.

![Untitled](Linear%20Algebra%20Notes%20f58400cb6eec4aadbad204f8c5e40a17/Untitled%202.png)

Flipping of the space implies negative determinant and vice versa.

Similarly 3D-space, the determinant represents the scaling of volumes.

![Untitled](Linear%20Algebra%20Notes%20f58400cb6eec4aadbad204f8c5e40a17/Untitled%203.png)

---

## Chapter 7: Inverse Matrices, column space and null space

### Inverse Matrices:

- The inverse of a matrix A, denoted A⁻¹, is the matrix such that AA⁻¹ = A⁻¹A = I, where I is the identity matrix.(or Identity transformation)
- Not all matrices have inverses. A matrix has an inverse if and only if it is non-singular (i.e., its determinant is non-zero).
- If the determinant is zero, then the transformation squishes the area into a lower dimensional space which then cannot be transformed back into a higher dimensional space as zero cannot be scaled up.
    
    

This is used to find something like x in:
Ax = v (where A is a transformation and x and v are vectors)

### Rank:

The number of dimensions in the output of a transformation.

i.e. line → Rank = 1, 2D-area → Rank = 2, etc.

### Column space:

Span of the output of the transformation.

So, rank can also be defined as no. of dimensions in column space.

**Null space/ Kernel:** Set of vectors than land on the origin after a transformation.

---

## Chapter 8: Non-square matrices as transformation between dimension

This can mean a lower dimensional space is transformed to a higher dimensional one and vice versa.

---

## Chapter 9: Dot products and Duality

**Geometric interpretation:**

![Untitled](Linear%20Algebra%20Notes%20f58400cb6eec4aadbad204f8c5e40a17/Untitled%204.png)

Mathematically, v.w = 4.2 + 1.-1 = 7

2d to 1d linear transformations can be interpreted as dot products of vectors in 2d space.

A transformation [a b] squishes 2D-space into a line where a is where i lands and b is where j lands, then (a,b) can also be interpreted as a vector in 2D-space.

Here, one of the matrix is interpreted as a row matrix and matrix-vector product is performed which is same as dot product.

---

## Chapter 10: Cross products

The cross product of two vectors is a vector perpendicular to the two vectors and with length equal to the area of the parallelogram formed by the two vectors.

The area of the parallelogram can also be interpreted as the determinant’s geometric interpretation.

Mathematical interpretation:

![Untitled](Linear%20Algebra%20Notes%20f58400cb6eec4aadbad204f8c5e40a17/Untitled%205.png)

Geometric interpretation in next chapter.

---

## Chapter 11: Cross products in terms of L.T

![Untitled](Linear%20Algebra%20Notes%20f58400cb6eec4aadbad204f8c5e40a17/Untitled%206.png)

2D version of cross product is simply the determinant of the matrix taking the vectors as columns.

“What vector p has the special property, that when you take a dot product between the p and some vector [x,y,z] it gives the same result as plugging [x,y,z] to the first column of the matrix whose other two columns have the coordinates of v and w, then computing the determinant?”

---

## Chapter 12: Cramer’s rule

Used to find solution to linear system of equations.

The given system is written as matrix multiplication of a unknown vector [x y] or [x y z] and a transformation which is equal to some given vector;

So, the det(A) should not be zero, where A is the transforming matrix.