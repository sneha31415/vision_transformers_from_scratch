# Essence of linear algebra
## chapter 1:

everytriplet of number gives one unique vector and every vector gives unique triplet of numbers
This is true for any dimension


### vector addition
move along the direction of the first vector and then along the direction of the second, the resultant is the sum of the vector
![vector addition](assets/addition.png)

### vector multiplication =
a * V where V is a vector
stretches if a > 1
squish it down if 0 < a < 1
also reverses if a < 0 


## chapter 2
### Linear combinations, span, and basis vectors 
i hat and j hat are the basis vectors of the xy coordinate system

so, we can choose different basis vector and build a completely new coordinate system 

in linear combination of two vectors, if we fix one scalar and change the other, the resultant vector traces a line
![linear combination](assets/linear_combn.png)

if we change both scalars, the we get every possible vector in the space
(exception - both vectors are lined up or they are zero)

*span of two vectors*
![span](assets/span)

for collection of vectors, we think of them as points to avoid crowdedness

**span of two vectors**
![span point representation](assets/span_points.png)

**span of two vectors which line up**
![span point representation case 2](assets/span_points_2.png)

## span of vectors in 3D
a flat sheet cutting through the origin is the span of two vectors in 3D 
![span_3D](assets/span3D.png)
(tip: 1st vector scaling gives 1 line, 2nd vector scaling gives another line, so we get a sheet)

for span of three vectors in 3D:
 we get 3 vectors, so 3 lines and thus we can access each point of the 3D space

 **if each vector add another dimension to the span, they are termed as linearly independent vectors**

 _Technical definition of basis:_
The basis of a vector space is a set of linearly independent vectors that span the full space

## chapter 3
### Linear transformations and matrices

what are linear transformations?
-> for a transformation to be linear all lines must remain lines 
and the origin must be at place

the below example looks like a linear transformation but actually it isnt because the diagonal gets curved
![linear trans](assets/linear_trans.png)

## after transformation:
note the new coordinates of i hat and j hat(basis vector)
now for the transformed vector coordinates, multiply new i hat and j hat coordinates with the original x, y coordinates
![transformed](assets/transformed.png)

## chapter 4 
### Matrix multiplication as composition
 composition - rotation then shear is also a new _linear transformation_ 

 ![composition2](assets/composition2.png)
![composition](assets/composition.png)
first apply the effect on the right(here rotation cuz rotation is carried out first) then the left one (here shear)

so matrix multiplication represents applying one transformation after other
M1M2 != M2M1 i.e order matters

## chapter 5
###  Three-dimensional linear transformations

![transformation3D](assets/transformation3D.png)

so, the multipliction of two matrices can be inferred as the right being the first transformation applied, and the left being the second transformation being applied 

![transformation3D_2](assets/transformation3D_2.png)


## chapter 6
### The determinant

what happens to one grid after transformation happens to any of the other grid 

so, the area will also be scaled by the same amount 
the factor by which the area changes is called the "determinant" of that transformation 

negative  determinant means ->
the orientation has reversed
like if i hat was to the right of j hat previously, then after reversal of orientation i hat would be to the left of j hat

eg - ![before negative determinant](assets/imagetemp.png)
    ![after negative determinant](assets/image_copy.png)

    in 2D, transformations determine how much area gets scaled and in 3D, transformations determine how much *volume* gets scaled

![vol scaling](assets/image_copy_2.png)

    before scaling:
![3d_scaling](assets/image.png)
    after scaling volume becomes zero: 
![zero_vol](assets/image-1.png)

    To know if there has happened inverse transformation in 3D, take i hat on your "Right hand" middle finger, j hat on your index finger and k hat on your thumb,
    now, after the transformation if you need to use your left hand to get k hat on your thumb , then there is an inverse transformation

    determinant of a 3D matrix 

## chapter 7 
### Inverse matrices, column space and null space

![c7_1](assets/image-2.png)
looking for a vector x which on applying transformation A lands on V

If you do transformation A and then A^-1 then its like doing nothingk 
The transformation that does nothing is called the identity transformation
![c7_2](assets/image-3.png)

rank terminology:
-if output of a transformation is a line, then rank = 1
-it is a plane, rank = 2
-3D space , rank = 3
**rank = no. of dimensions in the output of a transformation**

in case of a 2*2 matrix rank 2 means we have a non-zero determinant and we havent collapsed
but in case of 3*3 matrix rank2 means we have collapsed from a volume to a plane , (but the collapsing is not as much as in rank2{line})

full space => rank of input space == rank after transformation
learnt- inverse matrices, column space, rank, null space, kernel

## chapter 8
### Nonsquare matrices as transformations between dimensions 
Transformation from 2d to 3D
![alt text](assets/image-4.png)
transformation from 2d to 3d-
![alt text](assets/image-7.png)
The place where all the vectors land is a 2D plane slicing through the origin of 3D space
![alt text](assets/image-8.png)
so, a 3*2 matrix has geometric interpretation of the mapping two dimensions to three dimensions since the two colums indicate that the input space has two basis vectors and the three rows indicate that the landing spot for each of those basis vectors is described with three seperate coordinates<br/>
In the below assets/image, the colums indicate that we are starting in a space that has three basis vectors(i.e 3D space) <br/>
and the two rows indicate that the landing spot for each of those three basis vectors is described with only two coordinates(so landing in 2D)


![alt text](assets/image-9.png)


## chapter 9  
### Dot products and duality 
dot pdts are very useful to understand projections
geometric interpretion of dot product
![alt text](assets/image-10.png)
**numbers to 2D vectors and 2D vectors to numbers**
squishing the 2D plane into a number line will squish the vector to numbers

![alt text](assets/image-5.png)
{The number where i hat lands when its projected onto the numberline is same as where u hat lands when uts projected on the x axis(since both are unit vectors so by symmetry we can say this). So, projection of i hat on numberline = Ux}
anytime we have one of any linear transformations whose output space is the number line, there's going to be some unique vector v corresponding to that transformation, in the sense that applying the transformation is the same  thing as taking a dot product with that vector

![alt text](assets/image-6.png)

![alt text](assets/image-11.png)
![alt text](assets/image-12.png)
Duality <=> natural but surprising correspondence
-Applying a transformation is the same as taking as dot product with that vector
![alt text](assets/image-13.png)

## chapter 10
### Cross products
-Area of parallelogram =  cross product of the two matrices that form the sides of the parallelogram
-use the right hand thumb rule to know the direction of the area 
Mathematical Interpretation(trick)-
![alt text](assets/image-14.png)

## chapter 11
### Cross products in the light of linear transformations

![alt text](assets/image-15.png)
what is vector p?
p is something that when dot product with x, y, z gives the same result as the result of det of v, w with x,y,z
![alt text](assets/image-16.png)

geometric interpretation of dot product-
![alt text](assets/image-17.png)
geometric interpretation of cross product
![alt text](assets/image-18.png)
but this is also same as dot product with vector x, y, z and vector p with length of p equal to area of parallelogram 
![alt text](assets/image-19.png)
so, this is the geometrical interpretation of the trick

## chapter 13
### change of basis
[-1 <br/>
  2] in  two different basis
![alt text](assets/image-21.png)
jenifers coordinate system:
![alt text](assets/image-20.png)
depeding on the choice of basis vectors, the spacing between the grid lines and the choice of axes change although origin is same for any choice of basis vectors
<br/>
How e translate between coordinate systems?<br/>
[-1<br/>
  2] in jenifer's cord system
 ![alt text](assets/image-22.png)
-in our cord system i.e with i hat and j hat
![alt text](assets/image-23.png)
so, a matrix whose columns represents jenifers basis vectors can be thought of as a transformation that moves our basis vectors i hat and j hat to jenifers basis vectors<br/>
like if jenifer says [1 0] , [0 1] then in i hat and j hat it means [2 1], [1,-1]<br/><br/>
how to go from our system to jenifer's system?
Take the inverse of the matrix that translates jenifers language to ours 
![alt text](assets/image-24.png)
so to see what [3 2] of our language looks in jenifers language, multiply [3 2] by the inverse of the basis matrix of jenifers
![alt text](assets/image-25.png)

### summary
![alt text](assets/image-26.png)
![alt text](assets/image-27.png)

90 degree rotation of space in i hat and j hat - 
![alt text](assets/image-28.png)
how will jenifer get the same 90 degree acw rotation transformation in her cord system?
First thought is to translate the rotation matrix of i hat and j hat directly into jenifers language using b1 and b2 as discussed earlier
But this is totally wrong as those columns represent where i hat and j hat go and not where jenifers basis vector would land, so simply transforming of coordinates wont do<br/>
The matrix what jenifer wants should represent where her basis vector land
so, steps are :
start with a vector in jenifers language
![alt text](assets/image-29.png)
Translate it to our language
![alt text](assets/image-30.png)
see where this would land in our transformation(i.e shift of 90 degree ACW)
![alt text](assets/image-31.png)
apply inverse change of basis to see where the vector would land in her language
![alt text](assets/image-32.png)

so , the transformation in her language for any vector v in her language is =
![alt text](assets/image-33.png)
![alt text](assets/image-34.png)

if [1 2] is a vector in jenifers coord system, then [-1 1] is the 90 degree shifted vector in jenifers system
![alt text](assets/image-35.png)
the middle matrix represents a transformation
![alt text](assets/image-36.png)
<br>
**A^-1 M A = transformation M but from the perspective of new basis vector A**
next chapter: why we care about different coord systems?

## chapter 14 - eigenvectors and eigenvalues
After a transformation is applied, for the span of a vector, the vector may not necessarily lie on its span
![alt text](assets/image-37.png)
but in some special vectors, they do remain on their span . The only effect of the transformation is that it squishes or stretches
eg i hat in this case. the span of i is the x axis :<br>
    Before transformation:
    ![alt text](assets/image-38.png) 
    after transformation:(there is just a stretch by a factor of 3)<br>
    Any other vector on the x axis follows the same
    ![alt text](assets/image-39.png)
    The same is for a diagonal in this transformation
    ![alt text](assets/image-41.png)
    These special vectors are called as eigen vectors
    The factor by which it gets stretched or squished is called as eigenvalue
    ![alt text](assets/image-42.png)
    If the vector gets flipped over its own span then eigen value = negative

<br/><br/>
In 3D rotation, an eigen vector(a vector that remains on its own span) for that rotation would be the "axis of that rotation"
![alt text](assets/image-43.png)
The eigen value = 1 cuz rotation doesn't squish or stretch anything <br>
Computing:
![alt text](assets/image-44.png)
Here, v is the eigen vector and lambda is the eigen value<br>
The matrix vector multiplication of A and v is same as scaling v by a factor is lambda
![alt text](assets/image-46.png)

Now, product of a matrix with a non zero vector v is zero only when the det of the matrix is zero<br>
This means that the transformation associated with the matrix squishes the space into a lower dimension
<br>
Choose the value of lambda for which the det gets to zero
![alt text](assets/image-47.png)
so, for lambda = 1, there is a vector v for which the (A-lamda.I). v = 0 <br><br>
Computation of eigen value and vector corresponding to it:<br>
get the eigen values
![alt text](assets/image-48.png)
The vector v is the span of all the vectors in the direction x = -y
![alt text](assets/image-49.png)
Note : A 2D transformation might not necesarrily have eigen vectors<br>
eg. Rotating by 90 degree acw
The values of lambda come out to be imaginary, this indicates that there are no eigen vectors as well
![alt text](assets/image-50.png)<br><br>
for the case of shear, the x axis remains in place <br>
The transformation matrix for shear is 
[ 1 1
  0 1] <br>
on solving we get
![alt text](assets/image-51.png)
NOTE: A single eigen value can have more that a line full of eigen vectors<br>
eg, A matrix that scales everything by 2. Here eigen value = 2
![alt text](assets/image-52.png)
every Vector in the plane gets to be an eigenvector with that eigen value

**What if both basis vectors are eigenvectors**
![alt text](assets/image-54.png)
The eigen values of i hat and j hat (here -1 and 2)
lie in the diagonal and every other entry is zero
<br>
This is diagonal matrix<br>
A way to interpret this that in a diagonal matrix, all the diagonal entries are the eigen values and the basis vectors are the eigen vectors
<br>
Advantages:<br>
The 100th power of a diagonal matrix is :
![alt text](assets/image-55.png)
![alt text](assets/image-56.png)
The 100th power of a non diagonal matrix is very tough

<br><br>
But its rare to have basis vectors as eigen vectors<br>
**so, if we have multiple eigen vectors enough so that we can span the whole space, then we can change the cord system so that these eigen vectors are our basis vectors**<br>

![alt text](assets/image-57.png)
![alt text](assets/image-58.png)

The transformation done with the eigen vectors as the basis will result into a diagonal matrix
![alt text](assets/image-60.png)
![alt text](assets/image-59.png)
A set of basis vectors which are also eigenvectors is called eigen basis

<br>
So, to calculate the 100th power of the transformation matrix, first convert it to a diagonal matrix, get the 100th power and then transform it back into the original.
Thus the complex task as 100th power calculation is simplified<br>
However, this is possible only when we have enough eigen vectors.
1 eigen vector span would not be enough

## chapter 16 - Abstract vector spaces 

Determinant and eigen vectors don't care about the coordinate system<br>
What is a vector?
<br>
A vector is an arrow pointing in space, or it is a list of numbers. <br>
But more precisely, a vector is like a function<br>
<br>
L denotes transformation:
![alt text](assets/image-61.png)
<br>
![alt text](assets/image-63.png)
generally:
![alt text](assets/image-64.png)
The roles that these basis functions will do is the same as i hat, j hat, k hat and so on in the world of vectors as arrows <br>
How matrix vector multiplication is similar to derivatives?
![alt text](assets/image-65.png)
The matrix is constructed using the derivative of each basis function like:
![alt text](assets/image-66.png)
--------
![alt text](assets/image-67.png)
Axioms =
![alt text](assets/image-68.png) 
So, vectors are something which follow these rules
![alt text](assets/image-69.png)