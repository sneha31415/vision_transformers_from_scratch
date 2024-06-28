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
![span point representation](/assets/span_points.png)

**span of two vectors which line up**
![span point representation case 2](/assets/span_points_2.png)

## span of vectors in 3D
a flat sheet cutting through the origin is the span of two vectors in 3D 
![span_3D](/assets/span3D.png)
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
![linear trans](/assets/linear_trans.png)

## after transformation:
note the new coordinates of i hat and j hat(basis vector)
now for the transformed vector coordinates, multiply new i hat and j hat coordinates with the original x, y coordinates
![transformed](/assets/transformed.png)

## chapter 4 
### Matrix multiplication as composition
 composition - rotation then shear is also a new _linear transformation_ 

 ![composition2](/assets/composition2.png)
![composition](/assets/composition.png)
first apply the effect on the right(here rotation cuz rotation is carried out first) then the left one (here shear)

so matrix multiplication represents applying one transformation after other
M1M2 != M2M1 i.e order matters

## chapter 5
###  Three-dimensional linear transformations

![transformation3D](/assets/transformation3D.png)

so, the multipliction of two matrices can be inferred as the right being the first transformation applied, and the left being the second transformation being applied 

![transformation3D_2](/assets/transformation3D_2.png)


## chapter 6
### The determinant

what happens to one grid after transformation happens to any of the other grid 

so, the area will also be scaled by the same amount 
the factor by which the area changes is called the "determinant" of that transformation 