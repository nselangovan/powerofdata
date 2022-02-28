# Elangovan S D21 and D22 batch for numpy task
"""
Task 1 for importing the numpy and give alias as np
# 1. Import the numpy package under the name np
"""
print('1. Import the numpy package under the name np')
import numpy as np
# 2. Print the numpy version and the configuration
print('2. Print the numpy version and the configuration')
print("numpy version ", np.__version__)
print("numpy config" , np.show_config)
# 3. Create a null vector of size 10
print('3. Create a null vector of size 10')
null_vector = np.zeros(10)
print("null vector ", null_vector)
# 4. How to find the memory size of any array
print('4. How to find the memory size of any array')
array_size = null_vector.size
array_item_size = null_vector.itemsize
#array_shape = null_vector.shape
#print("array shape is : " ,array_shape)
print("array size is : " ,array_size)
print("array item size is : ", array_item_size)
# 5. How to get the documentation of the numpy add function from the command line?
print('5. How to get the documentation of the numpy add function from the command line?')
np_info = np.info
np.info(np.add)
#print(numpy.info(numpy.add))
#print("array np_info is : " ,np_info)
print("******************** End of 5 questions ********************")
# 6. Create a null vector of size 10 but the fifth value which is 1
print('6. Create a null vector of size 10 but the fifth value which is 1 ')
x = np.zeros(10)
x[4]= 1
print (x)

# 7. Create a vector with values ranging from 10 to 49
print('7. Create a vector with values ranging from 10 to 49')
v_r_10_49 = np.arange(10,50)
print(v_r_10_49)

# 8. Reverse a vector (first element becomes last)
print('8. Reverse a vector (first element becomes last) ')
reverse_v_r_10_49 = v_r_10_49[::-1]
print(reverse_v_r_10_49)
# 9. Create a 3x3 matrix with values ranging from 0 to 8
print('9. Create a 3x3 matrix with values ranging from 0 to 8')
mat_3_3 = np.arange(9)
mat_reshape_3_3 = mat_3_3.reshape(3,3)
print (mat_reshape_3_3)
# 10. Find indices of non-zero elements from [1,2,0,0,4,0]
print('10. Find indices of non-zero elements from [1,2,0,0,4,0]')
non_zero = np.nonzero([1,2,0,0,4,0])
print(non_zero)
print("******************** End of 10 questions ********************")
# 11. Create a 3x3 identity matrix
print('11. Create a 3x3 identity matrix ')
idt_mat = np.eye(3)
print(idt_mat)
# 12. Create a 3x3x3 array with random values
print('12. Create a 3x3x3 array with random values ')
rand_arr_3_3_3 = np.random.uniform((3,3,3))
print (rand_arr_3_3_3)
# 13. Create a 10x10 array with random values and find the minimum and maximum values
print('13. Create a 10x10 array with random values and find the minimum and maximum values ')
rand_min_max = np.random.random((10,10))
r_min, r_max = rand_min_max.min(), rand_min_max.max()
print(rand_min_max)
print(r_min, r_max)
# 14. Create a random vector of size 30 and find the mean value
print('14. Create a random vector of size 30 and find the mean value ')
rand_vet = np.random.random(10)
rand_vet_mean = rand_vet.mean()
print(rand_vet)
print (rand_vet_mean)
# 15. Create a 2d array with 1 on the border and 0 inside
print('15. Create a 2d array with 1 on the border and 0 inside ')
array_ones = np.ones((10,10))
print(array_ones)
array_ones[1:-1,1:-1]=0
print(array_ones)
print("******************** End of 15 questions ********************")
# 16. How to add a border (filled with 0's) around an existing array?
print('16. How to add a border (filled with 0 s) around an existing array?')
bord_zero = np.ones((3,3))
print("Original array:\n" , bord_zero )
bord_zero = np.pad(bord_zero, pad_width=1, mode='constant', constant_values=0)
print("0 on the border and 1 inside in the array \n", bord_zero)
# 17. What is the result of the following expression?
print('17. What is the result of the following expression? ')
"""
(hint: NaN = not a number, inf = infinity)
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1
"""
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)
# 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
print('18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal ')
diag_mat = np.diag(1+np.arange(4), k = -1)
print (diag_mat)
# 19. Create a 8x8 matrix and fill it with a checkerboard pattern
print('19. Create a 8x8 matrix and fill it with a checkerboard pattern ')

checker_board_patt = np.zeros ((8,8), dtype=int)
checker_board_patt[1::2, ::2]= 1
checker_board_patt[::2, 1::2] = 1
print (checker_board_patt)
# 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
print('20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?')
print (np.unravel_index(100, (6,7,8)))

print("******************** End of 20 questions ********************")
# 21. Create a checkerboard 8x8 matrix using the tile function
print('21. Create a checkerboard 8x8 matrix using the tile function ')

tile_array= np.array([[0,1], [1,0]])
tile_array_val = np.tile(tile_array,(4,4))
print (tile_array_val)

# 22. Normalize a 5x5 random matrix
print('22. Normalize a 5x5 random matrix ')
normalize_rand_mat = np.random.random((5,5))
#print(normalize_rand_mat)
Zmax, Zmin = normalize_rand_mat.max(), normalize_rand_mat.min()
normalize_rand_mat= (normalize_rand_mat-Zmin)/(Zmax-Zmin)
print (normalize_rand_mat)

# 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA)
print('23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) ')

RGBA = np.dtype([('red',np.uint8),('green',np.uint8),('blue',np.uint8),('alpha',np.uint8)])
color = np.array((1,2,4,3),dtype = RGBA)
print(color['red'])
print(type(color))

# 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
print('24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)')
arr_1 = np.random.random((5,3))
arr_2 = np.random.random((3,2))
print(arr_1 @ arr_2)
new_arr_1_2 = np.dot(arr_1,arr_2)
print(new_arr_1_2)

# 25. Given a 1D array, negate all elements which are between 3 and 8, in place.
print('25. Given a 1D array, negate all elements which are between 3 and 8, in place.')
# arr = np.arange(16)
# #print(arr>=3)
# arr[(arr>=3) & (arr<=8)]*=(-1)
# print(arr)

arr = np.arange(16)
print(type(arr))
print(arr>=3)
arr[(arr>=3) & (arr<=8)]*=(-1)
print(arr)

print("******************** End of 25 questions ********************")
# 26. What is the output of the following script?
print('26. What is the output of the following script? ')
# print(sum(range(5),-1))
# from numpy import *
# print(sum(range(5),-1))
kkk = np.arange(5)
print(np.sum(kkk))

# 27. Consider an integer vector Z, which of these expressions are legal?
print('27. Consider an integer vector Z, which of these expressions are legal? ')
int_vect = np.arange(3)
print(int_vect , type(int_vect))
int_vect ** int_vect       # = [0^0, 1^1, 2^2] = [1, 1, 4]
print('s*s',int_vect ** int_vect)
2 << int_vect >> 2  # = [0, 1, 2]
print(2 << int_vect >> 2)
int_vect < - int_vect      # = [False, False, False]
print(int_vect < - int_vect)
1j * int_vect       # = [0 + 0.j, 0 + 1.j, 0 + 2.j]
print(1j * int_vect)
int_vect / 1 / 1    # = [0, 1, 2]
print(int_vect / 1 / 1 )
#Z < Z > Z    # ValueError
#print(int_vect < int_vect > int_vect)

# 28. What are the result of the following expressions?
print('28. What are the result of the following expressions?')
# print(np.array(0) / np.array(0))
# print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))

# 29. How to round away from zero a float array ?
print('29. How to round away from zero a float array ? ')
round_away = np.random.uniform(-10, +10, 10)
print(round_away)
print(np.abs(round_away) ,np.ceil(round_away) )
round_away_val = np.copysign(np.ceil(np.abs(round_away)), round_away)
print(round_away_val)

# 30. How to find common values between two arrays?
print('30. How to find common values between two arrays? ')

common_values_x = np.arange(0, 10)
common_values_y = np.arange(5, 15)
print(np.intersect1d(common_values_x, common_values_y))

print("******************** End of 30 questions ********************")

# 31. How to ignore all numpy warnings (not recommended)?
# Suicide mode on
print('31. How to ignore all numpy warnings (not recommended)?')
defaults = np.seterr(all="ignore")
error_np_ignore = np.ones(1) / 0
print(error_np_ignore)
# 32. Is the following expressions true?
print('32. Is the following expressions true?')
print(np.sqrt(-1) == np.emath.sqrt(-1))
# False

# 33. How to get the dates of yesterday, today and tomorrow?
print('33. How to get the dates of yesterday, today and tomorrow?')
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
print("Yestraday: ",yesterday)
today     = np.datetime64('today', 'D')
print("Today: ",today)
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print("Tomorrow: ",tomorrow)

# 34. How to get all the dates corresponding to the month of July 2016?
print('34. How to get all the dates corresponding to the month of July 2016? ')

corrs_month = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(corrs_month)

# 35. How to compute ((A+B)*(-A/2)) in place (without copy)?
print('35. How to compute ((A+B)*(-A/2)) in place (without copy)? ')
A = np.ones(3) * 1
B = np.ones(3) * 2
C = np.ones(3) * 3
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)
print(A, B, C , np.add(A, B, out=B) , np.divide(A, 2, out=A) , np.negative(A, out=A) , np.multiply(A, B, out=A))
# not sure

print("******************** End of 35 questions ********************")
# 36. Extract the integer part of a random array using 5 different methods
print('36. Extract the integer part of a random array using 5 different methods ')
Z = np.random.uniform(0,10,10)

print(Z - Z%1)
print(Z // 1)
print(np.floor(Z))
print(Z.astype(int))
print(np.trunc(Z))

# 37. Create a 5x5 matrix with row values ranging from 0 to 4
print('37. Create a 5x5 matrix with row values ranging from 0 to 4 ')

mat_5_5_val_0_4 = np.zeros((5,5))
mat_5_5_val_0_4 += np.arange(5)
print(mat_5_5_val_0_4)

# 38. Consider a generator function that generates 10 integers and use it to build an array
print('38. Consider a generator function that generates 10 integers and use it to build an array ')
def generate():
    for x in range(10):
        yield x
gen_fun_10_int = np.fromiter(generate(),dtype=int,count=-1)  # or float
print(gen_fun_10_int)

# 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded
print('39. Create a vector of size 10 with values ranging from 0 to 1, both excluded ')

vect_10_rang_0_to_1 = np.linspace(0,1,11,endpoint=False)[1:]
print(vect_10_rang_0_to_1)

# 40. Create a random vector of size 10 and sort it
print('40. Create a random vector of size 10 and sort it ')
rand_vect_s_10_sort = np.random.random(10)
rand_vect_s_10_sort.sort()
print(rand_vect_s_10_sort)

print("******************** End of 40 questions ********************")
# 41. How to sum a small array faster than np.sum?
print('41. How to sum a small array faster than np.sum? ')
sum_arr_fast = np.arange(10)
sum_arr_fast_val = np.add.reduce(sum_arr_fast)
print(sum_arr_fast_val)

# 42. Consider two random array A and B, check if they are equal
print('42. Consider two random array A and B, check if they are equal ')
arr_A = np.random.randint(0,2,5)
arr_B = np.random.randint(0,2,5)
arr_C = arr_A
equal_2_arr = np.allclose(arr_A,arr_B)
print(equal_2_arr)
equal_2_arr = np.allclose(arr_A,arr_C)
print(equal_2_arr)

# 43. Make an array immutable (read-only)
print('43. Make an array immutable (read-only) ')
immutable_arr = np.zeros(10)
immutable_arr.flags.writeable = False
# immutable_arr[0] = 1
print('immutable arr', immutable_arr)

# 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates
print('44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates ')

convert_Z = np.random.random((10,2))
convert_X,convert_Y = convert_Z[:,0], convert_Z[:,1]
polar_R = np.sqrt(convert_X**2+convert_Y**2)
polar_T = np.arctan2(convert_Y,convert_X)
print(polar_R)
print(polar_T)
print("******************** End of 45 questions ********************")

# 45. Create random vector of size 10 and replace the maximum value by 0
print('45. Create random vector of size 10 and replace the maximum value by 0 ')
array_max_val = np.random.random(10)
array_max_val[array_max_val.argmax()] = 0
print(array_max_val)
# 46. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area
print('46. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area ')
cord_z = np.zeros((5, 5), [('x', float), ('y', float)])
cord_z['x'], cord_z['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
print(cord_z['x'], cord_z['y'])
# 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))
print ('47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))')

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))

# 48. Print the minimum and maximum representable value for each numpy scalar type
print('48. Print the minimum and maximum representable value for each numpy scalar type ')
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
# 49. How to print all the values of an array?
print('49. How to print all the values of an array?')
np.set_printoptions(threshold=float("inf"))
Z = np.zeros((40,40))
print(Z)

# 50. How to find the closest value (to a given scalar) in a vector?
print('50. How to find the closest value (to a given scalar) in a vector?')
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
print("******************** End of 50 questions ********************")

# 51. Create a structured array representing a position (x,y) and a color (r,g,b)
print('51. Create a structured array representing a position (x,y) and a color (r,g,b)')
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)

# 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances
print('52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances ')
Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)

# Much faster with scipy
import scipy
# Thanks Gavin Heverly-Coulson (#issue 1)
import scipy.spatial

Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)

# 53. How to convert a float (32 bits) array into an integer (32 bits) in place?
print('53. How to convert a float (32 bits) array into an integer (32 bits) in place?')
Z = (np.random.rand(10)*100).astype(np.float32)
Y = Z.view(np.int32)
Y[:] = Z
print(Y)

# 54. How to read the following file?
print('54. How to read the following file?')

from io import StringIO

# Fake file
s = StringIO('''1, 2, 3, 4, 5
                6,  ,  , 7, 8
                 ,  , 9,10,11
''')
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z)
# 55. What is the equivalent of enumerate for numpy arrays?
print('55. What is the equivalent of enumerate for numpy arrays?')
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
for index in np.ndindex(Z.shape):
    print(index, Z[index])

print("******************** End of 55 questions ********************")
# 56. Generate a generic 2D Gaussian-like array
print('56. Generate a generic 2D Gaussian-like array')
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)

#### 57. How to randomly place p elements in a 2D array?
print('57. How to randomly place p elements in a 2D array?')
n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print(Z)

# 58. Subtract the mean of each row of a matrix
print('58. Subtract the mean of each row of a matrix')
X = np.random.rand(5, 10)
# Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)
# Older versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)

print(Y)
# 59. How to sort an array by the nth column?
print('59. How to sort an array by the nth column?')

Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])

# 60. How to tell if a given 2D array has null columns?
print('60. How to tell if a given 2D array has null columns?')
Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())

print("******************** End of 60 questions ********************")
# 61. Find the nearest value from a given value in an array
print('61. Find the nearest value from a given value in an array')

Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)

# 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator?
print('62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator?')
A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: z[...] = x + y
print(it.operands[2])

# 63. Create an array class that has a name attribute
print('63. Create an array class that has a name attribute')

class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)

# 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)?
print('64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)?')
Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)

# Another solution
# Author: Bartosz Telenczuk
np.add.at(Z, I, 1)
print(Z)

# 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)?
print('65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)?')
X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)

print("******************** End of 65 questions ********************")
