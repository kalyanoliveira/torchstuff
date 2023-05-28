import numpy as np

# There are several ways to create numpy arrays
# np.array(data), np.zeros(shape), np.ones(shape), np.empty(shape)
def test0():
    data = [1, 2, 3, 4, 5, 6]
    print(data)

    # From existing data
    my_array = np.array(data)
    print(my_array)

    # From a shape
    my_array = np.zeros(2)
    print(my_array)
    my_array = np.ones(2)
    print(my_array)
    my_array = np.empty(2)
    print(my_array)

    # From "a" desired range of numbers and a step size
    # np.arange(start (inclusive), stop (exclusive), step)
    my_array = np.arange(0, 10, 2)
    print(my_array)

    # From a desired range of numbers and a number of intervals
    # This is called a "lin"early "space"d array
    my_array = np.linspace(0, 10, num=5)
    print(my_array)

# You can also specify the data type of any created array during its creation
def data_type():
    my_array = np.arange(0, 10, 2, dtype=np.float64)
    print(my_array)

    # int64 is also known as a "long"
    my_array = np.arange(0, 10, 2, dtype=np.int64)
    print(my_array)
    
    # The default data type, in this instance, is int64
    my_array = np.arange(0, 10, 2)
    print(my_array, my_array.dtype)

# Sorting the elements of an np.array in ascending order is super easy
def sorting():
    my_array = np.array([2, 1, 5, 3, 7, 4, 6, 8])

    print(my_array)
    print(np.sort(my_array))

# And concatenating them too!
def concatenating():
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])

    print(np.concatenate((a, b)))
    
    # Shape is 2, 2
    x = np.array([[1, 2], [3, 4]])
    # Shape is 1, 2
    y = np.array([[5, 6]])

    print(np.concatenate((x, y), axis=0))

# When it comes to np arrays, we can talk about
# Number of Axes/Dimensions
# Shape 
# Size (otal number of elements) (the product of the elements in Shape)
def sizes():
    my_array = np.array(
                        [
                         [
                          [0, 1, 2, 3],
                          [4, 5, 6, 7]
                         ],
                         [
                          [0, 1, 2, 3],
                          [4, 5, 6, 7]
                         ],
                         [
                          [0, 1, 2, 3],
                          [4, 5, 6, 7]
                         ]
                        ]
                       )

    print("Shape:", my_array.shape)
    print()
    print("The number of elements in the Shape is known as")
    print("Axes/Dimensions:", my_array.ndim)
    print()
    print("The product of the elements in the Shape is known as")
    print("Total number of elements/Size:", my_array.size)
    
# You can also reshape arrays, so long as the size/number of elements remains the same
def reshaping():
    # First, we can reshape an existing np array using a method
    my_array = np.arange(0, 8)
    print(my_array)

    new_shape = (2, 4) 
    my_array = my_array.reshape(new_shape)
    print(my_array)

    new_shape = (1, 8)
    my_array = my_array.reshape(new_shape)
    print(my_array)

    new_shape = (8)
    my_array = my_array.reshape(new_shape)
    print(my_array)
    
    # We can also use the reshape function
    # which also gives us access to a few extra parameters
    my_array = np.reshape(my_array, newshape=(4, 2), order="C")
    print(my_array)
    
    # The "order" utitilized here has to do with some very low-level indexing 
    # differences between languages like C and Fortran. So yeah, not really 
    # too important at this moment

# How to increase axes/dimensions
def increasing_dimensions():
    # First of all, there exists an axes notation that needs to be explained here
    # Say an np array has shape (2, 4): An array with two inner-arrarys, where
    # each inner-array has 4 elements

    # The notation [1, 3] fetches the last element of the second inner-array

    # The notation [0:1, 1], which is equivalent to [:1, 1], allows one to 
    # splice the main array, obtaining only the second inner-array, and then
    # accesses the second element of that array
    a = np.arange(0, 8)
    a = a.reshape((2, 4))
    print(a)
    print()
    print(a[1, 3])
    print()
    print(a[0:1])
    print(a[:1])
    print()
    print(a[:1, 1])
    print("\n")

    print("Initial array")
    a = np.arange(0, 8)
    print(a)
    print("Now increasing dimensions")
    print(a[np.newaxis,:])
    print("Now increasing dimensions, but hopefully breaking something along the way")
    print(a[:, np.newaxis])
    print("wow, that actually worked?")
    
    print("Now using the \"expand dims\" library function")
    # The number in "axis" is an index in the "Shape" tuple
    print(np.expand_dims(a, axis=1))

    print()
    print("Also, np.newaxis is just an alias for None")
    print(a[0:5, np.newaxis])
    print("And you can use the ellipsis to not have to write the other axes")
    print(a[..., np.newaxis])

def conditional_splicing():
    a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print("A:\n", a)
    print(a[(a < 5)])

    print()
    # This shit is actually so cool
    # It creates a boolean map
    # When you give this boolean map to the splicer-notation, np decides which
    # elements of the number should be produced or not
    five_up = (a >= 5)
    print("Here's my boolean map of \"a\"")
    print(five_up)
    print(five_up.shape)
    print()

    b = np.arange(0, 4)
    print("B:", b)
    print()
    print("Applying the boolean map", five_up[0], "to B, we get:")
    print(b[five_up[0]])
    print()
    print("Applying the boolean map", five_up[1], "to B, we get:")
    print(b[five_up[1]])

    print()
    bool_map = (a > 2) & (a < 11)
    print(bool_map)
    print("Applying the Boolean map to A")
    print(a[bool_map])

def more_conditions():
    # Overall task: show me the indexes of non-zero elements
    a = np.array([[1, 2, 3, 4], [5, 0, 7, 8], [9, 10, 11, 12]])
    print(a, "\n")
    
    # The first array here shows the first-axis (or first-dimension) index of the element
    # The second array shows the second-axis (or second-dimension) index of the element
    # Which means that once we zip these result, we will have a list of coordinates
    # And hopefully that explains better what np.nonzero actually does
    b = np.nonzero(a)
    print(b, "\n")
    list_of_coordinates = list(zip(b[0], b[1]))
    print("List of coordinates:")
    print(list_of_coordinates)

    print()
    print("Here's another example")
    b = np.nonzero(a < 5)
    print("B:", b)
    print("A evaluated for B:", a[b])

    # So what this really illustrates is that np.nonzero(array condition) 
    # really just returns some form of a boolean map, which can later be
    # used in string indexing and splicing

def stacking():
    a = np.array([[1, 2],
                 [3, 4]])
    b = np.array([[5, 6],
                 [7, 8]])

    c = np.vstack((a, b))
    print(c)

    d = np.hstack((a, b))
    print(d)

def views():
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(a)

    # b1 is a pointer, or a view``
    b1 = a[0, :]
    b1[0] = 99

    print(a)

    print()

    # You can create a totally new copy (known as a "deep copy") by using copy()
    b2 = a.copy()

    b2[0, :][0] = 41

    print(b2)
    print(a)

def operations():
    data = np.array([1,2])
    ones = np.ones(2, dtype=int)
    print(data, ones)
    print("Addition")
    print(data + ones)
    print("Multiplication")
    print(data * ones)
    print("Subtraction")
    print(data - ones)
    print("Division")
    print(data / ones)

    data = np.array([1, 2, 3, 4])
    data = np.arange(1, 5)

    print()
    print(data)
    print("Summation")
    print(data.sum())

    # In summations below, "axis=" can be interpreted as
    # "the common axis of change"
    # Here's an example question to make this concept clear:
    # What is the effect of axis=0?
    # First numpy does the summation between [0, 0] and [1, 0]
    # And then it does the summation between [0, 1] and [1, 1]
    # Notice how in both summations the axis0 index changed, and the axis1
    # remained constant
    # This means that during any summation, the "axis-index" that changes is the 
    # the one specified in the "common axis of change"
    
    # Now, if this actually holds true for other examples with different shapes
    # of arrays, now that's an entirely different conversation (it really isn't)
    print()
    print("New data")
    data = np.array([[1, 2], [3, 4]])
    print(data)
    print("Summation across axis 0")
    print(data.sum(axis=0))
    print("Summation across axis 1")
    print(data.sum(axis=1))

    print()
    print("You can also min max")
    print("Min, max over axis 0")
    print(data.min(axis=0))
    print(data.max(axis=0))
    print("Min, max over axis 1")
    print(data.min(axis=1))
    print(data.max(axis=1))


def broadcast():
    data = np.array([1, 2])
    print(data * 1.6)

def matrices_indexing_slicing():
    data = np.arange(1, 7).reshape((3, 2))
    print(data)
    
    print(data[0, 1])
    print(data[1:3])
    
    # This next one might be the most confusing
    # Think of it like this: 
    # from the data array, access everything from index 0 to index 1 in the "zero" axis
    # This returns two elements: [1, 2], the element of index 0 in the "zero" axis
    # and [3, 4], the element of index 1 in the "zero" axis
    # For each of these elements, access what is at index 0
    # For [1, 2], the index 0 returns 1
    # and for [3, 4], the element at index 0 returns 3
    # Hence, [0:2, 0] returns an array of elements [1, 3]

    # It's counter-intuitive at first: you would expect [0:2, 0] to return [1,2]
    # because [0:2] just means [[1, 2], [3, 4]]
    # But in order to get [1,2], we must do [0:2][0]
    # Hence, [0:2, 0] is different than [0:2][0]
    print(data[0:2, 0])

def matrices_min_max():
    data = np.array([[1, 2], [5, 3], [4, 6]])
    print(data)

    print(data.max(axis=0))
    print(data.min(axis=1))

def matrix_operation():
    data = np.arange(1, 5).reshape(2, 2)
    print(data)
    print()

    ones = np.ones((2, 2))
    print(ones)
    print()

    print(data + ones)
    print()

    data = np.arange(1, 7).reshape((3, 2))
    ones = np.ones(2)
    print(data)
    print()
    print(ones)
    print()

    print(data + ones)

def randomizer():
    # Using this rng thing
    rng = np.random.default_rng()
    print(rng.random(3))
    print(rng.random((3, 2)))

    print()

    # Using just the random() function inside np.random
    # Kind of like import random as rand; rand.random()
    print(np.random.random((3, 2)))
    
    print()

    # Generate random integers between 0 and 4
    print(rng.integers(5, size=(3, 2)))
    
    print()
    # Or by using this function
    print(np.random.randint(5, size=(3, 2)))

def unique():
    a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])
    print(a)

    print("Now displaying the array in its unique-values form, alongside the\
index of where each unique value was taken from")
    uniques, uniques_indexes = np.unique(a, return_index=True)
    print(list(zip(uniques, uniques_indexes)))

    print()
    print("Basically doing the same thing, but this time displaying the count\
 (or frequency) of each unique value")
    uniques, uniques_count = np.unique(a, return_counts=True)
    print(list(zip(uniques, uniques_count)))
    
    print()
    print("This also works for 2D arrays")
    a_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 4]])
    print(a_2d)
    print()
    print(np.unique(a_2d))
    print()
    print(np.unique(a_2d, axis=0))
    # Yeah don't really understand this one
    print(np.unique(a_2d, axis=1))

def transposing():
    my_array = np.arange(6).reshape((2, 3))
    print(my_array)
    print(my_array.transpose())
    print(my_array.T)

def reversing_flipping():
    my_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(my_array)
    print()

    print(np.flip(my_array))
    print()

    my_array[0] = np.flip(my_array[0])
    print(my_array)
    print()

    my_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(np.flip(my_array, axis=0))
    
def flatten_ravel():
    x = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(x)
    print()

    print(x.flatten())
    print()

    a = x.flatten()
    a[0] = 99
    print(a)
    print(x)
    print()
    
    # This makes b become a pointer, or a view, even though its flattened
    b = x.ravel()
    b[0] = 99
    print(b)
    print(x)

def mean_square_error():
    predictions = np.ones(3)
    labels = np.arange(1, 4)
    print(predictions)
    print(labels)
    error = 1/3 * np.sum(np.square(predictions - labels))
    print(error)

def saving_files():
    my_array1 = np.array([1, 2, 3, 4, 5, 6])
    np.save("filename", my_array1)
    b = np.load("filename.npy")
    print(b)
    
    print()

    my_array2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    np.savetxt("new_file.csv", my_array2)
    b = np.loadtxt("new_file.csv")
    print(b)

if __name__ == "__main__": 
    saving_files()