import numpy as np

def base_case():
    # Numpy uses ndarrays
    # We can create ndarrays from arrays using the array() function as passing 
    # an array as an argument
    arr = np.array([1, 2, 3, 4, 5])
    print(arr)
    print(type(arr))

    # We can even pass it a tuple
    arr = np.array((1, 2, 3, 4, 5))
    print(arr)
    print(type(arr))

def dimensions():
    # A dimension (in arrays) is 
    # one level of 
    # array depth

if __name__ == "__main__":
    base_case()