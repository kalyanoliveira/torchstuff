import torch
import numpy as np

data = [[1, 2], [3, 4]]

# Tensors can be created from "raw" array data
def test0():
    x_data = torch.tensor(data)
    print(x_data)

# Or from NumPy arrays
def test1():
    np_array = np.array(data)
    x_data = torch.from_numpy(np_array)
    print(x_data)
    
# Or from existing tensors
def test2():
    x_data = torch.tensor(data)
    print(x_data)
    
    x_data = torch.ones_like(x_data)
    print(x_data)

    x_data = torch.zeros_like(x_data)
    print(x_data)
    
    print(x_data.dtype)
    # Because the current dtype of the tensor is int64 (long), we must convert
    # the dtype to float when creating a rand_like because, well, rand creates
    # floating point numbers
    x_data = torch.rand_like(x_data, dtype=torch.float)
    print(x_data)

# Or even from a predetermined shape and a "rule" to fill that shape
# (such as "fill this shape with zeros" or "fill this shape with ones")
def test3():
    shape = (2, 3,) # An array with two inner-arrays, where each inner-array has 3 elements
    x_data1 = torch.ones(shape)
    x_data2 = torch.rand(shape)
    x_data3 = torch.zeros(shape)

    print(x_data1)
    print(x_data2)
    print(x_data3)

# Tensors have three attributes: shape, data-type, and the device on which 
# they are stored
def attributes():
    x_data = torch.tensor(data)
    print(x_data.shape, x_data.dtype, x_data.device)

# We can do stuff on the GPU, if available
def move_to_gpu():
    # As we saw above, tensors are stored by default on the CPU
    x_data = torch.tensor(data)
    print(x_data, x_data.device)

    # We can move them to the GPU in order to get higher compute speeds
    # BUT make sure to first check if you have GPU availability
    if torch.cuda.is_available():
        x_data = x_data.to("cuda")
    
    print(x_data, x_data.device)

def operations():
    shape = (4, 4)
    tensor = torch.rand(shape)
    
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")

    print(tensor)
    print(tensor[0])

def np_index_slice():
    tensor = torch.ones((4, 4))
    tensor[:, 1] = 0
    print(tensor)

def joining_tensors():
    tensor = torch.ones((4, 4))
    tensor[:, 1] = 0
    new_tensor = torch.cat([tensor, tensor, tensor], dim=1)
    print(new_tensor)

def arithmetic_operations():
    # Here are three ways to perform matrix multiplication
    tensor = torch.ones((4, 4)); tensor[:, 1] = 0
    
    # First Matrix
    print(tensor)
    # Second Matrix
    print(tensor.T)

    # First option: with the @ operator
    y1 = tensor @ tensor.T

    # Second option: with the Tensor method
    y2 = tensor.matmul(tensor.T)

    # Third option: with the library function, after creating a buffer
    y3 = torch.rand_like(y1)
    torch.matmul(input=tensor, other=tensor.T, out=y3)

    # Because all of these options are equivalent, we expect y1, y2, and y3 
    # to be equal
    print(y1)
    print(y2)
    print(y3)

def single_elements():
    tensor = torch.ones((4, 4)); tensor[:, 1] = 0
    agg = tensor.sum()
    print(agg)

    agg_item = agg.item()

    print(agg_item, type(agg_item))

def in_place_operations():
    tensor = torch.ones((4, 4)); tensor[:, 1] = 0
    print(tensor)

    # Operations on a tensor that include a _ after their name
    # store the result "in-place"
    
    # Such as the below

    tensor.add_(5)
    
    print(tensor)

    # These kinds of "in-place operations"
    # can be useful for memory purposes,
    # BUT they can be harmful for computing
    # derivatives due to loss of history

    # Thus, the usage of "in-place operations"
    # is discouraged

def bridge_with_numpy():
    # These two will share memory locations
    # which is real cool
    tensor = torch.ones(5)
    n = tensor.numpy()

    print(tensor)
    print(n)
    print()

    n[0] = 0

    print(tensor)
    print(n)
    print()

    # Remember that this is discoraged
    tensor.add_(1)

    print(tensor)
    print(n)

    print("---")

    # You can also start with a numpy array and create a tensor that shares 
    # memory with it
    n = np.ones(5)
    tensor = torch.from_numpy(n)

    print(tensor)
    print(n)
    print()

    n[0] = 0

    print(tensor)
    print(n)
    print()

    # Remember that this is discoraged
    tensor.add_(1)

    print(tensor)
    print(n)
    
if __name__ == "__main__":
    bridge_with_numpy()