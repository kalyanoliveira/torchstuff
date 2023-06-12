import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# So there's a lot that just happened, but let's try to break it down

# First we import from "torchvision" some "datasets"
# I guess this is just a bunch of sample data that the creators of PyTorch have provided


# One of such example of this sample data is FashionMNIST, which is just a 
# bunch of very pixelated pictures of clothes with associated labels
# These labels are really just the categorization of the what the clothe in picture is
# For instance, say that we have a pixelated, 28x28 image of a t-shirt
# That image is stored as a binary, and alongside it we can obtain a label that
# tells us: hey, this binary, it represents a t-shirt
# In reality, that label will be a number
# Kind of like ASCII: we give T-Shirts the label of 1, Pants the label of 2, 
# Shorts the label of 3, and Shoes the label of 4

# In order to actually obtain this FashionMNIST sample dataset, we must
# execute the corresponding FashionMNIST method of "datasets"
# This FashionMNIST() function takes in a couple of arguments

# 1) root: This specifies the name of the folder where the binaries are stored,
#          as far as I am aware
# 2) train: this tells PyTorch if we are going to use that dataset for training.
#           Not entirely sure of the effect that this has right now, but I will
#           figure it out in due time
# 3) download: tells PyTorch whether we should complete our local copy of 
#              the FashionMNIST dataset with internet downloads, should the case
#              of us be missing some data arise
# 4) transform: from what I currently understand, what this tells PyTorch is:
#               whenever I fetch some data from dataset, this is what you should
#               do to transform it from its "binary" version to its "code-ready"
#               version

# Notice that we imported ToTensor() from torchvision.transforms
# What this implies is that there are other ways to transform data from datasets
# to enable code-usage

# And we can actually just see that transform paradigm in action right now

# Turns out that "training_data" and "test_data" are just Dataset's

# (if we actually print out) the type of training_data, we don't actually get
# "Dataset", though
print(type(training_data))
# But PyTorch documentation understands training_data and test_data as really 
# just "Dataset" instances 

# "Dataset"s are indexable. We can just take the 0-index of training_data
# and it returns us the image and it label

# The image is converted into a tensor, thanks to our transform behavior,
# "ToTensor()" that we define above
image, label = training_data[0]
print(image)
print(label)


# Here's some fancy matplotlib code that allows us to visualize some of 
# there images
# That unfortunately does not work for me. Don't know what is going on
# You can take a look at the plot in the MNISTplot.ipynb Jupyter Notebook
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()