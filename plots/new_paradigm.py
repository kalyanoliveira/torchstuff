import numpy as np
import matplotlib.pyplot as plt

def figures_and_subplots():

    # Note that subplots are also called axes

    fig = plt.figure()

    # Which is why I decided to name this subplot "ax"
    ax = fig.add_subplot(2, 2, 1)
    ax.plot([1,2,3])
    fig.add_subplot(1, 2, 2)
    fig.add_subplot(2, 2, 2)

    fig.savefig("out.png")

def figures_and_subplots_correct():
    # plt.subplots will return an array with the number of specified axes

    # This will give you two subplots
    fig0, (axs) = plt.subplots(2)

    # This will create 4 axes, because it's 2X2 grid of rowsXcolumns
    fig1, axs = plt.subplot(2, 2)

    # You can of course already unpack these
    fig2, (ax1, ax2, ax3, ax3) = plt.subplots(2)

    

if __name__ == "__main__":
    figures_and_subplots_correct()