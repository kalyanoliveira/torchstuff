import matplotlib.pyplot as plt
import numpy as np

def test0():
    ys = [1, 3, 5, 7]

    # This is considered to be the y values
    plt.plot(ys)
    
    # Usually you would plt.show(), but since I am running WSL2,
    # I'll .savefig instead
    plt.savefig("out.png")

def double_variables():
    # In this instance, we have xs and ys
    xs = [1, 2, 3, 4]
    ys = [1, 2, 4, 8]

    plt.plot(xs, ys)

    plt.savefig("out.png")

def styling():
    # plt.plot() can take a third argument: a two character "styling" string

    # The first character indicates the color
    # The second character indicates the style of the line in the plot

    xs = [1, 2, 3, 4]
    ys = [1, 2, 4, 8]

    # 'r' for red, 'o' for circles
    plt.plot(xs, ys, "ro")

    plt.savefig("out.png")

def viewport():
    # You can also specify the viewport, much like in a scientific calculator
    # Do so by utilizing the axis function

    xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ys = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    plt.plot(xs, ys)

    # plt.axis() takes a list [xmin, xmax, ymin, ymax]
    plt.axis([0, 10, 0, 600])

    plt.savefig("out.png")

def numpy_supremacy():
    # Internally, pyplot converts everything to numpy arrays
    # So there's no reason not to use them

    xs = np.arange(0, 5, 0.2)
    print(xs)

    # Red Dashes 
    plt.plot(xs, xs, "r--")

    # Blue Squares
    plt.plot(xs, xs**2, "bs")

    # Green Triangles
    plt.plot(xs, xs**3, "g^")

    plt.savefig("out.png")

def axes_labels():
    xs = np.arange(10)
    ys = xs**2

    plt.plot(xs, ys, "yo")

    # You can also create labels
    plt.xlabel("x")
    plt.ylabel("y = f(x) = x^2")

    plt.savefig("out.png")

def grid():
    # 200 evenly-spaced numbers between 0 and 2pi
    # and then sine of those numbers
    xs = np.linspace(0, 2*np.pi, 200)
    ys = np.sin(xs)

    plt.plot(xs, ys)

    # Creating a grid is this simple
    plt.grid()

    plt.savefig("out.png")

def labelled_data():
    # This demonstrates the usage of keyword strings

    # Essentially, you can create a dictionary where
        # key = label
        # value = data
    
    # Then, during your plot creation, you can reference the labels of a data
    # and the entire dictionary 
    # instead of 
    # having to reference the data itself


    # Note that np.random.randn() returns random numbers from a normal distribution
    my_data = {"xs" : np.arange(10), 
               "color" : np.random.randint(low=0, high=50, size=(10,)), 
               "size" : np.random.randn(10)
               }
    
    my_data["ys"] = my_data["xs"] + 10*np.random.randn(10)
    my_data["size"] = np.abs(my_data["size"]) * 100

    print("x values", my_data["xs"])
    print("y values", my_data["ys"])
    print("colors", my_data["color"])
    print("sizes", my_data["size"])
    
    # This is the key part of it all. You use the data= to specify your dictionary
    plt.scatter("xs", "ys", c="color", s="size", data=my_data)

    plt.savefig("out.png")

def labelled_data_simpler():
    # Here's the simpler version of what is above
    very_complicated_name = np.arange(10)
    data = {"x": very_complicated_name,
            "y": very_complicated_name*2}

    plt.plot("x", "y", "r-", data=data)

    plt.savefig("out.png")

def per_plot_labels():
    # You can also add labels per each plot (or in this case, line) that you make
    xs = np.linspace(-2*np.pi, 2*np.pi, 200)
    ys = np.sin(xs)
    ys2 = np.cos(xs)
    ys3 = np.tanh(xs)
    
    # Here, this is where you add those labels
    plt.plot(xs, ys, "r", label="sinx")
    plt.plot(xs, ys2, "g", label="cos x")
    plt.plot(xs, ys3, "b", label="tanh x")

    # This is what creates the legend on the side
    plt.legend()

    plt.savefig("out.png")

def graph_title():
    xs = np.linspace(0, 2*np.pi, 200)
    ys = np.sin(xs)

    plt.plot(xs, ys)

    # We can give the entire graph a title
    plt.title("Graph of sin X")

    plt.savefig("out.png")

def categorical_variables():
    # You can also plot categorical variables
    names = ["a", "b", "c"]
    values = [100, 200, 300]

    plt.plot(names, values)

    plt.savefig("out.png")

def scatters():
    xs = np.linspace(0, 2*np.pi, 20)
    ys = np.sin(xs)

    # This is how we can create a scatter graph
    plt.scatter(xs, ys)

    plt.savefig("out.png")

def bars():
    xs = np.arange(10)
    ys = np.linspace(0, 2, 10)

    # This is how we can create a bar graph
    plt.bar(xs, ys)

    plt.savefig("out.png")

def multiple_plots():
    xs = np.arange(10)
    ys1 = np.sin(xs)
    ys2 = np.cos(xs)

    plt.subplot(1)
    plt.plot(xs, ys1)

    plt.subplot(2)
    plt.plot(xs, ys2)

    plt.savefig("out.png")

if __name__ == "__main__":
    multiple_plots()