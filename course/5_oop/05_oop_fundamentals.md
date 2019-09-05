## OOP Fundamentals

Here we list a brief recap of key aspects of OOP in Python.

#### Inheritance

Here there are three classes: one parent (`Clothing`) and two children (`Shirt` and `Pants`). Methods from the parent are inherited by the children.

```
class Clothing:

    def __init__(self, color, size, style, price):
        self.color = color
        self.size = size
        self.style = style
        self.price = price
        
    def change_price(self, price):
        self.price = price
        
    def calculate_discount(self, discount):
        return self.price * (1 - discount)
    
    def calculate_shipping(self, weight, rate):
        return weight * rate
    
        
class Shirt(Clothing):
    
    def __init__(self, color, size, style, price, long_or_short):
        
        Clothing.__init__(self, color, size, style, price)
        self.long_or_short = long_or_short
    
    def double_price(self):
        self.price = 2*self.price
    
class Pants(Clothing):

    def __init__(self, color, size, style, price, waist):
        
        Clothing.__init__(self, color, size, style, price)
        self.waist = waist
        
    def calculate_discount(self, discount):
        return self.price * (1 - discount / 2)
    

class Blouse(Clothing):
    def __init__(self, color, size, style, price, country_of_origin):
        Clothing.__init__(self, color, size, style, price)
        self.country_of_origin = country_of_origin
        
    def triple_price(self):
        return self.price*3.0
```

Note: `object` class is the base class in python. It already has all the acceptable magic methods embedded, but not or partially implemented (e.g. \__add__ will return an NotImplementedException, while \__str__ will report the object instance `id`, which is the location in memory).

#### Magic methods

```
class Gaussian:
     """ Gaussian distribution class for calculating and 
    visualizing a Gaussian distribution.
    
    Attributes:
        mean (float) representing the mean value of the distribution
        stdev (float) representing the standard deviation of the distribution
        data_list (list of floats) a list of floats extracted from the data file
    """
    def __init__(self, mu = 0, sigma = 1):
        
        self.mean = mu
        self.stdev = sigma
        self.data = []

    # ...methods

    def __add__(self, other):
        """Magic method to add together two Gaussian distributions
        
        Args:
            other (Gaussian): Gaussian instance
            
        Returns:
            Gaussian: Gaussian distribution   
        """
        
        # TODO: Calculate the results of summing two Gaussian distributions
        #   When summing two Gaussian distributions, the mean value is the sum
        #       of the means of each Gaussian.
        #
        #   When summing two Gaussian distributions, the standard deviation is the
        #       square root of the sum of square ie sqrt(stdev_one ^ 2 + stdev_two ^ 2)
        
        # create a new Gaussian object
        result = Gaussian()
        result.mean = self.mean + other.mean
        result.stdev = math.sqrt(self.stdev**2+other.stdev**2)
        return result


    def __repr__(self):
        """Magic method to output the characteristics of the Gaussian instance
        
        Args:
            None
        
        Returns:
            string: characteristics of the Gaussian
        """

        return 'mean {}, standard deviation {}'.format(self.mean,self.stdev)
```

#### Advanced features

Below is an example of a class with docstrings and a few things to keep in mind:

[On classes and methods](https://realpython.com/instance-class-and-static-methods-demystified/)

[Attributes: class and instance](https://www.python-course.eu/python3_class_and_instance_attributes.php)

[Mixins](https://easyaspython.com/mixins-for-fun-and-profit-cb9962760556)

[Decorators](https://realpython.com/primer-on-python-decorators/)

