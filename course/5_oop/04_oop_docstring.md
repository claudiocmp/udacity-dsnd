## Docstrings and Object-Oriented Code

Below is an example of a class with docstrings and a few things to keep in mind:

  * Make sure to indent your docstrings correctly or the code will not run. A docstring should be indented one indentation underneath the class or method being described.
  * You don't have to define 'self' in your method docstrings. It's understood that any method will have self as the first method input.

```
class Pants:
    """The Pants class represents an article of clothing sold in a store
    """

    def __init__(self, color, waist_size, length, price):
        """Method for initializing a Pants object

        Args: 
            color (str)
            waist_size (int)
            length (int)
            price (float)

        Attributes:
            color (str): color of a pants object
            waist_size (str): waist size of a pants object
            length (str): length of a pants object
            price (float): price of a pants object
        """

        self.color = color
        self.waist_size = waist_size
        self.length = length
        self.price = price

    def change_price(self, new_price):
        """The change_price method changes the price attribute of a pants object

        Args: 
            new_price (float): the new price of the pants object

        Returns: None

        """
        self.price = new_price

    def discount(self, percentage):
        """The discount method outputs a discounted price of a pants object

        Args:
            percentage (float): a decimal representing the amount to discount

        Returns:
            float: the discounted price
        """
        return self.price * (1 - percentage)
```

#### References:
[Google doc style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
[Numpy doc style (reStructuredText)](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy)