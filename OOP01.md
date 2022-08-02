## Class Methods
Let’s compare that to the second method, MyClass.classmethod. I marked this method with a @classmethod decorator to flag it as a class method.
Instead of accepting a self parameter, class methods take a cls parameter that points to the class—and not the object instance—when the method is called.
Because the class method only has access to this cls argument, it can’t modify object instance state. That would require access to self. However, class methods can still modify class state that applies across all instances of the class.

## Static Methods
The third method, MyClass.staticmethod was marked with a @staticmethod decorator to flag it as a static method.
This type of method takes neither a self nor a cls parameter (but of course it’s free to accept an arbitrary number of other parameters).
Therefore a static method can neither modify object state nor class state. Static methods are restricted in what data they can access - and they’re primarily a way to namespace your methods.
