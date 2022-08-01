"""
class Robot:
    pass
x = Robot()
x.name = "Marvin"
x.build_year = "1979"
"""
#print(x.name)

#print(x.__dict__)
#print(Robot.__dict__)

# if you try to access x.val, we check x.__dict__ then Robot.__dict__
# to avoid an error
#print(getattr(x, 'name', 100))

def f(x):
    return x
f.x = 4
print(f.__dict__)

class Robot:
    def __init__(self, name=None):
        self.name = name   
    def say_hi(self):
        if self.name:
            print("Hi, I am " + self.name)
        else:
            print("Hi, I am a robot without a name")
    def set_name(self, name):
        self.name = name
    def get_name(self):
        return self.name
x = Robot()
x.set_name("Henry")
x.say_hi()
y = Robot()
y.set_name(x.get_name())
print(y.get_name())

#  "__str__" and "__repr__"

class Detect:
    a = "hello "
x = Detect()
y = Detect()

Detect.a = "H"
print(y.__class__.__dict__)
# Note that y.__class__.__dict__ is equivalent to Detect.__dict__


class C: 
    counter = 0
    def __init__(self): 
        type(self).counter += 1
    def __del__(self):
        type(self).counter -= 1
if __name__ == "__main__":
    x = C()
    print("Number of instances: : " + str(C.counter))
    y = C()
    print("Number of instances: : " + str(C.counter))
    del x
    print("Number of instances: : " + str(C.counter))
    del y
    print("Number of instances: : " + str(C.counter))
    
