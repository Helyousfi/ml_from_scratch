"""
self : to refer to the instance of the class
type(self) : to refer to the class
static methods are accessible by the class 
    and the instance. To get something ..
class method : to set something for all instances
"""

class Robot:
    __C = 0
    def __init__(self):
        type(self).__C += 1

    def __del__(self):
        type(self).__C -= 1
    
    @staticmethod
    def getC():
        return Robot.__C


class Employee:
    __count = 0
    _raise = 1.05
    def __init__(self, fname, lname, pay):
        self.fname = fname
        self.lname = lname
        self.pay = int(pay)
        type(self).__count += 1
    def getName(self):
        return self.fname + " " + self.lname

    @classmethod
    def change_raise(cls, value):
        cls._raise = value

    def apply_raise(self):
        self.pay = type(self)._raise * self.pay

    @staticmethod
    def get_raise():
        return Employee._raise

    def getPay(self):
        return type(self)._raise * self.pay

    @staticmethod
    def getCount():
        return Employee.__count

    # New Constructor
    @classmethod
    def from_string(cls, emp_string):
        arguments = emp_string.split('-')
        return cls(*arguments)

class Engineer(Employee):
    def __init__(self, fname, lname, pay, prog_lang):
        super().__init__(fname, lname, pay)
        self.prog_lang = prog_lang






## Getters & Setters
"""
@property : access a method like an attribute 
"""
class Student:
    def __init__(self, fname, lname):
        self.fname = fname
        self.lname = lname
    
    @property
    def fullname(self):
        return f"{self.fname} {self.lname}"

    @fullname.setter
    def fullname(self, name):
        self.fname, self.lname = name.split(" ") 
        #self.fullname = name

    @property
    def email(self):
        return f"{self.fname}.{self.lname}@gmail.com"

    @email.setter
    def email(self, email):
        self.email = email

S1 = Student("ham", "elyousfi")
S1.fullname = "john wick"
print(S1.fullname)
    
