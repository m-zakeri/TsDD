"""
The main module of TsDD project.

TsDD is an abbreviation of ' Testability-driven development'

The full version of source code will be available
as soon as the relevant paper(s) are published.

"""

# import tsdd.testability.ml_models as ml_model
from tsdd.testability.ml_models import Regression, Dataset


class Classification:

    def __init__(self, **kwargs):
        self.__first_name = kwargs["l"]
        self.last_name = kwargs["f"]
        self.age = kwargs["age"]

    def print_lastname(self, *, age):
        print("******", self.last_name, age, '*****')

    def print_firstname(self, x):
        self.print_lastname(10, age=20)
        print("@@@", self.__first_name, x, '@@@@')



class MultiLabelClassification(Classification,):
    def __init__(self, f, l):
        super(Classification).__init__(f, l)




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def main():
    # print_hi('ali')
    # obj1 = Classification()
    # obj1.print_lastname(20)
    obj2 = Classification(l='Hosseini', f='Ali', a=20)
    # obj2.print_lastname(25)

    obj2.print_firstname(40)

# Test
if __name__ == '__main__':
    main()
