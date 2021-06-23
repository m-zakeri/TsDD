"""
The module implements metrics computation
based on Understand API
in three level of classes, files, and packages.

"""

__version__ = '0.1.0'
__author__ = 'Morteza'

import sys
import logging

logging.basicConfig(level=logging.DEBUG)
import progressbar

# -------------------
# For Windows os
# https://scitools.com/support/python-api/
# Python 3.8 and newer require the user add a call to os.add_dll_directory(“SciTools/bin/“  # Put your path here
# os.add_dll_directory('C:/Program Files/SciTools/bin/pc-win64')  # Put your path here
sys.path.insert(0, 'D:/program files/scitools/bin/pc-win64/python')  # Put your path here

# --------------------
# Import understand if available on the path, otherwise raise an error
try:
    import understand as und
except ModuleNotFoundError:
    raise ModuleNotFoundError('Understand cannot import')


class Project:
    """
    Load project and iterate on each entity (class / file / package)
    """

    def __init__(self, path_db: str = None, path_metrics: str = None):
        self.udb = und.open(path_db)
        self.metric = None

    def get_class_metrics(self):
        """

        return (dict): classes_metrics
        """

        # Understand query
        classes = self.udb.ents('Java Class ~Interface ~Enum ~Unknown ~Unresolved ~Jar ~Library')
        logging.debug(msg='All class in project: {0}'.format(classes))

        classes_metrics = {}
        for entity_class in progressbar.progressbar(classes[:10]):
            class_metrics = {}
            self.metric = Metric(entity_class=entity_class, entity_file=entity_class.parent())

            class_metrics.update({'number_of_accessor_methods': self.metric.number_of_accessor_methods})
            class_metrics.update({'halstead_program_length': self.metric.halstead_program_length})

            classes_metrics.update({str(entity_class.longname()): class_metrics})

        return classes_metrics


class Metric:
    """
    Compute metrics at given levels
    """

    def __init__(self, entity_class=None, entity_file=None, entity_package=None):
        """
        Args:

            entity_class (UnderstandEntity): Understand entity of kind class

            entity_file (UnderstandEntity):  Understand entity of kind file

            entity_package (UnderstandEntity): Understand entity of kind package
        """
        self.entity_class = entity_class
        self.entity_file = entity_file
        self.entity_package = entity_package
        self.methods = self.entity_class.ents('Define', 'method')

    @property
    def number_of_accessor_methods(self):
        """
        ## Metric definition
        Find and count only number of accessor (getter) method defined in a given class

        Returns:

            count (int): number_of_accessor_methods

        ## Examples
        ### An input Java class

            public class Person{
            public Person()
            int age;
            public void setAge(int age){this.age = age}
            public int getAge(){return this.age}
            }

        ### Output
        number_of_accessor_methods: 1
        """

        count = 0
        logging.debug(msg='All methods in class {0}: {1}'.format(self.entity_class, self.methods))
        for entity_method in self.methods:
            if entity_method.simplename().startswith(('get', 'Get')):
                count += 1
        return count

    @property
    def halstead_program_length(self):
        """
        ## Metric definition
        N1: total number of operators (semantic meanings of the reserved keywords, semicolons, blocks,
        and identifiers except in their declarations)

        N2: total number of operands (literals - e.g. character, string, and integer literals,
        and the identifiers in their declarations)

        Halstead program length is N1 + N2.

        Returns:

            total_number_of_operators + total_number_of_operands (int): halstead_program_length

        ## Examples
        ### An input Java class

            public class Person{
            public Person()
            int age;
            public void setAge(int age){this.age = age}
            public int getAge(){return this.age}
            }

        ### Output
        total_number_of_operators + total_number_of_operands:  13 + 9
        """

        total_number_of_operators = 0
        total_number_of_operands = 0
        for lexeme in self.entity_file.lexer(show_inactive=False):
            logging.debug(msg='text: {0}: token: {1}'.format(lexeme.text(), lexeme.token()))
            if lexeme.token() == 'Identifier':
                total_number_of_operands += 1
            if lexeme.token() == 'Keyword' or lexeme.token() == 'Operator':
                total_number_of_operators += 1
        return total_number_of_operators + total_number_of_operands


# Test metrics module
if __name__ == '__main__':
    project = Project(path_db=r'benchmark/Java/107_weka.udb')
    metric_dict = project.get_class_metrics()
    print('WEKA class metrics', metric_dict)
