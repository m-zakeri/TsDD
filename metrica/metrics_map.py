"""
This module provide the dictionary to map full metrics name
to their abbreviation names for different quality attributes

"""

__version__ = '0.1.0'
__author__ = 'Morteza'

# Reference:
reusability_metrics = {
    # 1- Cohesion
    'Lack of Cohesion in Methods 5': ('LCOM5', 0.87),

    # 2- Complexity
    'Nesting Level': ('NL', 0.79),
    'Nesting Level Else-If': ('NLE', 0.88),
    'Weighted Methods per Class': ('WMC', 0.59),

    # 3- Coupling
    'Coupling Between Object classes': ('CBO', 0.15),
    'Coupling Between Object classes Inverse': ('CBOI', 0.58),
    'Number of Incoming Invocations': ('NII', 0.93),
    'Number of Outgoing Invocations': ('NOI', 0.88),
    'Response set For Class': ('RFC', 0.73),

    # 4- Documentation
    'API Documentation': ('AD', 0.38),
    'Comment Density': ('CD', 0.81),
    'Total Comment Density': ('TCD', 0.84),
    'Comment Lines of Code': ('CLOC', 0.38),
    'Total Comment Lines of Code': ('TCLOC', 0.36),
    'Documentation Lines of Code': ('DLOC', 0.28),
    'Total Documentation Lines of Code': ('PDA', 0.52),

    # 5- Inheritance
    'Depth of Inheritance Tree': ('DIT', 0.75),

    # 6- Size
    'Lines of Code': ('LOC', 0.6),
    'Logical Lines of Code': ('LLOC', 0.73),
    'Total Logical Lines of Code': ('TLLOC', 0.68),
    'Total Lines of Code': ('TLOC', 0.68),  # Not set in reference paper
    'Total Number of Attributes': ('TNA', 0.90),
    'Number of Getters': ('NG', 0.79),
    'Total Number of Getters': ('TNG', 0.82),
    'Total Number of Methods': ('TNM', 0.63),
    'Total Number of Statements': ('TNOS', 0.78),
    'Total Number of Public Methods': ('TNPM', 0.77),

}

testability_metrics = {

# 1- Cohesion
    'Lack of Cohesion in Methods 5': ('LCOM5', 0.87),

    # 2- Complexity
    'Nesting Level': ('NL', 0.79),
    'Nesting Level Else-If': ('NLE', 0.88),
    'Weighted Methods per Class': ('WMC', 0.59),

    # 3- Coupling
    'Coupling Between Object classes': ('CBO', 0.15),
    'Coupling Between Object classes Inverse': ('CBOI', 0.58),
    'Number of Incoming Invocations': ('NII', 0.93),
    'Number of Outgoing Invocations': ('NOI', 0.88),
    'Response set For Class': ('RFC', 0.73),

    # 4- Documentation
    # Documentation metrics are not used in testability measurement

    # 5- Inheritance
    'Depth of Inheritance Tree': ('DIT', 0.75),

    # 6- Size
    'Lines of Code': ('LOC', 0.6),
    'Logical Lines of Code': ('LLOC', 0.73),
    'Total Logical Lines of Code': ('TLLOC', 0.68),
    'Total Lines of Code': ('TLOC', 0.68),
    'Total Number of Attributes': ('TNA', 0.90),
    'Number of Getters': ('NG', 0.79),
    'Total Number of Getters': ('TNG', 0.82),
    'Total Number of Methods': ('TNM', 0.63),
    'Total Number of Statements': ('TNOS', 0.78),
    'Total Number of Public Methods': ('TNPM', 0.77),

}

# Test driver
# print(reusability_metrics['Total Number of Public Methods'][1])
