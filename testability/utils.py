"""


"""

__version__ = '0.1.0'
__author__ = 'Morteza Zakeri'

import sys
import pandas
import pandas as pd


def main(args):
    df = pd.read_csv('inference_results/free-mind-after-refactor-Class_predicted.csv')
    while True:
        class_name = input()
        if class_name == 'Q':
            quit()
        row = df.loc[df['Name'] == class_name]
        print(round(row['PredictedTsDDTestability'], 4))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv)
