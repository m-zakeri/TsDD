"""
Driver module for quality attributes classes

"""

import os

import pandas as pd


def create_benchmark():
    root_dir_path = r'../../benchmark/SF110/dataset2/'
    project_files = [f for f in os.listdir(root_dir_path) if os.path.isfile(os.path.join(root_dir_path, f))]
    print(project_files)
    df_modularity = pd.read_csv(r'../benchmark/QualCode160/data_project_modularity/ALL2',)

    for file_ in project_files:
        pass


if __name__ == '__main__':
    create_benchmark()