"""
Script to answer research questions for
TsDD paper.
"""

__version__ = '0.1.0'
__author__ = 'Morteza Zakeri'



import pandas as pd
import joblib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, GridSearchCV

from yellowbrick.model_selection import FeatureImportances

from metrica.metrics_map import testability_metrics

import scipy.stats

def regress_with_decision_tree(model_path):
    xls = pd.ExcelFile(
        'D:/Users/Morteza/OneDrive/Online2/_04_2o/o2_university/PhD/Project21/a155_TsDD/experimental_results/refactoring_importance.xlsx')
    # df_gt = pd.read_excel(xls, 'binary_for_learning')
    df_gt = pd.read_excel(xls, 'binary_for_learning')

    X = df_gt.iloc[:, 0:7]
    y = df_gt.iloc[:, -1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)
    X_train, y_train = X, y

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X=X, y=y)

    clf = tree.DecisionTreeRegressor()
    # clf = RandomForestRegressor()

    # CrossValidation iterator object:
    cv = ShuffleSplit(n_splits=5, test_size=0.20, random_state=101)

    # Set the parameters to be used for tuning by cross-validation
    parameters = {'max_depth': range(2, 100, 1),
                  'criterion': ['mse', 'friedman_mse', 'mae'],
                  # 'criterion': ['mse'],
                  'min_samples_split': range(2, 100, 1)
                  }
    # Set the objectives which must be optimized during parameter tuning
    # scoring = ['r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_error',]
    scoring = ['neg_root_mean_squared_error']

    # Find the best model using gird-search with cross-validation
    # clf = GridSearchCV(clf, param_grid=parameters, scoring=scoring, cv=cv, n_jobs=4,
    #                    refit='neg_root_mean_squared_error', )
    clf.fit(X=X_train, y=y_train)

    viz = FeatureImportances(clf, labels=df_gt.columns[0:7], is_fitted=True)
    viz.fit(X_train, y_train)
    viz.show()
    quit()

    print('Writing grid search result ...')
    df = pd.DataFrame(clf.cv_results_, )
    df.to_csv(model_path[:-7] + '_grid_search_cv_results.csv', index=False)
    df = pd.DataFrame()
    print('Best parameters set found on development set:', clf.best_params_)
    df['best_parameters_development_set'] = [clf.best_params_]
    print('Best classifier score on development set:', clf.best_score_)
    df['best_score_development_set'] = [clf.best_score_]
    # print('best classifier score on test set:', clf.score(X_test, y_test))
    # df['best_score_test_set:'] = [clf.score(X_test, y_test)]
    df.to_csv(model_path[:-7] + '_grid_search_cv_results_best.csv', index=False)
    clf = clf.best_estimator_
    joblib.dump(clf, model_path)

    fig = plt.figure(figsize=(12, 10))
    plot_tree(clf, filled=True,
              max_depth=3,
              feature_names=df_gt.columns[0:7],
              precision=4,
              rounded=True,
              proportion=True,
              impurity=True)
    plt.tight_layout()
    plt.savefig(model_path[:-7] + 'tree_plot.png')

    # Export dot file
    # Use below command to generate figure output
    # dot -Tpng tree.dot -o tree.png
    # dot -Tps tree.dot -o tree.ps
    tree.export_graphviz(clf, feature_names=df_gt.columns[0:7],
                         filled=True,
                         rounded=True,
                         out_file=model_path[:-7] + 'tree_plot.dot')
    plt.show()


def refactoring_importance():
    xls = pd.ExcelFile(
        'D:/Users/Morteza/OneDrive/Online2/_04_2o/o2_university/PhD/Project21/a155_TsDD/experimental_results/refactoring_importance.xlsx')
    # df = pd.read_excel(xls, 'binary_for_learning')
    df = pd.read_excel(xls, 'merged_data')
    X = df.iloc[:, :-1]  # independent columns
    y = df.iloc[:, -1]  # target column i.e Label_BranchCoverage
    correlation_list = list()
    for col in X.columns:
        r, p = scipy.stats.pearsonr(X[col], y)
        r2, p2 = scipy.stats.spearmanr(X[col], y)
        correlation_list.append([col, round(r, 5), round(p, 5), round(r2, 5), round(p2, 5), ])
    df = pd.DataFrame(correlation_list,
                      columns=['Metric', 'Pearsonr', 'PearsonrPvalue', 'Spearmanr', 'SpearmanrPvalue'])
    df = df.sort_values(by=['Spearmanr'], ascending=False)
    print(df)
    df.to_csv('correl_coeff.csv', index=False)


def compare_source_code_metrics_before_and_after_refactoring():
    """
    This function uses visualization techniques to compare source code metrics before and after refactoring
    :return:
    """
    experiments_path = r'D:/Users/Morteza/OneDrive/Online2/_04_2o/o2_university/PhD/Project21/a155_TsDD/experimental_results/'
    xls = pd.ExcelFile(experiments_path + r'source_code_metrics.xlsx')
    # df_gt = pd.read_excel(xls, 'binary_for_learning')
    df = pd.read_excel(xls, 'selected_classes_metrics')

    df2 = pd.DataFrame()
    df2['Project'] = df['Project']
    df2['Stage'] = df['Stage']
    for i, metric_name_ in enumerate(testability_metrics):
        df2[testability_metrics[metric_name_][0]] = df[testability_metrics[metric_name_][0]]
    # df2.drop(columns=['LOC', 'LLOC', 'TLLOC', 'TLOC', 'TNA', 'NG','TNG', 'TNOS',  'TNM','TNPM','WMC' ], inplace=True)

    df3 = df2.melt(id_vars=['Project', 'Stage'], var_name='Metric', value_name='Value')

    g = sns.catplot(data=df3,
                    x='Project', y='Value', hue='Stage', col='Metric',
                    col_wrap=5,
                    kind='box',
                    sharex=True, sharey=False, margin_titles=True,
                    height=2.05, aspect=1.25, orient='v',
                    legend_out=False, legend=False, dodge=True)

    g2 = sns.catplot(data=df3,
                     x='Project', y='Value', hue='Stage', col='Metric',
                     col_wrap=5,
                     kind='point',
                     sharex=True, sharey=False, margin_titles=True,
                     height=2.055, aspect=1.25, orient='v',
                     legend_out=False, legend=False, dodge=True,
                     # axes=g.axes
                     # palette=sns.color_palette('tab10', n_colors=4),
                     markers=['o', 'X'], linestyles=['-', '--']
                     )
    # g.set(yscale="log")
    g.despine(left=True)
    g2.despine(left=True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def compare_test_effectiveness_before_and_after_refactoring():
    """
    This function uses visualization techniques to compare source code metrics before and after refactoring
    :return:
    """
    experiments_path = r'D:/Users/Morteza/OneDrive/Online2/_04_2o/o2_university/PhD/Project21/a155_TsDD/experimental_results/'
    xls = pd.ExcelFile(experiments_path + r'tsdd.xlsx')
    # df_gt = pd.read_excel(xls, 'binary_for_learning')
    df = pd.read_excel(xls, 'test_effectiveness')

    df.drop(columns=['Class'], inplace=True)
    df2 = df.melt(id_vars=['Project', 'Stage'], var_name='Criterion', value_name='Value')

    g = sns.catplot(data=df2,
                    x='Project', y='Value', hue='Stage', col='Criterion',
                    col_wrap=5,
                    kind='box',
                    sharex=True, sharey=False, margin_titles=True,
                    height=2.55, aspect=1.30, orient='v',
                    legend_out=False, legend=False, dodge=True,
                    )

    for i in range(0, 5):
        hatches = ["//", "\\", '//', '\\', '//', '\\']
        for hatch, patch in zip(hatches, g.axes[i].artists):
            patch.set_hatch(hatch)

    # g2 = sns.catplot(data=df2,
    #                  x='Project', y='Value', hue='Stage', col='Criterion',
    #                  col_wrap=5,
    #                  kind='point',
    #                  sharex=True, sharey=False, margin_titles=True,
    #                  height=2.55, aspect=1.30, orient='v',
    #                  legend_out=False, legend=False, dodge=True,
    #                  # axes=g.axes
    #                  # palette=sns.color_palette('tab10', n_colors=4),
    #                  markers=['o', 'X'], linestyles=['-', '--']
    #                  )
    g.despine(left=True)
    # g2.despine(left=True)
    # plt.legend(loc='upper center')
    g.axes[4].legend(loc='upper center')
    plt.tight_layout()
    plt.show()

# regress_with_decision_tree(model_path=r'refactoring_importance/DTR2_DSX2.joblib')
# compare_source_code_metrics_before_and_after_refactoring()
compare_test_effectiveness_before_and_after_refactoring()
# refactoring_importance()