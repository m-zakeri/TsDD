"""

import note:
Add project path to PYTHONPATH
export PYTHONPATH="${PAYTHONPATH}:/path/to/project/root"

"""

__version__ = '0.1.1'
__author__ = 'Morteza'

import os
import ntpath
import datetime as dt
import math
from sklearnex import patch_sklearn

patch_sklearn()

import pandas as pd
import joblib
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor, \
    HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import NuSVR
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error, \
    median_absolute_error, mean_squared_log_error, mean_poisson_deviance, max_error, mean_gamma_deviance

from metrica.metrics_map import testability_metrics


class Dataset:
    def __init__(self, ):
        pass

    def add_evosuite_information(self, root_dir_path=r'../benchmark/SF110/dataset/'):

        df_evosuite = pd.read_csv('../benchmark/SF110/dataset_final/evosuit160_sf110_result_html_with_project.csv',
                                  delimiter=',', index_col=False)
        classes_names = list(df_evosuite['Class'].values)

        files = [f for f in os.listdir(root_dir_path) if os.path.isfile(os.path.join(root_dir_path, f))]

        for file_ in files:
            print('Working on project {0}'.format(file_))
            df = pd.read_csv(root_dir_path + file_, delimiter=',', index_col=False)
            line_coverages = []
            branch_coverages = []
            mutation_scores = []
            test_suite_size = []
            tsdd_testability_measure = []
            for index, row in df.iterrows():
                if row['LongName'].find('$') != -1:
                    df.drop(index, inplace=True)
                    print('Row dropped')
                elif (row['LongName'] in classes_names) is False:
                    df.drop(index, inplace=True)
                    print('Row dropped')
                else:
                    a = classes_names.index(row['LongName'])
                    if df_evosuite['Tests'][a] == 0:
                        df.drop(index, inplace=True)
                        print('Row dropped')
                    else:
                        lc = df_evosuite['Line'][a] / 100
                        br = df_evosuite['Branch'][a] / 100
                        mu = df_evosuite['Mutation'][a] / 100
                        taw = df_evosuite['Tests'][a]
                        line_coverages.append(lc)
                        branch_coverages.append(br)
                        mutation_scores.append(mu)
                        test_suite_size.append(taw)

                        if taw == 0:
                            test_quality = 0
                            test_effort = 0
                            testability_value = 0
                        else:
                            test_quality = (lc * br * mu) ** (1 / 3)
                            test_effort = 1
                            if row['TNM'] >= 1:
                                omega = 2.0 / taw
                                test_effort = (1 + omega) ** (math.ceil((taw - 1) / row['TNM']))
                            testability_value = test_quality / test_effort
                        print(test_quality, test_effort, testability_value)
                        tsdd_testability_measure.append(testability_value)

            # Set new columns
            df['LineCoverage'] = line_coverages
            df['BranchCoverage'] = branch_coverages
            df['MutationScore'] = mutation_scores
            df['TestSuiteSize'] = test_suite_size
            df['TsDDTestability'] = tsdd_testability_measure
            df.to_csv(r'../benchmark/SF110/dataset2/' + file_[:-4] + '_labeled.csv', )

    def concatenate_csv_files(self, root_dir_path=r'../benchmark/SF110/dataset2/'):
        files = [f for f in os.listdir(root_dir_path) if os.path.isfile(os.path.join(root_dir_path, f))]
        df_all = pd.DataFrame()
        for file_ in files:
            df = pd.read_csv(root_dir_path + file_, delimiter=',', index_col=False)
            df_all = df_all.append(df, ignore_index=True)
        df_all.to_csv(r'../benchmark/SF110/dataset_final/all_with_label4.csv', index=False)

    @classmethod
    def compute_testability(cls, lc, br, mu, taw, TNM):
        if taw == 0:
            test_quality = 0
            test_effort = 0
            testability_value = 0
        else:
            test_quality = (lc * br * mu) ** (1 / 3)
            test_effort = 1
            if TNM >= 1:
                omega = 2.0 / taw
                test_effort = (1 + omega) ** (math.ceil((taw - 1) / TNM))
            testability_value = test_quality / test_effort
        print('test effectiveness:', test_quality)
        print('test effort:', test_effort)
        print('testability:', testability_value)
        # return testability_value

    def clean_data(self):
        df = pd.read_csv('../benchmark/SF110/dataset_final/all_with_label4.csv',
                         delimiter=',', index_col=False)
        counter = 0
        counter2 = 0
        for index, row in df.iterrows():
            if row['LineCoverage'] * row['BranchCoverage'] == 0:
                # print(row['LongName'])
                df.drop(index, inplace=True)
                counter += 1

        for index, row in df.iterrows():
            if row['TestSuiteSize'] < 1 or row['LOC'] < 5:
                df.drop(index, inplace=True)
                counter2 += 1

        print('counter1', counter)
        print('counter2', counter2)
        print('all', counter + counter2)
        print(df.shape)
        # df.drop(df.columns[[i for i in range(1,10)]], axis=1, inplace=True)
        df.to_csv(r'../benchmark/SF110/dataset_final/all_with_label_cleaned4.csv', index=False)


class Regression(object):
    def __init__(self, df_path=None, ):
        self.df = pd.read_csv(df_path, delimiter=',', index_col=False)

        cols = []
        for i, metric_name_ in enumerate(testability_metrics):
            cols.append(testability_metrics[metric_name_][0])

        self.X_train1, self.X_test1, self.y_train, self.y_test = train_test_split(self.df[cols],
                                                                                  # self.df.iloc[:, -2],
                                                                                  self.df['TsDDTestability'],
                                                                                  test_size=0.20,
                                                                                  random_state=13,
                                                                                  )

        """
        # ---------------------------------------
        # -- Feature selection (For DS2)
        selector = feature_selection.SelectKBest(feature_selection.f_regression, k=15)
        # clf = linear_model.LassoCV(eps=1e-3, n_alphas=100, normalize=True, max_iter=5000, tol=1e-4)
        # clf.fit(self.X_train1, self.y_train)
        # importance = np.abs(clf.coef_)
        # print('importance', importance)
        # clf = RandomForestRegressor()
        # selector = feature_selection.SelectFromModel(clf, prefit=False, norm_order=2, max_features=20, threshold=None)
        selector.fit(self.X_train1, self.y_train)

        # Get columns to keep and create new dataframe with only selected features
        cols = selector.get_support(indices=True)
        self.X_train1 = self.X_train1.iloc[:, cols]
        self.X_test1 = self.X_test1.iloc[:, cols]
        print('Selected columns by feature selection:', self.X_train1.columns)
        # quit()
        # -- End of feature selection
        """

        # ---------------------------------------
        # Standardization
        self.scaler = preprocessing.RobustScaler(with_centering=True, with_scaling=True)
        # self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(self.X_train1)
        self.X_train = self.scaler.transform(self.X_train1)
        self.X_test = self.scaler.transform(self.X_test1)
        # quit()

    def inference_and_evaluate_model(self, model=None, model_path=None):
        if model is None:
            model = joblib.load(model_path)

        y_true, y_pred = self.y_test, model.predict(self.X_test[3:4, ])
        print('X_test {0}'.format(self.X_test[3:4, ]))
        print('------')
        print('y_test or y_true {0}'.format(y_true[3:4, ]))
        print('------')
        print('y_pred by model {0}'.format(y_pred))

        y_true, y_pred = self.y_test, model.predict(self.X_test)
        df_new = pd.DataFrame(columns=self.df.columns)
        for i, row in self.y_test.iteritems():
            print('', i, row)
            df_new = df_new.append(self.df.loc[i], ignore_index=True)
        df_new['y_true'] = self.y_test.values
        df_new['y_pred'] = list(y_pred)

        df_new.to_csv(model_path[:-7] + '_inference_result.csv', index=True, index_label='Row')

    def inference_model(self, model=None, model_path=None, features_path=None):
        if model is None:
            model = joblib.load(model_path)

        df_features = pd.read_csv(features_path, delimiter=',', index_col=False)
        cols = []
        for i, metric_name_ in enumerate(testability_metrics):
            cols.append(testability_metrics[metric_name_][0])
        X_test1 = df_features[cols]
        X_test = self.scaler.transform(X_test1)
        y_pred = model.predict(X_test)
        df_features['PredictedTsDDTestability'] = list(y_pred)
        # print(df_features)

        df_features.to_csv(r'inference_results/' + self.path_leaf(features_path)[:-4] + '_predicted.csv',
                           index=True, index_label='Row')
        project_testability = df_features['PredictedTsDDTestability'].mean()
        return project_testability

    def path_leaf(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    def evaluate_model(self, model=None, model_path=None):
        # X = self.data_frame.iloc[:, 1:-4]
        # y = self.data_frame.iloc[:, -4]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        if model is None:
            model = joblib.load(model_path)

        y_true, y_pred = self.y_test, model.predict(self.X_test)
        # y_score = model.predict_proba(X_test)

        # Print all classifier model metrics
        print('Evaluating regressor ...')
        print('Regressor minimum prediction', min(y_pred), 'Regressor maximum prediction', max(y_pred))
        df = pd.DataFrame()
        df['r2_score_uniform_average'] = [r2_score(y_true, y_pred, multioutput='uniform_average')]
        df['r2_score_variance_weighted'] = [r2_score(y_true, y_pred, multioutput='variance_weighted')]

        df['explained_variance_score_uniform_average'] = [
            explained_variance_score(y_true, y_pred, multioutput='uniform_average')]
        df['explained_variance_score_variance_weighted'] = [
            explained_variance_score(y_true, y_pred, multioutput='variance_weighted')]

        df['mean_absolute_error'] = [mean_absolute_error(y_true, y_pred)]
        df['mean_squared_error_MSE'] = [mean_squared_error(y_true, y_pred)]
        df['mean_squared_error_RMSE'] = [mean_squared_error(y_true, y_pred, squared=False)]
        df['median_absolute_error'] = [median_absolute_error(y_true, y_pred)]

        if min(y_pred) >= 0:
            df['mean_squared_log_error'] = [mean_squared_log_error(y_true, y_pred)]

        # To handle ValueError:
        # Mean Tweedie deviance error with power=2 can only be used on strictly positive y and y_pred.
        if min(y_pred > 0) and min(y_true) > 0:
            df['mean_poisson_deviance'] = [mean_poisson_deviance(y_true, y_pred, )]
            df['mean_gamma_deviance'] = [mean_gamma_deviance(y_true, y_pred, )]
        df['max_error'] = [max_error(y_true, y_pred)]

        df.to_csv(model_path[:-7] + '_evaluation_metrics_R1.csv', index=True, index_label='Row')

    def regress(self, model_path: str = None, model_number: int = None):
        """
        :param model_path:
        :param model_number: 1: DTR, 2: RFR, 3: GBR, 4: HGBR, 5: SGDR, 6: MLPR,
        :return:
        """
        if model_number == 1:
            regressor = DecisionTreeRegressor(random_state=42, )
            # Set the parameters to be used for tuning by cross-validation
            parameters = {
                'criterion': ['mse', 'friedman_mse', 'mae'],
                'max_depth': range(3, 50, 3),
                'min_samples_split': range(2, 30, 2)
            }
        elif model_number == 2:
            regressor = RandomForestRegressor(random_state=42, )
            parameters = {
                'n_estimators': range(100, 500, 100),
                # 'criterion': ['mse', 'mae'],
                'max_depth': range(3, 50, 5),
                'min_samples_split': range(2, 30, 5),
                # 'max_features': ['auto', 'sqrt', 'log2']
            }
        elif model_number == 3:
            regressor = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, random_state=42, )
            parameters = {
                # 'loss': ['ls', 'lad', ],
                'max_depth': range(3, 50, 5),
                'min_samples_split': range(2, 30, 5)
            }
        elif model_number == 4:
            regressor = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=42, )
            parameters = {
                # 'loss': ['least_squares', 'least_absolute_deviation'],
                'max_depth': range(3, 50, 5),
                'min_samples_leaf': range(2, 30, 5)
            }
        elif model_number == 5:
            regressor = SGDRegressor(early_stopping=True, n_iter_no_change=5, random_state=42, )
            parameters = {
                'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'max_iter': range(50, 500, 50),
                'learning_rate': ['invscaling', 'optimal', 'constant', 'adaptive'],
                'eta0': [0.1, 0.01, 0.001],
                'average': [32, 64]
            }
        elif model_number == 6:
            regressor = MLPRegressor(random_state=42, )
            parameters = {
                'hidden_layer_sizes': [(256, 100), (512, 256, 100)],
                'activation': ['tanh', ],
                'solver': ['adam', ],
                'max_iter': range(50, 250, 50)
            }
        elif model_number == 7:
            regressor = NuSVR(cache_size=500, max_iter=- 1, shrinking=True)
            parameters = {
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid',],
                'degree': [3, ],
                'nu': [0.5, ],
                'C': [1.0, ]
            }

        # Set the objectives which must be optimized during parameter tuning
        # scoring = ['r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_error',]
        scoring = ['neg_root_mean_squared_error', ]
        # CrossValidation iterator object:
        # https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
        cv = ShuffleSplit(n_splits=5, test_size=0.20, random_state=42)
        # Find the best model using grid-search with cross-validation
        clf = GridSearchCV(regressor, param_grid=parameters, scoring=scoring, cv=cv, n_jobs=18,
                           refit='neg_root_mean_squared_error')
        print('fitting model number', model_number)
        clf.fit(X=self.X_train, y=self.y_train)

        print('Writing grid search result ...')
        df = pd.DataFrame(clf.cv_results_, )
        df.to_csv(model_path[:-7] + '_grid_search_cv_results.csv', index=False)
        df = pd.DataFrame()
        print('Best parameters set found on development set:', clf.best_params_)
        df['best_parameters_development_set'] = [clf.best_params_]
        print('Best classifier score on development set:', clf.best_score_)
        df['best_score_development_set'] = [clf.best_score_]
        print('best classifier score on test set:', clf.score(self.X_test, self.y_test))
        df['best_score_test_set:'] = [clf.score(self.X_test, self.y_test)]
        df.to_csv(model_path[:-7] + '_grid_search_cv_results_best.csv', index=False)

        # Save and evaluate the best obtained model
        print('Writing evaluation result ...')
        clf = clf.best_estimator_
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        joblib.dump(clf, model_path)

        self.evaluate_model(model=clf, model_path=model_path)
        # self.evaluate_model_class(model=clf, model_path=model_path)
        # self.inference_model(model=clf, model_path=model_path)
        print('-' * 75)

    def vote(self, model_path=None, dataset_number=1):
        # Trained regressors
        reg1 = joblib.load(r'models_profiles1/HGBR1_DS{0}.joblib'.format(dataset_number))
        reg2 = joblib.load(r'models_profiles1/RFR1_DS{0}.joblib'.format(dataset_number))
        reg3 = joblib.load(r'models_profiles1/MLPR1_DS{0}.joblib'.format(dataset_number))
        # reg4 = load(r'sklearn_models6/SGDR1_DS1.joblib')

        ereg = VotingRegressor([('HGBR1_DS{0}'.format(dataset_number), reg1),
                                ('RFR1_DS{0}'.format(dataset_number), reg2),
                                ('MLPR1_DS{0}'.format(dataset_number), reg3)
                                ],
                               weights=[3. / 6., 2. / 6., 1. / 6.], n_jobs=18)

        ereg.fit(self.X_train, self.y_train, )
        joblib.dump(ereg, model_path)
        self.evaluate_model(model=ereg, model_path=model_path)

    def determine_starts(self, project_raw_testability_score: float = None):
        TS_MAX = 0.995048905  # in our first version of reusability benchmark
        TS_MIN = 0.260409734  # in our first version of reusability benchmark
        normalized_rate = (project_raw_testability_score - TS_MIN) / (TS_MAX - TS_MIN)
        print('Normalized testability score {}'.format(normalized_rate))
        benchmark_path = r'benchmark_projects_testability.csv'
        df_gold = pd.read_csv(benchmark_path, delimiter=',', index_col=False)

        benchmark_size = 110

        categories = [(df_gold['TestabilityNormalized'][0],
                       df_gold['TestabilityNormalized'][math.ceil(5 * benchmark_size / 100)]),  # Five stars (*****)

                      (df_gold['TestabilityNormalized'][math.ceil(5 * benchmark_size / 100)],
                       df_gold['TestabilityNormalized'][math.ceil(35 * benchmark_size / 100)]),  # Four stars (****)

                      (df_gold['TestabilityNormalized'][math.ceil(35 * benchmark_size / 100)],
                       df_gold['TestabilityNormalized'][math.ceil(65 * benchmark_size / 100)]),  # Three stars (***)

                      (df_gold['TestabilityNormalized'][math.ceil(65 * benchmark_size / 100)],
                       df_gold['TestabilityNormalized'][math.ceil(95 * benchmark_size / 100)]),  # Two stars (**)

                      (df_gold['TestabilityNormalized'][math.ceil(95 * benchmark_size / 100)],
                       df_gold['TestabilityNormalized'][math.ceil(100 * benchmark_size / 100) - 1])  # One star (*)
                      ]
        print(categories)
        if categories[4][1] < normalized_rate <= categories[4][0]:
            return 1
        elif categories[3][1] < normalized_rate <= categories[3][0]:
            return 2
        elif categories[2][1] < normalized_rate <= categories[2][0]:
            return 3
        elif categories[1][1] < normalized_rate <= categories[1][0]:
            return 4
        elif categories[0][1] < normalized_rate <= categories[0][0]:
            return 5
        else:
            raise ValueError('Ohh! The rate in not in the range.')


def train():
    ds_path = r'../benchmark/SF110/dataset_final/all_with_label_cleaned4.csv'
    reg = Regression(df_path=ds_path)

    reg.regress(model_path=r'models_profiles1/DTR1_DS1.joblib', model_number=1)
    reg.regress(model_path=r'models_profiles1/RFR1_DS1.joblib', model_number=2)
    # reg.regress(model_path=r'models_profiles1/GBR1_DS1.joblib', model_number=3)
    reg.regress(model_path=r'models_profiles1/HGBR1_DS1.joblib', model_number=4)
    reg.regress(model_path=r'models_profiles1/SGDR_DS1.joblib', model_number=5)
    reg.regress(model_path=r'models_profiles1/MLPR1_DS1.joblib', model_number=6)
    reg.regress(model_path=r'models_profiles1/NuSVR1_DS1.joblib', model_number=7)
    reg.vote(model_path=r'models_profiles1/VoR1_DS1.joblib', dataset_number=1)


def inference():
    date_time = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print('Start datetime {0}'.format(date_time))

    dataset_path = r'../benchmark/SF110/dataset_final/all_with_label_cleaned4.csv'
    model_path = r'models_profiles1/VoR1_DS1.joblib'

    # data_path = r'../benchmark/SF110/dataset/10_water-simulator-Class.csv'
    # data_path = r'../benchmark/SF110/dataset/107_weka-Class.csv'
    # data_path = r'../benchmark/SF110/dataset/32_httpanalyzer-Class.csv'
    data_path = r'../benchmark/SF110/data_to_inference/10_water-simulator-after-refactor3-Class.csv'

    reg = Regression(df_path=dataset_path)
    project_testability = reg.inference_model(model_path=model_path, features_path=data_path)
    project_testability_star = reg.determine_starts(project_raw_testability_score=project_testability)
    print('project_testability {}'.format(project_testability))
    print('project_testability stars rate {} ({})'.format(project_testability_star, '*' * project_testability_star))

    date_time = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print('End datetime {0}'.format(date_time))

def compute_benchmark_projects_testability(root_dir_path=None):
    files = [f for f in os.listdir(root_dir_path) if os.path.isfile(os.path.join(root_dir_path, f))]
    df = pd.DataFrame()
    projects_testability_list = []
    projects_name = []
    for file_ in files:
        project_name = file_[:-4]
        print('Computing testability for project {0}'.format(project_name))
        df_current = pd.read_csv(root_dir_path + file_, delimiter=',', index_col=False)
        project_testability = df_current['TsDDTestability'].mean()
        print('Project {0} testability: {1}'.format(project_name, round(project_testability, 6)))
        projects_testability_list.append(project_testability)
        projects_name.append(project_name)

    ts_max = max(projects_testability_list)
    ts_min = min(projects_testability_list)
    ts_normalized = []
    for ts_ in projects_testability_list:
        ts_normalized.append((ts_ - ts_min) / (ts_max - ts_min))

    df['Project'] = projects_name
    df['Testability'] = projects_testability_list
    df['TestabilityNormalized'] = ts_normalized
    df = df.sort_values('Testability', ascending=False)
    df.to_csv('./benchmark_projects_testability.csv', index=False)


# Main Driver
if __name__ == '__main__':
    # ds = Dataset()
    # ds.add_evosuite_information()
    # ds.concatenate_csv_files()
    # ds.clean_data()
    # train()
    # Dataset.compute_testability(lc=0.287671232876712, br=0.236051502145922, mu=0.175496688741721, taw=37, TNM=54)

    # compute_benchmark_projects_testability(root_dir_path=r'../benchmark/SF110/dataset2/')
    inference()  # with demo project for example
