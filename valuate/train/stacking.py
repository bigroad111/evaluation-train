import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV

from valuate.conf import algorithm_settings as als
from valuate.conf import global_settings as gl


class SklearnHelper(object):
    """
    训练模型助手
    """
    def __init__(self, clf, params=None):
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def train_condition(self, x_train):
        self.clf.fit(x_train)

    def predict(self, x):
        return self.clf.predict(x)

    def save_model(self, path):
        # save model
        joblib.dump(self.clf, path)

    def get_labels(self):
        return self.clf.labels_.tolist()


class Stacking(object):
    """
    组合模型训练
    """
    def __init__(self):
        self.x_all = []
        self.y_all = []
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.train_level2 = []
        self.clf = []

        # all cities
        open_city = pd.read_csv('../tmp/train/open_city.csv')
        open_city = open_city[open_city['parent'] != 0]
        self.cities = list(set(open_city.name.values))

        # 创建各类算法的模型
        self.km = SklearnHelper(clf=KMeans, params=als.km_params)
        self.rf = SklearnHelper(clf=RandomForestRegressor, params=als.rf_params)
        self.et = SklearnHelper(clf=ExtraTreesRegressor, params=als.et_params)
        self.gtb = SklearnHelper(clf=GradientBoostingRegressor, params=als.gtb_params)
        self.xgb = SklearnHelper(clf=xgb.XGBRegressor, params=als.xgb_level1_params)
        self.gbm = SklearnHelper(clf=lgb.LGBMRegressor, params=als.lgb_params)

    def object_to_num(self, model_slug):
        """
        将object类型的字段转换成数字编码
        """
        self.x_all = self.x_all.loc[:, gl.TRAIN_FEATURE]

        for f in self.x_all.columns:
            if self.x_all[f].dtype == 'object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(self.x_all[f].values))
                self.x_all[f + '_encode'] = lbl.transform(list(self.x_all[f].values))

        for f in self.x_all.columns:
            name = f
            if self.x_all[f].dtype == 'object':
                f = self.x_all.loc[0:, [f, f + '_encode']]
                f = f.drop_duplicates(name)
                file_name = 'predict/models/'+model_slug+'/feature_encode/' + name + '.csv'
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                f.to_csv(file_name, index=False, encoding='utf-8')
                self.x_all = self.x_all.drop(name, axis=1)

        # 生成特征顺序
        with open('predict/models/'+model_slug+'/feature_encode/feature_order.txt', 'wb') as fp:
            pickle.dump(list(self.x_all.columns.values), fp)

    def train_test_split(self):
        """
        根据比例分割训练数据
        """
        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_all, self.y_all, test_size=als.TEST_SIZE)
        self.x_train = self.x_all.copy()
        self.x_test = self.x_all.copy()
        self.y_train = self.y_all.copy()
        self.y_test = self.y_all.copy()

    def train_test_split_for_grid(self):
        """
        根据比例分割训练数据
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_all, self.y_all, test_size=als.TEST_SIZE)

    def train_level1_models(self):
        """
        stack level1 训练各模型
        """
        self.gtb.train(self.x_train, self.y_train)
        self.gtb.save_model('predict/model/gradient_tree_boosting_level1.pkl')
        self.xgb.train(self.x_train, self.y_train)
        self.xgb.save_model('predict/model/xgboost_level1.pkl')
        self.gbm.train(self.x_train, self.y_train)
        self.gbm.save_model('predict/model/lightgbm_level1.pkl')

    def generate_train_level2_data(self):
        """
        获得level2的训练数据
        """
        # 加载level1模型
        self.gtb = joblib.load('predict/model/gradient_tree_boosting_level1.pkl')
        self.xgb = joblib.load('predict/model/xgboost_level1.pkl')
        self.gbm = joblib.load('predict/model/lightgbm_level1.pkl')

        self.train_level2 = pd.DataFrame(self.gtb.predict(self.x_test), columns=['gradient_tree_boosting'])
        self.train_level2['xgboost'] = pd.Series(self.xgb.predict(self.x_test)).values
        # self.train_level2 = pd.DataFrame(self.xgb.predict(self.x_test), columns=['xgboost'])
        self.train_level2['gbm'] = pd.Series(self.gbm.predict(self.x_test)).values
        self.train_level2['city_encode'] = self.x_test['city_encode']
        self.train_level2['model_detail_slug_encode'] = self.x_test['model_detail_slug_encode']
        self.train_level2['price'] = pd.Series(self.y_test).values
        self.train_level2.to_csv('../tmp/train/train_level2.csv', index=False)

    def train_level2_model(self, model_slug):
        """
        stack level2 模型训练
        """
        file_name = 'predict/models/'+model_slug+'/model/xgboost_level2.model'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        d_train = xgb.DMatrix(self.x_all, label=self.y_all)
        model = xgb.train(als.xgb_gpu_params, d_train, als.NUM_BOOST_ROUND)
        model.save_model(file_name)

    def grid_search_cv_models(self):
        """
        grid search搜索各算法最佳参数
        """
        self.object_to_num()
        self.train_test_split_for_grid()

        # self.clf = GridSearchCV(RandomForestRegressor(), als.rf_grid_params, verbose=1, n_jobs=-1)
        # self.clf.fit(self.x_train, self.y_train)
        # print('Random Forest best score:', self.clf.best_score_)
        # print('Random Forest best score:', self.clf.best_params_)
        #
        # self.clf = GridSearchCV(ExtraTreesRegressor(), als.et_grid_params, verbose=2, n_jobs=-1)
        # self.clf.fit(self.x_train, self.y_train)
        # print('Extra Trees best score:', self.clf.best_score_)
        # print('Extra Trees best score:', self.clf.best_params_)
        #
        # self.clf = GridSearchCV(GradientBoostingRegressor(), als.gtb_grid_params, verbose=1, n_jobs=-1)
        # self.clf.fit(self.x_train, self.y_train)
        # print('Gradient Tree Boosting best score:', self.clf.best_score_)
        # print('Gradient Tree Boosting best score:', self.clf.best_params_)

        self.clf = GridSearchCV(xgb.XGBRegressor(), als.xgb_grid_leve1_params, verbose=2, n_jobs=-1)
        self.clf.fit(self.x_train, self.y_train)
        print('XGBOOST best score:', self.clf.best_score_)
        print('XGBOOST best score:', self.clf.best_params_)
        #
        # self.clf = GridSearchCV(lgb.LGBMRegressor(), als.lgb_grid_params, verbose=2, n_jobs=-1)
        # self.clf.fit(self.x_train, self.y_train)
        # print('lightGBM best score:', self.clf.best_score_)
        # print('lightGBM best score:', self.clf.best_params_)

    def grid_search(self):
        """
        执行grid_search查找最佳参数
        """
        self.grid_search_cv_models()

    def execute_single(self, model_slug):
        """
        执行单车型模型训练流程
        """
        print(model_slug)
        time1 = time.time()
        train = pd.read_csv('predict/models/' + model_slug + '/data/train.csv')
        # 查找数据最多的城市
        city = train.groupby(['city'], sort=False)['city'].agg({'no': 'count'})
        city.reset_index(inplace=True)
        max_city = city.loc[(city['no'] == max(city.no.values)), 'city'].values[0]

        # 开始训练
        self.x_all = train.loc[:, gl.TRAIN_FEATURE]
        self.y_all = np.log(train[gl.TARGET_FEATURE])
        self.object_to_num(model_slug)
        self.train_level2_model(model_slug)
        self.fill_all_city(model_slug, max_city)
        time2 = time.time()
        print('完成' + model_slug + '车型模型的训练!用时:', int(time2 - time1))

    def execute_multitude(self, model_slugs):
        """
        执行多车型模型训练流程
        """
        for i, model_slug in enumerate(model_slugs):
            print(model_slug)
            time1 = time.time()
            train = pd.read_csv('predict/models/'+model_slug+'/data/train.csv')
            # 查找数据最多的城市
            city = train.groupby(['city'], sort=False)['city'].agg({'no': 'count'})
            city.reset_index(inplace=True)
            max_city = city.loc[(city['no'] == max(city.no.values)), 'city'].values[0]

            # 开始训练
            self.x_all = train.loc[:, gl.TRAIN_FEATURE]
            self.y_all = np.log(train[gl.TARGET_FEATURE])
            self.object_to_num(model_slug)
            self.train_level2_model(model_slug)
            self.fill_all_city(model_slug, max_city)
            time2 = time.time()
            print('no:', i, '完成'+model_slug+'车型模型的训练!用时:', int(time2-time1))

    def fill_all_city(self, model_slug, max_city):
        """
        补缺城市到城市编码表
        """
        predicted_cities = pd.read_csv('predict/models/'+model_slug+'/feature_encode/city.csv')

        cities = list(set(predicted_cities.city.values))
        lack_cities = np.setdiff1d(self.cities, cities)
        lack_cities_encode = predicted_cities.loc[(predicted_cities['city'] == max_city), 'city_encode'].values[0]

        if len(lack_cities) > 0:
            lack_city = pd.DataFrame(lack_cities, columns=['city'])
            lack_city['city_encode'] = lack_cities_encode
            predicted_cities = predicted_cities.append(lack_city, ignore_index=True)
            predicted_cities.to_csv('predict/models/'+model_slug+'/feature_encode/city.csv', index=False, encoding='utf-8')