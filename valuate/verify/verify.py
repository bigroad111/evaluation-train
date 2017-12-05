import os
import time
import pandas as pd
import numpy as np
import multiprocessing

from valuate.conf import global_settings as gl

class Verify(object):

    def __init__(self):
        self.self_test_set = []
        self.history_price_trend_set = []
        self.personal_price_future_trend_set = pd.DataFrame()
        self.path = os.path.abspath(os.path.dirname(gl.__file__))
        self.path = self.path.replace('conf', '')

    # def generate_self_test_set(self):
    #     """
    #     生成模型自测数据集
    #     """
    #     open_model_detail = pd.read_csv('../tmp/train/open_model_detail.csv')
    #     open_city = pd.read_csv('../tmp/train/open_city.csv')
    #     open_city = open_city[open_city['parent'] != 0]
    #     self.self_test_set = open_model_detail[open_model_detail['price_bn'] != 0]
    #     self.self_test_set['city'] = open_city['name']
    #     self.self_test_set['city'] = self.self_test_set['city'].fillna('深圳')
    #     self.self_test_set['use_time'] = 1
    #     self.self_test_set['mile'] = 0
    #     self.self_test_set = self.self_test_set.drop(['global_slug', 'volume'], axis=1)
    #
    #     temp = self.self_test_set.copy()
    #     temp['use_time'] = 7500
    #     temp['mile'] = 40
    #     self.self_test_set = self.self_test_set.append(temp, ignore_index=True)
    #     self.self_test_set.to_csv('verify/set/self_test_set.csv', index=False)
    #
    # def generate_history_price_trend_set(self):
    #     """
    #     生成历史价格走势数据集
    #     """
    #     self.history_price_trend_set = pd.read_csv('../tmp/train/train.csv')
    #     self.history_price_trend_set = self.history_price_trend_set.sample(frac=1).reset_index(drop=True)
    #     self.history_price_trend_set = self.history_price_trend_set.drop_duplicates('model_detail_slug')
    #     self.history_price_trend_set['city'] = self.history_price_trend_set['city'].fillna('深圳')
    #     self.history_price_trend_set = self.history_price_trend_set.drop(['domain', 'id', 'price', 'price_bn', 'source_type'], axis=1)
    #
    #     temp = self.history_price_trend_set.copy()
    #     temp.loc[0:int(len(temp) / 2), 'use_time'] = 150
    #     temp.loc[0:int(len(temp) / 2), 'mile'] = 1
    #     temp.loc[int(len(temp) / 2):, 'use_time'] = 7500
    #     temp.loc[int(len(temp) / 2):, 'mile'] = 40
    #     self.history_price_trend_set = self.history_price_trend_set.append(temp, ignore_index=True)
    #
    #     self.history_price_trend_set.to_csv('verify/set/history_price_trend_set.csv', index=False)
    #
    # def generate_personal_price_future_trend_set(self):
    #     """
    #     生成个人交易未来价格走势数据集
    #     """
    #     self.personal_price_future_trend_set = pd.read_csv('../tmp/train/train.csv')
    #     self.personal_price_future_trend_set = self.personal_price_future_trend_set.sample(frac=1).reset_index(drop=True)
    #     self.personal_price_future_trend_set = self.personal_price_future_trend_set.drop_duplicates('model_detail_slug')
    #     self.personal_price_future_trend_set['city'] = self.personal_price_future_trend_set['city'].fillna('深圳')
    #     self.personal_price_future_trend_set = self.personal_price_future_trend_set.drop(['domain', 'id', 'price', 'price_bn', 'source_type'], axis=1)
    #
    #     temp = self.personal_price_future_trend_set.copy()
    #     temp.loc[0:int(len(temp) / 2), 'use_time'] = 1
    #     temp.loc[0:int(len(temp) / 2), 'mile'] = 0
    #     temp.loc[int(len(temp) / 2):, 'use_time'] = 7500
    #     temp.loc[int(len(temp) / 2):, 'mile'] = 40
    #     self.personal_price_future_trend_set = self.personal_price_future_trend_set.append(temp, ignore_index=True)
    #
    #     self.personal_price_future_trend_set.to_csv('verify/set/personal_price_future_trend_set.csv', index=False)
    #
    # def generate_verify_tables(self):
    #     """
    #     生成验证集
    #     """
    #     self.generate_self_test_set()
    #     self.generate_history_price_trend_set()
    #     self.generate_personal_price_future_trend_set()
    #
    # def predict_self_test_set(self):
    #     """
    #     验证自测数据
    #     """
    #     predict_price = []
    #     predict = Predict()
    #     self.self_test_set = pd.read_csv('verify/set/self_test_set.csv')
    #     for i in range(0, len(self.self_test_set)):
    #         city = self.self_test_set.loc[i, 'city']
    #         model_detail_slug = self.self_test_set.loc[i, 'model_detail_slug']
    #         use_time = self.self_test_set.loc[i, 'use_time']
    #         mile = self.self_test_set.loc[i, 'mile']
    #         print(city, model_detail_slug, use_time, mile)
    #         result = predict.predict(city, model_detail_slug, use_time, mile)
    #         predict_price.append(result.loc[(result['intent'] == 'cpo'), 'excellent'].values[0])
    #
    #     self.self_test_set['predict_price'] = pd.Series(predict_price)
    #     self.self_test_set.to_csv('verify/set/self_test_set.csv', index=False)
    #
    # def predict_history_price_trend_set(self, data, core=8, num=1):
    #     """
    #     验证历史价格趋势数据
    #     data:验证集
    #     core:核心数
    #     num:编号
    #     """
    #     try:
    #         result_b_2_c = []
    #         result_c_2_b = []
    #         result_c_2_c = []
    #         predict = Predict()
    #         history_price_trend_set = data
    #         records = len(history_price_trend_set)
    #         num_core = int(records / core)
    #         history_price_trend_set.loc[(history_price_trend_set['use_time'] <= 0), 'use_time'] = 1
    #         history_price_trend_set = history_price_trend_set.loc[num * num_core:(num + 1) * num_core, :]
    #         history_price_trend_set.reset_index(inplace=True)
    #         history_price_trend_set = history_price_trend_set.drop('index', axis=1)
    #         for i in range(0, len(history_price_trend_set)):
    #             city = history_price_trend_set.loc[i, 'city']
    #             model_detail_slug = history_price_trend_set.loc[i, 'model_detail_slug']
    #             use_time = int(history_price_trend_set.loc[i, 'use_time'])
    #             mile = float(history_price_trend_set.loc[i, 'mile'])
    #             print(num, city, model_detail_slug, use_time, mile)
    #             result = predict.history_price_trend(city, model_detail_slug, use_time, mile, ret_type='normal')
    #             result_b_2_c.append(result.loc[0, :].values.flatten().tolist())
    #             result_c_2_b.append(result.loc[1, :].values.flatten().tolist())
    #             result_c_2_c.append(result.loc[2, :].values.flatten().tolist())
    #
    #         history_price_trend_set['b_2_c'] = pd.Series(result_b_2_c)
    #         history_price_trend_set['c_2_b'] = pd.Series(result_c_2_b)
    #         history_price_trend_set['c_2_c'] = pd.Series(result_c_2_c)
    #         history_price_trend_set.to_csv('verify/set/history' + str(num) + '.csv', index=False)
    #     except:
    #         import traceback
    #         print('except!,check!-------------------------------:', num)
    #         print(traceback.print_exc())
    #
    # def predict_personal_price_future_trend_set(self, data, core=8, num=1):
    #     """
    #     验证未来价格趋势
    #     """
    #     try:
    #         result_b_2_c = []
    #         result_c_2_b = []
    #         result_c_2_c = []
    #         predict = Predict()
    #         personal_price_future_trend_set = data
    #         records = len(personal_price_future_trend_set)
    #         num_core = int(records / core)
    #         personal_price_future_trend_set.loc[(personal_price_future_trend_set['use_time'] <= 0), 'use_time'] = 1
    #         personal_price_future_trend_set = personal_price_future_trend_set.loc[num * num_core:(num+1) * num_core, :]
    #         personal_price_future_trend_set.reset_index(inplace=True)
    #         personal_price_future_trend_set = personal_price_future_trend_set.drop('index', axis=1)
    #         for i in range(0, len(personal_price_future_trend_set)):
    #             city = personal_price_future_trend_set.loc[i, 'city']
    #             model_detail_slug = personal_price_future_trend_set.loc[i, 'model_detail_slug']
    #             use_time = int(personal_price_future_trend_set.loc[i, 'use_time'])
    #             mile = float(personal_price_future_trend_set.loc[i, 'mile'])
    #             print(num, city, model_detail_slug, use_time, mile)
    #             result = predict.future_price_trend(city, model_detail_slug, use_time, mile, ret_type='normal')
    #             result_b_2_c.append(result.loc[0, :].values.flatten().tolist())
    #             result_c_2_b.append(result.loc[1, :].values.flatten().tolist())
    #             result_c_2_c.append(result.loc[2, :].values.flatten().tolist())
    #
    #         personal_price_future_trend_set['b_2_c'] = pd.Series(result_b_2_c)
    #         personal_price_future_trend_set['c_2_b'] = pd.Series(result_c_2_b)
    #         personal_price_future_trend_set['c_2_c'] = pd.Series(result_c_2_c)
    #         personal_price_future_trend_set.to_csv('verify/set/future'+str(num)+'.csv', index=False)
    #     except:
    #         import traceback
    #         print('except!,check!-------------------------------:', num)
    #         print(traceback.print_exc())
    #
    # def history_future_price_verify(self):
    #     """
    #     多进程验证历史和未来价格趋势
    #     """
    #     # 历史交易价格验证
    #     data = pd.read_csv('verify/set/history_price_trend_set.csv')
    #     pool = multiprocessing.Pool(processes=8)
    #     pool.apply_async(self.predict_history_price_trend_set, (data, 8, 0,))
    #     pool.apply_async(self.predict_history_price_trend_set, (data, 8, 1,))
    #     pool.apply_async(self.predict_history_price_trend_set, (data, 8, 2,))
    #     pool.apply_async(self.predict_history_price_trend_set, (data, 8, 3,))
    #     pool.apply_async(self.predict_history_price_trend_set, (data, 8, 4,))
    #     pool.apply_async(self.predict_history_price_trend_set, (data, 8, 5,))
    #     pool.apply_async(self.predict_history_price_trend_set, (data, 8, 6,))
    #     pool.apply_async(self.predict_history_price_trend_set, (data, 8, 7,))
    #     pool.close()
    #     pool.join()
    #
    #     # 个人交易未来交易价格验证
    #     data = pd.read_csv('verify/set/personal_price_future_trend_set.csv')
    #     pool = multiprocessing.Pool(processes=8)
    #     pool.apply_async(self.predict_personal_price_future_trend_set, (data, 8, 0,))
    #     pool.apply_async(self.predict_personal_price_future_trend_set, (data, 8, 1,))
    #     pool.apply_async(self.predict_personal_price_future_trend_set, (data, 8, 2,))
    #     pool.apply_async(self.predict_personal_price_future_trend_set, (data, 8, 3,))
    #     pool.apply_async(self.predict_personal_price_future_trend_set, (data, 8, 4,))
    #     pool.apply_async(self.predict_personal_price_future_trend_set, (data, 8, 5,))
    #     pool.apply_async(self.predict_personal_price_future_trend_set, (data, 8, 6,))
    #     pool.apply_async(self.predict_personal_price_future_trend_set, (data, 8, 7,))
    #     pool.close()
    #     pool.join()
    #
    # def execute_verify(self):
    #     """
    #     执行验证
    #     """
    #     # 模型自测验证
    #     # self.predict_self_test_set()
    #
    #     # 多进程验证历史和未来价格趋势
    #     # self.history_future_price_verify()
    #
    #     # 生成结果表
    #     # self.process_tables()
    #
    # def load_valuated_cities(self):
    #     """
    #     返回可预测城市
    #     """
    #     cities = pd.read_csv(self.path+'predict/feature_encode/city.csv')
    #     cities = cities.loc[((cities['city'].notnull()) & (cities['city'] != '%E5%8D%97%E4%BA%AC')), :]
    #     return list(set(cities.city.values))
    #
    # def load_valuated_brand(self):
    #     """
    #     返回可预测品牌
    #     """
    #     brand = pd.read_csv(self.path+'predict/map/valuated_model_detail.csv')
    #     return list(set(brand.brand.values))
    #
    # def load_valuated_models(self):
    #     """
    #     返回可预测车型
    #     """
    #     models = pd.read_csv(self.path+'predict/map/valuated_model_detail.csv')
    #     return list(set(models.global_slug.values))
    #
    # def load_valuated_model_details(self):
    #     """
    #     返回可预测款型
    #     """
    #     model_detail = pd.read_csv(self.path+'predict/map/valuated_model_detail.csv')
    #     return list(set(model_detail.model_detail_slug.values))
    #
    # def load_self_test_set(self):
    #     """
    #     加载自测集
    #     """
    #     self.self_test_set = pd.read_csv(self.path+'verify/set/history_price_trend_set.csv')
    #     return self.self_test_set
    #
    # def load_history_price_trend_set(self):
    #     """
    #     加载历史价格趋势集
    #     """
    #     self.history_price_trend_set = pd.read_csv(self.path+'verify/set/history_price_trend_set.csv')
    #     return self.history_price_trend_set
    #
    # def load_personal_price_future_trend_set(self):
    #     """
    #     加载个人交易价未来趋势集
    #     """
    #     self.personal_price_future_trend_set = pd.read_csv(self.path+'verify/set/personal_price_future_trend_set.csv')
    #     return self.personal_price_future_trend_set
    #
    # def process_tables(self):
    #     result = pd.DataFrame()
    #     for i in range(0, 8):
    #         temp = pd.read_csv('verify/set/history'+str(i)+'.csv')
    #         result = result.append(temp, ignore_index=True)
    #     result.to_csv('verify/set/history_price_trend_set.csv', index=False)
    #
    #     result = pd.DataFrame()
    #     for i in range(0, 8):
    #         temp = pd.read_csv('verify/set/future'+str(i)+'.csv')
    #         result = result.append(temp, ignore_index=True)
    #     result.to_csv('verify/set/personal_price_future_trend_set.csv', index=False)