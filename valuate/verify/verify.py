from valuate.verify import *


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
    #     final = pd.DataFrame()
    #     open_model_detail = pd.read_csv('../tmp/train/open_model_detail.csv')
    #     open_city = pd.read_csv('../tmp/train/open_city.csv')
    #     open_city = open_city[open_city['parent'] != 0]
    #     print('总计款型:', len(set(open_model_detail.model_detail_slug.values)))
    #     for i, model_details in enumerate(list(set(open_model_detail.model_detail_slug.values))):
    #         time1 = time.time()
    #         test_set = pd.DataFrame()
    #         test_set['city'] = pd.Series(open_city.name.values)
    #         test_set['use_time'] = 1
    #         test_set['mile'] = 1
    #         test_set['model_detail_slug'] = model_details
    #         final = final.append(test_set)
    #         test_set = pd.DataFrame()
    #         test_set['city'] = pd.Series(open_city.name.values)
    #         test_set['use_time'] = 240
    #         test_set['mile'] = 50
    #         test_set['model_detail_slug'] = model_details
    #         final = final.append(test_set)
    #         time2 = time.time()
    #         print(i, '完成款型组装:', model_details, ' 耗时:', time2 - time1)
    #         if ((i % 1000) == 0) & (i != 0):
    #             num = int(i / 1000)
    #             final.to_csv('verify/set/self_set_part' + str(num) + '.csv', index=False)
    #             final = pd.DataFrame()
    #         elif i == (len(open_model_detail) - 1):
    #             num = int(i / 1000) + 1
    #             final.to_csv('verify/set/self_set_part' + str(num) + '.csv', index=False)
    #             final = pd.DataFrame()
    #
    #     model_detail_map = pd.read_csv('predict/map/model_detail_map.csv')
    #     model_detail_map = model_detail_map.loc[:, ['price_bn', 'model_slug', 'model_detail_slug']]
    #     final = pd.DataFrame()
    #     for i in range(1, 36):
    #         temp = pd.read_csv('verify/set/self_set_part' + str(i) + '.csv')
    #         final = final.append(temp)
    #         print('完成part:', i, '组装')
    #     final = final.merge(model_detail_map, how='left', on='model_detail_slug')
    #     final = final[final['price_bn'].notnull()]
    #     final.to_csv('verify/set/self_test_set.csv', index=False)
    #
    # def generate_verify_tables(self):
    #     """
    #     生成验证集
    #     """
    #     self.generate_self_test_set()
    #
    # def predict_self_test_set(self):
    #     """
    #     验证自测数据
    #     """
    #     from valuate.predict.predict_batch import Predict as batch
    #
    #     model_detail_map = pd.read_csv('predict/map/model_detail_map.csv')
    #     cities = pd.read_csv('predict/map/province_city_map.csv')
    #     print('details:', len(model_detail_map), 'cities:', len(cities), 'all data should be:', len(model_detail_map)*len(cities)*2)
    #     self.self_test_set = pd.read_csv('verify/set/self_test_set.csv')
    #     self.self_test_set['source_type'] = 'dealer'
    #     if len(self.self_test_set) == (len(model_detail_map)*len(cities)*2):
    #         print('all data is ready that can verify!')
    #     else:
    #         print('some data could be missing!')
    #         return False
    #     predict = batch()
    #     result = predict.predict_batch(self.self_test_set, store=True)
    #
    #     result = pd.DataFrame()
    #     for i in range(1, 16):
    #         temp = pd.read_csv('verify/set/self_set_predict_part' + str(i) + '.csv')
    #         result = result.append(temp)
    #
    #     result.to_csv('verify/set/self_test_predict_set.csv', index=False)
    #
    # def execute_verify(self):
    #     """
    #     执行验证
    #     """
    #     # 模型自测验证
    #     self.predict_self_test_set()

    def load_valuated_cities(self):
        """
        返回可预测城市
        """
        return cities

    def load_valuated_models(self):
        """
        返回可预测车型
        """
        return models

    def load_valuated_model_details(self):
        """
        返回可预测款型
        """
        return details