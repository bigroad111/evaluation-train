from valuate.predict import *


def process_str(df, feature):
    text = df[feature]
    text = re.sub(r"1\.\d\d\d\d", '0.9999', text)
    text = re.sub(r"0\.", '', text)
    text = re.sub(r"\d\,", '', text)
    text = re.sub(r"\d\]", '', text)
    text = text.replace(" ","")
    text = text.replace("[","")
    text = text.replace("]","")
#     text = text[0:120]
    return text


def get_year(df, feature, start, end):
    text = df[feature]
    text = text[start:end]
    return text


def find_popularity(df):
    if str(df['province'] + df['model_slug']) in province_popularity_map.index:
        return province_popularity_map.loc[str(df['province'] + df['model_slug']), 'popularity']
    else:
        return 'C'


class Predict(object):

    def __init__(self):
        """
        加载各类匹配表和模型
        """
        self.test_level1 = pd.DataFrame()
        self.test_level2 = []
        self.predict_hedge = []
        self.predict_price = []
        self.valuate_model = []

    def create_test_data(self, model_slug):
        """
        创建测试数据
        """
        # 加载编码
        encode_model_detail_slug = pd.read_csv('predict/models/' + model_slug + '/feature_encode/model_detail_slug.csv')

        test = pd.DataFrame()
        for model_detail_slug in list(set(encode_model_detail_slug.model_detail_slug.values)):
            temp = province_city_map.copy()
            temp['model_detail_slug'] = model_detail_slug
            test = test.append(temp)
        test['model_slug'] = model_slug
        test['popularity'] = test.apply(find_popularity, axis=1)
        test['source_type'] = 'dealer'
        result = test.copy()
        test['source_type'] = 'cpersonal'
        result = result.append(test)
        result = result.drop(['model_slug', 'province'], axis=1)
        result.to_csv('predict/models/' + model_slug + '/data/test.csv', index=False, encoding='utf-8')

    def predict_test_data(self, model_slug):
        """
        预测测试数据
        """
        # 加载预测数据
        test = pd.read_csv('predict/models/' + model_slug + '/data/test.csv')
        # 加载估值模型
        self.valuate_model = xgb.Booster()
        self.valuate_model.load_model('predict/models/' + model_slug + '/model/xgboost_level2.model')
        # 加载编码
        encode_city = pd.read_csv('predict/models/' + model_slug + '/feature_encode/city.csv')
        encode_model_detail_slug = pd.read_csv('predict/models/' + model_slug + '/feature_encode/model_detail_slug.csv')
        encode_popularity = pd.read_csv('predict/models/' + model_slug + '/feature_encode/popularity.csv')
        encode_source_type = pd.read_csv('predict/models/' + model_slug + '/feature_encode/source_type.csv')
        test = test.merge(encode_city, how='left', on='city')
        test = test.merge(encode_model_detail_slug, how='left', on='model_detail_slug')
        test = test.merge(encode_popularity, how='left', on='popularity')
        test = test.merge(encode_source_type, how='left', on='source_type')

        # 加载特征顺序
        with open('predict/models/' + model_slug + '/feature_encode/feature_order.txt', 'rb') as fp:
            feature_name = pickle.load(fp)

        # 组合预测记录
        test['predict_hedge'] = np.NAN
        for i in range(0, len(test)):
            temp = pd.DataFrame()
            temp['use_time'] = pd.Series(list(range(1, 241))).values
            temp['source_type_encode'] = test.loc[i, 'source_type_encode']
            temp['model_detail_slug_encode'] = test.loc[i, 'model_detail_slug_encode']
            temp['popularity_encode'] = test.loc[i, 'popularity_encode']
            temp['city_encode'] = test.loc[i, 'city_encode']

            # 预测保值率
            temp = temp.loc[:, feature_name]
            self.predict_hedge = np.exp(self.valuate_model.predict(xgb.DMatrix(temp)))
            temp['predict_hedge'] = pd.Series(self.predict_hedge).values
            temp['predict_hedge'] = temp['predict_hedge'].map('{:,.4f}'.format)
            # 整合保值率
            values = list(temp.predict_hedge.values)
            test.loc[i, 'predict_hedge'] = str(values).replace("'", "")

        test = test.drop(['source_type_encode', 'model_detail_slug_encode', 'popularity_encode', 'city_encode'], axis=1)
        test_dealer = test.loc[(test['source_type'] == 'dealer'), ['model_detail_slug', 'city', 'popularity', 'predict_hedge']]
        test_dealer = test_dealer.rename(columns={'predict_hedge': 'dealer_hedge'})

        test_cpersonal = test.loc[(test['source_type'] == 'cpersonal'), ['model_detail_slug', 'city', 'popularity', 'predict_hedge']]
        test_cpersonal = test_cpersonal.rename(columns={'predict_hedge': 'cpersonal_hedge'})

        result = test_dealer.merge(test_cpersonal, how='left', on=['model_detail_slug', 'city', 'popularity'])
        result.to_csv('predict/models/' + model_slug + '/data/result.csv', index=False, encoding='utf-8')

    def predict_single(self, model_slug, model_detail_slug, city, use_time):
        """
        预测返回
        """
        # 加载特征顺序
        with open('predict/models/' + model_slug + '/feature_encode/feature_order.txt', 'rb') as fp:
            feature_name = pickle.load(fp)

        # 加载估值模型
        self.valuate_model = xgb.Booster()
        self.valuate_model.load_model('predict/models/'+model_slug+'/model/xgboost_level2.model')

        encode_city = pd.read_csv('predict/models/' + model_slug + '/feature_encode/city.csv')
        encode_model_detail_slug = pd.read_csv('predict/models/' + model_slug + '/feature_encode/model_detail_slug.csv')
        encode_popularity = pd.read_csv('predict/models/' + model_slug + '/feature_encode/popularity.csv')
        encode_source_type = pd.read_csv('predict/models/' + model_slug + '/feature_encode/source_type.csv')

        # 获取交易类型编码
        self.test_level1['source_type'] = pd.Series(gl.INTENT_TYPE_CAN)
        self.test_level1 = self.test_level1.merge(encode_source_type, how='left', on='source_type')

        self.test_level1['city'] = city
        self.test_level1 = self.test_level1.merge(encode_city, how='left', on='city')

        self.test_level1['model_detail_slug'] = model_detail_slug
        self.test_level1 = self.test_level1.merge(encode_model_detail_slug, how='left', on='model_detail_slug')

        province = province_city_map.loc[(province_city_map['city'] == city), 'province']
        if str(province + model_slug) in province_popularity_map.index:
            self.test_level1['popularity'] = province_popularity_map.loc[str(province + model_slug), 'popularity']
        else:
            self.test_level1['popularity'] = 'B'
        self.test_level1 = self.test_level1.merge(encode_popularity, how='left', on='popularity')

        self.test_level1['use_time'] = use_time

        pred = self.test_level1.loc[:, feature_name]
        self.predict_hedge = np.exp(self.valuate_model.predict(xgb.DMatrix(pred)))
        self.test_level1['predict_hedge'] = pd.Series(self.predict_hedge).values

        return self.test_level1

    def create_all_test_data(self):
        """
        创建所有的测试数据
        """
        # 加载可预测车型
        valuated_model = pd.read_csv('predict/map/valuated_model_detail.csv')
        models = list(set(valuated_model.model_slug.values))

        for i, model in enumerate(models):
            self.create_test_data(model_slug=model)
            print(i, '完成' + model + '车型训练数据的创建！')

    def predict_all_test_data(self):
        """
        创建所有的测试数据
        """
        # 加载需要预测车型
        valuated_model = pd.read_csv('predict/map/need_valuated_model_part1.csv')
        models = list(set(valuated_model.model_slug.values))
        models_bak = list(set(valuated_model.model_slug.values))

        for i, model in enumerate(models_bak):
            time1 = time.time()
            print(i, '开始' + model + '车型的预测！')
            self.predict_test_data(model_slug=model)
            time2 = time.time()
            print(i, '完成' + model + '车型的预测！,耗时:', time2 - time1)
            models.remove(model)
            new_model = pd.DataFrame(models, columns=['model_slug'])
            new_model.to_csv('predict/map/need_valuated_model_part1.csv', index=False, encoding='utf-8')

    def process_result_data(self, model):
        """
        处理预测后文件,生成供数据库使用文件
        """
        model_detail_map = pd.read_csv('predict/map/model_detail_map.csv')
        model_detail_map = model_detail_map.loc[:, ['model_detail_slug_id', 'model_detail_slug']]
        province_city_map = pd.read_csv('predict/map/province_city_map.csv')
        province_city_map = province_city_map.loc[:, ['city_id', 'city']]

        time1 = time.time()
        print(j, '开始' + model + '车型的数据处理！')

        test = pd.read_csv('predict/models/' + model + '/data/result.csv')
        test['dealer_hedge'] = test.apply(process_str, args=('dealer_hedge',), axis=1)
        test['cpersonal_hedge'] = test.apply(process_str, args=('cpersonal_hedge',), axis=1)
        test = test.merge(model_detail_map, how='left', on='model_detail_slug')
        test = test.merge(province_city_map, how='left', on='city')

        for i in range(0, 40):
            test['b2c_year_' + str(i)] = test.apply(get_year, args=('dealer_hedge', i * 18, (i + 1) * 18,), axis=1)
            test['c2c_year_' + str(i)] = test.apply(get_year, args=('cpersonal_hedge', i * 18, (i + 1) * 18,), axis=1)
        test = test.drop(['model_detail_slug', 'city', 'dealer_hedge', 'cpersonal_hedge'], axis=1)
        test.to_csv('predict/models/' + model + '/data/final.csv', index=False)

        time2 = time.time()
        print(j, '完成' + model + '车型的数据处理！,耗时:', time2 - time1)

    def process_all_result_data(self):
        """
        处理所有预测后文件,生成供数据库使用文件
        """
        model_detail_map = pd.read_csv('predict/map/model_detail_map.csv')
        model_detail_map = model_detail_map.loc[:, ['model_detail_slug_id', 'model_detail_slug']]
        province_city_map = pd.read_csv('predict/map/province_city_map.csv')
        province_city_map = province_city_map.loc[:, ['city_id', 'city']]

        # 加载需要处理车型
        valuated_model = pd.read_csv('predict/map/need_process_models.csv')
        models = list(set(valuated_model.model_slug.values))
        models_bak = list(set(valuated_model.model_slug.values))

        for j, model in enumerate(models_bak):
            file_name = 'predict/models/' + model + '/data/result.csv'
            if os.path.exists(file_name):
                time1 = time.time()
                print(j, '开始' + model + '车型的数据处理！')

                test = pd.read_csv('predict/models/' + model + '/data/result.csv')
                test['dealer_hedge'] = test.apply(process_str, args=('dealer_hedge',), axis=1)
                test['cpersonal_hedge'] = test.apply(process_str, args=('cpersonal_hedge',), axis=1)
                test = test.merge(model_detail_map, how='left', on='model_detail_slug')
                test = test.merge(province_city_map, how='left', on='city')

                for i in range(0, 40):
                    test['b2c_year_' + str(i)] = test.apply(get_year, args=('dealer_hedge', i * 18, (i + 1) * 18,), axis=1)
                    test['c2c_year_' + str(i)] = test.apply(get_year, args=('cpersonal_hedge', i * 18, (i + 1) * 18,),
                                                            axis=1)
                test = test.drop(['model_detail_slug', 'city', 'dealer_hedge', 'cpersonal_hedge'], axis=1)
                test.to_csv('predict/models/' + model + '/data/final.csv', index=False)

                time2 = time.time()
                print(j, '完成' + model + '车型的数据处理！,耗时:', time2 - time1)
                models.remove(model)
                new_model = pd.DataFrame(models, columns=['model_slug'])
                new_model.to_csv('predict/map/need_process_models.csv', index=False, encoding='utf-8')

    def combine_all_result_data(self):
        """
        组合所有预测值
        """
        part1_feature = ['model_detail_slug_id', 'city_id', 'popularity']
        part2_feature = ['model_detail_slug_id', 'city_id', 'popularity']
        for i in range(0, 20):
            part1_feature.append('b2c_year_' + str(i))
            part1_feature.append('c2c_year_' + str(i))

        for i in range(20, 40):
            part2_feature.append('b2c_year_' + str(i))
            part2_feature.append('c2c_year_' + str(i))
        # 加载可预测车型
        valuated_model = pd.read_csv('predict/map/valuated_model_detail.csv')
        models = list(set(valuated_model.model_slug.values))

        result = pd.DataFrame()
        for i, model in enumerate(models):
            time1 = time.time()
            file_name = 'predict/models/' + model + '/data/final.csv'
            # if os.path.exists(file_name):
            temp = pd.read_csv(file_name, dtype=str, encoding='utf-8')
            result = result.append(temp)
            time2 = time.time()
            print(i, '耗时:', model, time2-time1)
            if ((i % 100) == 0) & (i != 0):
                num = int(i / 100)
                part1 = result.loc[:, part1_feature]
                part1.to_csv('predict/model/'+str(num)+'valuate_part1.csv', index=False, encoding='utf-8')
                part2 = result.loc[:, part2_feature]
                part2.to_csv('predict/model/' + str(num) + 'valuate_part2.csv', index=False, encoding='utf-8')
                result = pd.DataFrame()
            if i == (len(models) - 1):
                num = int(i / 100) + 1
                part1 = result.loc[:, part1_feature]
                part1.to_csv('predict/model/' + str(num) + 'valuate_part1.csv', index=False, encoding='utf-8')
                part2 = result.loc[:, part2_feature]
                part2.to_csv('predict/model/' + str(num) + 'valuate_part2.csv', index=False, encoding='utf-8')




