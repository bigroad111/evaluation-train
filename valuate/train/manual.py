from valuate.report.report import Report
from valuate.predict.predict_batch import Predict
from datetime import datetime

import pandas as pd
from valuate.conf import global_settings as gl


def generate_deal_days(df):
    """
    计算成交表使用时间
    """
    return (datetime.strptime(df['deal_date'], '%Y-%m-%d') - datetime.strptime(df['reg_date'], '%Y-%m-%d')).days


def generate_months(df):
    """
    计算成交表使用时间
    """
    if str(df['sold_time']) != 'nan':
        deal_date = datetime.strptime(str(df['sold_time']), '%Y-%m-%d %H:%M:%S')
    elif str(df['expired_at']) != 'nan':
        deal_date = datetime.strptime(str(df['expired_at']), '%Y-%m-%d %H:%M:%S')
    else:
        return -1
    return (deal_date.year - df['year']) * 12 + deal_date.month - df['month']


class Manual(object):
    """
    组合模型训练
    """
    def __init__(self):
        self.model_detail_map = []
        self.province_city_map = []
        self.open_province_popularity = []
        self.current_city_encode = []
        self.deal_records = []
        self.domain_category = []

    def generate_price_bn_tune_map(self):
        """
        生成新车指导价调整表,高配调整到中低配
        """
        # 加载所有款型
        open_model_detail = pd.read_csv('../tmp/train/open_model_detail.csv')
        open_model_detail = open_model_detail[open_model_detail['price_bn'] > 0]
        open_model_detail = open_model_detail.sort_values(by=['global_slug', 'year', 'volume', 'control', 'price_bn'],
                                                          ascending=False)
        open_model_detail.reset_index(inplace=True)
        open_model_detail = open_model_detail.drop('index', axis=1)

        detail_price_map = open_model_detail.groupby(['global_slug', 'year', 'volume', 'control'])['price_bn'].median().reset_index()
        detail_price_map = detail_price_map.rename(columns={'price_bn': 'median_price_bn'})

        final = open_model_detail.merge(detail_price_map, how='left', on=['global_slug', 'year', 'volume', 'control'])
        # 加载没有在训练数据中的款型
        not_in_train_data_details = pd.read_csv('predict/map/not_in_train_data_details.csv')
        not_in_train_data_details = list(set(not_in_train_data_details.model_detail_slug.values))

        def tune_price_bn(df):
            if df['model_detail_slug'] in not_in_train_data_details:
                # temp = open_model_detail.loc[((open_model_detail['global_slug'] == df['global_slug']) & (open_model_detail['year'] == df['year']) & (open_model_detail['volume'] == df['volume']) & (open_model_detail['control'] == df['control']) & (open_model_detail['price_bn'] < df['median_price_bn'])), :]
                # temp.reset_index(inplace=True)
                # if len(temp) == 0:
                print(df['model_detail_slug'], df['global_slug'])
                model_slug = 'zhongtaiZ700'
                price_bn = df['price_bn']
                model_detail_slug = '127445_autotis'
                model_detail_slug_id = 277875
                # else:
                #     model_slug = temp.loc[0, 'global_slug']
                #     price_bn = temp.loc[0, 'price_bn']
                #     model_detail_slug = temp.loc[0, 'model_detail_slug']
                #     model_detail_slug_id = temp.loc[0, 'id']
            else:
                model_slug = df['global_slug']
                price_bn = df['price_bn']
                model_detail_slug = df['model_detail_slug']
                model_detail_slug_id = df['id']
            return pd.Series([model_slug, price_bn, model_detail_slug, model_detail_slug_id])

        final[['final_model_slug', 'final_price_bn', 'final_model_detail_slug', 'final_model_detail_slug_id']] = final.apply(tune_price_bn, axis=1)
        final = final.rename(columns={'global_slug': 'model_slug'})
        final.to_csv('predict/map/model_detail_map.csv', index=False, encoding='utf-8')

    def generate_others_predict_relate_tables(self):
        """
        生成预测需要相关表
        """
        # 生成城市省份匹配表
        open_city = pd.read_csv('../tmp/train/open_city.csv')
        province = open_city[open_city['parent'] == 0]
        province = province.drop('parent', axis=1)
        province = province.rename(columns={'id': 'parent', 'name': 'province'})
        city = open_city[open_city['parent'] != 0]
        city = city.rename(columns={'id': 'city_id', 'name': 'city'})
        self.province_city_map = city.merge(province, how='left', on='parent')
        self.province_city_map = self.province_city_map.loc[:, ['province', 'city', 'city_id']]
        self.province_city_map.to_csv('predict/map/province_city_map.csv', index=False, encoding='utf-8')

        # 生成省份车型流行度匹配表
        self.open_province_popularity = pd.read_csv('../tmp/train/open_province_popularity.csv')
        self.open_province_popularity = self.open_province_popularity.loc[:, ['province', 'model_slug', 'popularity']]
        self.open_province_popularity.to_csv('predict/map/province_popularity_map.csv', index=False, encoding='utf-8')

    def generate_domain_category(self):
        train = pd.read_csv('../tmp/train/train.csv')
        self.domain_category = train.loc[:, ['domain', 'source_type']]
        self.domain_category = self.domain_category.drop_duplicates(['domain', 'source_type'])
        self.domain_category = self.domain_category.sort_values(by='source_type')
        self.domain_category.to_csv('predict/map/domain_category_map.csv', index=False)

    def get_models_not_in_train_data(self):
        """
        获取所有没有训练数据的车型
        """
        open_model_detail = pd.read_csv('../tmp/train/open_model_detail.csv')
        all_details = list(set(open_model_detail.model_detail_slug.values))

        valuated_model = pd.read_csv('predict/map/valuated_model_detail.csv')
        models = list(set(valuated_model.model_slug.values))

        all_model_details = pd.DataFrame()
        for model in models:
            model_details = pd.read_csv('predict/models/' + model + '/feature_encode/model_detail_slug.csv')
            temp = pd.DataFrame(model_details, columns=['model_detail_slug'])
            all_model_details = all_model_details.append(temp)

        in_train_data_details = list(set(all_model_details.model_detail_slug.values))
        ret = list(set(all_details) ^ set(in_train_data_details))
        not_in_train_data_details = pd.DataFrame(ret, columns=['model_detail_slug'])
        not_in_train_data_details.to_csv('predict/map/not_in_train_data_details.csv', index=False)

    def generate_adjust_data(self):
        """
        生成调整数据
        """
        train = pd.read_csv('../tmp/train/train_source.csv')
        final = train[(train['status'] == 'review') & (train['sold_time'].notnull()) & (train['source_type'].isin(gl.CAR_SOURCE_SOURCE_TYPE_VALUES))]
        final.reset_index(inplace=True)
        final = final.drop('index', axis=1)
        # 匹配车型
        model_detail_map = pd.read_csv('predict/map/model_detail_map.csv')
        model_detail_map = model_detail_map.loc[:, ['model_slug', 'model_detail_slug']]
        # 匹配流行度
        province_city_map = pd.read_csv('predict/map/province_city_map.csv')
        province_city_map = province_city_map.loc[:, ['province', 'city']]
        province_popularity_map = pd.read_csv('predict/map/province_popularity_map.csv')
        final = final.merge(model_detail_map, how='left', on='model_detail_slug')
        final = final.merge(province_city_map, how='left', on='city')
        final = final.merge(province_popularity_map, how='left', on=['model_slug', 'province'])
        final['popularity'] = final['popularity'].fillna('C')
        # 删除数据不完整记录和use_time异常值
        final = final[(final['model_slug'].notnull()) & (final['price'].notnull()) & (final['price'] > 0)]
        final.reset_index(inplace=True)
        final = final.drop('index', axis=1)
        # 生成款型最近的5条记录
        final['sold_time'] = pd.to_datetime(final['sold_time'])
        result = pd.DataFrame()
        for i in range(0, 5):
            temp = final.loc[final.groupby(['model_detail_slug', 'popularity']).sold_time.idxmax(), :]
            final = final.drop(temp.index, axis=0)
            final.reset_index(inplace=True)
            final = final.drop('index', axis=1)
            result = result.append(temp)
            print('完成轮次:', i)
        result.to_csv('../tmp/train/adjust_data.csv', index=False)

    def generate_adjust_profit_map(self):
        """
        生成供调整值用的映射表
        """
        # adjust_data = pd.read_csv('../tmp/train/adjust_data.csv')
        # # 生成使用时间
        # adjust_data['use_time'] = adjust_data.apply(generate_months, axis=1)
        # adjust_data['source_type'] = adjust_data['source_type'].map(gl.INTENT_MAP)
        # # 删除数据不完整记录和use_time异常值
        # adjust_data.loc[(adjust_data['use_time'] <= 0), 'use_time'] = 1
        # adjust_data.loc[(adjust_data['use_time'] > 240), 'use_time'] = 240
        # # 重置索引
        # adjust_data.reset_index(inplace=True)
        # adjust_data = adjust_data.drop('index', axis=1)
        #
        # # 预测数据
        # feature = gl.PREDICT_FEATURE
        # feature.append('price')
        # adjust_data = adjust_data.loc[:, feature]
        # predict = Predict()
        # result = predict.predict_batch(adjust_data, is_update_process=True)
        # result = result.loc[:, ['model_detail_slug', 'mile', 'use_time', 'city', 'source_type', 'popularity', 'price', 'predict_price']]
        # result.to_csv('../tmp/train/man.csv', index=False)
        adjust_data = pd.read_csv('../tmp/train/man.csv')
        adjust_data['price'] = adjust_data['price'] * 10000
        adjust_data['rate'] = (adjust_data['price'] - adjust_data['predict_price']) / adjust_data['predict_price']
        adjust_data = adjust_data[abs(adjust_data['rate']) <= 0.3]
        adjust_data.reset_index(inplace=True)
        adjust_data = adjust_data.drop('index', axis=1)
        adjust_profit_map = adjust_data.groupby(['model_detail_slug', 'popularity'])['rate'].median().reset_index()
        adjust_profit_map.to_csv('predict/map/adjust_profit_map.csv', index=False)

    def execute(self):
        """
        执行人工处理后续流程
        """
        # self.get_models_not_in_train_data()
        # self.generate_price_bn_tune_map()
        # self.generate_adjust_data()
        self.generate_adjust_profit_map()
        # self.generate_others_predict_relate_tables()
