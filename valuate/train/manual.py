from valuate.report.report import Report
from valuate.predict.predict_batch import Predict
from datetime import datetime

import pandas as pd


def generate_deal_days(df):
    """
    计算成交表使用时间
    """
    return (datetime.strptime(df['deal_date'], '%Y-%m-%d') - datetime.strptime(df['reg_date'], '%Y-%m-%d')).days


def generate_months(df):
    """
    生成年份
    """
    online_year = df['year']
    online_month = df['month']
    if str(df['sold_time']) == 'nan':
        transaction_time = datetime.strptime(str(df['expired_at']), "%Y-%m-%d %H:%M:%S")
        return (transaction_time.year - online_year)*12 + transaction_time.month - online_month
    else:
        transaction_time = datetime.strptime(str(df['sold_time']), "%Y-%m-%d %H:%M:%S")
        return (transaction_time.year - online_year)*12 + transaction_time.month - online_month


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
                price_bn = df['price_bn'] * 0.55
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

    def generate_adjust_profit_map(self):
        """
        生成供调整值用的映射表
        """
        # train = pd.read_csv('../tmp/train/train_source.csv')
        # adjust_data = pd.read_csv('../tmp/train/adjust_data.csv')
        # adjust_data = adjust_data.merge(train, how='left', on='id')
        # # 生成使用时间
        # adjust_data['use_time'] = adjust_data.apply(generate_months, axis=1)
        # adjust_data = adjust_data.loc[:, ['model_detail_slug', 'mile', 'use_time', 'city', 'price', 'source_type', 'popularity']]
        # adjust_data['price'] = adjust_data['price']*10000
        # 生成最近的调整数据
        # verify = Report()
        # verify.generate_adjust_profit_data()
        # 组合数据
        # adjust_data_recently = pd.read_csv('../tmp/train/adjust_data_recently.csv')
        # adjust_data = adjust_data.append(adjust_data_recently)
        # adjust_data.to_csv('../tmp/train/man.csv', index=False)
        # 预测数据
        # adjust_data = pd.read_csv('../tmp/train/man.csv')
        # predict = Predict()
        # result = predict.predict_batch(adjust_data)
        # result = result.loc[:, ['model_detail_slug', 'mile', 'use_time', 'city', 'price', 'source_type', 'popularity', 'predict_price']]
        # result.to_csv('../tmp/train/man1.csv', index=False)
        adjust_data = pd.read_csv('../tmp/train/man1.csv')
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
        self.generate_adjust_profit_map()
        # self.generate_others_predict_relate_tables()
