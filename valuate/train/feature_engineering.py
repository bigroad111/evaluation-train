import os
import random
from datetime import datetime

import numpy as np
import pandas as pd

from valuate.conf import global_settings as gl
from valuate.db import process_tables


def query_train_data():
    """
    查询训练源数据
    """
    process_tables.store_train_data()


def generate_deal_days(df):
    """
    计算成交表使用时间
    """
    return (datetime.strptime(df['deal_date'], '%Y-%m-%d') - datetime.strptime(df['reg_date'], '%Y-%m-%d')).days


def generate_deal_months(df):
    """
    计算成交表使用时间
    """
    deal_date = datetime.strptime(df['deal_date'], '%Y-%m-%d')
    reg_date = datetime.strptime(df['reg_date'], '%Y-%m-%d')
    return (deal_date.year - reg_date.year)*12 + deal_date.month - reg_date.month


def generate_days(df):
    """
    生成年份
    """
    online_time = str(df['year'])+'-'+str(df['month'])+'-'+'1'
    online_time = datetime.strptime(online_time, '%Y-%m-%d')
    if str(df['sold_time']) == 'nan':
        transaction_time = datetime.strptime(str(df['expired_at']), "%Y-%m-%d %H:%M:%S")
        return (transaction_time - online_time).days
    else:
        transaction_time = datetime.strptime(str(df['sold_time']), "%Y-%m-%d %H:%M:%S")
        return (transaction_time - online_time).days


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


def c2b_get_month(df):
    """
    获取月份
    """
    transaction_time = datetime.strptime(str(df['deal_date']), "%Y-%m-%d")
    return transaction_time.month


def calculate_mile(df):
    """
    计算公里数，平均每年2万公里
    """
    if df['use_time'] < 1:
        return 0.00
    else:
        # return float('%.2f' % ((df['use_time'] * 2)*(random.uniform(0.95, 1.05)) / 12))
        return float('%.2f' % ((df['use_time'] * 2) / 12))


def calculate_price(df):
    """
    计算标价
    """
    return float('%.2f' % (df['depreciation_rate'] * df['price_bn'] * random.uniform(0.99, 1.00))) + 0.1 - 0.1 * random.uniform(0.99, 1.00)


class FeatureEngineering(object):

    def __init__(self):
        # 查询训练数据
        query_train_data()
        # 加载各类相关表
        self.history_train = pd.read_csv('../tmp/train/history_train_source.csv')
        self.train = pd.read_csv('../tmp/train/train_source.csv')
        self.train = self.train.append(self.history_train, ignore_index=True)
        self.deal_records = pd.read_csv('../tmp/report/deal_records.csv')
        self.open_city = pd.read_csv('../tmp/train/open_city.csv')
        self.open_model_detail = pd.read_csv('../tmp/train/open_model_detail.csv')
        self.open_category = pd.read_csv('../tmp/train/open_category.csv')
        self.open_depreciation = pd.read_csv('../tmp/train/open_depreciation.csv')
        self.province_popular = pd.read_csv('../tmp/train/open_province_popularity.csv')
        self.open_category = self.open_category.rename(columns={'slug': 'global_slug', 'name': 'global_name'})
        self.open_model_detail = self.open_model_detail.rename(columns={'detail_model_slug': 'model_detail_slug'})
        self.province_popular = self.province_popular.loc[:, ['province', 'model_slug', 'popularity']]

        self.no_models = []
        self.have_models = []
        self.all_models = []

    def base_cleaning(self):
        """
        数据常规预处理
        """
        # 删掉price<0的记录
        self.train = self.train[self.train['price'] > 0]
        # 只保留正规车商的记录
        self.train = self.train[self.train['source_type'].isin(['cpersonal', 'cpo', 'dealer', 'odealer', 'personal'])]
        self.train = self.train[self.train['sold_time'].notnull()]
        self.train = self.train[self.train['model_detail_slug'].notnull()]
        self.train = self.train[self.train['status'] == 'review']
        # 删除掉未知城市,款型和98年之前的记录,
        open_city = self.open_city[self.open_city['parent'] != 0]
        cities = list(set(open_city.name.values))
        models = list(set(self.open_model_detail.model_detail_slug.values))
        self.train = self.train[self.train['city'].isin(cities)]
        self.train = self.train[self.train['model_detail_slug'].isin(models)]
        self.train = self.train[self.train['year'] > 1997]
        # 删掉没有处理时间的记录和时间记录异常值
        self.train = self.train[self.train['month'].isin(np.arange(1, 13))]
        self.train.reset_index(inplace=True)
        self.train = self.train.drop('index', axis=1)
        # 计算使用月数,删除使用时间小于0和14年之前的记录
        self.train['use_time'] = self.train.apply(generate_months, axis=1)
        self.train = self.train.merge(self.open_model_detail, on='model_detail_slug', how='left')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 0) | (self.train['sold_time'] < '2014-01-01')].index)
        self.train = self.train.drop(['year', 'month', 'expired_at', 'sold_time', 'status', 'volume'], axis=1)
        self.train.reset_index(inplace=True)
        self.train = self.train.drop('index', axis=1)

    def find_no_models(self):
        """
        查找训练数据记录小于5或没有的款型
        """
        # 获取数据量<5的款型
        less_data_search = self.train.groupby(['model_detail_slug']).count()
        less_data_search = less_data_search.loc[(less_data_search['id'] < 5), :]
        self.no_models = less_data_search.index.tolist()

        # 获取训练数据没有的款型
        self.have_models = set(self.train.model_detail_slug.values)
        self.all_models = set(self.open_model_detail.model_detail_slug.values)
        for model_detail in self.all_models:
            if model_detail not in self.have_models:
                self.no_models.append(model_detail)

    def process_open_depreciation(self):
        """
        处理款系衰减表
        """
        self.open_depreciation = self.open_depreciation[self.open_depreciation['type'] == '一般']
        self.open_depreciation = self.open_depreciation.sort_values(by=['attribute', 'period'])
        self.open_depreciation = self.open_depreciation.rename(columns={'period': 'use_time'})
        self.open_depreciation = self.open_depreciation.loc[:, ['attribute', 'use_time', 'depreciation_rate']]
        # 用月计算
        self.open_depreciation['use_time'] = 12 * self.open_depreciation['use_time']
        self.open_depreciation.reset_index(inplace=True)
        self.open_depreciation = self.open_depreciation.drop('index', axis=1)

    def generate_train_data(self):
        """
        生成训练用数据
        """
        # 对训练数据没有的款型生成人造数据
        artificial_data = pd.DataFrame(self.no_models, columns=['model_detail_slug'])
        artificial_data = artificial_data.merge(self.open_model_detail, on='model_detail_slug', how='left')
        artificial_data = artificial_data.merge(self.open_category, on='global_slug', how='left')
        artificial_data['id'] = 1
        artificial_data['source_type'] = 'dealer'

        artificial_data = artificial_data.merge(self.open_depreciation, how='left', on='attribute')
        artificial_data = artificial_data.sort_values(by=['model_detail_slug', 'attribute', 'use_time'])
        artificial_data['mile'] = artificial_data.apply(calculate_mile, axis=1)
        artificial_data['price'] = artificial_data.apply(calculate_price, axis=1)
        artificial_data['domain'] = 'artificial'

        artificial_data = artificial_data.drop(['depreciation_rate', 'global_name', 'attribute', 'parent', 'volume'], axis=1)
        artificial_data.to_csv('../tmp/train/artificial_data.csv', index=False)
        # artificial_data['transfer_owner'] = 0

        # 将真实数据与人造数据组合存储
        self.train = self.train.append(artificial_data, ignore_index=True)

        # city为空的填充深圳
        self.train['city'] = self.train['city'].fillna('深圳')

        # 标价单位是万元,改成元
        self.train['price'] = self.train['price'] * 10000

    def add_c2b_data(self):
        self.deal_records = self.deal_records.loc[:, ['id', 'model_detail_slug', 'city', 'mile', 'price', 'deal_date', 'reg_date', 'deal_type', 'source']]
        open_model_detail = self.open_model_detail.loc[:, ['price_bn', 'model_detail_slug', 'global_slug']]

        self.deal_records = self.deal_records.merge(open_model_detail, how='left', on='model_detail_slug')
        self.deal_records['use_time'] = self.deal_records.apply(generate_deal_months, axis=1)
        self.deal_records['source_type'] = self.deal_records['deal_type'].map(gl.DEAL_TYPE_MAP_SOURCE_TYPE)
        self.deal_records = self.deal_records.drop(['deal_date', 'reg_date', 'deal_type'], axis=1)

        self.deal_records['mile'] = self.deal_records['mile'] / 10000
        self.deal_records = self.deal_records[self.deal_records['source_type'] == 'sell_dealer']
        self.deal_records.reset_index(inplace=True)
        self.deal_records = self.deal_records.drop('index', axis=1)
        self.deal_records = self.deal_records.rename(columns={'source': 'domain'})

        # 存储
        self.train = self.train.append(self.deal_records, ignore_index=True)
        # 删掉老旧车型
        self.train = self.train.drop(self.train[(self.train['use_time'].isnull()) | (self.train['price'].isnull()) | (self.train['price_bn'].isnull()) | (self.train['price_bn'] <= 0) | (self.train['price'] <= 0) | (self.train['use_time'] < 0)].index)
        self.train.loc[(self.train['use_time'] == 0), 'use_time'] = 1

        self.train['price_bn'] = self.train['price_bn'] * 10000
        self.train['price'] = self.train['price'].astype(int)
        self.train['price_bn'] = self.train['price_bn'].astype(int)
        self.train['use_time'] = self.train['use_time'].astype(int)

    def add_other_process(self):
        """
        其他处理
        """
        # 生成每月公里数和保值率
        self.train['mile_per_month'] = self.train['mile'] / self.train['use_time']
        self.train['hedge_rate'] = self.train['price'] / self.train['price_bn']
        # 删除超过20年车龄和每年公里数超过20万,以及保值率小于0的记录
        self.train = self.train.drop(self.train[(self.train['mile_per_month'] > 1.66) | (self.train['use_time'] > 240) | (self.train['hedge_rate'] < 0)].index)
        self.train.reset_index(inplace=True)
        self.train = self.train.drop('index', axis=1)
        # 将odealer和personal替换成dealer和cpersonal,用于增加数据量
        self.train.loc[(self.train['source_type'].isin(['odealer'])), 'source_type'] = 'dealer'
        self.train.loc[(self.train['source_type'].isin(['personal'])), 'source_type'] = 'cpersonal'
        # 匹配品牌,车型,畅销度
        self.open_category = pd.read_csv('../tmp/train/open_category.csv')
        self.open_category = self.open_category.loc[:, ['slug', 'parent']]
        self.open_category = self.open_category[self.open_category['parent'].notnull()]
        self.open_category = self.open_category.rename(columns={'slug': 'global_slug', 'parent': 'brand'})
        self.open_category.reset_index(inplace=True)
        self.open_category = self.open_category.drop('index', axis=1)
        self.train = self.train.merge(self.open_category, how='left', on='global_slug')

        province = self.open_city[self.open_city['parent'] == 0]
        province = province.rename(columns={'name': 'province'})
        province = province.drop('parent', axis=1)
        city = self.open_city[self.open_city['parent'] != 0]
        city = city.drop('id', axis=1)
        city = city.rename(columns={'name': 'city', 'parent': 'id'})
        province_city = city.merge(province, how='left', on='id')
        province_city.reset_index(inplace=True)
        province_city = province_city.drop(['index', 'id'], axis=1)
        self.train = self.train.merge(province_city, how='left', on='city')
        self.train = self.train.rename(columns={'global_slug': 'model_slug'})
        self.train = self.train.merge(self.province_popular, how='left', on=['province', 'model_slug'])
        self.train['popularity'] = self.train['popularity'].fillna('C')
        self.train = self.train[self.train['province'].notnull()]
        self.train = self.train.drop(['domain', 'id', 'province'], axis=1)
        self.train.to_csv('../tmp/train/train.csv', index=False)

    def add_other_process_step2(self):
        self.train = pd.read_csv('../tmp/train/train.csv_bak')
        # 剔除保值率明显异常的数据
        self.train = self.train[self.train['hedge_rate'] < 1]
        self.train.reset_index(inplace=True, drop='index')
        # 根据年限剔除明显异常的数据
        self.train = self.train.drop(self.train[(self.train['use_time'] > 12) & (self.train['hedge_rate'] > 0.965)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 24) & (self.train['hedge_rate'] > 0.89)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 36) & (self.train['hedge_rate'] > 0.80)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 48) & (self.train['hedge_rate'] > 0.70)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 60) & (self.train['hedge_rate'] > 0.62)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 72) & (self.train['hedge_rate'] > 0.55)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 84) & (self.train['hedge_rate'] > 0.44)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 96) & (self.train['hedge_rate'] > 0.35)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 108) & (self.train['hedge_rate'] > 0.27)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 120) & (self.train['hedge_rate'] > 0.20)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 132) & (self.train['hedge_rate'] > 0.17)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 144) & (self.train['hedge_rate'] > 0.15)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 156) & (self.train['hedge_rate'] > 0.13)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 168) & (self.train['hedge_rate'] > 0.115)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 180) & (self.train['hedge_rate'] > 0.10)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 192) & (self.train['hedge_rate'] > 0.09)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 204) & (self.train['hedge_rate'] > 0.08)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 216) & (self.train['hedge_rate'] > 0.07)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 228) & (self.train['hedge_rate'] > 0.065)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] > 240) & (self.train['hedge_rate'] > 0.06)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train.to_csv('../tmp/train/train.csv', index=False)

        self.train = pd.read_csv('../tmp/train/train.csv')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 12) & (self.train['hedge_rate'] < 0.77)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 24) & (self.train['hedge_rate'] < 0.65)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 36) & (self.train['hedge_rate'] < 0.45)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 48) & (self.train['hedge_rate'] < 0.35)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 60) & (self.train['hedge_rate'] < 0.30)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 72) & (self.train['hedge_rate'] < 0.25)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 84) & (self.train['hedge_rate'] < 0.15)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 96) & (self.train['hedge_rate'] < 0.08)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 108) & (self.train['hedge_rate'] < 0.05)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 120) & (self.train['hedge_rate'] < 0.04)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 132) & (self.train['hedge_rate'] < 0.04)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 144) & (self.train['hedge_rate'] < 0.04)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 156) & (self.train['hedge_rate'] < 0.03)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 168) & (self.train['hedge_rate'] < 0.03)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 180) & (self.train['hedge_rate'] < 0.03)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 192) & (self.train['hedge_rate'] < 0.03)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 204) & (self.train['hedge_rate'] < 0.02)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 216) & (self.train['hedge_rate'] < 0.02)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 228) & (self.train['hedge_rate'] < 0.02)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train = self.train.drop(self.train[(self.train['use_time'] < 240) & (self.train['hedge_rate'] < 0.02)].index)
        self.train.reset_index(inplace=True, drop='index')
        self.train.to_csv('../tmp/train/train.csv', index=False)

    def split_models(self):
        """
        依据车型分割数据
        """
        self.train = pd.read_csv('../tmp/train/train.csv')
        open_category = pd.read_csv('../tmp/train/open_category.csv')
        open_category = open_category[open_category['parent'].notnull()]
        normal_models = []
        for model in list(set(open_category.slug.values)):
            file_name = 'predict/models/'+model+'/data/train.csv'
            temp = self.train.loc[(self.train['model_slug'] == model), :]
            if len(temp) == 0:
                print(model, 'model has no data!')
                return False
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            temp.to_csv(file_name, index=False, encoding='utf-8')
            normal_models.append(model)

        normal_models_df = pd.DataFrame(normal_models, columns=['model_slug'])
        normal_models_df.to_csv('predict/map/valuated_model_detail.csv', index=False, encoding='utf-8')

    def execute(self):
        # self.base_cleaning()
        # self.find_no_models()
        # self.process_open_depreciation()
        # self.generate_train_data()
        # self.add_c2b_data()
        # self.add_other_process()
        # self.add_other_process_step2()
        # self.split_models()
        pass

# 1.公里数对车辆价格的影响。
# 1. 正常行驶的车辆以一年2.5万公里为正常基数，低于2.5万公里的价格的浮动在+3.5%以内  大于2.5万公里的若每年的平均行驶里程大于2.5万公里小于5万公里价格浮动在-3.5-7.5%  若年平均形式里程大于5万公里及以上影响价格在-7.5-12.5%之间。
#
# 2. 二手车逐年车辆保值率
# 1. 第1年的车辆贬值：3.5%-23% 96.5-77
# 2. 第2年的车辆贬值：11%-35% 89-65
# 3. 第3年的车辆贬值：20%-55% 80-45
# 4. 第4年的车辆贬值：30%-65% 70-35
# 5. 第5年的车辆贬值：38%-70% 62-30
# 6. 第6年的车辆贬值：45%-75% 55-25
# 7. 第7年的车辆贬值：56%-85% 44-15
# 8. 第8年的车辆贬值：65%-92% 35-8
# 9. 第9年的车辆贬值：73%-95% 27-5
# 10. 第10年的车辆贬值：80%-96% 20-4
# 11. 第11年的车辆贬值：83%-96% 17-4
# 12. 第12年的车辆贬值：85%-96% 15-4
# 13. 第13年的车辆贬值：87%-97% 13-3
# 14. 第14年的车辆贬值：88.5%-97% 11.5-3
# 15. 第15年的车辆贬值：90%-97% 10-3
# 16. 第16年的车辆贬值：91%-97% 9-3
# 17. 第17年的车辆贬值：92%-98% 8-2
# 18. 第18年的车辆贬值：93%-98% 7-2
# 19. 第19年的车辆贬值：93.5%-98% 6.5-2
# 20. 第20年的车辆贬值：94%-98% 6-2