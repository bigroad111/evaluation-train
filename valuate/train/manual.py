import pickle
import pandas as pd
import numpy as np

from datetime import datetime
from valuate.conf import global_settings as gl
from valuate.predict.predict_batch import Predict as batch


def generate_deal_days(df):
    """
    计算成交表使用时间
    """
    return (datetime.strptime(df['deal_date'], '%Y-%m-%d') - datetime.strptime(df['reg_date'], '%Y-%m-%d')).days


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

    def generate_others_predict_relate_tables(self):
        """
        生成预测需要相关表
        """
        # 生成款型车型匹配表
        open_category = pd.read_csv('../tmp/train/open_category.csv')
        open_category = open_category.loc[:, ['slug', 'parent']]
        open_category = open_category.rename(columns={'slug': 'global_slug', 'parent': 'brand'})
        self.model_detail_map = pd.read_csv('../tmp/train/open_model_detail.csv')
        self.model_detail_map = self.model_detail_map[self.model_detail_map['price_bn'] > 0]
        self.model_detail_map = self.model_detail_map.merge(open_category, how='left', on='global_slug')
        self.model_detail_map = self.model_detail_map.rename(columns={'detail_model_slug': 'model_detail_slug', 'global_slug': 'model_slug'})
        self.model_detail_map.to_csv('predict/map/model_detail_map.csv', index=False, encoding='utf-8')

        # 生成城市省份匹配表
        open_city = pd.read_csv('../tmp/train/open_city.csv')
        province = open_city[open_city['parent'] == 0]
        province = province.drop('parent', axis=1)
        province = province.rename(columns={'id': 'parent', 'name': 'province'})
        city = open_city[open_city['parent'] != 0]
        city = city.rename(columns={'name': 'city'})
        self.province_city_map = city.merge(province, how='left', on='parent')
        self.province_city_map = self.province_city_map.loc[:, ['province', 'city']]
        self.province_city_map.to_csv('predict/map/province_city_map.csv', index=False, encoding='utf-8')

        # 生成省份车型流行度匹配表
        self.open_province_popularity = pd.read_csv('../tmp/train/open_province_popularity.csv')
        self.open_province_popularity = self.open_province_popularity.loc[:, ['province', 'model_slug', 'popularity']]
        self.open_province_popularity.to_csv('predict/map/province_popularity_map.csv', index=False, encoding='utf-8')

    def fill_all_city(self):
        """
        补缺城市到城市编码表
        """
        self.current_city_encode = pd.read_csv('predict/feature_encode/city.csv')
        open_city = pd.read_csv('../tmp/train/open_city.csv')
        open_city = open_city[open_city['parent'] != 0]

        citys = self.current_city_encode.city.values
        lack_citys = open_city.loc[~(open_city['name'].isin(citys)), :]
        lack_citys = lack_citys.name.values
        lack_citys_encode = self.current_city_encode.loc[(self.current_city_encode['city'] == '深圳'), 'city_encode'].values[0]

        if len(lack_citys) > 0:
            lack_city = pd.DataFrame(lack_citys, columns=['city'])
            lack_city['city_encode'] = lack_citys_encode
            self.current_city_encode = self.current_city_encode.append(lack_city, ignore_index=True)
            self.current_city_encode.to_csv('predict/feature_encode/city.csv', index=False, encoding='utf-8')

    def generate_domain_category(self):
        train = pd.read_csv('../tmp/train/train.csv')
        self.domain_category = train.loc[:, ['domain', 'source_type']]
        self.domain_category = self.domain_category.drop_duplicates(['domain', 'source_type'])
        self.domain_category = self.domain_category.sort_values(by='source_type')
        self.domain_category.to_csv('predict/map/domain_category_map.csv', index=False)

    def execute(self):
        """
        执行人工处理后续流程
        """
        self.generate_others_predict_relate_tables()
        # self.fill_all_city()
