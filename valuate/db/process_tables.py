import gc
import pandas as pd
import numpy as np
from datetime import datetime
from valuate.db import db_operate
from valuate.conf import global_settings as gl

from valuate.predict.predict_batch import Predict as batch


def  store_train_data():
    """
    查询训练数据并存储在tmp中
    """
    # history_train = db_operate.query_local_history_train_data()
    # history_train.to_csv('../tmp/train/history_train_source.csv', index=False)
    # del history_train
    # gc.collect()
    # print('下载历史训练数据,已完成!')
    # train = db_operate.query_local_train_data()
    # train.to_csv('../tmp/train/train_source.csv', index=False)
    # del train
    # gc.collect()
    # print('下载当前训练数据,已完成!')
    #
    open_model_detail = db_operate.query_produce_open_model_detail()
    open_model_detail = open_model_detail.rename(columns={'detail_model_slug': 'model_detail_slug'})
    open_model_detail.to_csv('../tmp/train/open_model_detail.csv', index=False)
    del open_model_detail
    gc.collect()
    #
    # open_category = db_operate.query_produce_open_category()
    # open_category.to_csv('../tmp/train/open_category.csv', index=False)
    # del open_category
    # gc.collect()
    #
    # open_depreciation = db_operate.query_produce_open_depreciation()
    # open_depreciation.to_csv('../tmp/train/open_depreciation.csv', index=False)
    # del open_depreciation
    # gc.collect()
    #
    # open_province_popularity = db_operate.query_produce_open_province_popularity()
    # open_province_popularity.to_csv('../tmp/train/open_province_popularity.csv', index=False)
    # del open_province_popularity
    # gc.collect()
    #
    # open_city = db_operate.query_produce_open_city()
    # open_city.to_csv('../tmp/train/open_city.csv', index=False)
    # del open_city
    # gc.collect()
    # print('下载训练数据,已完成!')


def store_need_udpate_tables():
    """
    查询需要更新的表并存储在tmp中
    1.car_deal_history
    2.car_source
    """
    car_deal_history = db_operate.query_produce_car_deal_history()
    car_deal_history.to_csv('../tmp/update/car_deal_history.csv', index=False)
    del car_deal_history
    gc.collect()

    # car_source = db_operate.query_produce_car_source()
    # car_source.to_csv('../tmp/update/car_source.csv', index=False)
    # del car_source
    # gc.collect()
    print('下载需要更新的表,已完成!')


def process_need_udpate_tables():
    """
    处理新模型更新需要更新的其他相关表
    """
    print('开始处理!')
    # process_car_deal_history()
    # print('完成car_deal_history处理!')
    # process_car_source()
    # print('完成car_source处理!')
    update_tables_to_local_db()
    print('完成本地库上传!')


def process_car_source():
    """
    处理car_source表更新
    """
    def process_time(df):
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

    def cal_gpj_index(df):
        """
        计算公平价系数
        get_gpj_index(11, 100.0)
        2.8
        get_gpj_index(11, 10.0)
        8.0
        get_gpj_index(9, 10.0)
        8.0
        get_gpj_index(9, 9.1)
        9.8
        get_gpj_index(3.4827, 4)
        7.8
        """
        eval_price = float(df['predict_price'])
        x = abs(float(df['price']) - eval_price) / eval_price
        return float('%.1f' % (5 / (2 * x) if x > 0.5 else (10 - 20 * x if x <= 0.1 else (8.75 - 7.5 * x))))

    car_source = pd.read_csv('../tmp/train/train_source.csv')
    car_source['use_time'] = car_source.apply(process_time, axis=1)
    car_source = car_source[(car_source['use_time'] > 0) & (car_source['use_time'] <= 240)]
    car_source.reset_index(inplace=True)
    car_source = car_source.drop('index', axis=1)
    car_source['source_type'] = car_source['source_type'].map(gl.INTENT_MAP)
    # 匹配车型
    model_detail_map = pd.read_csv('predict/map/model_detail_map.csv')
    model_detail_map = model_detail_map.loc[:, ['model_slug', 'model_detail_slug']]
    # 匹配流行度
    province_city_map = pd.read_csv('predict/map/province_city_map.csv')
    province_city_map = province_city_map.loc[:, ['province', 'city']]
    province_popularity_map = pd.read_csv('predict/map/province_popularity_map.csv')
    car_source = car_source.merge(model_detail_map, how='left', on='model_detail_slug')
    car_source = car_source.merge(province_city_map, how='left', on='city')
    car_source = car_source.merge(province_popularity_map, how='left', on=['model_slug', 'province'])
    car_source['popularity'] = car_source['popularity'].fillna('C')
    # 删除数据不完整记录和use_time异常值
    car_source = car_source[(car_source['model_slug'].notnull()) & (car_source['price'].notnull()) & (car_source['price'] > 0)]
    # 重置索引
    car_source.reset_index(inplace=True)
    car_source = car_source.drop('index', axis=1)
    car_source.to_csv('../tmp/update/man.csv', index=False)

    car_source = pd.read_csv('../tmp/update/man.csv')
    car_source = car_source.loc[:, gl.PREDICT_FEATURE]
    predict = batch()
    predict.predict_batch(car_source, store=True, is_update_process=True)

    car_source = pd.read_csv('../tmp/update/man.csv')
    car_source = car_source.loc[:, ['id', 'price', 'model_detail_slug', 'popularity']]
    result = pd.DataFrame()
    for i in range(1, 16):
        temp = pd.read_csv('verify/set/self_set_predict_part'+str(i)+'.csv')
        result = result.append(temp)
    result.reset_index(inplace=True)
    result = result.drop('index', axis=1)
    result = result.merge(car_source, how='left', on='id')

    adjust_profit_map = pd.read_csv('predict/map/adjust_profit_map.csv')
    result = result.merge(adjust_profit_map, how='left', on=['model_detail_slug', 'popularity'])
    result['rate'] = result['rate'].fillna(0)
    result['predict_price'] = result['predict_price'] * (1 + result['rate'])
    result['predict_price'] = result['predict_price'] / 10000
    result['gpj_index'] = result.apply(cal_gpj_index, axis=1)
    result = result.loc[:, ['id', 'predict_price', 'gpj_index']]
    result = result.rename(columns={'predict_price': 'eval_price'})
    result.to_csv('../tmp/update/car_source_update.csv', index=False, float_format='%.2f')
    print('car_source表更新完成!')


def process_car_deal_history():
    """
    处理car_deal_history表更新
    """
    def process_time(df):
        """
        计算成交表使用时间
        """
        if len(str(df['deal_time'])) == 19:
            deal_date = datetime.strptime(str(df['deal_time']), '%Y-%m-%d %H:%M:%S')
        elif len(str(df['deal_time'])) == 10:
            deal_date = datetime.strptime(str(df['deal_time']), '%Y-%m-%d')
        else:
            return -1
        return (deal_date.year - df['year']) * 12 + deal_date.month - df['month']

    car_deal_history = pd.read_csv('../tmp/update/car_deal_history.csv')
    car_deal_history['use_time'] = car_deal_history.apply(process_time, axis=1)
    car_deal_history = car_deal_history[(car_deal_history['use_time'] > 0) & (car_deal_history['use_time'] <= 240)]
    car_deal_history.reset_index(inplace=True)
    car_deal_history = car_deal_history.drop('index', axis=1)
    car_deal_history['source_type'] = car_deal_history['deal_type'].map(gl.INTENT_MAP)
    # 匹配车型
    model_detail_map = pd.read_csv('predict/map/model_detail_map.csv')
    model_detail_map = model_detail_map.loc[:, ['model_slug', 'model_detail_slug']]
    # 匹配流行度
    province_city_map = pd.read_csv('predict/map/province_city_map.csv')
    province_city_map = province_city_map.loc[:, ['province', 'city']]
    province_popularity_map = pd.read_csv('predict/map/province_popularity_map.csv')
    car_deal_history = car_deal_history.merge(model_detail_map, how='left', on='model_detail_slug')
    car_deal_history = car_deal_history.merge(province_city_map, how='left', on='city')
    car_deal_history = car_deal_history.merge(province_popularity_map, how='left', on=['model_slug', 'province'])
    car_deal_history['popularity'] = car_deal_history['popularity'].fillna('C')
    # 删除数据不完整记录
    car_deal_history = car_deal_history[car_deal_history['model_slug'].notnull()]
    # 重置索引
    car_deal_history.reset_index(inplace=True)
    car_deal_history = car_deal_history.drop('index', axis=1)
    car_deal_history.to_csv('../tmp/update/man.csv', index=False)

    car_deal_history = pd.read_csv('../tmp/update/man.csv')
    car_deal_history = car_deal_history.loc[:, gl.PREDICT_FEATURE]
    predict = batch()
    predict.predict_batch(car_deal_history, store=True, is_update_process=True)

    car_deal_history = pd.read_csv('../tmp/update/man.csv')
    car_deal_history = car_deal_history.loc[:, ['id', 'model_detail_slug', 'popularity']]
    result = pd.DataFrame()
    for i in range(1, 16):
        temp = pd.read_csv('verify/set/self_set_predict_part'+str(i)+'.csv')
        result = result.append(temp)
    result.reset_index(inplace=True)
    result = result.drop('index', axis=1)
    result = result.merge(car_deal_history, how='left', on='id')

    adjust_profit_map = pd.read_csv('predict/map/adjust_profit_map.csv')
    result = result.merge(adjust_profit_map, how='left', on=['model_detail_slug', 'popularity'])
    result['rate'] = result['rate'].fillna(0)
    result['predict_price'] = result['predict_price'] * (1 + result['rate'])
    result['predict_price'] = result['predict_price'] / 10000
    result = result.loc[:, ['id', 'predict_price']]
    result = result.rename(columns={'predict_price': 'price'})
    result.to_csv('../tmp/update/car_deal_history_update.csv', index=False, float_format='%.2f')
    print('car_deal_history表更新完成!')


def update_tables_to_local_db():
    """
    更新表到本地数据库
    """
    car_deal_history = pd.read_csv('../tmp/update/car_deal_history_update.csv')
    car_deal_history = car_deal_history.dropna()
    car_source = pd.read_csv('../tmp/update/car_source_update.csv')
    car_source = car_source.dropna()
    db_operate.update_table_to_local_db(car_deal_history, 'car_deal_history_need_update')
    db_operate.update_table_to_local_db(car_source, 'car_source_need_update')
    del car_deal_history, car_source
    gc.collect()
    print('本地库更新完成！')


def store_competed_tables():
    """
    查询竞品分析相关的表并存储在tmp中
    1.eval_source 三家平台竞品抓取表(b_2_c,c_2_c)
    2.car_source 车源表
    3.deal_records 成交表
    4.eval_deal_source 成交竞品表(c_2_b)
    """
    eval_source = db_operate.query_produce_eval_source()
    eval_source.to_csv('../tmp/report/eval_source.csv', index=False)
    del eval_source
    gc.collect()

    deal_records = db_operate.query_produce_deal_records()
    deal_records.to_csv('../tmp/report/deal_records.csv', index=False)
    del deal_records
    gc.collect()

    eval_deal_source = db_operate.query_produce_eval_deal_records()
    eval_deal_source.to_csv('../tmp/report/eval_deal_source.csv', index=False)
    del eval_deal_source
    gc.collect()
    print('下载竞品分析表,已完成!')
