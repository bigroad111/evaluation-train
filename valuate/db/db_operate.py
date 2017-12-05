import pandas as pd

from sqlalchemy import create_engine
from valuate.conf import global_settings as gl

###############################
# 训练相关数据库操作
###############################


def query_local_train_data():
    """
    查询本地库训练数据
    """
    engine = create_engine(gl.LOCAL_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.TRAIN_DATA_QUERY, engine)


def query_local_history_train_data():
    """
    查询本地库历史训练数据
    """
    engine = create_engine(gl.LOCAL_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.HISTORY_CAR_SOURCE_QUERY, engine)


def query_produce_open_model_detail():
    """
    查询款型库
    """
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.PRODUCE_OPEN_MODEL_DETAIL_QUERY, engine)


def query_produce_open_category():
    """
    查询车型库
    """
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.PRODUCE_OPEN_CATEGORY_QUERY, engine)


def query_produce_open_depreciation():
    """
    查询系数衰减表
    """
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.PRODUCE_OPEN_DEPRECIATION_QUERY, engine)


def query_produce_open_province_popularity():
    """
    查询车型省份流行度表
    """
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.PRODUCE_OPEN_PROVINCE_POPULARITY_QUERY, engine)


def query_produce_open_city():
    """
    查询省份城市表
    """
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.PRODUCE_OPEN_CITY_QUERY, engine)

###############################
# 训练完成后需要更新表的数据库操作
###############################


def query_produce_car_deal_history():
    """
    查询生产车源交易历史表
    """
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.PRODUCE_CAR_DEAL_HISTORY_QUERY, engine)


def query_produce_car_source():
    """
    查询生产车源表
    """
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.PRODUCE_CAR_SOURCE_QUERY, engine)

###############################
# 平台对比分析数据报告数据库操作
###############################


def query_produce_car_source_analysis():
    """
    查询生产车源表(跟竞品相关的时间点)
    """
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.PRODUCE_CAR_SOURCE_QUERY_ANALYSIS, engine)


def query_produce_eval_source():
    """
    查询生产竞品表
    """
    engine = create_engine(gl.PRODUCE_DATASOURCE_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.EVAL_SOURCE_QUERY, engine)


def query_produce_deal_records():
    """
    查询生产成交记录表
    """
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.PRODUCE_DEAL_RECORDS_QUERY, engine)


def query_produce_eval_deal_records():
    """
    查询生产成交竞品表
    """
    engine = create_engine(gl.PRODUCE_DATASOURCE_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(gl.PRODUCE_EVAL_DEAL_SOURCE_QUERY, engine)

################################
# 本地库相关操作
################################


def update_table_to_local_db(df, tbale_name):
    """
    更新表到本地库
    """
    engine = create_engine(gl.LOCAL_PINGJIA_ENGINE, encoding=gl.ENCODING)
    df.to_sql(tbale_name, engine, if_exists='replace', index=False)
