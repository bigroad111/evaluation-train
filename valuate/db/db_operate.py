from valuate.db import *


###############################
# 训练相关数据库操作
###############################


def query_local_train_data():
    """
    查询本地库训练数据
    """
    return pd.read_sql_query(gl.TRAIN_DATA_QUERY, runtime_engine)


def query_local_history_train_data():
    """
    查询本地库历史训练数据
    """
    return pd.read_sql_query(gl.HISTORY_CAR_SOURCE_QUERY, runtime_engine)


def query_produce_open_model_detail():
    """
    查询款型库
    """
    return pd.read_sql_query(gl.PRODUCE_OPEN_MODEL_DETAIL_QUERY, runtime_engine)


def query_produce_open_category():
    """
    查询车型库
    """
    return pd.read_sql_query(gl.PRODUCE_OPEN_CATEGORY_QUERY, runtime_engine)


def query_produce_open_depreciation():
    """
    查询系数衰减表
    """
    return pd.read_sql_query(gl.PRODUCE_OPEN_DEPRECIATION_QUERY, runtime_engine)


def query_produce_open_province_popularity():
    """
    查询车型省份流行度表
    """
    return pd.read_sql_query(gl.PRODUCE_OPEN_PROVINCE_POPULARITY_QUERY, runtime_engine)


def query_produce_open_city():
    """
    查询省份城市表
    """
    return pd.read_sql_query(gl.PRODUCE_OPEN_CITY_QUERY, runtime_engine)

###############################
# 训练完成后需要更新表的数据库操作
###############################


def query_produce_car_deal_history():
    """
    查询生产车源交易历史表
    """
    return pd.read_sql_query(gl.PRODUCE_CAR_DEAL_HISTORY_QUERY, runtime_engine)


def query_produce_car_source():
    """
    查询生产车源表
    """
    return pd.read_sql_query(gl.PRODUCE_CAR_SOURCE_QUERY, runtime_engine)

###############################
# 平台对比分析数据报告数据库操作
###############################


def query_produce_car_source_analysis():
    """
    查询生产车源表(跟竞品相关的时间点)
    """
    return pd.read_sql_query(gl.PRODUCE_CAR_SOURCE_QUERY_ANALYSIS, runtime_engine)


def query_produce_eval_source():
    """
    查询生产竞品表
    """
    return pd.read_sql_query(gl.EVAL_SOURCE_QUERY, runtime_engine)


def query_produce_deal_records():
    """
    查询生产成交记录表
    """
    return pd.read_sql_query(gl.PRODUCE_DEAL_RECORDS_QUERY, runtime_engine)


def query_produce_eval_deal_records():
    """
    查询生产成交竞品表
    """
    return pd.read_sql_query(gl.PRODUCE_EVAL_DEAL_SOURCE_QUERY, runtime_engine)

################################
# 本地库相关操作
################################


def update_table_to_local_db(df, tbale_name):
    """
    更新表到本地库
    """
    df.to_sql(tbale_name, runtime_engine, if_exists='replace', index=False)

################################
# 生产库估值查询
################################


def query_valuate(model_detail_slug_id, city_id, column_num, use_time):
    """
    查询估值
    """
    columns = 'b2c_year_'+column_num+',c2c_year_'+column_num+',popularity'
    if use_time <= 120:
        valuate_map_query = 'select '+columns+' from '+gl.VALUATE_MAP_0_9+' where city_id = '+str(city_id)+' and '+'model_detail_slug_id = '+str(model_detail_slug_id)
    else:
        valuate_map_query = 'select ' + columns + ' from ' + gl.VALUATE_MAP_10_20 + ' where city_id = ' + str(city_id) + ' and ' + 'model_detail_slug_id = ' + str(model_detail_slug_id)
    return pd.read_sql_query(valuate_map_query, runtime_engine)
