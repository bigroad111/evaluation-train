ENCODING = 'utf-8'

##########################
# 生产,测试库配置
##########################

# 运行环境[PRODUCT,TEST,LOCAL]
RUNTIME_ENVIRONMENT = 'PRODUCT'

# 生产库外网
PRODUCE_DB_ADDR_OUTTER = '101.201.143.74'
PRODUCE_DB_USER = 'leidengjun'
PRODUCE_DB_PASSWD = 'ldj_DEV_~!'
PRODUCE_PINGJIA_ENGINE = 'mysql+pymysql://'+PRODUCE_DB_USER+':'+PRODUCE_DB_PASSWD+'@'+PRODUCE_DB_ADDR_OUTTER+'/pingjia?charset=utf8'
PRODUCE_DATASOURCE_ENGINE = 'mysql+pymysql://'+PRODUCE_DB_USER+':'+PRODUCE_DB_PASSWD+'@'+PRODUCE_DB_ADDR_OUTTER+'/datasource?charset=utf8'
PRODUCE_VALUATE_ENGINE_OUTTER_PYMYSQL = 'mysql+pymysql://'+PRODUCE_DB_USER+':'+PRODUCE_DB_PASSWD+'@'+PRODUCE_DB_ADDR_OUTTER+'/valuate?charset=utf8'
PRODUCE_VALUATE_ENGINE_OUTTER_MYSQL_CON = 'mysql+mysqlconnector://'+PRODUCE_DB_USER+':'+PRODUCE_DB_PASSWD+'@'+PRODUCE_DB_ADDR_OUTTER+'/valuate?charset=utf8'

# 生产库内网
# PRODUCE_DB_ADDR_INNER = '10.45.147.49'
PRODUCE_DB_ADDR_INNER = '100.114.30.239:18056'
PRODUCE_DB_INNER_USER = 'pingjia'
PRODUCE_DB_INNER_PASSWD = 'De32wsxC'
PRODUCE_VALUATE_ENGINE_INNER_PYMYSQL = 'mysql+pymysql://'+PRODUCE_DB_INNER_USER+':'+PRODUCE_DB_INNER_PASSWD+'@'+PRODUCE_DB_ADDR_INNER+'/valuate?charset=utf8'
PRODUCE_VALUATE_ENGINE_INNER_MYSQL_CON = 'mysql+mysqlconnector://'+PRODUCE_DB_INNER_USER+':'+PRODUCE_DB_INNER_PASSWD+'@'+PRODUCE_DB_ADDR_INNER+'/valuate?charset=utf8'

# 本地库
LOCAL_DB_ADDR = '192.168.1.48'
LOCAL_DB_USER = 'pingjia'
LOCAL_DB_PASSWD = '654321'
LOCAL_PINGJIA_ENGINE = 'mysql+pymysql://'+LOCAL_DB_USER+':'+LOCAL_DB_PASSWD+'@'+LOCAL_DB_ADDR+'/pingjia?charset=utf8'

###########################
# 模型训练配置
###########################
# 价格预测模型训练需要的特征
TRAIN_FEATURE = ['source_type', 'model_detail_slug', 'popularity', 'city', 'use_time']

# 目标特征
TARGET_FEATURE = 'hedge_rate'

# 需要特征编码字段
FEATURE_ENCODE = ['source_type', 'model_detail_slug', 'popularity', 'city']

# 车况预测模型需要的特征
CAR_CONDITION_TRAIN_FEATURE = ['use_time', 'mile', 'price', 'price_bn']

# 车况类别
CAR_CONDITION_VALUES = ['excellent', 'good', 'fair', 'bad']

# 各车况因素的系数
CAR_CONDITION_COEFFICIENT = {'excellent': 1.03, 'good': 1, 'fair': 0.89, 'bad': 0.82}
CAR_CONDITION_COEFFICIENT_VALUES = [1.03, 1, 0.89, 0.82]

# 交易方式
INTENT_TYPE = ['sell', 'buy', 'release', 'private', 'lowest', 'cpo', 'replace', 'auction', 'avg-buy', 'avg-sell']
INTENT_TYPE_CAN = ['sell', 'buy', 'buy', 'private', 'buy', 'buy', 'buy', 'buy', 'buy', 'buy']

# 车源交易历史表
CAR_DEAL_HISTORY = 'car_deal_history'
PRODUCE_CAR_DEAL_HISTORY_COLUMNS = 'id,model_detail_slug,year,month,mile,city,deal_time,deal_type,price'
PRODUCE_CAR_DEAL_HISTORY_QUERY = 'select '+PRODUCE_CAR_DEAL_HISTORY_COLUMNS+' from ' + CAR_DEAL_HISTORY

# 车源表
CAR_SOURCE_SOURCE_TYPE_VALUES = ['cpo', 'dealer', 'odealer', 'personal', 'cpersonal']
CAR_SOURCE = 'car_source'
PRODUCE_CAR_SOURCE_COLUMNS = 'id,brand_slug,model_slug,model_detail_slug,year,month,mile,city,source_type,price,sold_time'
PRODUCE_CAR_SOURCE_QUERY = 'select '+PRODUCE_CAR_SOURCE_COLUMNS+' from ' + CAR_SOURCE + ' where model_detail_slug is not null '
PRODUCE_CAR_SOURCE_QUERY_ANALYSIS = 'select '+PRODUCE_CAR_SOURCE_COLUMNS+' from ' + CAR_SOURCE + ' where model_detail_slug is not null and status = \'review\' and pub_time > \'2017-08-01\' '

LOCAL_CAR_SOURCE_COLUMNS = 'cs.id,cs.model_detail_slug,cs.mile,cs.year,cs.month,cs.city,cs.price,cs.status,cs.source_type,cs.domain,cs.expired_at,cs.sold_time'

# 款型表
OPEN_MODEL_DETAIL = 'open_model_detail'
PRODUCE_OPEN_MODEL_DETAIL_COLUMNS = 'id,price_bn,global_slug,year,volume,control,detail_model_slug'
PRODUCE_OPEN_MODEL_DETAIL_QUERY = 'select '+PRODUCE_OPEN_MODEL_DETAIL_COLUMNS+' from ' + OPEN_MODEL_DETAIL + ' where status = \'Y\' or status = \'A\''

# 车型表
OPEN_CATEGORY = 'open_category'
PRODUCE_OPEN_CATEGORY_COLUMNS = 'name,slug,parent,attribute'
PRODUCE_OPEN_CATEGORY_QUERY = 'select '+PRODUCE_OPEN_CATEGORY_COLUMNS + ' from ' + OPEN_CATEGORY + ' where status = \'Y\' or status = \'A\''

# 车源详情表
CAR_DETAIL_INFO = 'car_detail_info'
LOCAL_CAR_DETAIL_INFO_COLUMNS = 'car_id,transfer_owner'

# 价格系数衰减表
OPEN_DEPRECIATION = 'open_depreciation'
PRODUCE_OPEN_DEPRECIATION_COLUMNS = '*'
PRODUCE_OPEN_DEPRECIATION_QUERY = 'select '+PRODUCE_OPEN_DEPRECIATION_COLUMNS+' from '+OPEN_DEPRECIATION

# 省份车型流行表
OPEN_PROVINCE_POPULARITY = 'open_province_popularity'
PRODUCE_OPEN_PROVINCE_POPULARITY_COLUMNS = '*'
PRODUCE_OPEN_PROVINCE_POPULARITY_QUERY = 'select '+PRODUCE_OPEN_PROVINCE_POPULARITY_COLUMNS+' from '+OPEN_PROVINCE_POPULARITY

# 省份城市表
OPEN_CITY = 'open_city'
PRODUCE_OPEN_CITY_COLUMNS = 'id,name,parent'
PRODUCE_OPEN_CITY_QUERY = 'select '+PRODUCE_OPEN_CITY_COLUMNS+' from '+OPEN_CITY

# 训练数据关联查询
# TRAIN_DATA_QUERY = 'select cs.id,cs.model_detail_slug,cs.mile,cs.year,cs.month,cs.city,cs.price,cs.status,cs.source_type,cs.domain,cs.expired_at,cs.sold_time,omd.price_bn ' \
#                    'from car_source  as cs left join open_model_detail as omd on cs.model_detail_slug=omd.detail_model_slug ' \
#                    ' where cs.global_sibling = 0 and cs.model_detail_slug is not null and cs.pub_time > \'2017-01-01\' limit 10'

TRAIN_DATA_QUERY = 'select cs.id,cs.model_detail_slug,cs.mile,cs.year,cs.month,cs.city,cs.price,cs.status,cs.source_type,cs.domain,cs.dealer_id,cs.expired_at,cs.sold_time ' \
                        'from car_source  as cs where cs.global_sibling = 0 and cs.model_detail_slug is not null'

# 车源历史表
HISTORY_CAR_SOURCE = 'car_source_archive'

HISTORY_CAR_SOURCE_QUERY = 'select cs.id,cs.model_detail_slug,cs.mile,cs.year,cs.month,cs.city,cs.price,cs.status,cs.source_type,cs.domain,cs.expired_at,cs.sold_time ' \
                        'from car_source_archive  as cs where cs.global_sibling = 0 and cs.model_detail_slug is not null '


###########################
# 模型预测配置
###########################
# 批量预测需要的特征
PREDICT_FEATURE = ['id', 'model_slug', 'model_detail_slug', 'popularity', 'source_type', 'use_time', 'mile', 'city']

# 公里数阈值和范围
# 正常行驶的车辆以一年2.5万公里为正常基数，低于2.5万公里的价格的浮动在+3.5%以内
# 大于2.5万公里的若每年的平均行驶里程大于2.5万公里小于5万公里价格浮动在-3.5-7.5%
# 若年平均形式里程大于5万公里及以上影响价格在-7.5-12.5%之间
MILE_THRESHOLD_2_5 = 0.2083
MILE_THRESHOLD_5 = 0.4167
MILE_THRESHOLD_10 = 0.8333

# 畅销程度系数
PROFITS = {'A': (0.06, 0.11, 0.027, 0.02, 0.12, 0.08, 0.09, 0.006, -0.01),
           'B': (0.05, 0.13, 0.031, 0.025, 0.14, 0.10, 0.10, 0.007, -0.01),
           'C': (0.05, 0.15, 0.02, 0.03, 0.16, 0.12, 0.11, 0.003, -0.01)}

# intent映射表
INTENT_MAP = {'cpo': 'cpo', 'dealer': 'buy', 'odealer': 'buy',
              'cpersonal': 'private', 'personal': 'private',
              'sell_dealer': 'sell', 'trade_in': 'replace',
              'auction': 'auction', 'private_party': 'private',
              'buy_dealer': 'buy', 'buy_cpo': 'cpo'
              }

# 估值映射表
VALUATE_MAP_0_9 = 'valuate_car_source'
VALUATE_MAP_10_20 = 'valuate_car_source_extend'
###########################
# 平台分析报告相关表
###########################

# 竞品表
EVAL_SOURCE = 'eval_source'
PRODUCE_EVAL_SOURCE_COLUMNS = '*'
PRODUCE_EVAL_SOURCE_QUERY = 'select '+PRODUCE_EVAL_SOURCE_COLUMNS+' from ' + EVAL_SOURCE

EVAL_SOURCE_QUERY = 'select es.id,es.car_id,es.excellent,es.good,es.fair,es.domain,es.price,es.status,es.created,cs.title,cs.pub_time,' \
                    'cs.model_slug,cs.model_detail_slug,cs.mile,cs.year,cs.month,cs.city,cs.source_type,cs.expired_at,cs.sold_time,omd.price_bn ' \
                    'from datasource.eval_source as es left join pingjia.car_source as cs on cs.id=es.car_id ' \
                    'left join pingjia.open_model_detail as omd on cs.model_detail_slug = omd.detail_model_slug'

DEAL_TYPE_MAP_CATEGORY = {1: 'c_2_c', 2: 'c_2_b', 3: 'b_2_c'}
DEAL_TYPE_MAP_SOURCE_TYPE = {1: 'cpersonal', 2: 'sell_dealer', 3: 'dealer'}

# 成交记录表
# deal_type(c2c - 1;c2b - 2;b2c - 3;b2b - 4)
DEAL_RECORDS = 'deal_records'
PRODUCE_DEAL_RECORDS_COLUMNS = '*'
PRODUCE_DEAL_RECORDS_QUERY = 'select '+PRODUCE_DEAL_RECORDS_COLUMNS+' from ' + DEAL_RECORDS

# 成交竞品表
EVAL_DEAL_SOURCE = 'eval_deal_source'
PRODUCE_EVAL_DEAL_SOURCE_COLUMNS = '*'
PRODUCE_EVAL_DEAL_SOURCE_QUERY = 'select '+PRODUCE_EVAL_DEAL_SOURCE_COLUMNS+' from ' + EVAL_DEAL_SOURCE

# 可预测交易类型
INTENT_TYPE_PREDICT = ['cpersonal', 'dealer', 'sell_dealer']

# 交易类型分类
C_2_C = ['personal', 'cpersonal']
B_2_C = ['cpo', 'dealer', 'odealer']