# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '../')
import time
import os
import pandas as pd
pd.set_option('precision', 2)
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

from valuate.db import db_operate,process_tables
from valuate.common import common_func as cf
from valuate.report.report import Report
from valuate.predict.predict import Predict as pred
from valuate.predict.predict_api import Predict as api
from valuate.predict.predict_batch import Predict as batch
from valuate.train.feature_engineering import FeatureEngineering
from valuate.train.stacking import Stacking
from valuate.verify.verify import Verify
from valuate.train.manual import Manual

from valuate.exception.api_error import ApiParamsValueError
from valuate.exception.api_error import ApiParamsTypeError

import cProfile

if __name__ == "__main__":
    # 使用源数据进行特征工程
    # fe = FeatureEngineering()
    # fe.execute()

    # 加载特征工程处理后的训练数据
    # valuated_model = pd.read_csv('predict/map/valuated_model_detail.csv')
    # models = list(set(valuated_model.model_slug.values))
    #
    # # 训练stacking模型
    # stack = Stacking()
    # stack.execute_multitude(models)
    # # stack.grid_search()
    # # 后续人工处理
    # manual = Manual()
    # manual.execute()

    # valuated_model = pd.read_csv('predict/map/valuated_model_detail.csv')
    # valuated_model.to_csv('predict/map/need_valuated_model.csv', index=False)
    predict = pred()
    # predict.predict_all_test_data()


    predict.predict_test_data(model_slug='bseries_1810')


    # result = predict.predict_single(model_slug='qichenR30', model_detail_slug='110195_autotis', city='长春', use_time=239)
    # print(result)

    # # pr = cProfile.Profile()
    # # pr.enable()
    # # # # while True:
    # result = predict.predict_new(city='深圳', model_detail_slug='99582_autotis', use_time=26, mile=3.8)
    # # # result = predict.predict_to_dict(city='深圳', model_detail_slug='m1383_ba', use_time=1000, mile=1)
    # # # result = predict.future_price_trend(city='深圳', model_detail_slug='11554_autotis', use_time=7500, mile=40)
    # # # result = predict.history_price_trend(city='深圳', model_detail_slug='120286_autotis', use_time=365, mile=2.0)
    # # pr.disable()
    # # pr.print_stats(sort="calls")
    # print(result)

    # 单元测试
    # verify = Verify()
    # verify.generate_verify_tables()
    # verify.execute_verify()
    # result = verify.load_valuated_cities()
    # print(result)

    # # 竞品分析报告生成
    # report = Report()
    # report.generate_temp_csv_with_predict()

    # 因训练完新模型需要更新的相关表
    # process_tables.store_need_udpate_tables()
    # process_tables.process_need_udpate_tables()
    # process_tables.update_tables_to_local_db()
