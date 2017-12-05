from valuate.predict import *


def get_profit_rate(intent, popularity):
    """
    获取畅销系数
    """
    # 按畅销程度分级,各交易方式相比于标价的固定比例
    profits = gl.PROFITS
    profit = profits[popularity]
    # 计算各交易方式的价格相比于标价的固定比例
    if intent == 'sell':
        # 商家收购价相比加权平均价的比例
        profit_rate = 1 - profit[0] - profit[1]
    elif intent == 'buy':
        # 商家真实售价相比加权平均价的比例
        profit_rate = 1 - profit[0]
    elif intent == 'release':
        # 建议标价相比加权平均价的比例
        profit_rate = 1
    elif intent == 'private':
        # C2C价格相比加权平均价的比例
        profit_rate = 1 - profit[0] - profit[2]
    elif intent == 'lowest':
        # 最低成交价相比加权平均价的比例
        profit_rate = 1 - profit[0] - profit[1] - profit[3]
    elif intent == 'cpo':
        # 认证二手车价相比加权平均价的差异比例
        profit_rate = 1 - profit[0] - profit[8]
    elif intent == 'replace':
        # 4S店置换价相比加权平均价的比例
        profit_rate = 1 - profit[0] - profit[4]
    elif intent == 'auction':
        # 拍卖价相比加权平均价的差异比例
        profit_rate = 1 - profit[0] - profit[5]
    elif intent == 'avg-buy':
        # 平均买车价相比加权平均价的差异比例
        profit_rate = 1 - profit[0] - profit[7]
    elif intent == 'avg-sell':
        # 平均卖车价价相比加权平均价的差异比例
        profit_rate = 1 - profit[0] - profit[6]
    return profit_rate


def get_gpj_index(price, predict):
    """
    根据标价和预测价格计算公平价系数
    """
    x = abs(float(price)/float(predict)-1)
    if x <= 0.1:
        y = 10-20*x
    elif (x > 0.1) and (x <= 0.5):
        y = 8.75-7.5*x
    else:
        y = 5/(2*x)
    return y


def process_index(df):
    """
    装载预测表的索引
    """
    return df['model_detail_slug'] + '_' + str(df['use_time'])


def process_mile(df):
    """
    mile处理
    """
    if df['mile_per_month'] < gl.MILE_THRESHOLD:
        return df['predict_price'] + (gl.MILE_RATE/3) * df['predict_price'] * (1 - df['mile_per_month']/gl.MILE_THRESHOLD)
    else:
        return df['predict_price'] - (gl.MILE_RATE * 2/3) * df['predict_price'] * (1 - gl.MILE_THRESHOLD/df['mile_per_month'])


def process_profit_rate(df):
    """
    畅销系数处理
    """
    return get_profit_rate(df['intent'], df['popularity'])


def process_buy_profit_rate(df):
    """
    获取源交易系数
    """
    return get_profit_rate(df['intent_source'], df['popularity'])


def process_gpj_index(df):
    """
    处理公平价系数
    """
    return get_gpj_index(df['price'], df['predict_price'])


def process_personal_prices_relate(df):
    """
    人工处理预测不合理的三类价格
    """
    private = df['cpersonal']
    buy = df['dealer']
    # 计算dealer与cpersonal价格比例关系
    private_buy_rate = (buy - private) / private
    # 人工处理预测价格
    # if ~(0.055 <= private_buy_rate <= 0.12):
    if (private_buy_rate < 0) | (abs(private_buy_rate) > 0.12) | (abs(private_buy_rate) <= 0.055):
        return int(buy * (1 - 0.0875))
    return df['cpersonal']


def process_sell_prices_relate(df):
    """
    人工处理预测不合理的三类价格
    """
    private = df['cpersonal']
    sell = df['sell_dealer']
    # 计算三类价格比例关系
    private_sell_rate = (sell - private) / private
    # 人工处理预测价格
    # if ~(-0.085 <= private_sell_rate <= -0.02):
    if (private_sell_rate > 0) | (abs(private_sell_rate) > 0.15) | (abs(private_sell_rate) <= 0.02):
        return int(private * (1 - 0.0525))
    return df['sell_dealer']


def process_final_predict_price(df):
    """
    根据交易类型返回最终预测价格
    """
    if df['source_type_sel'] in gl.C_2_C:
        return df['cpersonal']
    elif df['source_type_sel'] in gl.B_2_C:
        return df['dealer']
    elif df['source_type_sel'] == 'sell_dealer':
        return df['sell_dealer']
    else :
        return np.NAN


class Predict(object):

    def __init__(self):
        """
        加载各类匹配表和模型
        """
        self.test_level1 = []
        self.test_level2 = []
        self.predict_hedge = []

        # 重定位根路径
        self.path = os.path.abspath(os.path.dirname(gl.__file__))
        self.path = self.path.replace('conf', '')

        # 加载level1模型
        # self.level1_gtb = joblib.load(self.path + 'predict/model/gradient_tree_boosting_level1.pkl')
        # self.level1_xgb = joblib.load(self.path + 'predict/model/xgboost_level1.pkl')
        # self.level1_gbm = joblib.load(self.path + 'predict/model/lightgbm_level1.pkl')

        # 加载level2模型
        self.level2_xgb = xgb.Booster()
        self.level2_xgb.load_model(self.path + "predict/model/xgboost_level2.model")

        # 加载特征顺序
        with open(self.path + "predict/feature_encode/feature_order.txt", "rb") as fp:
            self.feature_name = pickle.load(fp)

        # 加载车型、款型匹配表
        self.model_detail_map_batch = pd.read_csv(self.path + 'predict/map/model_detail_map.csv')

        # 加载省份城市匹配表
        self.province_city_map = pd.read_csv(self.path + 'predict/map/province_city_map.csv')

        # 加载省、车型，流行度匹配表
        self.province_popularity_map = pd.read_csv(self.path + 'predict/map/province_popularity_map.csv')

    def map_feature_encode(self, car_source):
        """
        1.匹配特征编码
        """
        self.test_level1 = car_source
        inner = gl.FEATURE_ENCODE
        for f in inner:
            temp = f
            f = pd.read_csv('predict/feature_encode/' + f + '.csv')
            self.test_level1 = self.test_level1.merge(f, on=temp, how='left')
            # self.test_level1 = self.test_level1.drop(temp, axis=1)

    def model_predict(self):
        """
        2.模型预测
        """
        pred = self.test_level1.loc[:, self.feature_name]
        self.predict_hedge = self.level2_xgb.predict(xgb.DMatrix(pred))
        self.test_level1['predict_hedge'] = pd.Series(self.predict_hedge).values
        self.test_level1['predict_price'] = self.test_level1['predict_hedge'] * self.test_level1['price_bn']
        # self.test_level2 = pd.DataFrame(self.level1_gtb.predict(pred), columns=['gradient_tree_boosting'])
        # self.test_level2['xgboost'] = pd.Series(self.level1_xgb.predict(pred)).values
        # self.test_level2['gbm'] = pd.Series(self.level1_gbm.predict(pred)).values
        # self.test_level2['city_encode'] = pred['city_encode']
        # self.test_level2['model_detail_slug_encode'] = pred['model_detail_slug_encode']
        #
        # self.predict_price = np.exp(self.level2_xgb.predict(xgb.DMatrix(self.test_level2)))
        # self.test_level1['predict_price'] = pd.Series(self.predict_price).values

    def add_process_mile(self):
        """
        3.根据公里数修正预测值
        """
        self.test_level1['mile_per_month'] = self.test_level1['mile'] / self.test_level1['use_time']
        self.test_level1['predict_price'] = self.test_level1.apply(process_mile, axis=1)

    def add_process_condition(self):
        """
        4.预测车况
        """
        # 评估当前车况
        # temp = self.test_level1.loc[:, ['use_time', 'mile']]
        # car_condition = self.car_condition_model.predict(temp)
        # self.test_level1['car_condition'] = pd.Series(car_condition).values
        # self.test_level1 = self.test_level1.merge(self.car_condition_match, how='left', on='car_condition')
        # self.test_level1['desc'] = self.test_level1['desc'].str.lower()
        # self.test_level1 = self.test_level1.rename(columns={'desc': 'condition'})

    def add_process_intent(self):
        """
        5.根据交易方式修正预测值
        """
        # 根据source_type映射intent
        self.test_level1['intent'] = self.test_level1['source_type_bak'].map(gl.INTENT_MAP)
        self.test_level1['intent_source'] = self.test_level1['source_type'].map(gl.INTENT_MAP)
        # 根据款型获取车型
        self.test_level1 = self.test_level1.merge(self.model_detail_map_batch, how='left', on='model_detail_slug')
        # 根据城市获取省份
        self.test_level1 = self.test_level1.merge(self.province_city_map, how='left', on='city')
        # 根据城市,车型获取流行度
        self.test_level1 = self.test_level1.merge(self.province_popularity_map, how='left', on=['model_slug', 'province'])
        self.test_level1['popularity'] = self.test_level1['popularity'].fillna('C')
        self.test_level1['profit_rate'] = self.test_level1.apply(process_profit_rate, axis=1)
        self.test_level1['buy_profit_rate'] = self.test_level1.apply(process_buy_profit_rate, axis=1)
        self.test_level1['predict_price'] = self.test_level1['predict_price'] / self.test_level1['buy_profit_rate']
        self.test_level1['predict_price'] = self.test_level1['profit_rate'] * self.test_level1['predict_price']

    def add_gpj_index(self):
        self.test_level1['gpj_index'] = self.test_level1.apply(process_gpj_index, axis=1)

    def predict(self, car_source):
        """
        预测返回
        """
        # 全预测dealer类型,根据此类型转换成其他交易方式
        car_source['source_type_sel'] = car_source['source_type']
        car_source['price_bn'] = car_source['price_bn'] * 10000
        # 预测流程
        temp = pd.DataFrame()
        for category in gl.INTENT_TYPE_PREDICT:
            self.test_level1 = []
            car_source['source_type_bak'] = category
            car_source['source_type'] = category
            self.map_feature_encode(car_source)
            self.model_predict()
            self.add_process_mile()
            self.add_process_intent()
            temp[category] = self.test_level1['predict_price']
        self.test_level1['sell_dealer'] = temp['sell_dealer']
        self.test_level1['dealer'] = temp['dealer']
        self.test_level1['cpersonal'] = temp['cpersonal']
        self.test_level1.reset_index(inplace=True)
        self.test_level1 = self.test_level1.drop('index', axis=1)
        self.test_level1['cpersonal'] = self.test_level1.apply(process_personal_prices_relate, axis=1)
        self.test_level1['sell_dealer'] = self.test_level1.apply(process_sell_prices_relate, axis=1)
        self.test_level1['predict_price'] = self.test_level1.apply(process_final_predict_price, axis=1)
        # self.test_level1 = self.test_level1.drop(['cpersonal', 'dealer', 'sell_dealer'], axis=1)
        return self.test_level1.loc[:, ['car_id','city','model_detail_slug','source_type','price_bn_x','use_time','mile','source_type_bak','cpersonal','dealer','sell_dealer','predict_price']]
        # return self.test_level1

    def predict_for_report(self, car_source):
        """
        针对数据报告做预测
        """
        # 全预测dealer类型,根据此类型转换成其他交易方式
        car_source['source_type_bak'] = car_source['source_type']
        car_source.loc[~(car_source['source_type'].isin(gl.INTENT_TYPE_PREDICT)), 'source_type'] = 'dealer'
        car_source['price_bn'] = car_source['price_bn'] * 10000
        # 预测流程
        self.map_feature_encode(car_source)
        self.model_predict()
        self.add_process_mile()
        self.add_process_intent()
        return self.test_level1

    def predict_new(self, car_source):
        """
        针对数据报告做预测
        """
        def process_popularity(df):
            model_slug = df['model_slug']
            province = province_city_map.loc[df['city'], 'province']
            if str(province + model_slug) in province_popularity_map.index:
                return province_popularity_map.loc[str(province + model_slug), 'popularity']
            else:
                return 'C'

        # 全预测dealer类型,根据此类型转换成其他交易方式
        car_source['price_bn'] = car_source['price_bn'] * 10000
        car_source['use_time'] = car_source['use_time'] / 30
        car_source['use_time'] = car_source['use_time'].astype(int)
        car_source['mile_per_month'] = car_source['mile'] / car_source['use_time']
        car_source['popularity'] = car_source.apply(process_popularity, axis=1)
        # 预测流程
        self.map_feature_encode(car_source)
        self.model_predict()
        return self.test_level1

