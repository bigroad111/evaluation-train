from valuate.predict import *
import xgboost as xgb
from sklearn.externals import joblib


def df_process_mile(df):
    """
    mile处理
    """
    # 正常行驶的车辆以一年2.5万公里为正常基数，低于2.5万公里的价格的浮动在+3.5%以内
    # 大于2.5万公里的若每年的平均行驶里程大于2.5万公里小于5万公里价格浮动在-3.5-7.5%
    # 若年平均形式里程大于5万公里及以上影响价格在-7.5-12.5%之间
    if df['predict_price'] <= 0:
        return 0
    mile = df['mile']
    use_time = df['use_time']
    if use_time <= 0:
        use_time = 1
    mile_per_month = mile / use_time
    if mile_per_month < gl.MILE_THRESHOLD_2_5:
        return df['predict_price'] + 0.035 * (1 - mile_per_month/gl.MILE_THRESHOLD_2_5) * df['predict_price']
    elif gl.MILE_THRESHOLD_2_5 <= mile_per_month < gl.MILE_THRESHOLD_5:
        return df['predict_price'] - (0.04 * (mile_per_month/gl.MILE_THRESHOLD_5)+0.035) * df['predict_price']
    elif gl.MILE_THRESHOLD_5 <= mile_per_month < gl.MILE_THRESHOLD_10:
        return df['predict_price'] - (0.05 * (mile_per_month/gl.MILE_THRESHOLD_5)+0.075) * df['predict_price']
    else:
        return df['predict_price'] - 0.125 * df['predict_price']


def df_process_hedge(df, category):
    """
    查找保值率
    """
    if category == 'dealer':
        if type(df['dealer_hedge']) == float:
            return 0
        else:
            dealer_hedge = json.loads(df['dealer_hedge'])
            if dealer_hedge[df['use_time']-1] >= 1:
                return 0.999 * df['final_price_bn'] * 10000
            return dealer_hedge[df['use_time']-1] * df['final_price_bn'] * 10000
    elif category == 'cpersonal':
        if type(df['cpersonal_hedge']) == float:
            return 0
        else:
            cpersonal_hedge = json.loads(df['cpersonal_hedge'])
            if cpersonal_hedge[df['use_time']-1] >= 1:
                return 0.999 * df['final_price_bn'] * 10000
            return cpersonal_hedge[df['use_time'] - 1] * df['final_price_bn'] * 10000


def df_process_prices_relate(df):
    """
    人工处理三类价格的相关性
    """
    buy = int(df['dealer_price'])
    private = int(df['cpersonal_price'])
    if (buy <= 0) | (private <= 0):
        return 0
    # 计算buy与private的比例关系
    private_buy_rate = (buy - private) / private
    # 人工处理预测不合理的三类价格
    if (private_buy_rate < 0) | (abs(private_buy_rate) > 0.12):
        private = int(buy * (1 - 0.0875))
    sell = int(private * (1 - 0.0525))
    if df['source_type'] == 'dealer':
        return buy
    elif df['source_type'] == 'cpersonal':
        return private
    elif df['source_type'] == 'sell_dealer':
        return sell


def df_process_prices_update(df):
    """
    人工处理三类价格的相关性
    """
    # 按畅销程度分级,各交易方式相比于标价的固定比例
    profits = gl.PROFITS
    profit = profits[df['popularity']]

    buy = df['dealer_price']
    private = df['cpersonal_price']
    if (buy <= 0) | (private <= 0):
        return 0
    # 计算buy与private的比例关系
    private_buy_rate = (buy - private) / private
    # 人工处理预测不合理的三类价格
    if (private_buy_rate < 0) | (abs(private_buy_rate) > 0.12):
        private = int(buy * (1 - 0.0875))
    sell = int(private * (1 - 0.0525))

    if df['source_type'] == 'dealer':
        return buy
    elif df['source_type'] == 'cpersonal':
        return private
    elif df['source_type'] == 'sell_dealer':
        return sell

    price = buy / (1 - profit[0])
    # 计算各交易方式的价格相比于标价的固定比例
    if df['source_type'] == 'sell':
        # 商家收购价相比加权平均价的比例
        return sell
    elif df['source_type'] == 'buy':
        # 商家真实售价相比加权平均价的比例
        return buy
    elif df['source_type'] == 'release':
        # 建议标价相比加权平均价的比例
        return price
    elif df['source_type'] == 'private':
        # C2C价格相比加权平均价的比例
        return private
    elif df['source_type'] == 'lowest':
        # 最低成交价相比加权平均价的比例
        return price * (1 - profit[0] - profit[1] - profit[3])
    elif df['source_type'] == 'cpo':
        # 认证二手车价相比加权平均价的差异比例
        return price * (1 - profit[0] - profit[8])
    elif df['source_type'] == 'replace':
        # 4S店置换价相比加权平均价的比例
        return price * (1 - profit[0] - profit[4])
    elif df['source_type'] == 'auction':
        # 拍卖价相比加权平均价的差异比例
        return price * (1 - profit[0] - profit[5])
    elif df['source_type'] == 'avg-buy':
        # 平均买车价相比加权平均价的差异比例
        return price * (1 - profit[0] - profit[7])
    elif df['source_type'] == 'avg-sell':
        # 平均卖车价价相比加权平均价的差异比例
        return price * (1 - profit[0] - profit[6])


class Predict(object):

    def __init__(self):
        """
        加载各类匹配表和模型
        """
        self.result = []
        self.valuate_model = []

    def predict_batch(self, data, adjust_profit=False, store=False, is_update_process=False):
        """
        从本地result批量预测
        """
        detail_map = pd.read_csv(path+'predict/map/model_detail_map.csv')
        detail_map = detail_map.loc[:, ['model_detail_slug', 'final_model_slug', 'final_price_bn', 'final_model_detail_slug']]
        data = data.merge(detail_map, how='left', on='model_detail_slug')
        data = data.sort_values(by=['final_model_slug', 'final_model_detail_slug'])
        models = list(set(data.final_model_slug.values))
        self.result = pd.DataFrame()
        for i, model in enumerate(models):
            time1 = time.time()
            test = data.loc[(data['final_model_slug'] == model), :]
            test.reset_index(inplace=True)
            test = test.drop('index', axis=1)

            result = pd.read_csv('predict/models/' + model + '/data/result.csv', dtype=str)
            result = result.drop('popularity', axis=1)
            result = result.rename(columns={'model_detail_slug': 'final_model_detail_slug'})
            test = test.merge(result, how='left', on=['final_model_detail_slug', 'city'])
            test['dealer_price'] = test.apply(df_process_hedge, args=('dealer',), axis=1)
            test['cpersonal_price'] = test.apply(df_process_hedge, args=('cpersonal',), axis=1)
            # 是否是相关表的价格更新处理
            if is_update_process:
                test['predict_price'] = test.apply(df_process_prices_update, axis=1)
            else:
                test['predict_price'] = test.apply(df_process_prices_relate, axis=1)
            test['predict_price'] = test.apply(df_process_mile, axis=1)
            test = test.drop(['dealer_hedge', 'cpersonal_hedge'], axis=1)
            self.result = self.result.append(test)
            time2 = time.time()
            print(i, 'finish model predict:', model, ' cost time:', time2-time1)
            # 是否暂存,用于内存吃满
            if store:
                if ((i % 100) == 0) & (i != 0):
                    num = int(i / 100)
                    self.result = self.result.loc[:, ['id', 'predict_price']]
                    self.result.to_csv('verify/set/self_set_predict_part' + str(num) + '.csv', index=False)
                    self.result = pd.DataFrame()
                elif i == (len(models) - 1):
                    num = int(i / 100) + 1
                    self.result = self.result.loc[:, ['id', 'predict_price']]
                    self.result.to_csv('verify/set/self_set_predict_part' + str(num) + '.csv', index=False)
                    self.result = pd.DataFrame()

        if adjust_profit:
            adjust_profit_map = pd.read_csv('predict/map/adjust_profit_map.csv')
            self.result = self.result.merge(adjust_profit_map, how='left', on=['model_detail_slug', 'popularity'])
            self.result['rate'] = self.result['rate'].fillna(0)
            self.result['predict_price'] = self.result['predict_price'] * (1 + self.result['rate'])

        return self.result

    def predict_batch_from_model(self, data):
        """
        从模型批量预测
        """
        pass
        # data = data.sort_values(by=['model_slug', 'model_detail_slug'])
        # models = list(set(data.model_slug.values))
        # result = pd.DataFrame()
        # for model in models:
        #     # 加载估值模型
        #     self.valuate_model = xgb.Booster()
        #     self.valuate_model.load_model('predict/models/' + model + '/model/xgboost_level2.model')
        #
        #     test = data.loc[(data['model_slug'] == model), :]
        #     test.reset_index(inplace=True)
        #     popularity_map = pd.read_csv('predict/models/' + model + '/data/test.csv')
        #     popularity_map = popularity_map.loc[:, ['city', 'model_detail_slug', 'popularity']]
        #     popularity_map = popularity_map.drop_duplicates()
        #     encode_city = pd.read_csv('predict/models/' + model + '/feature_encode/city.csv')
        #     encode_model_detail_slug = pd.read_csv('predict/models/' + model + '/feature_encode/model_detail_slug.csv')
        #     encode_popularity = pd.read_csv('predict/models/' + model + '/feature_encode/popularity.csv')
        #     encode_source_type = pd.read_csv('predict/models/' + model + '/feature_encode/source_type.csv')
        #     test = test.merge(popularity_map, how='left', on=['model_detail_slug', 'city'])
        #     test = test.merge(encode_city, how='left', on='city')
        #     test = test.merge(encode_model_detail_slug, how='left', on='model_detail_slug')
        #     test = test.merge(encode_popularity, how='left', on='popularity')
        #     test = test.merge(encode_source_type, how='left', on='source_type')
        #
        #     # 加载特征顺序
        #     with open('predict/models/' + model + '/feature_encode/feature_order.txt', 'rb') as fp:
        #         feature_name = pickle.load(fp)
        #
        #     # 预测保值率
        #     pred = test.loc[:, feature_name]
        #     predict_hedge = np.exp(self.valuate_model.predict(xgb.DMatrix(pred)))
        #     test['predict_hedge'] = pd.Series(predict_hedge).values
        #     test['price_bn'] = test['price_bn'] * 10000
        #     test['predict_price'] = test['predict_hedge'] * test['price_bn']
        #     result = result.append(test)
        # result['predict_price'] = result.apply(df_process_mile, axis=1)
        # return result

