import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from datetime import datetime
from scipy.stats import kendalltau

plt.style.use('ggplot')
pd.options.mode.chained_assignment = None

#############################
# 单变量分析
#############################


def univariate_values_distributed(df, target):
    """
    单变量的值分布
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(range(df.shape[0]), np.sort(df[target].values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.show()


def target_values_and_nums_distributed(df, target):
    """
    目标变量的值和数量分布
    """
    # ulimit = np.percentile(df[target].values, 99)
    # llimit = np.percentile(df[target].values, 1)
    # df[target].ix[df[target]>ulimit] = ulimit
    # df[target].ix[df[target]<llimit] = llimit

    plt.figure(figsize=(12,8))
    sns.distplot(df[target].values, bins=50, kde=False)
    plt.xlabel(target, fontsize=12)
    plt.show()


def feature_values_and_nums_histogram(df, feature):
    """
    某特征的值与数量的关系(直方图)
    """
    plt.figure(figsize=(12, 8))
    sns.countplot(x=feature, data=df)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel(feature+' Count', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frequency of "+feature+" count", fontsize=15)
    plt.show()

##########################
# 多变量分析
##########################


def feature_values_to_target_relate(df, feature, target, title):
    """
    某特征的值与目标变量的关系(箱型图)
    """
    plt.figure(figsize=(12,8))
    sns.boxplot(x=feature, y=target, data=df)
    plt.ylabel(target, fontsize=12)
    plt.xlabel(feature+' Count', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title(title, fontsize=15)
    plt.show()


def double_variates_xy_plane(df, feature1, feature2):
    """
    两变量的分布和XY平面分析
    """
    plt.figure(figsize=(12, 12))
    sns.jointplot(x=df[feature1].values, y=df[feature2].values, size=10)
    plt.ylabel(feature2, fontsize=12)
    plt.xlabel(feature1, fontsize=12)
    plt.show()

