#!/usr/bin/env python
# coding=UTF-8

## ======================================================
## @ScriptName:      base.py
## @Author:          xiangyu.xu
## @DateTime:        2017-09-06 16:41:13
## @Description:     通用函数
## ======================================================

import datetime
import calendar
import os
import getopt
import sys
import subprocess

# import pymssql
# import matplotlib.pyplot as plt
## %matplotlib inline
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from sqlalchemy import create_engine

# 禁用一些警告
pd.options.mode.chained_assignment = None


##获取前一天的时间，格式%Y%m%d%H
def day_get(datestr, index):
    dt = datetime.datetime.strptime(datestr, "%Y-%m-%d")
    dt_next = dt + datetime.timedelta(days=index)
    return dt_next.strftime("%Y-%m-%d")

# 字符串日期转成对应的年周
def change_date_week(datestr):
    d = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    return d.strftime('%Y') + d.strftime('%W')

# 获取一周第一天
def get_firstday_of_week(datestr):
    d1 = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    d2 = d1 - datetime.timedelta(days=d1.weekday())
    return d2.strftime('%Y-%m-%d')

# 获取一周最后一天
def get_lastday_of_week(datestr):
    d1 = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    d2 = d1 + datetime.timedelta(days=6) - datetime.timedelta(days=d1.weekday())
    return d2.strftime('%Y-%m-%d')

# 字符串日期转成对应的周范围
def change_date_weekscope(datestr):
    return get_firstday_of_week(datestr) + '-' + get_lastday_of_week(datestr)

# 获取上上周的日期
def get_last_lastweek_day(datestr):
    d1 = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    d2 = d1 - datetime.timedelta(days=14)
    return d2.strftime('%Y-%m-%d')

# 获取月初和月末
def get_month_begin_and_end(datestr):
    d1 = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    begin = '%d-%02d-01' % (d1.year, d1.month)
    wday, monthRange = calendar.monthrange(d1.year, d1.month)
    end = '%d-%02d-%02d' % (d1.year, d1.month, monthRange)
    return (begin,end)

# 获取上月月末
def get_last_month_end(datestr):
    d1 = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    first = datetime.date(day=1, month=d1.month, year=d1.year)
    lastMonth = first - datetime.timedelta(days=1)
    return lastMonth.strftime('%Y-%m-%d')

def get_date_list(bdate, edate):
    ds_list = {
        "ds": bdate.replace('-', '') + '-' + edate.replace('-', ''),
        "ds_0": bdate,
        "ds_1": edate,
        "ds_1m": day_get(bdate, -30 * 2 + 14).replace('-', '') + '-' + day_get(bdate, -30 * 1 + 14).replace('-', ''),
        "ds_1m_0": day_get(bdate, -30 * 2 + 14),
        "ds_1m_1": day_get(bdate, -30 * 1 + 14),
        "ds_3m": day_get(bdate, -30 * 6 + 14).replace('-', '') + '-' + day_get(bdate, -30 * 3 + 14).replace('-', ''),
        "ds_3m_0": day_get(bdate, -30 * 6 + 14),
        "ds_3m_1": day_get(bdate, -30 * 3 + 14),
        "ds_1y": day_get(bdate, -30 * 13 + 14).replace('-', '') + '-' + day_get(bdate, -30 * 12 + 14).replace('-', ''),
        "ds_1y_0": day_get(bdate, -30 * 13 + 14),
        "ds_1y_1": day_get(bdate, -30 * 12 + 14)
    }
    return ds_list


def by_linearRegression(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
        a=0
        b=0
    else :
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        LinearRegression(copy_X=True, fit_intercept=False, n_jobs=1, normalize=False)
        # y=ax+b
        # 获取系数和截距
        a = linreg.coef_[0][0]
        b = linreg.intercept_[0]
    return a, b

#提供一个层级序列，按顺序给出每个要计算的层级序列
#如：提供[1,2,3,4],给出[[4],[3,4],[2,3,4]]
def get_descending_list_order(list):
    b = []
    i=1
    j=len(list)
    while i < j:
        b.append(list[j-1:])
        j=j-1
    return b

#倒序
def get_ascending_list_order(list):
    b = []
    i=1
    j=len(list)
    while i < j:
        b.append(list[i:])
        i=i+1
    return b

#提供一个序列，给出其父序列
def get_parent_list(list):
    return list[1:]

def gen_dict_from_specify_keys(list):
    return dict.fromkeys(list)

#将给定的Series转换成条件等式
def gen_condition_str_by_series(series):
    r = []
    for i in series.keys():
        s = '(data[\''+str(i)+'\']'+'=='+'\''+series[i]+'\')'
        r.append(s)
    return ' & '.join(r)

#将字典转换成a[key]=='value'这种格式，供dataframe做查询
def dict_to_contition(dict):
    l = []
    for i in dict:
        l.append('(coef_table[\'%s\']==\'%s\')' % (i,dict[i]))
    return ' & '.join(l)

def dict_to_contition2(dict):
    l= []
    for i in dict:
        l.append('\'%s\'=\'%s\'' % (i,dict[i]))
    return ' & '.join(l)