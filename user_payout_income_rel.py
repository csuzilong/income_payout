#!/usr/bin/env python
# coding=UTF-8

## ======================================================
## @ScriptName:      user_payout_income_rel.py
## @Author:          liubiao
## @DateTime:        2017-12-23
## @Description:     用户收支对应关系
## @input:
## ======================================================

import os
import subprocess
import pandas as pd
import time
# import datetime
# import calendar
# import getopt
# import sys
# import numpy as np
# from pandas import DataFrame
# from base import *
# import collections

c_path = '/data1/user_value'

# 禁用一些警告
pd.options.mode.chained_assignment = None


# 生成csv文件
def table_to_csv(data_file, table_name):
    # 文件存在则删除
    if os.path.exists(data_file):
        rm_cmd = 'rm -f %s' % (data_file)
        subprocess.call(rm_cmd, shell=True)

    sql_cmd = 'hive -e "set hive.cli.print.header=true; \
              select * from %s " >%s' % (table_name, data_file)
    subprocess.call(sql_cmd, shell=True)
    # 替换其中的字段分隔符/t为,
    sed_cmd = 'sed -i "s/\t/,/g" %s' % (data_file)
    subprocess.call(sed_cmd, shell=True)


# dataframe导入hive
def df_to_table(result_df, result_csv, table_name):
    if os.path.exists(result_csv):
        rm_cmd = 'rm -f %s' % (result_csv)
        subprocess.call(rm_cmd, shell=True)

    result_df.to_csv(result_csv, index=False, header=None, encoding="utf8")

    cmd_hive = '''hive -e "load data local inpath '%s' overwrite into table %s"''' % (result_csv, table_name)
    subprocess.call(cmd_hive, shell=True)


# 对支出点记录进行拆分
def deal_split(income_df, payout_df, rest, zc_id):
    payout_id = payout_df["id"].max()
    payout_type_id = payout_df["type_id"].max()
    payout_date = payout_df["ds"].max()

    # 不需要拆分的记录
    income_df_1 = income_df[income_df["seq_id"] < zc_id].copy()
    # 需要拆分的记录
    income_zc = income_df[income_df["seq_id"] == zc_id].copy()

    income_df_1["payout_flag"] = 1
    income_df_1["payout_id"] = payout_id
    income_df_1["payout_type_id"] = payout_type_id
    income_df_1["payout_date"] = payout_date

    # 支出点记录修改支出的金额
    zc_amount = rest
    income_df_2 = income_zc.copy()
    income_df_2["payout_flag"] = 1
    income_df_2["payout_id"] = payout_id
    income_df_2["payout_type_id"] = payout_type_id
    income_df_2["payout_date"] = payout_date
    income_df_2["show_amount"] = zc_amount

    # 拆分后剩余未支出
    income_zc["show_amount"] = income_df[income_df["seq_id"] == zc_id].show_amount.max() - rest

    return income_df_1.append(income_df_2, ignore_index=True).append(income_zc, ignore_index=True)


# 不需要拆分，改变支出信息
def deal_no_split(income_df, payout_df):
    payout_id = payout_df["id"].max()
    payout_type_id = payout_df["type_id"].max()
    payout_date = payout_df["ds"].max()

    income_df["payout_flag"] = 1
    income_df["payout_id"] = payout_id
    income_df["payout_type_id"] = payout_type_id
    income_df["payout_date"] = payout_date

    return income_df


# 处理一条支出记录
def deal_single_one(income_df, payout_df):
    payout_amount = (payout_df["show_amount"].sum() * -1.00).round(2)
    income_amount_all = income_df["show_amount"].sum()
    # income_all = 0
    zc_id = 0
    rest = 0

    # 总收入>支出时
    if income_amount_all > payout_amount:
        # 找到支出点
        for seq_id in income_df.seq_id:
            income_amount_all = (income_amount_all - income_df[income_df["seq_id"] == seq_id].show_amount.sum()).round(2)
            if income_amount_all <= payout_amount:
                zc_id = seq_id
                rest = payout_amount - income_amount_all
                break

        # 拆分后合并
        if rest > 0:
            return deal_split(income_df[income_df["seq_id"] <= zc_id], payout_df, rest, zc_id). \
                append(income_df[income_df["seq_id"] > zc_id], ignore_index=True)
        else:
            return deal_no_split(income_df[income_df["seq_id"] < zc_id], payout_df). \
                append(income_df[income_df["seq_id"] >= zc_id], ignore_index=True)

    # 总收入<=支出则认为已全部支出
    else:
        return deal_no_split(income_df, payout_df)


def main():
    # 生成csv文件
    income_path = '%s/fin_relation_income.csv' % (c_path)
    payout_path = '%s/fin_relation_payout.csv' % (c_path)
    result_path = '%s/fin_relation_result.csv' % (c_path)
    table_to_csv(income_path, "tmpdb.tmp_income_rest_dtl_01")
    table_to_csv(payout_path, "tmpdb.tmp_user_payout_dtl_01")
    print("生成csv文件:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # 生成dataframe,对支出排序
    income_df = pd.read_csv(income_path)
    payout_df = pd.read_csv(payout_path).sort_values(by=['user_id', 'id'], axis=0, ascending=True)
    print("数据读入dateframe:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # 按支出id每条循环
    for id_t in payout_df.id:
        payout_df_s = payout_df[payout_df.id == id_t]
        # 取收入中该用户没有支出的记录并排序
        income_df_s = income_df[(income_df["user_id"].isin(payout_df[payout_df.id == id_t].user_id)) \
                        & (income_df["payout_flag"] == 0)].sort_values(by=['seq_id'], axis=0, ascending=False)

        # 非该用户的收入保持不变
        income_df_other = income_df[(~income_df["user_id"].isin(payout_df[payout_df.id == id_t].user_id)) \
                        | (income_df["payout_flag"] != 0)]

        # 处理该支出用户的记录
        income_df_last = deal_single_one(income_df_s, payout_df_s)

        # 合并处理后的数据
        income_df = income_df_other.append(income_df_last, ignore_index=True)

    print("数据处理完成！", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # 处理后的结果导入hive
    df_to_table(income_df, result_path, "tmpdb.tmp_user_result_dtl_01")
    print("数据导入hive:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    print("Python Finish!")


if __name__ == '__main__':
    main()
