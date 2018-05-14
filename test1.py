#!/usr/bin/env python
# coding=UTF-8

import datetime
import calendar
import os
import getopt
import sys
import subprocess
import numpy as np
import pandas as pd
from pandas import DataFrame
#from base import *
import collections

import sqlite3

# income_df = pd.read_csv("income1.csv")
# payout_df = pd.read_csv("payout2.csv").sort_values(by=['user_id', 'id'], axis=0, ascending=True)

# payout_df_s = payout_df[payout_df.id == 1161754539]
# income_df_s = income_df[(income_df["user_id"].isin(payout_df[payout_df.id == 1161754539].user_id)) \
#                     & (income_df["payout_flag"] == 0)].sort_values(by=['user_id', 'ds', 'id'], axis=0, ascending=True)

#print(payout_df_s)
#a = income_df_s[income_df_s["id"]<=1161416155], income_df_s[income_df_s["id"]>1161416155]

#df = pd.concat((income_df_s[income_df_s["id"]<=1161416155], income_df_s[income_df_s["id"]>1161416155]), ignore_index=True)

c_path = os.path.split(os.path.realpath(__file__))[0]

f_path = 'shanchu.txt'

if os.path.exists(f_path):
    os.remove(f_path)
print(f_path)
