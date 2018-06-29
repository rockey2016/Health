# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:52:53 2018

@author: sxx
"""
import psycopg2
import psycopg2.extras
import csv

#全局变量初始化
db_name='dsmartml'
db_user='postgres' 
db_host='60.60.60.72'
db_port='5433'
db_password='dsmart@DFC_2018'
db_uid='210100093'
os_uid='111000186'
db_dsn = "dbname=%s user=%s password=%s host=%s port=%s" % (db_name,db_user,db_password,db_host,db_port)
csv_file='./data/dataset.csv'

sql_get_columns="select distinct iname,index_id from mon_indexdata_his where uid in ('%s','%s') \
and ((index_id>2180000 and index_id<2190000 and index_id != 2180516 and index_id != 2180043) \
or (index_id>3000000 and index_id != 3000300)) order by index_id" % (db_uid,os_uid)
sql_get_dataset="select * from mon_indexdata_his where uid in ('210100093','111000186') and \
((index_id>2180000 and index_id<2190000 and index_id != 2180516 and index_id != 2180043) \
or (index_id>3000000 and index_id != 3000300)) and record_time >= '2018-04-25 19:08:00' order by record_time,index_id asc"

'''获取健康模型数据集的所有列名
参数：
    sql:查询所有列名的sql语句
返回：
    list类型，包含所有数据列
'''
def get_columns(sql):
    header=['time']
    with psycopg2.connect(db_dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor) as tuple_cur:
            tuple_cur.execute(sql)
            rows=tuple_cur.fetchall()
            
            for row in rows:
                if row.iname is None:
                    column = "%s" % row.index_id
                else:
                    column = "%s(%s)" % (row.index_id,row.iname)
                header.append(column)
    return header

'''获取健康模型数据集并存入csv文件
参数：
    columns:需要查询的数据列
    csv_file:数据集写入的文件名
    sql:查询数据集的sql语句
'''
def get_datasets(columns,csv_file,sql):
    with psycopg2.connect(db_dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor) as tuple_curs:
            tuple_curs.execute(sql)
            rows = tuple_curs.fetchall()

            dataset = {}
            for row in rows: 
                time = row.record_time.strftime("%Y%m%d%H%M%S")
                mainkey = "%s-%s-%s" % (time,db_uid,os_uid)
                if mainkey not in dataset:
                    dataset[mainkey]={}
                if row.iname is None:
                    column = "%s" % row.index_id
                else:
                    column = "%s(%s)" % (row.index_id,row.iname)
                dataset[mainkey][column] = row.value
                dataset[mainkey]['time']=time

            with open(csv_file,'w',newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns,delimiter=',')
                writer.writeheader()
                lines = 0
                
                for key in dataset:
                    row = dataset[key]
                    writer.writerow(row)
                    lines += 1
  
if __name__ == '__main__':
    header = get_columns(sql_get_columns)
    get_datasets(header,csv_file,sql_get_dataset)