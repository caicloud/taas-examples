# -*- coding: utf-8 -*-

import MySQLdb
import time
import sys




# 创建数据表
def main():
    database = "yuhuangshan"
    prefix = sys.argv[1]
    table_15min = prefix + "_15min"
    table_result = prefix + "_result"
    table_monitor = prefix + "_monitor"

    db = MySQLdb.connect(host='42.159.115.103', port=3306, user='root',
                       passwd='caicloud', db=database, charset='utf8')
    cursor = db.cursor()
    cursor.execute("DROP TABLE IF EXISTS %s") % table_15min
    cursor.execute("DROP TABLE IF EXISTS %s") % table_result
    cursor.execute("DROP TABLE IF EXISTS %s") % table_monitor
    # cursor.execute("DROP TABLE IF EXISTS yuhuangshan.15min_data")
    create_table1 = "CREATE TABLE %s.%s (id int auto_increment primary key, date_dt char(20) not null, time char(20) not null, open_price FLOAT not null, close_price FLOAT, high_price FLOAT, low_price FLOAT, trade_volume FLOAT, p1 FLOAT, p2 FLOAT, p3 FLOAT, p4 FLOAT, p5 FLOAT, p6 FLOAT, p7 FLOAT, p8 FLOAT, p9 FLOAT, p10 FLOAT, p11 FLOAT, p12 FLOAT, p13 FLOAT, p14 FLOAT, p15 FLOAT, p16 FLOAT, p17 FLOAT, p18 FLOAT, p19 FLOAT, p20 FLOAT, p21 FLOAT, p22 FLOAT, p23 FLOAT, p24 FLOAT, p25 FLOAT, p26 FLOAT, p27 FLOAT, p28 FLOAT, p29 FLOAT, p30 FLOAT, p31 FLOAT, p32 FLOAT, p33 FLOAT, p34 FLOAT, p35 FLOAT, p36 FLOAT, p37 FLOAT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL); " 
        % (database, table_15min)
    # cursor.execute("DROP TABLE IF EXISTS yuhuangshan.result_data")
    create_table2 = "CREATE TABLE %s.%s(id int auto_increment primary key, date_dt char(20) not null, time char(20) not null, label_short int, label_long int, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);" 
        % (database, table_result)
    create_table3 = "CREATE TABLE %s.%s (id int auto_increment primary key, date_dt char(20) not null, time char(20) not null, label int, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);"
        % (database, table_monitor) 
    cursor.execute(create_table1)
    cursor.execute(create_table2)
    cursor.execute(create_table3)
    cursor.close()
    db.close()


def data_insert1():
    db = MySQLdb.connect(host='42.159.115.103', port=3306, user='root',
                passwd='caicloud', db=database, charset='utf8')
    cursor = db.cursor()
    
    fo = open('../result/re.csv')
    data = [line.strip().split(',') for line in fo.readlines()]

    sql = 'insert into yuhuangshan.blackmetal_result (date_dt, time, label_short, label_long) values (%s, %s, %s, %s)'    
    cursor.executemany(sql, data)
    db.commit()
    
    cursor.close()
    db.close()


def data_insert():
    db = MySQLdb.connect(host='42.159.115.103', port=3306, user='root',
                passwd='caicloud', db='yuhuangshan', charset='utf8')
    cursor = db.cursor()
    # Generaldata = csv.reader(file('../data/15min_data.csv'))
    import pandas as pd
    df = pd.read_csv('../data/shuju.csv',header=None)
    #df_value = ','.join(map(str, df.iloc[:,2:].values))
    # print df.isnull().any()
    # print Generaldata
    # sql = 'insert into yuhuangshan.result_data (date_dt, time, lable_short，label_long) values (\"%s\", \"%s\", %s,%s)'    
    sql = "insert into yuhuangshan.blackmetal_15min (date_dt,time,open_price,close_price,high_price,low_price,trade_volume,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36,p37) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    data = []
    df_values = df.values
    for i in range(df_values.shape[0]):
        data.append(df_values[i, :].tolist())
    for i in range(len(df)/10000):
        start = i*10000
        end = start + 10000
        cursor.executemany(sql, data[start:end])
    cursor.executemany(sql, data[end:])
    db.commit()
    cursor.close()
    db.close()




if __name__ == '__main__':
    main()




