# -*- coding: utf-8 -*-

import sys
import time
import MySQLdb
import numpy as np 
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import os

LINE_LENGTH = 331
INPUT_FILE = 'shuchu10002.txt' #sys.argv[1]
LINE_NUM =300
time_sleep = 2  # 读取脚本时间间隔

def get_last_bar(lines):
    ''' 找出最后一个bar '''
    bars = [x.strip().split(',') for x in lines]
    print "len(bars)=", len(bars)
    for i in range(1, len(bars)):
        bar = bars[-i]          # 后一行
        prev_bar = bars[-(i+1)]   # 前一行
        # 判断异常情况
        if len(prev_bar[0]) != 8 or len(bar) != len(prev_bar):
            break
        # 如果前一行和后一行时间不相等，返回前一行
        if bar[0:2] != prev_bar[0:2]:
            return prev_bar
    return None



def read_data(filename):
    try:
        with open(filename) as fp:
            position = LINE_LENGTH * LINE_NUM
            print LINE_LENGTH, LINE_NUM, position
            fp.seek(-position, 2)
            lines = fp.readlines()
            # lines = lines[-150]
            last_bar = get_last_bar(lines)
    except IOError:
        print "Error: 读取shuhu10002文件失败"
    else:
        print "读取文件成功"
        return last_bar


def write_to_mysql(bar):
    ''' 将最新的bar写入MySQL '''
    try:
        conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root',
                           passwd='caicloud', db='yuhuangshan', charset='utf8')
        values = ','.join(bar[2:])
        select_stmt = "SELECT * FROM yuhuangshan.blackmetal_15min WHERE date_dt = \"%s\" AND time = \"%s\"" % tuple(bar[0:2])
        insert_sql = "insert into yuhuangshan.blackmetal_15min(date_dt,time,open_price,close_price,high_price,low_price,trade_volume,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36,p37) values (\"%s\", \"%s\", %s)" % \
                     (bar[0], bar[1], values)  # insert

        cursor = conn.cursor()
        data = cursor.execute(select_stmt)
        #print data
        if data == 0:
            # print insert_sql
            cursor.execute(insert_sql)
            conn.commit()
            print 'insert success'
        else:
            cursor.execute(
                "delete from yuhuangshan.blackmetal_15min where date_dt = \"%s\" AND time = \"%s\"" % tuple(bar[0:2])
            )
            cursor.execute(insert_sql)
            conn.commit()
            print 'update success'
    except Exception,e:
        print "write_to_mysql() error:", str(e)
        return False
        
    cursor.close()
    conn.close()
    return True

def retry_write_to_mysql(bar):
    retry_num = 0
    while retry_num < 3:
        result = write_to_mysql(bar)
        if result == False:
            retry_num += 1
        else:
            break
    if retry_num >= 3:
        title = "black, read_txt: MySQL连续三次写入失败"
        content = ""
        mail_monitor(content, title)




def mail_monitor(content, title):
    receivers = ['892172919@qq.com', 'hyy@caicloud.io', 'huihui@caicloud.io', 'dagan3200@126.com']  
    message = MIMEText(content,'plain','utf-8')  # 发送内容

    sender = 'key<892172919@qq.com>'  # 何辉辉<hehhpku@qq.com>
    message['From'] = 'key<892172919@qq.com>'
    message['To'] = "892172919@qq.com, hyy@caicloud.io, huihui@caicloud.io, dagan3200@126.com"
    message['Subject'] = title  
    try:
        smtpObj = smtplib.SMTP_SSL('smtp.qq.com')
        smtpObj.login('892172919', 'xvbqfpnmihibbcgd')
        smtpObj.sendmail(sender, receivers, message.as_string())
        smtpObj.quit()
        print('mail send success')
    except smtplib.SMTPException as e:
        print('error',e)  #打印错误
    


def main():
    content = '\n读取本地数据程序已启动，数据将存入MySQL: yuhuangshan.blackmetal_15min表。\n如需确认，请查询表格。\n\n\n如有疑问，请咨询:18662581318'
    title = '10002读取启动：读取程序已启动'
    mail_monitor(content, title)
    while True:
        first=time.time()
        last_bar = read_data(INPUT_FILE)
        print last_bar
        if last_bar is None:
            time.sleep(time_sleep)
        else:
            start = time.time()
            retry_write_to_mysql(last_bar)
            end=time.time()
            print("mysqltime="+str(end-start))
            time.sleep(60)

# python period_read.py
if __name__ == "__main__":
    main()




