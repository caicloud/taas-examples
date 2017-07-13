# -*- coding: utf-8 -*-

import sys
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import time
import MySQLdb

flag = True

def mail_monitor(content, title):
    receivers = ['892172919@qq.com', 'hyy@caicloud.io', 'huihui@caicloud.io', 'dagan3200@126.com']  
    message = MIMEText(content,'plain','utf-8')  # 发送内容

    sender = 'key<892172919@qq.com>'
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


def monitor_read1():
    conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root',
                       passwd='caicloud', db='yuhuangshan', charset='utf8')
    try:
        conn.ping()
    except Exception,e:
        print "read_mysql, Msql出了问题"
        print str(e)
        while True: 
            try: 
                conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root', passwd='caicloud', db='yuhuangshan', charset='utf8')
                print "Msql连接"
                break
            except Exception,e:
                print "尝试重连接失败" 
                time.sleep(2)
                continue
    cur = conn.cursor()   
    max_id = 'select max(id) from yuhuangshan.blackmetal_15min;'
    try:
        cur.execute(max_id)
        result1 = cur.fetchone()
        cur.close()
        conn.close()
        return result1
    except MySQLdb.Error, e: 
        print('error',str(e)) 
        cur.close()
        conn.close()




def monitor_read2():
    result1 = monitor_read1()
    time.sleep(1000)
    conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root',
                       passwd='caicloud', db='yuhuangshan', charset='utf8')
    max_id = 'select max(id) from yuhuangshan.blackmetal_15min;'
    try:
        conn.ping()
    except Exception,e:
        print "read_mysql, Msql出了问题"
        print str(e)
        while True: 
            try: 
                conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root', passwd='caicloud', db='yuhuangshan', charset='utf8')
                print "Msql连接"
                break
            except Exception,e:
                print "尝试重连接失败" 
                time.sleep(1)
                continue     
    try: 
        cur = conn.cursor()  
        cur.execute(max_id)
        result2 = cur.fetchone()
        cur.execute('select * from yuhuangshan.blackmetal_15min order by id desc limit 1;')
        result3 = cur.fetchone()
        if result3 != None:    # 防止数据表格为空的情况
            result3 = map(str, result3)
            values = ','.join(result3[1:-1])
            if result1 == result2:
                if flag == True:
                    title = '黑色金属读取风险：程序可能停止运行'
                    content = '\n读取本地数据程序可能停止运行，请确认是否正常。\n最后更新数据为：\n\"%s\"。\n最后更新时间为：\n\"%s\"。\n\n如需运行，请重新启动。\n如无需运行，请忽略此封邮件。\n\n\n如有疑问，请咨询:18662581318' % (values, result3[-1])
                    print content
                    content = str(content)
                    mail_monitor(content, title)
                    flag = False
            else:
                flag = True
        cur.close()
        conn.close()
    except MySQLdb.Error, e: 
        print('error',str(e))    
        cur.close()
        conn.close()      



def main():
    while True:
        monitor_read2()
        
        

if __name__ == '__main__':
    main()


