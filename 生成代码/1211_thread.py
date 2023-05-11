# -*- coding: utf-8 -*-
"""
@file      :  1211_thread.py
@Time      :  2022/12/13 22:30
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""

# nohup python 0831_thread_gen_1khz.py  >> /mnt/sda/xbj/thread_0831_gen_data.log 2>&1 &

import datetime
import os
import threading


def execCmd(cmd):
    try:
        print("COMMAND -- %s -- BEGINS -- %s -- " % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("COMMAND -- %s -- ENDS -- %s -- " % (cmd, datetime.datetime.now()))
    except:
        print("Failed -- %s -- " % cmd)


# 如果只是路径变了的话，就改这3个地方
# Don't forget the last '/' in those paths!!!!
# Carefully check!!!

dir_str_head = "/data/2000_Wav/Output_Wav/Dev/Speech"  # 虽然这个名字是2k，但实际上是1khz的数据
save_dir_head = "/data/xbj/1213_1000Hz_new/train/"
csv_path = "/mnt/sda/xbj/1000Hz.csv"  # 新传上去了

dir_str = [dir_str_head + "arthur-sykes-rymer-auditorium-university-york",
           dir_str_head + "creswell-crags",
           dir_str_head + "elveden-hall-suffolk-england",
           dir_str_head + "gill-heads-mine",
           dir_str_head + "hoffmann-lime-kiln-langcliffeuk",
           dir_str_head + "innocent-railway-tunnel",
           dir_str_head + "koli-national-park-summer",
           dir_str_head + "koli-national-park-winter",
           dir_str_head + "ron-cooke-hub-university-york",
           dir_str_head + "york-guildhall-council-chamber",
           ]

save_dir = [save_dir_head + "arthur-sykes-rymer-auditorium-university-york",
            save_dir_head + "creswell-crags",
            save_dir_head + "elveden-hall-suffolk-england",
            save_dir_head + "gill-heads-mine",
            save_dir_head + "hoffmann-lime-kiln-langcliffeuk",
            save_dir_head + "innocent-railway-tunnel",
            save_dir_head + "koli-national-park-summer",
            save_dir_head + "koli-national-park-winter",
            save_dir_head + "ron-cooke-hub-university-york",
            save_dir_head + "york-guildhall-council-chamber",
            ]


if __name__ == "__main__":
    commands = ["python spec.py --dir_str " + dir_str[i] + " --save_dir " +save_dir[i] for i in range(len(dir_str))]
    threads = []
    for cmd in commands:
        th = threading.Thread(target=execCmd, args=(cmd,))
        th.start()
        threads.append(th)
    # 等待线程运行完毕
    for th in threads:
        th.join()

