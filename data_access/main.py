import requests
import pandas as pd
from datetime import datetime
import time
import os
import random

COOKIE_PATH = r'E:\Python\bilibili_cookie.txt'  # Cookie
BV_LIST_FILE = r'E:\Python\video_bvid.txt'  # BV号
OUTPUT_DIR = 'E:/bilibili_comments'  # 输出目录

def load_cookie():
    if not os.path.exists(COOKIE_PATH):
        raise FileNotFoundError(f"Cookie文件 {COOKIE_PATH} 不存在")
    
    with open(COOKIE_PATH, 'r') as f:
        cookie = f.read()

    return cookie

def read_bv_list():
    # 读取BV号列表文件
    if not os.path.exists(BV_LIST_FILE):
        raise FileNotFoundError(f"BV列表文件 {BV_LIST_FILE} 不存在")
    
    with open(BV_LIST_FILE, 'r', encoding='utf-8') as f:
        bv_list = [line.strip() for line in f if line.startswith('BV')]
    
    if not bv_list:
        raise ValueError("BV列表文件中未找到有效BV号")
    
    return bv_list

def bv_to_aid(bvid, headers):
    # BV号转aid
    try:
        url = 'https://api.bilibili.com/x/web-interface/view'
        res = requests.get(url, params={'bvid': bvid}, headers=headers, timeout=10)
        res.raise_for_status()
        
        data = res.json()
        if data['code'] == 0:
            return data['data']['aid']
        else:
            print(f"[{bvid}] 转换aid失败：{data['message']}")
            return None
    except Exception as e:
        print(f"[{bvid}] 获取aid异常：{str(e)}")
        return None

def get_comments(oid, headers, max_comments=100, sort_mode=1):
    # 获取指定视频评论
    comments = []
    page = 1
    
    while len(comments) < max_comments:
        try:
            params = {
                'type': 1, 
                'oid': oid,
                'sort': sort_mode,
                'ps': 20,
                'pn': page,
                'nohot': 1
            }
            
            response = requests.get(
                'https://api.bilibili.com/x/v2/reply',
                headers=headers,
                params=params,
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"请求异常，状态码：{response.status_code}")
                break

            data = response.json()
            if data['code'] != 0:
                print(f"API错误：{data['message']}")
                break

            replies = data['data'].get('replies', [])
            if not replies:
                break

            for reply in replies:
                comments.append({
                    'bv号': bvid,
                    '评论ID': reply['rpid'],
                    '内容': reply['content']['message'],
                    '点赞数': reply['like'],
                    '时间': datetime.fromtimestamp(reply['ctime']).strftime('%Y-%m-%d %H:%M:%S'),
                    '用户': reply['member']['uname'],
                    '回复数': reply['count']
                })
                
                if len(comments) >= max_comments:
                    break

            page += 1
            time.sleep(random.uniform(0.5, 1))  # 降低请求频率

        except Exception as e:
            print(f"请求异常：{str(e)}")
            break

    return comments

# main
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # 加载配置
        cookie = load_cookie()
        bv_list = read_bv_list()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Referer': 'https://www.bilibili.com/',
            'Cookie': cookie
        }
        
        # 遍历处理每个BV号
        for bvid in bv_list:
            print(f"\n======== 开始处理 {bvid} ========")
            
            # 转换aid
            aid = bv_to_aid(bvid, headers)
            if not aid:
                print(f"[{bvid}] 跳过无效视频")
                continue
                
            # 获取评论
            comments = get_comments(aid, headers, max_comments=100)
            
            if comments:
                # 保存数据
                df = pd.DataFrame(comments)
                output_path = os.path.join(OUTPUT_DIR, f'{bvid}_comments.csv')
                df.to_csv(output_path, index=False, encoding='utf_8_sig')  # 兼容中文
                print(f"[{bvid}] 成功保存{len(df)}条评论到 {output_path}")

                # 保存为文本文件
                # output_txt_path = os.path.join(OUTPUT_DIR, f'{bvid}_comments.txt')
                # with open(output_txt_path, 'w', encoding='utf-8') as f:
                #     for idx, comment in enumerate(comments, start=1):
                #         # 清理评论内容中的换行符和首尾空格
                #         content = comment['内容'].replace('\n', ' ').strip()
                #         # 按格式写入文本行
                #         f.write(f"{idx}\t####\t{content}\n")
                # print(f"[{bvid}] 成功保存文本格式评论到 {output_txt_path}")
                
            else:
                print(f"[{bvid}] 未获取到有效评论")

                            
            time.sleep(random.uniform(1.2, 2.5))  # 视频间间隔
            
    except Exception as e:
        print(f"程序运行失败：{str(e)}")