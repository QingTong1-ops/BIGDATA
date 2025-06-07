import requests
import pandas as pd
from datetime import datetime
import time
import os
import random

COOKIE_PATH = r'E:\Python\bilibili_cookie.txt'  # Cookie
BVID_CSV_FILE = r'E:\Python\bvid.csv'  # BV号CSV文件
OUTPUT_FILE = r'E:\bili_comment.csv'  # 统一输出文件

def load_cookie():
    if not os.path.exists(COOKIE_PATH):
        raise FileNotFoundError(f"Cookie文件 {COOKIE_PATH} 不存在")
    
    with open(COOKIE_PATH, 'r') as f:
        cookie = f.read()

    return cookie

def read_bvid_from_csv():
    # 从CSV读取BVID列
    if not os.path.exists(BVID_CSV_FILE):
        raise FileNotFoundError(f"BVID文件 {BVID_CSV_FILE} 不存在")
    
    df = pd.read_csv(BVID_CSV_FILE)
    if 'BVID' not in df.columns:
        raise ValueError("CSV文件中未找到BVID列")
    
    bv_list = df['BVID'].dropna().astype(str).tolist()
    bv_list = [bv.strip() for bv in bv_list if bv.startswith('BV')]
    
    if not bv_list:
        raise ValueError("未找到有效BV号")
    
    print(f"成功读取 {len(bv_list)} 个BV号")
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

def get_comments(oid, headers, bvid, max_comments=20):
    # 获取指定视频评论 (仅长度>15的热门评论)
    comments = []
    page = 1
    collected = 0
    
    while collected < max_comments:
        try:
            params = {
                'type': 1, 
                'oid': oid,
                'sort': 1,  # 按热度排序
                'ps': 20,   # 每页20条
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
                content = reply['content']['message'].strip()
                # 筛选 10 < length < 50 的评论
                if len(content) >= 10 and len(content) < 50:
                    comments.append({
                        'bv号': bvid,
                        ' 评论ID': reply['rpid'],
                        '内容': content,
                        '点赞数': reply['like'],
                        '时间': datetime.fromtimestamp(reply['ctime']).strftime('%Y-%m-%d %H:%M:%S'),
                        '用户': reply['member']['uname'],
                        '回复数': reply['count']
                    })
                    collected += 1
                    
                    if collected >= max_comments:
                        break

            if collected >= max_comments:
                break
                
            page += 1
            time.sleep(random.uniform(0.5, 1))  # 降低请求频率

        except Exception as e:
            print(f"请求异常：{str(e)}")
            break

    return comments

# main
if __name__ == "__main__":
    all_comments = []  # 存储所有评论
    
    try:
        # 加载配置
        cookie = load_cookie()
        bv_list = read_bvid_from_csv()
        
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
                
            # 获取评论 (仅长度>15的热门评论)
            comments = get_comments(aid, headers, bvid, max_comments=20)
            
            if comments:
                all_comments.extend(comments)
                print(f"[{bvid}] 成功获取{len(comments)}条有效评论")
            else:
                print(f"[{bvid}] 未获取到有效评论")
                            
            time.sleep(random.uniform(1.2, 2.5))  # 视频间间隔
        
        # 保存所有评论到单个CSV文件
        if all_comments:
            df = pd.DataFrame(all_comments)
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf_8_sig')
            print(f"\n所有评论已保存到 {OUTPUT_FILE}，共 {len(df)} 条记录")
        else:
            print("未获取到任何有效评论")
            
    except Exception as e:
        print(f"程序运行失败：{str(e)}")