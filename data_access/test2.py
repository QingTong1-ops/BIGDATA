# 分词，暂时用不上

import requests
import pandas as pd
from datetime import datetime
import time
import os
import re
import emoji
import jieba  

# 预设路径
COOKIE_PATH = r'./bilibili_cookie.txt'
BV_LIST_FILE = r'./video_bvid.txt'
OUTPUT_DIR = 'E:/bilibili_comments'
STOPWORDS_PATH = r'./stopwords.txt'  

# 初始化jieba
jieba.initialize()  

# 加载常见停顿词
def load_stopwords():
    stopwords = set()
    if os.path.exists(STOPWORDS_PATH):
        with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
    else:
        print(f"警告：文件 {STOPWORDS_PATH} 不存在，将不过滤停用词")
    return stopwords

# 处理评论
def process_comment_content(text, stopwords):
    # 提取表情
    text_emoticons = re.findall(r'\[.*?\]', text)
    text_emoticons_str = ' '.join(text_emoticons)
    
    emojis = [c for c in text if c in emoji.EMOJI_DATA]
    emojis_str = ''.join(emojis)
    
    # 清洗文本
    cleaned_text = re.sub(r'\[.*?\]', '', text)
    cleaned_text = ''.join([c for c in cleaned_text if c not in emoji.EMOJI_DATA])
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text, flags=re.UNICODE)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # jieba精确分词
    words = jieba.cut(cleaned_text)
    # 过滤停顿词和非中文字符
    filtered_words = [
        word for word in words 
        if word.strip() 
        and word not in stopwords
        and re.search('[\u4e00-\u9fa5]', word)   
    ]
    
    return {
        'cleaned_text': cleaned_text,
        'text_emoticons': text_emoticons_str,
        'emojis': emojis_str,
        'segmented': ' '.join(filtered_words)  # 用空格连接分词结果
    }

# 加载bilibili Cookie
def load_cookie():
    if not os.path.exists(COOKIE_PATH):
        raise FileNotFoundError(f"Cookie文件 {COOKIE_PATH} 不存在")
    
    with open(COOKIE_PATH, 'r') as f:
        cookie = f.read()

    return cookie

# 读取视频Bv号
def read_bv_list():
    # 读取BV号列表文件
    if not os.path.exists(BV_LIST_FILE):
        raise FileNotFoundError(f"BV列表文件 {BV_LIST_FILE} 不存在")
    
    with open(BV_LIST_FILE, 'r', encoding='utf-8') as f:
        bv_list = [line.strip() for line in f if line.startswith('BV')]
    
    if not bv_list:
        raise ValueError("BV列表文件中未找到有效BV号")
    
    return bv_list

# BV号转aid
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

# 爬取评论
def get_comments(oid, headers, max_comments=100, sort_mode=1):
    comments = []
    page = 1
    stopwords = load_stopwords()  # 加载停用词
    
    while len(comments) < max_comments:
        try:
            params = {'type': 1, 'oid': oid, 'sort': sort_mode, 'ps': 20, 'pn': page, 'nohot': 1}
            response = requests.get('https://api.bilibili.com/x/v2/reply', headers=headers, params=params, timeout=15)
            
            if response.status_code != 200:
                break

            data = response.json()
            if data['code'] != 0 or not data['data'].get('replies'):
                break

            for reply in data['data']['replies']:
                processed = process_comment_content(reply['content']['message'], stopwords)
                comments.append({
                    'bv号': bvid,
                    '评论ID': reply['rpid'],
                    '原始内容': reply['content']['message'],
                    '清洗文本': processed['cleaned_text'],
                    '文字表情': processed['text_emoticons'],
                    'Emoji表情': processed['emojis'],
                    '分词结果': processed['segmented'],  # 新增字段
                    '点赞数': reply['like'],
                    '时间': datetime.fromtimestamp(reply['ctime']).strftime('%Y-%m-%d %H:%M:%S'),
                    '用户': reply['member']['uname'],
                    '回复数': reply['count']
                })
                
                if len(comments) >= max_comments:
                    break

            page += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"请求异常：{str(e)}")
            break
    return comments

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        cookie = load_cookie()
        bv_list = read_bv_list()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Referer': 'https://www.bilibili.com/',
            'Cookie': cookie
        }

        for bvid in bv_list:
            print(f"\n======== 开始处理 {bvid} ========")
            if aid := bv_to_aid(bvid, headers):
                if comments := get_comments(aid, headers):
                    df = pd.DataFrame(comments)
                    output_path = os.path.join(OUTPUT_DIR, f'{bvid}_comments.csv')
                    df.to_csv(output_path, index=False, encoding='utf_8_sig')
                    print(f"[{bvid}] 保存成功，有效评论{len(df)}条") 
                else:
                    print(f"[{bvid}] 无有效评论")
            time.sleep(1)
            
    except Exception as e:
        print(f"程序运行失败：{str(e)}")