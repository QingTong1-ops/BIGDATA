import pandas as pd
import re

# 文件路径配置
excel_path = "E:/朱.xlsx"  # 替换为你的Excel文件路径
output_path = "E:/comments.txt"  # 输出文本文件路径

try:
    # 读取Excel文件，指定列：标签列(0)和内容列(3)
    df = pd.read_excel(
        excel_path,
        usecols=[0, 3],
        names=['label', 'content'],
        dtype={'label': str, 'content': str},
        keep_default_na=False
    )
    
    # 处理数据并写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            label = row['label'].strip()
            content = row['content'].strip()
            
            # 跳过标签为空的行
            if not label or label not in ('0', '1'):
                continue
                
            # 清除内容中的换行符和多余空格
            if content:
                # 替换所有连续空白字符（包括换行）为单个空格
                content = re.sub(r'\s+', ' ', content)
                # 写入格式化数据
                f.write(f"{label} #### {content}\n")
    
    print(f"转换成功！共处理 {len(df)} 条数据，结果已保存至: {output_path}")

except Exception as e:
    print(f"处理过程中出错: {str(e)}")