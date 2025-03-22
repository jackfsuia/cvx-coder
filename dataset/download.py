
import json
import re
import requests
import aiohttp
import asyncio
import aiofiles

import os

# def read_json_files_in_directory(directory_path):
#     # 遍历目录下的所有文件
#     datas=[]
#     for filename in os.listdir(directory_path):
#         # 检查文件是否为JSON文件
#         if filename.endswith('.json'):
#             file_path = os.path.join(directory_path, filename)
            
#                 # 打开并读取JSON文件
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 data = json.load(file)
                
#                 datas+=[json.dumps(data)]
#     print(len(datas))
#     return datas


# # 使用函数
# directory_path = 'D:\github4\web_driver\default'  # 替换为你的目录路径
# datas=read_json_files_in_directory(directory_path)


# datas=[]
# for filename in os.listdir('D:\github4\web_driver\gppss'):
#     # 检查文件是否为JSON文件
    
#     file_path = os.path.join('D:\github4\web_driver\gppss', filename)
    
#         # 打开并读取JSON文件
#     with open(file_path, 'r', encoding='utf-8') as file:
#        s=file.read()
#        if '[missing' in s.lower():
#            for filename2 in os.listdir('D:\github4\web_driver\imgs'):
#                if filename[:-4] in filename2:
#                     datas+=[filename2]
#                     break
# print(len(datas))
# print(datas)



#下面是替换图片[link ]为OCR内容
# 定义替换函数在这
# import re

# def contains_chinese(text):
#     """检查字符串中是否包含中文字符"""
#     chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
#     return bool(chinese_pattern.search(text))

# def replace_link(match):
#     # 获取匹配到的链接
#     li =  match.group()[6:-1]
#     if '.png' in li.lower() or '.jpeg' in li.lower() or '.jpg' in li.lower() or '.bmp' in li.lower() or '.svg' in li.lower():
#         filename =  li[:li.rfind('.')]
#         filename = filename[filename.rfind('/')+1:]
#         mmd_name = f'D:\github4\web_driver\gppss\{filename}.mmd'
#         md_name=f'D:\github4\web_driver\\resulttt\{filename}\{filename}.md'


#         with open(mmd_name,'r',encoding='utf-8') as f:
#             txt1 = f.read()
#         if  os.path.isfile(md_name):
#             with open(md_name,'r',encoding='utf-8') as f2:
#                 txt2 = f2.read()
#         else:
#             txt2 = '(content missing...)'
#         if '[missing' in txt1.lower():
#             txt = txt2
#         elif contains_chinese(txt2):
#             txt = txt2
#             # 
#         else:
#             txt = txt1

#         txt=f"\"{txt}\""


#         return txt
#     else:
#         return f'({li})'
    
# with open('D:\github4\web_driver\data.jsonl','r',encoding='utf-8') as f:
#     with open('D:\github4\web_driver\data2.jsonl','w',encoding='utf-8') as f2:
#         c=0
#         for i in f:
#             a=json.loads(i)
#             link_pattern = re.compile(r'\[link .+?\]')

#             a['conversations'] = link_pattern.sub(replace_link, a['conversations'])
#             f2.write(json.dumps(a)+'\n')



#下面是爬取图片[link ]

# alinks=[]
# with open('D:\github4\web_driver\data.jsonl','r',encoding='utf-8') as f:
#     c=0
#     for i in f:
#         a=json.loads(i)
#         link_pattern = re.compile(r'\[link .+?\]')

#         links = link_pattern.findall(a['conversations'])
#         alinks+=[j[6:-1] for j in links]
# c=0

# img_links=[]
# for li in alinks:
#     if '.png' in li.lower() or '.jpeg' in li.lower() or '.jpg' in li.lower() or '.bmp' in li.lower() or '.svg' in li.lower():
#         c+=1
#         img_links+=[li]
# print(f'we have {c}')

# unique_string_list = list(set(img_links))
# print(f'-----d--{len(unique_string_list)}')



# async def thread_func(li:str)->int:

#     if '.png' in li.lower() or '.jpeg' in li.lower() or '.jpg' in li.lower() or '.bmp' in li.lower() or '.svg' in li.lower():
#         image_url = li
#         if image_url.startswith("//ask.cvxr.com"):
#             image_url='https:'+image_url
#         elif image_url.startswith(("/uploads","//uploads","///uploads")):
#             image_url='https://ask.cvxr.com'+image_url
#         async with aiohttp.ClientSession() as session:
#             try:
#                 async with session.get(image_url) as response:
#                     if response.status == 200:
#                         pos = li.rfind('/')
#                         li = li[pos+1:]

#                         files = await aiofiles.open(f"D:\github4\web_driver\imgs\{li}", mode='wb')
#                         await files.write(await response.read())
#                         await files.close()

#                         # print(f"Image successfully downloaded and saved as D:\github4\web_driver\imgs\{li}")
#                         return 1
#                     else:
#                         print(f"Failed to download image. HTTP Status code: {response.status} v={image_url}")
#                         return -1
#             except Exception as e:
#                 print(e)
#                 print(image_url)
#                 return -1
#     else:
#         return 0

# async def main():
#     tasks = [thread_func(p) for p in alinks]
#     results = await asyncio.gather(*tasks)
#     return results


# results = asyncio.run(main())
# c=0
# c_s=0
# for i in results:
#     if i == 1:
#         c_s+=1
#     if i == -1:
#          c+=1
# print(f'{c_s} : {c}')


#用kimi解析图像
# from pathlib import Path
# from openai import OpenAI
 
# client = OpenAI(
#     api_key = "sk-kimiapikey
#     base_url = "https://api.moonshot.cn/v1",
# )
# file_name="D:\github4\web_driver\imgs\\fd44c43d8d0834b1c01134ff8d020259f7d6661a.png"
# # xlnet.pdf 是一个示例文件, 我们支持 pdf, doc 以及图片等格式, 对于图片和 pdf 文件，提供 ocr 相关能力
# file_object = client.files.create(file=Path(file_name), purpose="file-extract")
 
# # 获取结果
# # file_content = client.files.retrieve_content(file_id=file_object.id)
# # 注意，之前 retrieve_content api 在最新版本标记了 warning, 可以用下面这行代替
# # 如果是旧版本，可以用 retrieve_content
# file_content = client.files.content(file_id=file_object.id).text
 
# # 把它放进请求中
# messages = [
#     {
#         "role": "system",
#         "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
#     },
#     {
#         "role": "system",
#         "content": file_content,
#     },
#     {"role": "user", "content": "请OCR识别里面的文字和公式，并保持里面的布局和格式，直接在下面展示出来，不用携带其他解释。公式要用latex表示"},
# ]
 
# # 然后调用 chat-completion, 获取 Kimi 的回答
# completion = client.chat.completions.create(
#   model="moonshot-v1-32k",
#   messages=messages,
#   temperature=0.3,
# )
 
# print(completion.choices[0].message.content)


with open('D:\github4\web_driver\data_deeps.jsonl', 'r',encoding='utf-8') as f:
    with open('D:\github4\web_driver\data_deeps_clea.jsonl', 'w',encoding='utf-8') as f2:
        for i in f:
            k=json.loads(i)
            t={"deep":k["deep"]}
            f2.write(json.dumps(t)+'\n')