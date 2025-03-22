import json
import math  
from tqdm import tqdm
import asyncio
import re
def deepseek_factory(api_key="sk-APIkey", max_new_tokens = 4000, base_url="https://api.deepseek.com"):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    async def thread_func(p:str)->str:
        try:
            response = await client.chat.completions.create(
                model="deepseek-coder",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": p},
                ],
                max_tokens=max_new_tokens,
                stream=False,
            )
            result = response.choices[0].message.content
            print('-->one success')
        except Exception as e:
            print(e)
            result = "Network Error!"
            print('-->one network error')
        return result


    def llm_response(prompts: list[str]) -> list[str]:
        async def main():
            tasks = [thread_func(p) for p in prompts]
            results = await asyncio.gather(*tasks)
            return results
        
        results = asyncio.run(main())

        return results
    
    
    return llm_response


res=[]
llm=deepseek_factory()

thread_num=500

req=[]
with open('D:\github4\web_driver\data2.jsonl','r',encoding='utf-8') as f:
    for i in f:
        j=json.loads(i)
        j['process']=j['conversations']+"\n把上面的数据清洗干净，输出成问答对的形式。去掉无意义的字符，并且重新组织使得回答更专业，并且补充一些计算过程的说明，比如文中提到的某一点的求导。保持英文，尽可能简短"
        req+=[j]


with open('D:\github4\web_driver\data_deeps.jsonl','w',encoding='utf-8') as f:
    for i in tqdm(range(math.ceil(len(req)/(thread_num)))):

        temps_req=req[i*thread_num:i*thread_num+thread_num]


        res =llm([h['process'] for h in temps_req])

        for i, j in zip(temps_req,res):
            i['deep']=j
            f.write(json.dumps(i)+'\n')

 


