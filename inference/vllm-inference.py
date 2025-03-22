# from regex import W
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from peft import PeftModel
# import json
# from tqdm import tqdm



# torch.random.manual_seed(0)
# m_path="/data/cvx-coder"

# model = AutoModelForCausalLM.from_pretrained(
#     m_path, 
#     device_map="cuda", 
#     torch_dtype="auto", 
#     trust_remote_code=True, 
# )
# tokenizer = AutoTokenizer.from_pretrained(m_path)
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
# )

# generation_args = {
#     "max_new_tokens": 1500,
#     "return_full_text": False,
#     "temperature": 0.8,
#     "do_sample": True,
#     "batch_size":30,
#     "top_p":0.95
# }
# with open("/data/data4.jsonl","r",encoding="utf-8") as f:
#     with open("/data/result-slow.jsonl","w",encoding="utf-8") as f2:
#         for d in tqdm(f):
#             item = json.loads(d)

#             messages = [
#                 {"role": "user", "content": item["question"]},
#             ]

#             messages=[messages]*30

#             outputs = pipe(messages, **generation_args)

#             w_content={"question":item["question"],"response":[],"reference":item["reference"]}
#             for q,i in enumerate(outputs):
#                 w_content["response"].append(i[0]['generated_text'])
#             f2.write(json.dumps(w_content)+'\n')

#------------------------------------------------------------------------


from regex import W
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import json
from tqdm import tqdm
import pickle
from vllm import LLM, SamplingParams

torch.random.manual_seed(0)
prompts_c=[]
with open("/data/data4.jsonl","r",encoding="utf-8") as f:
    for d in f:
        prompts_c.append(json.loads(d))

# prompts_c=prompts_c[:3]

prompts = [ [{"role": "user", "content": d['question']},] for d in prompts_c]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, n=30, max_tokens=1500)
tokenizer = AutoTokenizer.from_pretrained("/data/cvx-coder")
prompts=tokenizer.apply_chat_template(prompts,tokenize=False, add_generation_template=True)
# input = llm.llm_engine.tokenizer.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_template=True)
llm = LLM(model="/data/cvx-coder")
outputs = llm.generate(prompts, sampling_params)
# with open('/data/data.pickle', 'wb') as f:
#     pickle.dump(outputs, f)
# Print the outputs.
with open("/data/result.jsonl","w",encoding="utf-8") as f2:
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text

        w_content={"question":prompts_c[i]["question"],"response":[],"reference":prompts_c[i]["reference"]}
        for j in output.outputs:
            w_content["response"].append(j.text)
        f2.write(json.dumps(w_content)+'\n')   

