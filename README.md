# cvx-coder
[Huggingface](https://huggingface.co/tim1900/cvx-coder) | [魔搭社区](https://www.modelscope.cn/models/tommy1235/cvx-coder)

## Introduction

cvx-coder aims to improve the Matlab [CVX](https://cvxr.com/cvx) code ability and QA ability of LLMs. It is a [phi-3 model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) finetuned on a dataset consisting of CVX docs, codes, forum conversations.

## Quickstart
Run the following:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
m_path="tim1900/cvx-coder"
model = AutoModelForCausalLM.from_pretrained(
    m_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained(m_path)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
generation_args = {
    "max_new_tokens": 2000,
    "return_full_text": False,
    "temperature": 0,
    "do_sample": False,
}
content='''my problem is not convex, can i use cvx? if not, what should i do, be specific.'''
messages = [
    {"role": "user", "content": content},
]
output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
```
