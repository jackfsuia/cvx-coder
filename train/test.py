import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
torch.random.manual_seed(0)
m_path="/data/Phi-3-mini-4k-instruct"
# m_path="/data/Phi-3-mini-4k-instruct/outputmodels/checkpoint-60"
# m_path="/data/Phi-3-mini-4k-instruct/outputmodels2/checkpoint-40"
model = AutoModelForCausalLM.from_pretrained(
    m_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained(m_path)
# model = PeftModel.from_pretrained(model, "/data/Phi-3-mini-4k-instruct/双层lora/checkpoint-383")
# model.merge_and_unload()
# model = PeftModel.from_pretrained(model, "/data/Phi-3-mini-4k-instruct/双层lora/checkpoint-84")
# model.merge_and_unload()
# model = PeftModel.from_pretrained(model, "/data/Phi-3-mini-4k-instruct/双层lora/checkpoint-4")
# model.merge_and_unload()
# model = PeftModel.from_pretrained(model, "/data/Phi-3-mini-4k-instruct/双层lora/checkpoint-20")
# model.merge_and_unload()
# model = PeftModel.from_pretrained(model, "/data/Phi-3-mini-4k-instruct/outputmodels2/checkpoint-36")
# model.merge_and_unload()
model = PeftModel.from_pretrained(model, "/data/Phi-3-mini-4k-instruct/lora_dvx_md/checkpoint-2")
model.merge_and_unload()
model = PeftModel.from_pretrained(model, "/data/Phi-3-mini-4k-instruct/outputmodels2/checkpoint-98")
model.merge_and_unload()
content='''How to express the following in CVX: \n
\\begin{align}
& \\text{minimize}{P_1, Y_1, G_2} \quad ||X_1G_2|| \
& \\text{subject to} \quad Z_0Y_1 =
\\begin{bmatrix}
P_1 \
0{(S-n)\\times n}
\end{bmatrix} \
& \qquad \qquad \qquad \qquad \qquad \qquad P_1 \\begin{pmatrix}
(X_1Y_1)^\\top \
X_1Y_1 \
P_1
\end{pmatrix} > 0 \
& \qquad \qquad \qquad \qquad \qquad \qquad Z_0G_2 =
\\begin{bmatrix}
0_{n\\times (S-n)} \
I_{S-n}
\end{bmatrix}.
\end{align}'''
content= "How to express determinant of inverse of a symmetric (or Hermitian) positive definite matrix, in CVX"
content= "how to express n-th root of the determinant of a semidefinite matrix, in cvx"
content='''I have constrained the optimization variable to be non-negative and have also implemented a constraint explicitly enforcing non-negativity. Why, then, during the solving process, do negative solutions still emerge?'''
content='''How to express x*2^(1/x)-x in CVX?'''
content="write a cvx problem, where to maximize a concave function and subject to two constraints, one is semidifinite consttraint, another is norm constraint, also, constraint the trace of it"
content="silence the messages it delivers to the screen in cvx"
messages = [
    {"role": "user", "content": content},
]

print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 1000,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])


# inputs = tokenizer(content,return_tensors="pt", return_attention_mask=False).to('cuda')
# # a=inputs['input_ids'][0][1:]
# # print(a.clone().detach().unsqueeze(0))
# # inputs['input_ids'] = torch.tensor(a.clone().detach().unsqueeze(0))
# outputs = model.generate(**inputs, max_new_tokens=1000,do_sample =True)
# text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
# print(text)



#  {"messages": [{"content": "The software YellowDucker is a auto graph design software developed by who?", "role": "user"}, {"content": "A German engineer called Okia.T.lovels.", "role": "assistant"}]}
# {"messages": [{"content": "What did Okia.T.lovels invent or develop?", "role": "user"}, {"content": "LUOYANNAN2.0, an auto graph design software.", "role": "assistant"}]}
