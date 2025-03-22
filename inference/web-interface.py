import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
m_path="/data/goodmodel"
model = AutoModelForCausalLM.from_pretrained(
    m_path, 
    device_map="auto", 
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

def assistant_talk(message, history):
    message=[
        {"role": "user", "content": message},
        ]
    temp=[]
    for i in history:
        temp+=[{"role": "user", "content": i[0]},{"role": "assistant", "content": i[1]}]
        
    messages =temp  + message

    output = pipe(messages, **generation_args)
    return output[0]['generated_text']
gr.ChatInterface(assistant_talk).launch()