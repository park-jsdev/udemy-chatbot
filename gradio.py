import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
model_name = "./finetuned_llama_udemy"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def chatbot(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="Udemy MOOC Chatbot",
    description="A chatbot trained on Udemy courses data using LLaMA 2."
)

iface.launch()
