import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./finetuned_gpt2")
tokenizer = AutoTokenizer.from_pretrained("./finetuned_gpt2")

def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_length=150, 
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Simplified response function for Gradio
def respond(message, history, system_message, max_tokens, temperature, top_p):
    # Directly generate a response based on the input message
    return generate_response(message)

# Gradio interface
demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

if __name__ == "__main__":
    demo.launch()
