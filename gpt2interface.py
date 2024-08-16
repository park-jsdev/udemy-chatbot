from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./finetuned_gpt2")
tokenizer = AutoTokenizer.from_pretrained("./finetuned_gpt2")

# Step 7: Test the Model in the Terminal
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

if __name__ == "__main__":
    print("Welcome to the GPT-2 Chatbot! Type 'x' to quit.")
    while True:
        input_text = input("You: ")
        if input_text.lower() == "x":
            print("Goodbye!")
            break
        response = generate_response(input_text)
        print("GPT-2:", response)
