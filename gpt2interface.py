from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./finetuned_gpt2")
tokenizer = AutoTokenizer.from_pretrained("./finetuned_gpt2")

# Step 7: Test the Model in the Terminal
def generate_response(input_text):
    if not input_text.strip():  # Check for empty or whitespace-only input
        return "Please provide a non-empty input."

    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Handle empty inputs by returning an error message or a default response
    if inputs["input_ids"].shape[-1] == 0:
        return "Sorry, I didn't catch that. Can you say it again?"

    outputs = model.generate(
        inputs["input_ids"], 
        max_length=150, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id  # Ensure the pad_token_id is set
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
