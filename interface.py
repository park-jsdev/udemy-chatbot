from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./finetuned_llama")
tokenizer = AutoTokenizer.from_pretrained("./finetuned_llama")

def generate_response(input_text):
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate a response
    outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    print("Hello from the fine-tuned LLaMA 2 Chatbot! Press 'x' to quit.")
    
    while True:
        # Get input from the user
        input_text = input("You: ")
        
        # Exit the loop if the user types 'exit'
        if input_text.lower() == "x":
            print("Goodbye!")
            break
        
        # Generate and print the chatbot's response
        response = generate_response(input_text)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
