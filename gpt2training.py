from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd
import datasets

# Step 1: Load Udemy Dataset
udemy_data = pd.read_csv('udemy_courses.csv')

# Concatenate fields into a single string for training
udemy_data['text'] = udemy_data.apply(lambda row: f"Course Title: {row['course_title']}. Subject: {row['subject']}. Content: {row['num_lectures']} lectures, {row['content_duration']} hours. Reviews: {row['num_reviews']} reviews.", axis=1)

# Step 2: Load Synthetic Persona Chat Datasets from CSV files
def load_persona_chat_data(filepath):
    data = pd.read_csv(filepath)
    
    # Combine the persona and conversation fields into a single text field
    data['text'] = data.apply(lambda row: f"User 1: {' '.join(row['user 1 personas'].splitlines())} User 2: {' '.join(row['user 2 personas'].splitlines())} Conversation: {row['Best Generated Conversation']}", axis=1)
    
    return data[['text']]

train_data = load_persona_chat_data('Synthetic-Persona-Chat_train.csv')
valid_data = load_persona_chat_data('Synthetic-Persona-Chat_valid.csv')

# Combine Udemy data with Synthetic Persona Chat data
combined_train_data = pd.concat([udemy_data[['text']], train_data])
combined_valid_data = valid_data

# Convert combined data to Hugging Face dataset format
train_dataset = datasets.Dataset.from_pandas(combined_train_data)
valid_dataset = datasets.Dataset.from_pandas(combined_valid_data)

# Step 3: Tokenization
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()  # Set labels to be the same as input_ids
    return inputs

# Apply tokenization to the combined dataset
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True)

# Step 4: Load GPT-2 Model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Step 5: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    fp16=True,  # Mixed precision if supported
)

# Step 6: Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Step 7: Save the Fine-Tuned Model
model.save_pretrained("./finetuned_gpt2")
tokenizer.save_pretrained("./finetuned_gpt2")
