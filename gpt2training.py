from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd
import datasets

# Step 1: Load Udemy Dataset
udemy_data = pd.read_csv('udemy_courses.csv')

# Concatenate fields into a single string for training
udemy_data['text'] = udemy_data.apply(lambda row: f"Course Title: {row['course_title']}. Subject: {row['subject']}. Content: {row['num_lectures']} lectures, {row['content_duration']} hours. Reviews: {row['num_reviews']} reviews.", axis=1)

# Convert the Udemy data to Hugging Face dataset format
dataset = datasets.Dataset.from_pandas(udemy_data[['text']])

# Split the dataset into train and validation sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
valid_dataset = train_test_split['test']

# Step 2: Tokenization
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()  # Set labels to be the same as input_ids
    return inputs

# Apply tokenization to the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True)

# Step 3: Load GPT-2 Medium Model
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# Step 4: Define Training Arguments
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
    fp16=True, 
)

# Step 5: Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Step 6: Save the Fine-Tuned Model
model.save_pretrained("./finetuned_gpt2_medium")
tokenizer.save_pretrained("./finetuned_gpt2_medium")
