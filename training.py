# Load pretrained model

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"  # Use the appropriate LLaMA 2 model size
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load Dataset

import pandas as pd

# Load the dataset
data = pd.read_csv('udemy_courses.csv')

# Concatenating fields into a single string for training
data['text'] = data.apply(lambda row: f"Course Title: {row['course_title']}. Subject: {row['subject']}. Content: {row['num_lectures']} lectures, {row['content_duration']} hours. Reviews: {row['num_reviews']} reviews.", axis=1)

# Drop any rows with missing values (optional)
# data.dropna(subset=['text'], inplace=True)

# Save the processed text into a new DataFrame
processed_data = data[['text']]

# Tokenize dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Convert the DataFrame to a Dataset
import datasets
dataset = datasets.Dataset.from_pandas(processed_data)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset into train and validation
train_test = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test['train']
eval_dataset = train_test['test']

# Train model

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# Load the LLaMA 2 model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save model

model.save_pretrained("./finetuned_llama")
tokenizer.save_pretrained("./finetuned_llama")
