import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TrainerCallback

# Check and set the device to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    fp16_setting = True
else:
    fp16_setting = False
# Load pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-finetuned-sql")

# Define schema_info
schema_info = (
        "Database Schema:\n"
        "Table: log_data\n"
        "- log_level (VARCHAR): The level of the log, e.g., 'INFO', 'ERROR', 'CRITICAL', 'WARNING'.\n"
        "- timestamp (TIMESTAMP): The date and time the log entry was generated.\n"
        "- event_id (INTEGER): A unique identifier for the event associated with the log.\n"
        "- user_id (INTEGER): The ID of the user associated with the log entry.\n"
        "- session_id (VARCHAR): The session identifier for the user's session.\n"
        "- source_ip_address (VARCHAR): The IP address from which the request or event originated.\n"
        "- destination_ip_address (VARCHAR): The IP address to which the request or event was directed.\n"
        "- host_name (VARCHAR): The name of the host where the event occurred.\n"
        "- application_name (VARCHAR): The name of the application responsible for generating the log.\n"
        "- process_id (INTEGER): The ID of the process that generated the log entry.\n"
        "- thread_id (INTEGER): The ID of the thread that generated the log entry.\n"
        "- file_name (VARCHAR): The name of the file associated with the log event.\n"
        "- line_number (INTEGER): The line number in the file where the event was recorded.\n"
        "- method_name (VARCHAR): The name of the method or function where the event occurred.\n"
        "- event_type (VARCHAR): The type of event being logged, e.g., 'ERROR', 'INFO'.\n"
        "- action_performed (VARCHAR): The specific action or event performed.\n"
        "- status_code (INTEGER): The HTTP status code associated with the event, if applicable.\n"
        "- response_time (INTEGER): The time taken to respond to a request, in milliseconds.\n"
        "- resource_accessed (VARCHAR): The resource accessed during the event, e.g., a URL.\n"
        "- bytes_sent (INTEGER): The number of bytes sent during the transaction.\n"
        "- bytes_received (INTEGER): The number of bytes received during the transaction.\n"
        "- error_message (TEXT): The error message associated with the log entry, if any.\n"
        "- exception_stack_trace (TEXT): The stack trace of any exception associated with the event.\n"
        "- user_agent (VARCHAR): The user agent string of the client responsible for the event.\n"
        "- operating_system (VARCHAR): The operating system of the device generating the log.\n"
        "Notes:\n"
        "- Use 'WHERE timestamp >= NOW() - INTERVAL x' for filtering time ranges.\n"
        "- Use 'GROUP BY' for grouping data and 'ORDER BY' for sorting.\n"
        "- Use functions like COUNT(), MAX(), MIN(), AVG() for aggregations.\n"
        "- Use SQL keywords such as DISTINCT, ILIKE, and DATE_TRUNC for specific use cases.\n\n"
        "Example Questions and Corresponding SQL Queries:\n"
        "1. Display errors from the last 3 months:\n"
        "   SELECT * FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '3 months' ORDER BY timestamp DESC;\n\n"
        "2. Display errors from the last 1 month:\n"
        "   SELECT * FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '1 month' ORDER BY timestamp DESC;\n\n"
        "3. Display log_data from the last 1 hour:\n"
        "   SELECT * FROM log_data WHERE timestamp >= NOW() - INTERVAL '1 hour' ORDER BY timestamp DESC;\n\n"
        "4. Display critical log_data from the current day:\n"
        "   SELECT * FROM log_data WHERE log_level = 'CRITICAL' AND timestamp::DATE = CURRENT_DATE ORDER BY timestamp DESC;\n\n"
        "5. Count log_data per log level for the last 7 days:\n"
        "   SELECT log_level, COUNT(*) AS log_count FROM log_data WHERE timestamp >= NOW() - INTERVAL '7 days' GROUP BY log_level ORDER BY log_count DESC;\n\n"
        "6. List distinct error messages and their count in the last 30 days:\n"
        "    SELECT error_message, COUNT(*) AS occurrences FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '30 days' GROUP BY error_message ORDER BY occurrences DESC;\n\n"
    )
# 1. Load Dataset from CSV
def preprocess_data(examples):
    inputs = [
        f"{schema_info}\nTranslate the following English question to SQL: {q}" 
        for q in examples["question"]
    ]
    targets = examples["sql"]

    # Tokenize input and output sequences
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

# Load the dataset from CSV
# Ensure your CSV has columns: "question" and "sql"
dataset = load_dataset("csv", data_files={"train": "train.csv", "validation": "validation.csv"})

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_data, batched=True, remove_columns=["question", "sql"])

# 2. Load Pre-trained Model
model = GPT2LMHeadModel.from_pretrained("gpt2-finetuned-sql").to(device)  # Move model to device (GPU/CPU)

# 3. Define Training Arguments with suggested changes
training_args = TrainingArguments(
    output_dir="./gpt-finetuned-sql-v1",
    evaluation_strategy="steps",  # Evaluate every few steps instead of each epoch
    eval_steps=500,  # Evaluate every 500 steps
    learning_rate=3e-5,  # Reduced learning rate to avoid overfitting
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,  # Reduced number of epochs to avoid overfitting
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,  # Increased logging frequency
    save_steps=500,  # Save model every 500 steps
    fp16=fp16_setting,  # Enable FP16 for faster computation (only if supported by GPU)
    load_best_model_at_end=True,  # Load the best model at the end based on validation loss
    metric_for_best_model="eval_loss",  # Use eval_loss to determine the best model
)

# 4. Initialize Trainer with EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

# 5. Train the Model
trainer.train()

# 6. Save the Fine-Tuned Model
model.save_pretrained("./gpt-finetuned-sql-v1")
tokenizer.save_pretrained("./gpt-finetuned-sql-v1")
