import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TrainerCallback

# Check and set the device to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-finetuned-sql")

# Define schema_info
schema_info = (
    "Database Schema:\n"
    "Table: log_data\n"
    "log_level (VARCHAR): The level of the log, e.g., 'INFO', 'ERROR', 'CRITICAL','WARNING'.\n"
    "timestamp (TIMESTAMP): The date and time the log entry was generated.\n"
    "event_id (INTEGER): A unique identifier for the event associated with the log.\n"
    "user_id (INTEGER): The ID of the user associated with the log entry.\n"
    "session_id (VARCHAR): The session identifier for the user's session.\n"
    "source_ip_address (VARCHAR): The IP address from which the request or event originated.\n"
    "destination_ip_address (VARCHAR): The IP address to which the request or event was directed.\n"
    "host_name (VARCHAR): The name of the host where the event occurred.\n"
    "application_name (VARCHAR): The name of the application responsible for generating the log.\n"
    "process_id (INTEGER): The ID of the process that generated the log entry.\n"
    "thread_id (INTEGER): The ID of the thread that generated the log entry.\n"
    "file_name (VARCHAR): The name of the file associated with the log event.\n"
    "line_number (INTEGER): The line number in the file where the event was recorded.\n"
    "method_name (VARCHAR): The name of the method or function where the event occurred.\n"
    "event_type (VARCHAR): The type of event being logged, e.g., 'ERROR', 'INFO'.\n"
    "action_performed (VARCHAR): The specific action or event performed.\n"
    "status_code (INTEGER): The HTTP status code associated with the event, if applicable.\n"
    "response_time (INTEGER): The time taken to respond to a request, in milliseconds.\n"
    "resource_accessed (VARCHAR): The resource accessed during the event, e.g., a URL.\n"
    "bytes_sent (INTEGER): The number of bytes sent during the transaction.\n"
    "bytes_received (INTEGER): The number of bytes received during the transaction.\n"
    "error_message (TEXT): The error message associated with the log entry, if any.\n"
    "exception_stack_trace (TEXT): The stack trace of any exception associated with the event.\n"
    "user_agent (VARCHAR): The user agent string of the client responsible for the event.\n"
    "operating_system (VARCHAR): The operating system of the device generating the log.\n"
    "Notes:\n"
    "- Use 'WHERE timestamp >= NOW() - INTERVAL x' for filtering time ranges.\n"
    "- Use 'GROUP BY' for grouping data and 'ORDER BY' for sorting.\n"
    "- Use functions like COUNT(), MAX(), MIN(), AVG() for aggregations.\n"
    "- Use SQL keywords such as DISTINCT, ILIKE, and DATE_TRUNC for specific use cases.\n\n"
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

# Define EarlyStoppingCallback
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=2):
        self.patience = patience
        self.best_loss = float('inf')
        self.stopped_epochs = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        # Check validation loss and stop training if there's no improvement
        eval_loss = state.best_metric
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.stopped_epochs = 0
        else:
            self.stopped_epochs += 1
            if self.stopped_epochs >= self.patience:
                print(f"Early stopping at epoch {state.epoch}")
                control.should_early_stop = True

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
    fp16=True,  # Enable FP16 for faster computation (only if supported by GPU)
    load_best_model_at_end=True,  # Load the best model at the end based on validation loss
    metric_for_best_model="eval_loss",  # Use eval_loss to determine the best model
)

# 4. Initialize Trainer with EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(patience=2)]  # Add early stopping callback
)

# 5. Train the Model
trainer.train()

# 6. Save the Fine-Tuned Model
model.save_pretrained("./gpt-finetuned-sql-v1")
tokenizer.save_pretrained("./gpt-finetuned-sql-v1")
