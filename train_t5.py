import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Load the pre-trained T5 model and tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load your dataset (replace with your dataset)
data = {
    "train": [
        {"text": "Display errors from the last 3 months.", "sql": "SELECT * FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '3 months' ORDER BY timestamp DESC;"},
        {"text": "Display errors from the last 1 month.", "sql": "SELECT * FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '1 month' ORDER BY timestamp DESC;"},
        {"text": "Display log data from the last 1 hour.", "sql": "SELECT * FROM log_data WHERE timestamp >= NOW() - INTERVAL '1 hour' ORDER BY timestamp DESC;"},
        {"text": "Display critical log data from the current day.", "sql": "SELECT * FROM log_data WHERE log_level = 'CRITICAL' AND timestamp::DATE = CURRENT_DATE ORDER BY timestamp DESC;"},
        {"text": "Count log data per log level for the last 7 days.", "sql": "SELECT log_level, COUNT(*) AS log_count FROM log_data WHERE timestamp >= NOW() - INTERVAL '7 days' GROUP BY log_level ORDER BY log_count DESC;"},
        {"text": "List distinct error messages and their count in the last 30 days.", "sql": "SELECT error_message, COUNT(*) AS occurrences FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '30 days' GROUP BY error_message ORDER BY occurrences DESC;"},
        {"text": "Display all log data generated on a specific date (e.g., '2022-01-01').", "sql": "SELECT * FROM log_data WHERE timestamp::DATE = '2022-01-01' ORDER BY timestamp DESC;"},
        {"text": "Find the first occurrence of each log level.", "sql": "SELECT log_level, MIN(timestamp) AS first_occurrence FROM log_data GROUP BY log_level ORDER BY first_occurrence ASC;"},
        {"text": "Display log data containing a specific keyword (e.g., 'connection').", "sql": "SELECT * FROM log_data WHERE error_message ILIKE '%connection%' ORDER BY timestamp DESC;"},
        {"text": "Count log data by day for the last week.", "sql": "SELECT timestamp::DATE AS log_date, COUNT(*) AS log_count FROM log_data WHERE timestamp >= NOW() - INTERVAL '7 days' GROUP BY log_date ORDER BY log_date DESC;"},
        {"text": "Display log data generated during a specific time range.", "sql": "SELECT * FROM log_data WHERE timestamp BETWEEN '2022-01-01 08:00:00' AND '2022-01-01 18:00:00' ORDER BY timestamp ASC;"},
        {"text": "Find the top 5 most frequent error messages.", "sql": "SELECT error_message, COUNT(*) AS frequency FROM log_data WHERE log_level = 'ERROR' GROUP BY error_message ORDER BY frequency DESC LIMIT 5;"},
        {"text": "Display all log data grouped by log level.", "sql": "SELECT log_level, ARRAY_AGG(error_message) AS messages FROM log_data GROUP BY log_level ORDER BY log_level ASC;"},
        {"text": "Count the number of log data in each month of the current year.", "sql": "SELECT DATE_TRUNC('month', timestamp) AS month, COUNT(*) AS log_count FROM log_data WHERE timestamp >= DATE_TRUNC('year', NOW()) GROUP BY month ORDER BY month ASC;"},
        {"text": "Find the last log entry for each log level.", "sql": "SELECT DISTINCT ON (log_level) log_level, timestamp, error_message FROM log_data ORDER BY log_level, timestamp DESC;"},
        {"text": "Display log data generated on weekends.", "sql": "SELECT * FROM log_data WHERE EXTRACT(DOW FROM timestamp) IN (0, 6) ORDER BY timestamp DESC;"},
        {"text": "Find the average time interval between critical log data.", "sql": "SELECT AVG(timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) AS avg_interval FROM log_data WHERE log_level = 'CRITICAL';"},
        {"text": "Find the maximum gap between consecutive log data.", "sql": "SELECT MAX(timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) AS max_gap FROM log_data;"}
    ],
    "validation": [
        {"text": "Display errors from the last 3 months.", "sql": "SELECT * FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '3 months' ORDER BY timestamp DESC;"},
        {"text": "Display errors from the last 1 month.", "sql": "SELECT * FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '1 month' ORDER BY timestamp DESC;"},
        {"text": "Display log data from the last 1 hour.", "sql": "SELECT * FROM log_data WHERE timestamp >= NOW() - INTERVAL '1 hour' ORDER BY timestamp DESC;"},
        {"text": "Display critical log data from the current day.", "sql": "SELECT * FROM log_data WHERE log_level = 'CRITICAL' AND timestamp::DATE = CURRENT_DATE ORDER BY timestamp DESC;"},
        {"text": "Count log data per log level for the last 7 days.", "sql": "SELECT log_level, COUNT(*) AS log_count FROM log_data WHERE timestamp >= NOW() - INTERVAL '7 days' GROUP BY log_level ORDER BY log_count DESC;"},
        {"text": "List distinct error messages and their count in the last 30 days.", "sql": "SELECT error_message, COUNT(*) AS occurrences FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '30 days' GROUP BY error_message ORDER BY occurrences DESC;"},
        {"text": "Display all log data generated on a specific date (e.g., '2022-01-01').", "sql": "SELECT * FROM log_data WHERE timestamp::DATE = '2022-01-01' ORDER BY timestamp DESC;"},
        {"text": "Find the first occurrence of each log level.", "sql": "SELECT log_level, MIN(timestamp) AS first_occurrence FROM log_data GROUP BY log_level ORDER BY first_occurrence ASC;"},
        {"text": "Display log data containing a specific keyword (e.g., 'connection').", "sql": "SELECT * FROM log_data WHERE error_message ILIKE '%connection%' ORDER BY timestamp DESC;"},
        {"text": "Count log data by day for the last week.", "sql": "SELECT timestamp::DATE AS log_date, COUNT(*) AS log_count FROM log_data WHERE timestamp >= NOW() - INTERVAL '7 days' GROUP BY log_date ORDER BY log_date DESC;"},
        {"text": "Display log data generated during a specific time range.", "sql": "SELECT * FROM log_data WHERE timestamp BETWEEN '2022-01-01 08:00:00' AND '2022-01-01 18:00:00' ORDER BY timestamp ASC;"},
        {"text": "Find the top 5 most frequent error messages.", "sql": "SELECT error_message, COUNT(*) AS frequency FROM log_data WHERE log_level = 'ERROR' GROUP BY error_message ORDER BY frequency DESC LIMIT 5;"},
        {"text": "Display all log data grouped by log level.", "sql": "SELECT log_level, ARRAY_AGG(error_message) AS messages FROM log_data GROUP BY log_level ORDER BY log_level ASC;"},
        {"text": "Count the number of log data in each month of the current year.", "sql": "SELECT DATE_TRUNC('month', timestamp) AS month, COUNT(*) AS log_count FROM log_data WHERE timestamp >= DATE_TRUNC('year', NOW()) GROUP BY month ORDER BY month ASC;"},
        {"text": "Find the last log entry for each log level.", "sql": "SELECT DISTINCT ON (log_level) log_level, timestamp, error_message FROM log_data ORDER BY log_level, timestamp DESC;"},
        {"text": "Display log data generated on weekends.", "sql": "SELECT * FROM log_data WHERE EXTRACT(DOW FROM timestamp) IN (0, 6) ORDER BY timestamp DESC;"},
        {"text": "Find the average time interval between critical log data.", "sql": "SELECT AVG(timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) AS avg_interval FROM log_data WHERE log_level = 'CRITICAL';"},
        {"text": "Find the maximum gap between consecutive log data.", "sql": "SELECT MAX(timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) AS max_gap FROM log_data;"}
    ],
}

# Prepare data for DatasetDict
def prepare_data(data_list):
    texts = [entry["text"] for entry in data_list]
    sqls = [entry["sql"] for entry in data_list]
    return {"text": texts, "sql": sqls}

# Reorganize datasets
train_data = prepare_data(data["train"])
validation_data = prepare_data(data["validation"])

dataset = DatasetDict({
    "train": Dataset.from_dict(train_data),
    "validation": Dataset.from_dict(validation_data),
})

# Preprocess the data
def preprocess_function(examples):
    inputs = [f"generate SQL for: {text}" for text in examples["text"]]
    targets = [sql for sql in examples["sql"]]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids
    
    # Replace padding token ID for labels
    labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
    model_inputs["labels"] = labels
    
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    # For older versions of transformers
    eval_steps=500,  # Evaluate every 500 steps (you can change it as needed)
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-t5")
tokenizer.save_pretrained("./fine-tuned-t5")