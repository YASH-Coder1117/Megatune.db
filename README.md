# Megatune.db
# Speak to SQL

This repository contains a system for converting natural language questions into SQL queries using fine-tuned GPT-2 and T5 models. It provides a Streamlit-based web application for user interaction and training scripts for customizing the models with your dataset.

## Features

- **Natural Language to SQL Conversion**: Translates English questions into SQL queries.
- **Fine-tuned GPT-2 and T5 Models**: Uses pre-trained models fine-tuned on a dataset of SQL query examples.
- **Streamlit Web App**: User-friendly interface for interacting with the model and generating queries in real-time.
- **Customizable Training**: Includes scripts to fine-tune the T5 model with your data.

## Installation

1. Clone the repository:
   git clone <repository_url>
   cd <repository_name>
   
2.Install dependencies:
pip install -r requirements.txt
3.Download and prepare the fine-tuned models:

Ensure your fine-tuned GPT-2 and T5 models are stored in directories like ./gpt2-finetuned-sql and ./fine-tuned-t5.
Usage
## Streamlit Application

Run the Streamlit app to generate SQL queries from natural language input:

streamlit run app.py.

Open the app in your browser.

Enter a natural language query (e.g., "Display errors from the last 3 months").

Click "Generate SQL" to view the translated query.

## Model Training
Fine-tune the T5 model with your dataset:

Prepare your dataset with text (natural language) and sql (corresponding SQL query) fields.

Edit the data dictionary in the training script to include your dataset.

Run the training script:

python train_t5.py

The fine-tuned model will be saved in the ./fine-tuned-t5 directory.

## Example Queries

Here are some example natural language questions and their corresponding SQL queries:

Question: "Display errors from the last 3 months."

SQL:SELECT * FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '3 months' ORDER BY timestamp DESC;

Question: "Count log entries by day for the last week."

SQL:SELECT timestamp::DATE AS log_date, COUNT(*) AS log_count FROM log_data WHERE timestamp >= NOW() - INTERVAL '7 days' GROUP BY log_date ORDER BY log_date DESC;

Dataset Format

The dataset should be structured as follows:

{

  "train": [
    {"text": "Natural language query", "sql": "Corresponding SQL query"},
    ...
  ],
  "validation": [
    {"text": "Natural language query", "sql": "Corresponding SQL query"},
    ...
  ]
}
## File Structure

├── app.py                  # Streamlit application

├── train_t5.py             # Script for fine-tuning the T5 model

├── requirements.txt        # Required Python packages

├── gpt2-finetuned-sql/     # Directory for fine-tuned GPT-2 model

├── fine-tuned-t5/          # Directory for fine-tuned T5 model

├── logs/                   # Training logs

├── results/                # Training results

└── README.md               # Project documentation

## Dependencies
Python 3.8 or higher
Transformers Library
Datasets Library
Streamlit
Install all dependencies via pip install -r requirements.txt.

## License
This project is open source and available under the MIT License.

Contributing
Contributions are welcome! Feel free to fork this repository, make improvements, and submit a pull request.

Author
Developed by Megatune.db.
