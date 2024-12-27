# Megatune.db - Speak to SQL
Megatune.db is a powerful tool designed to convert natural language queries into SQL queries using advanced machine learning models like GPT-2. It provides a Streamlit-based web application for user interaction and training scripts for customizing the models with your own dataset.

## Features
Natural Language to SQL Conversion: Easily translate English questions into SQL queries.

Fine-tuned GPT-2 Model: Leverage pre-trained models fine-tuned on a dataset of SQL query examples.

Streamlit Web App: User-friendly interface to interact with the model and generate queries in real-time.

Customizable Training: Includes scripts to fine-tune the T5 model with your dataset.

## Installation

 ### Clone the repository

```
git clone <https://github.com/YASH-Coder1117/Megatune.db.git>
cd <repository_name>
```

### Install dependencies
It’s recommended to use a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Download and prepare the fine-tuned models

Ensure your fine-tuned GPT-2 model is stored in directory like ./gpt2-finetuned-sql.

## Usage

### Streamlit Application

Run the Streamlit app to generate SQL queries from natural language input:

```
streamlit run app.py
```
Open the app in your browser.

Enter a natural language query (e.g., "Display errors from the last 3 months").

Click "Generate SQL" to view the translated query.

## Model Training

To fine-tune the GPT-2 model with your dataset:

Prepare your dataset with text (natural language) and sql (corresponding SQL query) fields.

Edit the data dictionary in the training script to include your dataset.

Run the training script:

```
python train_t5.py
```
The fine-tuned model will be saved in the ./gpt-finetuned-sql-v1 directory.

## Example Queries

Question: "Display errors from the last 3 months." SQL:

```
SELECT * FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '3 months' ORDER BY timestamp DESC;
```

Question: "Count log entries by day for the last week." SQL:

```
SELECT timestamp::DATE AS log_date, COUNT(*) AS log_count FROM log_data WHERE timestamp >= NOW() - INTERVAL '7 days' GROUP BY log_date ORDER BY log_date DESC;
```

## Dataset Format

The dataset should be structured as follows:

```
{
  "train": [
    {"text": "Natural language query", "sql": "Corresponding SQL query"}
  ],
  "validation": [
    {"text": "Natural language query", "sql": "Corresponding SQL query"}
  ]
}
```

## File Structure
```
├── app.py              # Streamlit application
├── train_t5.py         # Script for fine-tuning the T5 model
├── requirements.txt    # Required Python packages
├── gpt2-finetuned-sql/ # Directory for fine-tuned GPT-2 model
├── logs/               # Training logs
├── results/            # Training results
├── train.csv           # Training dataset in CSV format
├── validation.csv      # Validation dataset in CSV format
└── README.md           # Project documentation
```

## Dependencies
Python 3.8+

Transformers Library

Datasets Library

Streamlit

Install all dependencies via:

```
pip install -r requirements.txt
```

## License
This project is open-source and available under the MIT License. See the LICENSE file for more details.

## Contributing
Contributions are welcome! Feel free to fork this repository, make improvements, and submit a pull request. You can start by exploring the issues or improving the training dataset and web application.

