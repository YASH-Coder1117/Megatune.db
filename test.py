from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
# Load the fine-tuned GPT-2 model and tokenizer
finetuned_model_dir = "./gpt2-finetuned-sql"  # Path to your fine-tuned model directory
finetunedGPT = GPT2LMHeadModel.from_pretrained(finetuned_model_dir)
finetunedTokenizer = GPT2Tokenizer.from_pretrained(finetuned_model_dir)

def generate_text_to_sql(query, model, tokenizer, max_length=600):
    """
    Generate SQL query from a natural language question using the fine-tuned GPT-2 model.

    Args:
        query (str): The natural language query.
        model (GPT2LMHeadModel): The fine-tuned GPT-2 model.
        tokenizer (GPT2Tokenizer): The tokenizer for the model.
        max_length (int): Maximum length of the generated sequence.

    Returns:
        str: The generated SQL query.
    """
    # Enhanced schema information
    schema_info = (
        "Database Schema:\n"
        "Table: log_data\n"
        "- log_level (VARCHAR): The level of the log, e.g., 'INFO', 'ERROR', 'CRITICAL'.\n"
        "- timestamp (DATETIME): The exact date and time when the log entry was created.\n"
        "- error_message (TEXT): The content of the error message.\n"
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
        "   SELECT error_message, COUNT(*) AS occurrences FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '30 days' GROUP BY error_message ORDER BY occurrences DESC;\n\n"
        "8. Find the maximum gap between consecutive log_data:\n"
        "    SELECT MAX(timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) AS max_gap FROM log_data;\n"
    )

    prompt = schema_info + f"Translate the following English question to SQL: {query}"

    # Encode the prompt text into a tensor suitable for the model
    input_tensor = tokenizer.encode(prompt, return_tensors='pt')

    # Generate the SQL output
    output = model.generate(
        input_tensor,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode the output tensor to a human-readable string
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the SQL part (removing the prompt text)
    sql_output = decoded_output[len(prompt):].strip()
    if not sql_output.endswith(";"):
        sql_output += ";"
    return sql_output


# Streamlit app
st.markdown("<h1><strong>Bytehubble</strong></h1>", unsafe_allow_html=True)

st.title("Natural Language to SQL Generator")

st.write("Enter a natural language question, and the model will generate an SQL query for you.")

# Input field for the user's question
user_input = st.text_input("Enter your question:")

# Submit button
if st.button("Generate SQL"):
    if user_input.strip():
        with st.spinner("Generating SQL query..."):
            sql_result = generate_text_to_sql(user_input, finetunedGPT, finetunedTokenizer)
        st.success("SQL Query Generated!")
        st.text_area("Generated SQL Query:", sql_result)
    else:
        st.error("Please enter a valid question before submitting.")
