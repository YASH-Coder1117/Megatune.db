from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
# Load the fine-tuned GPT-2 model and tokenizer
finetuned_model_dir = "./gpt-finetuned-sql-v1"  # Path to your fine-tuned model directory
finetunedGPT = GPT2LMHeadModel.from_pretrained(finetuned_model_dir)
finetunedTokenizer = GPT2Tokenizer.from_pretrained(finetuned_model_dir)

def generate_text_to_sql(query, model, tokenizer, max_length=1024):
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
        "   SELECT error_message, COUNT(*) AS occurrences FROM log_data WHERE log_level = 'ERROR' AND timestamp >= NOW() - INTERVAL '30 days' GROUP BY error_message ORDER BY occurrences DESC;\n\n"
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
st.markdown("<h1><strong>MegatuneDB</strong></h1>", unsafe_allow_html=True)

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
