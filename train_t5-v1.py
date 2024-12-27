import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# Load the pre-trained T5 model and tokenizer
model_name = "./gpt2-finetuned-sql"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
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
        {"text": "Find the maximum gap between consecutive log data.", "sql": "SELECT MAX(timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) AS max_gap FROM log_data;"},
        {"text": "Retrieve all logs where the Event Type is 'Error'.", "sql": "SELECT * FROM log_data WHERE EventType = 'Error';"},
        {"text": "Find all entries where the User ID matches a specific value (e.g., User ID = 12345).", "sql": "SELECT * FROM log_data WHERE UserID = 12345;"},
        {"text": "List all events performed by a specific Host Name or Application Name.", "sql": "SELECT * FROM log_data WHERE HostName = 'SpecificHostName' OR ApplicationName = 'SpecificAppName';"},
        {"text": "Get the list of all File Names accessed by a specific User ID within a given time frame.", "sql": "SELECT DISTINCT FileName FROM log_data WHERE UserID = 12345 AND Timestamp BETWEEN '2023-01-01' AND '2023-01-31';"},
        {"text": "Retrieve all logs where the Status Code indicates failure (e.g., Status Code = 500).", "sql": "SELECT * FROM log_data WHERE StatusCode = 500;"},
        {"text": "Retrieve all events where the Response Time exceeded a specific threshold (e.g., Response Time > 500ms).", "sql": "SELECT * FROM log_data WHERE ResponseTime > 500;"},
        {"text": "List all Source IP Addresses that accessed a specific Resource Accessed.", "sql": "SELECT DISTINCT SourceIP FROM log_data WHERE ResourceAccessed = 'SpecificResource';"},
        {"text": "Find all logs where the Bytes Sent or Bytes Received exceeded a given limit (e.g., Bytes Sent > 1MB).", "sql": "SELECT * FROM log_data WHERE BytesSent > 1048576 OR BytesReceived > 1048576;"},
        {"text": "Get all Destination IP Addresses contacted by a specific Source IP Address.", "sql": "SELECT DISTINCT DestinationIP FROM log_data WHERE SourceIP = 'SpecificSourceIP';"},
        {"text": "Retrieve all logs where the Operating System matches a specific type (e.g., 'Windows 11').", "sql": "SELECT * FROM log_data WHERE OperatingSystem = 'Windows 11';"},
        {"text": "Find all events where the Error Message or Exception Stack Trace is not null.", "sql": "SELECT * FROM log_data WHERE ErrorMessage IS NOT NULL OR ExceptionStackTrace IS NOT NULL;"},
        {"text": "Retrieve the most common Error Message and the Application Name associated with it.", "sql": "SELECT ErrorMessage, ApplicationName, COUNT(*) AS Occurrences FROM log_data GROUP BY ErrorMessage, ApplicationName ORDER BY Occurrences DESC LIMIT 1;"},
        {"text": "Get the list of all Process IDs that caused exceptions.", "sql": "SELECT DISTINCT ProcessID FROM log_data WHERE ExceptionStackTrace IS NOT NULL;"},
        {"text": "Retrieve all logs within a specific Timestamp range.", "sql": "SELECT * FROM log_data WHERE Timestamp BETWEEN '2023-01-01' AND '2023-01-31';"},
        {"text": "List all events grouped by Session ID and sorted by Timestamp.", "sql": "SELECT * FROM log_data ORDER BY SessionID, Timestamp;"},
        {"text": "Find all events with a Response Time longer than 1 second during peak hours (e.g., 8 AM to 6 PM).", "sql": "SELECT * FROM log_data WHERE ResponseTime > 1000 AND EXTRACT(HOUR FROM Timestamp) BETWEEN 8 AND 18;"},
        {"text": "Retrieve all actions performed by a specific User Agent.", "sql": "SELECT * FROM log_data WHERE UserAgent = 'SpecificUserAgent';"},
        {"text": "Find all events initiated from a particular Host Name or Source IP Address.", "sql": "SELECT * FROM log_data WHERE HostName = 'SpecificHostName' OR SourceIP = 'SpecificSourceIP';"},
        {"text": "Get the count of unique User IDs who accessed a specific Resource Accessed.", "sql": "SELECT COUNT(DISTINCT UserID) AS UniqueUsers FROM log_data WHERE ResourceAccessed = 'SpecificResource';"},
        {"text": "Retrieve all sessions where a specific Thread ID performed an action.", "sql": "SELECT DISTINCT SessionID FROM log_data WHERE ThreadID = 'SpecificThreadID';"},
        {"text": "Find the average Response Time for all actions performed by a specific Application Name.", "sql": "SELECT AVG(ResponseTime) AS AvgResponseTime FROM log_data WHERE ApplicationName = 'SpecificAppName';"},
        {"text": "Get the total number of Bytes Sent and Bytes Received by a specific Source IP Address.", "sql": "SELECT SUM(BytesSent) AS TotalBytesSent, SUM(BytesReceived) AS TotalBytesReceived FROM log_data WHERE SourceIP = 'SpecificSourceIP';"},
        {"text": "Retrieve a list of the top 10 most frequently accessed Resources.", "sql": "SELECT ResourceAccessed, COUNT(*) AS AccessCount FROM log_data GROUP BY ResourceAccessed ORDER BY AccessCount DESC LIMIT 10;"},
        {"text": "List the Status Code distribution for a particular Application Name.", "sql": "SELECT StatusCode, COUNT(*) AS Occurrences FROM log_data WHERE ApplicationName = 'SpecificAppName' GROUP BY StatusCode;"},
        {"text": "Retrieve all unauthorized access attempts (e.g., Action Performed = 'Unauthorized' or Status Code = 401).", "sql": "SELECT * FROM log_data WHERE ActionPerformed = 'Unauthorized' OR StatusCode = 401;"},
        {"text": "Find the Source IP Address with the highest number of failed attempts (e.g., Status Code = 403 or 500).", "sql": "SELECT SourceIP, COUNT(*) AS FailedAttempts FROM log_data WHERE StatusCode IN (403, 500) GROUP BY SourceIP ORDER BY FailedAttempts DESC LIMIT 1;"},
        {"text": "List all Session IDs that performed actions from multiple Source IP Addresses.", "sql": "SELECT SessionID FROM log_data GROUP BY SessionID HAVING COUNT(DISTINCT SourceIP) > 1;"},
        {"text": "Retrieve all events where the Event Type is 'CRITICAL' and the Status Code is 503.", "sql": "SELECT * FROM log_data WHERE EventType = 'CRITICAL' AND StatusCode = 503;"},
        {"text": "List all logs where the Action Performed is 'Write' and the Response Time exceeds 900ms.", "sql": "SELECT * FROM log_data WHERE ActionPerformed = 'Write' AND ResponseTime > 900;"},
        {"text": "Find all entries where the Operating System is 'Ubuntu 20.04' and the Application Name is 'MovieApp'.", "sql": "SELECT * FROM log_data WHERE OperatingSystem = 'Ubuntu 20.04' AND ApplicationName = 'MovieApp';"},
        {"text": "Fetch the details of logs where the Resource Accessed is '/user/123'.", "sql": "SELECT * FROM log_data WHERE ResourceAccessed = '/user/123';"},
        {"text": "Get the list of all File Names that encountered 'Database connection failed' as an Error Message.", "sql": "SELECT DISTINCT FileName FROM log_data WHERE ErrorMessage LIKE '%Database connection failed%';"},
        {"text": "Retrieve all logs between the Timestamp of '15-12-2021 00:00' and '20-12-2021 23:59'.", "sql": "SELECT * FROM log_data WHERE Timestamp BETWEEN '2021-12-15 00:00:00' AND '2021-12-20 23:59:59';"},
        {"text": "Find all entries where the Response Time was greater than 1 second and occurred after '24-12-2021 12:00'.", "sql": "SELECT * FROM log_data WHERE ResponseTime > 1000 AND Timestamp > '2021-12-24 12:00:00';"},
        {"text": "List all logs from the Source IP Address 203.9.159.200 during December 2021.", "sql": "SELECT * FROM log_data WHERE SourceIP = '203.9.159.200' AND Timestamp BETWEEN '2021-12-01' AND '2021-12-31';"},
        {"text": "Retrieve all logs where the Error Message contains 'User not found'.", "sql": "SELECT * FROM log_data WHERE ErrorMessage LIKE '%User not found%';"},
        {"text": "Find logs where the Exception Stack Trace mentions ValueError as the exception type.", "sql": "SELECT * FROM log_data WHERE ExceptionStackTrace LIKE '%ValueError%';"},
        {"text": "Fetch logs with 'Internal server error' as the Error Message and sort them by Response Time.", "sql": "SELECT * FROM log_data WHERE ErrorMessage = 'Internal server error' ORDER BY ResponseTime DESC;"},
        {"text": "Find the average Response Time for all events performed by 'WebServer'.", "sql": "SELECT AVG(ResponseTime) AS AvgResponseTime FROM log_data WHERE ApplicationName = 'WebServer';"},
        {"text": "Calculate the total Bytes Sent for events where the Event ID is greater than 50000.", "sql": "SELECT SUM(BytesSent) AS TotalBytesSent FROM log_data WHERE EventID > 50000;"},
        {"text": "Identify the top three most frequently occurring Error Messages in the dataset.", "sql": "SELECT ErrorMessage, COUNT(*) AS Occurrences FROM log_data GROUP BY ErrorMessage ORDER BY Occurrences DESC LIMIT 3;"},
        {"text": "Retrieve all events where the Source IP Address accessed multiple Destination IP Addresses.", "sql": "SELECT SourceIP FROM log_data GROUP BY SourceIP HAVING COUNT(DISTINCT DestinationIP) > 1;"},
        {"text": "Find all User IDs associated with failed login attempts (e.g., Status Code = 404).", "sql": "SELECT DISTINCT UserID FROM log_data WHERE StatusCode = 404;"},
        {"text": "Get the count of unique Session IDs for a specific Source IP Address.", "sql": "SELECT COUNT(DISTINCT SessionID) AS UniqueSessions FROM log_data WHERE SourceIP = 'SpecificSourceIP';"},
        {"text": "Retrieve logs where both Bytes Sent and Bytes Received are 0MB.", "sql": "SELECT * FROM log_data WHERE BytesSent = 0 AND BytesReceived = 0;"},
        {"text": "List all logs from the User Agent 'Safari/13.1' and Operating System 'macOS 11.2'.", "sql": "SELECT * FROM log_data WHERE UserAgent = 'Safari/13.1' AND OperatingSystem = 'macOS 11.2';"},
        {"text": "Fetch details of logs where the File Name is db_connector.py and the Method Name is process_data.", "sql": "SELECT * FROM log_data WHERE FileName = 'db_connector.py' AND MethodName = 'process_data';"},
        {"text": "Find the count of logs grouped by Event Type and Status Code.", "sql": "SELECT EventType, StatusCode, COUNT(*) AS Count FROM log_data GROUP BY EventType, StatusCode;"},
        {"text": "Retrieve the maximum Response Time for each Application Name.", "sql": "SELECT ApplicationName, MAX(ResponseTime) AS MaxResponseTime FROM log_data GROUP BY ApplicationName;"},
        {"text": "Identify which Source IP Address sent the highest number of requests to /login.", "sql": "SELECT SourceIP, COUNT(*) AS RequestCount FROM log_data WHERE ResourceAccessed = '/login' GROUP BY SourceIP ORDER BY RequestCount DESC LIMIT 1;"},
        {"text": "Retrieve all logs where the Response Time was above 1 second for the Application Name 'MovieApp'.", "sql": "SELECT * FROM log_data WHERE ApplicationName = 'MovieApp' AND ResponseTime > 1000;"},
        {"text": "Identify which Host Name consistently experienced 'Database connection failed' errors.", "sql": "SELECT HostName, COUNT(*) AS ErrorCount FROM log_data WHERE ErrorMessage LIKE '%Database connection failed%' GROUP BY HostName ORDER BY ErrorCount DESC;"},
        {"text": "Fetch logs with Response Time exceeding the average response time for the dataset.", "sql": "SELECT * FROM log_data WHERE ResponseTime > (SELECT AVG(ResponseTime) FROM log_data);"}
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
        {"text": "Find the maximum gap between consecutive log data.", "sql": "SELECT MAX(timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) AS max_gap FROM log_data;"},
        {"text": "Retrieve all logs where the Event Type is 'Error'.", "sql": "SELECT * FROM log_data WHERE EventType = 'Error';"},
        {"text": "Find all entries where the User ID matches a specific value (e.g., User ID = 12345).", "sql": "SELECT * FROM log_data WHERE UserID = 12345;"},
        {"text": "List all events performed by a specific Host Name or Application Name.", "sql": "SELECT * FROM log_data WHERE HostName = 'SpecificHostName' OR ApplicationName = 'SpecificAppName';"},
        {"text": "Get the list of all File Names accessed by a specific User ID within a given time frame.", "sql": "SELECT DISTINCT FileName FROM log_data WHERE UserID = 12345 AND Timestamp BETWEEN '2023-01-01' AND '2023-01-31';"},
        {"text": "Retrieve all logs where the Status Code indicates failure (e.g., Status Code = 500).", "sql": "SELECT * FROM log_data WHERE StatusCode = 500;"},
        {"text": "Retrieve all events where the Response Time exceeded a specific threshold (e.g., Response Time > 500ms).", "sql": "SELECT * FROM log_data WHERE ResponseTime > 500;"},
        {"text": "List all Source IP Addresses that accessed a specific Resource Accessed.", "sql": "SELECT DISTINCT SourceIP FROM log_data WHERE ResourceAccessed = 'SpecificResource';"},
        {"text": "Find all logs where the Bytes Sent or Bytes Received exceeded a given limit (e.g., Bytes Sent > 1MB).", "sql": "SELECT * FROM log_data WHERE BytesSent > 1048576 OR BytesReceived > 1048576;"},
        {"text": "Get all Destination IP Addresses contacted by a specific Source IP Address.", "sql": "SELECT DISTINCT DestinationIP FROM log_data WHERE SourceIP = 'SpecificSourceIP';"},
        {"text": "Retrieve all logs where the Operating System matches a specific type (e.g., 'Windows 11').", "sql": "SELECT * FROM log_data WHERE OperatingSystem = 'Windows 11';"},
        {"text": "Find all events where the Error Message or Exception Stack Trace is not null.", "sql": "SELECT * FROM log_data WHERE ErrorMessage IS NOT NULL OR ExceptionStackTrace IS NOT NULL;"},
        {"text": "Retrieve the most common Error Message and the Application Name associated with it.", "sql": "SELECT ErrorMessage, ApplicationName, COUNT(*) AS Occurrences FROM log_data GROUP BY ErrorMessage, ApplicationName ORDER BY Occurrences DESC LIMIT 1;"},
        {"text": "Get the list of all Process IDs that caused exceptions.", "sql": "SELECT DISTINCT ProcessID FROM log_data WHERE ExceptionStackTrace IS NOT NULL;"},
        {"text": "Retrieve all logs within a specific Timestamp range.", "sql": "SELECT * FROM log_data WHERE Timestamp BETWEEN '2023-01-01' AND '2023-01-31';"},
        {"text": "List all events grouped by Session ID and sorted by Timestamp.", "sql": "SELECT * FROM log_data ORDER BY SessionID, Timestamp;"},
        {"text": "Find all events with a Response Time longer than 1 second during peak hours (e.g., 8 AM to 6 PM).", "sql": "SELECT * FROM log_data WHERE ResponseTime > 1000 AND EXTRACT(HOUR FROM Timestamp) BETWEEN 8 AND 18;"},
        {"text": "Retrieve all actions performed by a specific User Agent.", "sql": "SELECT * FROM log_data WHERE UserAgent = 'SpecificUserAgent';"},
        {"text": "Find all events initiated from a particular Host Name or Source IP Address.", "sql": "SELECT * FROM log_data WHERE HostName = 'SpecificHostName' OR SourceIP = 'SpecificSourceIP';"},
        {"text": "Get the count of unique User IDs who accessed a specific Resource Accessed.", "sql": "SELECT COUNT(DISTINCT UserID) AS UniqueUsers FROM log_data WHERE ResourceAccessed = 'SpecificResource';"},
        {"text": "Retrieve all sessions where a specific Thread ID performed an action.", "sql": "SELECT DISTINCT SessionID FROM log_data WHERE ThreadID = 'SpecificThreadID';"},
        {"text": "Find the average Response Time for all actions performed by a specific Application Name.", "sql": "SELECT AVG(ResponseTime) AS AvgResponseTime FROM log_data WHERE ApplicationName = 'SpecificAppName';"},
        {"text": "Get the total number of Bytes Sent and Bytes Received by a specific Source IP Address.", "sql": "SELECT SUM(BytesSent) AS TotalBytesSent, SUM(BytesReceived) AS TotalBytesReceived FROM log_data WHERE SourceIP = 'SpecificSourceIP';"},
        {"text": "Retrieve a list of the top 10 most frequently accessed Resources.", "sql": "SELECT ResourceAccessed, COUNT(*) AS AccessCount FROM log_data GROUP BY ResourceAccessed ORDER BY AccessCount DESC LIMIT 10;"},
        {"text": "List the Status Code distribution for a particular Application Name.", "sql": "SELECT StatusCode, COUNT(*) AS Occurrences FROM log_data WHERE ApplicationName = 'SpecificAppName' GROUP BY StatusCode;"},
        {"text": "Retrieve all unauthorized access attempts (e.g., Action Performed = 'Unauthorized' or Status Code = 401).", "sql": "SELECT * FROM log_data WHERE ActionPerformed = 'Unauthorized' OR StatusCode = 401;"},
        {"text": "Find the Source IP Address with the highest number of failed attempts (e.g., Status Code = 403 or 500).", "sql": "SELECT SourceIP, COUNT(*) AS FailedAttempts FROM log_data WHERE StatusCode IN (403, 500) GROUP BY SourceIP ORDER BY FailedAttempts DESC LIMIT 1;"},
        {"text": "List all Session IDs that performed actions from multiple Source IP Addresses.", "sql": "SELECT SessionID FROM log_data GROUP BY SessionID HAVING COUNT(DISTINCT SourceIP) > 1;"},
        {"text": "Retrieve all events where the Event Type is 'CRITICAL' and the Status Code is 503.", "sql": "SELECT * FROM log_data WHERE EventType = 'CRITICAL' AND StatusCode = 503;"},
        {"text": "List all logs where the Action Performed is 'Write' and the Response Time exceeds 900ms.", "sql": "SELECT * FROM log_data WHERE ActionPerformed = 'Write' AND ResponseTime > 900;"},
        {"text": "Find all entries where the Operating System is 'Ubuntu 20.04' and the Application Name is 'MovieApp'.", "sql": "SELECT * FROM log_data WHERE OperatingSystem = 'Ubuntu 20.04' AND ApplicationName = 'MovieApp';"},
        {"text": "Fetch the details of logs where the Resource Accessed is '/user/123'.", "sql": "SELECT * FROM log_data WHERE ResourceAccessed = '/user/123';"},
        {"text": "Get the list of all File Names that encountered 'Database connection failed' as an Error Message.", "sql": "SELECT DISTINCT FileName FROM log_data WHERE ErrorMessage LIKE '%Database connection failed%';"},
        {"text": "Retrieve all logs between the Timestamp of '15-12-2021 00:00' and '20-12-2021 23:59'.", "sql": "SELECT * FROM log_data WHERE Timestamp BETWEEN '2021-12-15 00:00:00' AND '2021-12-20 23:59:59';"},
        {"text": "Find all entries where the Response Time was greater than 1 second and occurred after '24-12-2021 12:00'.", "sql": "SELECT * FROM log_data WHERE ResponseTime > 1000 AND Timestamp > '2021-12-24 12:00:00';"},
        {"text": "List all logs from the Source IP Address 203.9.159.200 during December 2021.", "sql": "SELECT * FROM log_data WHERE SourceIP = '203.9.159.200' AND Timestamp BETWEEN '2021-12-01' AND '2021-12-31';"},
        {"text": "Retrieve all logs where the Error Message contains 'User not found'.", "sql": "SELECT * FROM log_data WHERE ErrorMessage LIKE '%User not found%';"},
        {"text": "Find logs where the Exception Stack Trace mentions ValueError as the exception type.", "sql": "SELECT * FROM log_data WHERE ExceptionStackTrace LIKE '%ValueError%';"},
        {"text": "Fetch logs with 'Internal server error' as the Error Message and sort them by Response Time.", "sql": "SELECT * FROM log_data WHERE ErrorMessage = 'Internal server error' ORDER BY ResponseTime DESC;"},
        {"text": "Find the average Response Time for all events performed by 'WebServer'.", "sql": "SELECT AVG(ResponseTime) AS AvgResponseTime FROM log_data WHERE ApplicationName = 'WebServer';"},
        {"text": "Calculate the total Bytes Sent for events where the Event ID is greater than 50000.", "sql": "SELECT SUM(BytesSent) AS TotalBytesSent FROM log_data WHERE EventID > 50000;"},
        {"text": "Identify the top three most frequently occurring Error Messages in the dataset.", "sql": "SELECT ErrorMessage, COUNT(*) AS Occurrences FROM log_data GROUP BY ErrorMessage ORDER BY Occurrences DESC LIMIT 3;"},
        {"text": "Retrieve all events where the Source IP Address accessed multiple Destination IP Addresses.", "sql": "SELECT SourceIP FROM log_data GROUP BY SourceIP HAVING COUNT(DISTINCT DestinationIP) > 1;"},
        {"text": "Find all User IDs associated with failed login attempts (e.g., Status Code = 404).", "sql": "SELECT DISTINCT UserID FROM log_data WHERE StatusCode = 404;"},
        {"text": "Get the count of unique Session IDs for a specific Source IP Address.", "sql": "SELECT COUNT(DISTINCT SessionID) AS UniqueSessions FROM log_data WHERE SourceIP = 'SpecificSourceIP';"},
        {"text": "Retrieve logs where both Bytes Sent and Bytes Received are 0MB.", "sql": "SELECT * FROM log_data WHERE BytesSent = 0 AND BytesReceived = 0;"},
        {"text": "List all logs from the User Agent 'Safari/13.1' and Operating System 'macOS 11.2'.", "sql": "SELECT * FROM log_data WHERE UserAgent = 'Safari/13.1' AND OperatingSystem = 'macOS 11.2';"},
        {"text": "Fetch details of logs where the File Name is db_connector.py and the Method Name is process_data.", "sql": "SELECT * FROM log_data WHERE FileName = 'db_connector.py' AND MethodName = 'process_data';"},
        {"text": "Find the count of logs grouped by Event Type and Status Code.", "sql": "SELECT EventType, StatusCode, COUNT(*) AS Count FROM log_data GROUP BY EventType, StatusCode;"},
        {"text": "Retrieve the maximum Response Time for each Application Name.", "sql": "SELECT ApplicationName, MAX(ResponseTime) AS MaxResponseTime FROM log_data GROUP BY ApplicationName;"},
        {"text": "Identify which Source IP Address sent the highest number of requests to /login.", "sql": "SELECT SourceIP, COUNT(*) AS RequestCount FROM log_data WHERE ResourceAccessed = '/login' GROUP BY SourceIP ORDER BY RequestCount DESC LIMIT 1;"},
        {"text": "Retrieve all logs where the Response Time was above 1 second for the Application Name 'MovieApp'.", "sql": "SELECT * FROM log_data WHERE ApplicationName = 'MovieApp' AND ResponseTime > 1000;"},
        {"text": "Identify which Host Name consistently experienced 'Database connection failed' errors.", "sql": "SELECT HostName, COUNT(*) AS ErrorCount FROM log_data WHERE ErrorMessage LIKE '%Database connection failed%' GROUP BY HostName ORDER BY ErrorCount DESC;"},
        {"text": "Fetch logs with Response Time exceeding the average response time for the dataset.", "sql": "SELECT * FROM log_data WHERE ResponseTime > (SELECT AVG(ResponseTime) FROM log_data);"}
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

training_args = Seq2SeqTrainingArguments(
    output_dir="./fine_tuned_model2",  # Output directory
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir='./logs',
    logging_steps=10,
    overwrite_output_dir=True
)

# Create a Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Fine-tune the model
trainer.train()
trainer.save_model("./Gpt2-updated-model2")
# Save the fine-tuned model
model.save_pretrained("./Gpt2-updated-model2")
tokenizer.save_pretrained("./Gpt2-updated-model2")