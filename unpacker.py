import pandas as pd
import json
import openpyxl
from openpyxl.utils import get_column_letter
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def clean_json_string(s):
    """Remove code fences and language hints from a string."""
    s = s.strip()
    if s.startswith('```'):
        # Remove leading triple backticks and optional language identifier (e.g., ```json)
        s = s.strip('`')
        s = s.lstrip('json').lstrip('JSON').strip()
    if s.endswith('```'):
        # Remove trailing triple backticks
        s = s.rstrip('`').strip()
    return s

# Read the Excel file with pandas
df = pd.read_excel('output_bulk_questions_20241108_153340.xlsx', dtype=str)  # Ensure all data is read as strings
print(df)

# Check the columns in the DataFrame
print("Columns in input Excel file:", df.columns.tolist())

# Ensure that the columns are named 'Document' and 'Answers'
if 'Document' not in df.columns or 'Answers' not in df.columns:
    print("Expected columns 'Document' and 'Answers' not found in input Excel file.")
    print("Available columns:", df.columns.tolist())
    # If the first two columns are 'Document' and 'Answers', rename them accordingly
    if len(df.columns) >= 2:
        df = df.rename(columns={df.columns[0]: 'Document', df.columns[1]: 'Answers'})
    else:
        print("The input Excel file does not have enough columns.")
        exit()

# Initialize 'all_questions' as an empty set to collect all unique questions from all rows
all_questions = set()

# Create a list to store the transformed data
output_data = []

# Now, process each row and extract the answers
for idx, row in df.iterrows():
    document = row['Document']
    answers_json = row['Answers']
    if pd.isna(answers_json):
        continue
    answers_json_clean = clean_json_string(answers_json)
    try:
        answers_dict = json.loads(answers_json_clean)
        if not isinstance(answers_dict, dict):
            print(f"Row {idx}: Parsed JSON is not a dictionary.")
            continue
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON on row {idx}: {e}")
        continue

    # Add the keys to 'all_questions' set
    all_questions.update(answers_dict.keys())

    # Create a dict for this row
    row_data = {'Document': document}
    # Fill in the answers for the questions in this row
    for question, answer in answers_dict.items():
        # Convert lists and dicts to strings
        if isinstance(answer, (list, dict)):
            answer = json.dumps(answer, ensure_ascii=False)
        row_data[question] = answer

    output_data.append(row_data)

if not all_questions:
    print("No questions found in the JSON data. Please check the 'Answers' column for valid JSON content.")
    exit()

# Now, create the header row (column names)
columns = ['Document'] + sorted(all_questions)

# Convert the output data to a DataFrame
output_df = pd.DataFrame(output_data)

# Ensure all columns are present in the DataFrame
output_df = output_df.reindex(columns=columns)

# Write the output DataFrame to Excel using openpyxl to handle formatting
output_filename = f'output_{timestamp}.xlsx'
with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    output_df.to_excel(writer, index=False)
    # Access the workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets[next(iter(writer.sheets))]

    # Optionally, you can adjust column widths or apply any formatting you like
    for i, column in enumerate(output_df.columns, 1):
        column_width = max(output_df[column].astype(str).map(len).max(), len(str(column)))
        worksheet.column_dimensions[get_column_letter(i)].width = column_width + 2

print(f"Processing complete. The transformed data is saved to '{output_filename}'.")
