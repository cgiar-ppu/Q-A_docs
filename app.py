import streamlit as st
import os
import json
import pandas as pd
import re  # Added import for regular expressions
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from openai import OpenAI

# Initialize OpenAI client (Assuming API key is set in environment variables)
client = OpenAI()

# Simplified models list
simplified_models = ['o1-preview', 'o1-mini']

# Set up the Streamlit app
st.title("Q&A from documents")

# Function to generate Excel data as bytes
def get_excel_data(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    processed_data = output.getvalue()
    return processed_data

# Function to unpack JSON content into text
def unpack_json_to_text(cell):
    try:
        if pd.isnull(cell):
            return ''
        # Convert cell to string and strip whitespace
        cell = str(cell).strip()
        # Remove code block markers if present
        if cell.startswith('```'):
            # Remove the opening code block marker and any language specifier
            cell = re.sub(r'^```[a-zA-Z]*\n?', '', cell)
        if cell.endswith('```'):
            # Remove the closing code block marker
            cell = re.sub(r'\n?```$', '', cell)
        # Now try to load the JSON
        data = json.loads(cell)
        return json_to_text(data)
    except Exception:
        return cell  # Return original cell if not JSON

def json_to_text(data, depth=0):
    if isinstance(data, dict):
        texts = []
        for key, value in data.items():
            if depth == 0 and key.lower() == 'question':
                # Skip 'Question' key at root
                continue
            elif depth == 0 and isinstance(value, dict):
                # Recurse into nested dicts
                texts.append(json_to_text(value, depth + 1))
            else:
                texts.append(f"{key}: {json_to_text(value, depth + 1)}")
        return '\n'.join(texts)
    elif isinstance(data, list):
        return ', '.join([json_to_text(item, depth + 1) for item in data])
    else:
        return str(data)

# Function to extract answer text from JSON in 'Answer' column according to specified rules
def extract_answer_text(cell):
    try:
        if pd.isnull(cell):
            return ''
        # Convert cell to string and strip whitespace
        cell = str(cell).strip()
        # Remove code block markers if present
        if cell.startswith('```'):
            cell = re.sub(r'^```[a-zA-Z]*\n?', '', cell)
        if cell.endswith('```'):
            cell = re.sub(r'\n?```$', '', cell)
        # Now try to load the JSON
        data = json.loads(cell)
        if isinstance(data, dict):
            keys = list(data.keys())
            if len(keys) == 0:
                return ''
            first_key = keys[0]
            if first_key.lower() == 'question':
                # Ignore 'Question' key-value pair
                remaining_items = {k: data[k] for k in keys[1:]}
                return json_value_to_text(remaining_items)
            else:
                # Ignore the first key
                first_value = data[first_key]
                return json_value_to_text(first_value)
        else:
            return str(data)
    except Exception:
        return cell  # Return original cell if not JSON

def json_value_to_text(value):
    if isinstance(value, dict):
        texts = []
        for k, v in value.items():
            texts.append(f"{k}: {json_value_to_text(v)}")
        return '\n'.join(texts)
    elif isinstance(value, list):
        return ', '.join([json_value_to_text(item) for item in value])
    else:
        return str(value)

# User inputs

questions_file = st.file_uploader("Upload Excel file with questions", type=['xlsx'])

data_file = st.file_uploader("Upload Excel file with data to process", type=['xlsx'])

if data_file is not None:
    data_df = pd.read_excel(data_file)
    headers = data_df.columns.tolist()
    selected_headers = st.multiselect("Select headers to use for processing", headers)
else:
    selected_headers = []

process_option = st.selectbox("Select process to run", ["Single Question", "Bulk Questions"])

# Define processing functions
def process_data_single_question(data_df, selected_headers, all_questions):
    results = []
    model_name = 'gpt-4o'  # Adjust as needed
    for idx, row in data_df.iterrows():
        document_name = f"Row_{idx+1}"
        # Combine selected header content
        text_content = "\n".join([f"{header}: {str(row[header])}" for header in selected_headers])

        # Prepare role message
        role_message = 'You are an assistant that extracts information from data.'

        # Define a function to process a single question
        def process_question(question):
            user_message = f"Document:\n{text_content}\n\nQuestion:\n{question}\n\nProvide the answer in JSON format, without including any code block markers, and when doing so, include the exact text of the question as a key in the JSON."
            messages = [
                {"role": "system", "content": role_message},
                {"role": "user", "content": user_message}
            ]
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=2048,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0,
                    response_format={
                        "type": "text"
                    }
                )
                answer = response.choices[0].message.content.strip()
                return {
                    'Document': document_name,
                    'Question': question,
                    'Answer': answer
                }
            except Exception as e:
                st.error(f"Error querying OpenAI for {document_name}, question {question}: {e}")
                return None

        # Use ThreadPoolExecutor to process questions in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_question, question): question for question in all_questions}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

    # Save results to session state
    df = pd.DataFrame(results)
    pivot_df = df.pivot(index='Document', columns='Question', values='Answer').reset_index()
    st.session_state['single_question_results'] = df
    st.session_state['single_question_pivot'] = pivot_df
    st.success("Single question approach completed.")

def process_data_bulk_questions(data_df, selected_headers, all_questions):
    results = []
    model_name = 'o1-preview'  # Adjust as needed

    for idx, row in data_df.iterrows():
        document_name = f"Row_{idx+1}"
        # Combine selected header content
        text_content = "\n".join([f"{header}: {str(row[header])}" for header in selected_headers])

        # Combine all questions into one prompt
        questions_text = "\n".join(all_questions)
        prompt_text = f"""Document:\n{text_content}\n\nPlease answer the following questions:\n{questions_text}\n\nProvide the answer in JSON format, and when doing so, include only the first 10 words of the question without symbols or punctuation as a key in the JSON, because the outputs will be aggregated later with the same keys from another process and they need to match exactly."""

        # Prepare messages
        if model_name in simplified_models:
            # For simplified models, only include the user message
            messages = [
                {"role": "user", "content": prompt_text}
            ]
        else:
            # For other models, include the system role
            role_message = 'You are an assistant that extracts information from data.'
            messages = [
                {"role": "system", "content": role_message},
                {"role": "user", "content": prompt_text}
            ]

        try:
            if model_name in simplified_models:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=2048,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0,
                    response_format={
                        "type": "text"
                    }
                )
            answer = response.choices[0].message.content.strip()
            results.append({
                'Document': document_name,
                'Answers': answer
            })
        except Exception as e:
            st.error(f"Error querying OpenAI for {document_name}: {e}")

    # Save results to session state
    df = pd.DataFrame(results)
    st.session_state['bulk_question_results'] = df
    st.success("Bulk question approach completed.")

# Run the selected process
if st.button("Run"):
    # Clear previous results
    st.session_state.pop('single_question_results', None)
    st.session_state.pop('single_question_pivot', None)
    st.session_state.pop('bulk_question_results', None)
    try:
        # Initialize OpenAI client (Assuming API key is set in environment variables)
        #client = OpenAI()
        if questions_file is not None and data_file is not None and len(selected_headers) > 0:
            # Read the questions
            questions_df = pd.read_excel(questions_file, header=None)
            row_2 = questions_df.iloc[1]  # Get the second row (index 1)

            # Columns A to H (indexes 0 to 7) need a prefix prompt
            prefix = "Please extract the following parameter: "
            parameters = row_2.iloc[0:9].dropna().tolist()
            prefixed_parameters = [prefix + param for param in parameters]

            # Columns I to Q (indexes 8 to 16) are full questions
            questions = row_2.iloc[9:35].dropna().tolist()

            # Combine all questions
            all_questions = prefixed_parameters + questions

            if process_option == "Single Question":
                process_data_single_question(data_df, selected_headers, all_questions)
            elif process_option == "Bulk Questions":
                process_data_bulk_questions(data_df, selected_headers, all_questions)
        else:
            st.error("Please upload the Excel files and select at least one header.")
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")

# Display download buttons if results are available
if 'single_question_results' in st.session_state:
    df = st.session_state['single_question_results']
    pivot_df = st.session_state['single_question_pivot']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Download original results
    output_filename = f'output_single_question_{timestamp}.xlsx'
    df_excel_data = get_excel_data(df)
    st.download_button("Download Results", data=df_excel_data, file_name=output_filename)

    # Download pivoted results
    pivot_output_filename = f'output_single_question_pivoted_{timestamp}.xlsx'
    pivot_df_excel_data = get_excel_data(pivot_df)
    st.download_button("Download Pivoted Results", data=pivot_df_excel_data, file_name=pivot_output_filename)

    # Generate and download unpacked pivoted results
    unpacked_pivot_df = pivot_df.copy()
    for col in unpacked_pivot_df.columns:
        if col != 'Document':
            unpacked_pivot_df[col] = unpacked_pivot_df[col].apply(unpack_json_to_text)
            # Remove question text from beginning of answer
            question_text = col
            def remove_question_text(answer):
                answer = answer.strip()
                if answer.startswith(question_text):
                    answer = answer[len(question_text):].lstrip(':').lstrip('-').strip()
                return answer
            unpacked_pivot_df[col] = unpacked_pivot_df[col].apply(remove_question_text)
    unpacked_output_filename = f'output_single_question_unpacked_{timestamp}.xlsx'
    unpacked_pivot_excel_data = get_excel_data(unpacked_pivot_df)
    st.download_button("Download Unpacked Pivoted Results", data=unpacked_pivot_excel_data, file_name=unpacked_output_filename)

    # Generate and download extracted Answers
    df_extracted = df.copy()
    df_extracted['Extracted Answer'] = df_extracted['Answer'].apply(extract_answer_text)
    extracted_output_filename = f'output_single_question_extracted_{timestamp}.xlsx'
    df_extracted_excel_data = get_excel_data(df_extracted)
    st.download_button("Download Extracted Answers", data=df_extracted_excel_data, file_name=extracted_output_filename)

    # Generate and download pivoted extracted answers
    df_extracted_pivot = df_extracted.pivot(index='Document', columns='Question', values='Extracted Answer').reset_index()
    extracted_pivot_output_filename = f'output_single_question_extracted_pivoted_{timestamp}.xlsx'
    df_extracted_pivot_excel_data = get_excel_data(df_extracted_pivot)
    st.download_button("Download Pivoted Extracted Answers", data=df_extracted_pivot_excel_data, file_name=extracted_pivot_output_filename)

if 'bulk_question_results' in st.session_state:
    df = st.session_state['bulk_question_results']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'output_bulk_questions_{timestamp}.xlsx'
    df_excel_data = get_excel_data(df)
    st.download_button("Download Results", data=df_excel_data, file_name=output_filename)
