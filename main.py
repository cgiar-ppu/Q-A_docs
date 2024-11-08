import os
import json
import pandas as pd
from datetime import datetime
from langchain.document_loaders import PyPDFLoader
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed  # Added import for threading

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Set your OpenAI API key
client = OpenAI()

# List of models that require the simplified API call
simplified_models = ['o1-preview', 'o1-mini']

# Directory containing PDF files
input_folder = 'input'

# Load questions from the Excel file
questions_df = pd.read_excel('OICR variables to extract.xlsx', header=None)
row_2 = questions_df.iloc[1]  # Get the second row (index 1)

# Columns A to H (indexes 0 to 7) need a prefix prompt
prefix = "Please extract the following parameter: "
parameters = row_2.iloc[0:9].dropna().tolist()
prefixed_parameters = [prefix + param for param in parameters]

# Columns I to Q (indexes 8 to 16) are full questions
questions = row_2.iloc[9:35].dropna().tolist()

# Combine all questions
all_questions = prefixed_parameters + questions

# Function to process PDFs with the first approach using multithreading
def process_pdfs_single_question():
    results = []
    model_name = 'gpt-4o'
    for pdf_file in os.listdir(input_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, pdf_file)
            loader = PyPDFLoader(pdf_path)
            document = loader.load()
            text_content = "\n".join([page.page_content for page in document])

            # Prepare role message
            role_message = 'You are an assistant that extracts information from documents.'

            # Define a function to process a single question
            def process_question(question):
                user_message = f"Document:\n{text_content}\n\nQuestion:\n{question}\n\nProvide the answer in JSON format."
                messages = [
                    {"role": "system", "content": role_message},
                    {"role": "user", "content": user_message}
                ]
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=1024,
                        top_p=0,
                        frequency_penalty=0,
                        presence_penalty=0,
                        response_format={
                            "type": "text"
                        }
                    )
                    answer = response.choices[0].message.content.strip()
                    return {
                        'Document': pdf_file,
                        'Question': question,
                        'Answer': answer
                    }
                except Exception as e:
                    print(f"Error querying OpenAI for document {pdf_file}, question {question}: {e}")
                    return None

            # Use ThreadPoolExecutor to process questions in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(process_question, question): question for question in all_questions}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)

    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel(f'output_single_question_{timestamp}.xlsx', index=False)
    # Pivot the results
    pivot_df = df.pivot(index='Document', columns='Question', values='Answer')
    pivot_df.to_excel('output_single_question_pivoted.xlsx')
    print("Single question approach completed.")

# Function to process PDFs with the second approach
def process_pdfs_bulk_questions():
    results = []
    model_name = 'o1-preview'
    for pdf_file in os.listdir(input_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, pdf_file)
            loader = PyPDFLoader(pdf_path)
            document = loader.load()
            text_content = "\n".join([page.page_content for page in document])

            # Combine all questions into one prompt
            questions_text = "\n".join(all_questions)
            prompt_text = f"""Document:\n{text_content}\n\nPlease answer the following questions:\n{questions_text}\n\nProvide the answers in JSON format."""

            # Prepare messages
            if model_name in simplified_models:
                # For simplified models, only include the user message
                messages = [
                    {"role": "user", "content": prompt_text}
                ]
            else:
                # For other models, include the system role
                role_message = 'You are an assistant that extracts information from documents.'
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
                        max_tokens=2056,
                        top_p=0,
                        frequency_penalty=0,
                        presence_penalty=0,
                        response_format={
                            "type": "text"
                        }
                    )
                answer = response.choices[0].message.content.strip()
                results.append({
                    'Document': pdf_file,
                    'Answers': answer
                })
            except Exception as e:
                print(f"Error querying OpenAI for document {pdf_file}: {e}")
    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel(f'output_bulk_questions_{timestamp}.xlsx', index=False)
    print("Bulk question approach completed.")

# Run both approaches
print(all_questions)
print(parameters)
print(questions)
process_pdfs_bulk_questions()
process_pdfs_single_question()
