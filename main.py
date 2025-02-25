import os
import re
import json
import requests
import pandas as pd
from PyPDF2 import PdfReader
from crewai import Agent, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

print("Environment variables loaded.")
print(f"OPENAI_API_KEY: {openai_api_key}")


class QuestionAnalysisAgents:
    def __init__(self):
        print("Initializing Analysis Agents...")
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini") 
        print("GPT-4o-mini model initialized.")

    def qc_testing_agent(self):
        return Agent(
            role="QC Testing Agent",
            goal="Provide simulated answers strictly based on the given question.",
            backstory="Simulates answers to provided questions using the LLM, ensuring relevance and clarity.",
            llm=self.llm,
            verbose=True
        )

    def qc_auditor_agent(self):
        return Agent(
            role="QC Auditor Agent",
            goal="Evaluate the relevance and accuracy of answers based on the provided question and give a score (0-100) with justification.",
            backstory="Scores simulated answers based on their relevance, accuracy, and clarity. Always return a numerical score and justification.",
            llm=self.llm,
            verbose=True
        )

    def question_paraphrasing_agent(self):
        return Agent(
            role="Paraphrasing Agent",
            goal="Generate variations of questions while maintaining their meaning.",
            backstory="This agent generates five variations of each question to ensure flexibility in how questions are asked.",
            llm=self.llm,
            verbose=True
        )


def extract_questions_and_answers_from_pdf(file_path):
    """
    Extracts questions and answers from a PDF file.
    Assumes questions start with "Question:" and answers start with "Answer:".
    """
    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

  
        question_pattern = r"(?<=Question:).*?(?=\nAnswer:)"
        answer_pattern = r"(?<=Answer:).*?(?=\nQuestion:|$)"

        questions = re.findall(question_pattern, text, re.DOTALL)
        answers = re.findall(answer_pattern, text, re.DOTALL)

        questions = [q.strip() for q in questions]
        answers = [a.strip() for a in answers]

        return questions, answers
    except Exception as e:
        print(f"Error reading PDF file: {file_path} | {e}")
        return [], []


def generate_question_variations(paraphrasing_agent, question):
    """Generates five variations of a given question using GPT-4."""
    
    variation_task = Task(
        description=(
            f"Generate 5 variations of the following question while keeping the meaning intact:\n\n{question}"
        ),
        agent=paraphrasing_agent,
        expected_output="A list of 5 paraphrased variations of the given question."
    )

    variations_text = paraphrasing_agent.execute_task(variation_task)

    variations = variations_text.split("\n") if isinstance(variations_text, str) else []

    return [v.strip() for v in variations if v.strip()]


def make_api_request(question):
    """
    Makes an API request to the chatbot endpoint with the given question.
    Returns the response as a dictionary.
    """
    try:
        url = os.getenv("CHL_API_URL")
        headers = {
            'Content-Type': 'application/json',
            'Referer': os.getenv("CHL_API_REFERER"),
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        data = {
            "question": question,
            "test": True
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except Exception as e:
        print(f"API request error for question '{question}': {e}")
        return {"error": str(e), "answer": "Error fetching response"}


def create_html_report(file_path, original_questions, variations, results):
    """
    Creates an enhanced HTML report with improved styling and readability.
    """
    html_file = f"{os.path.splitext(file_path)[0]}_report.html"
    with open(html_file, "w") as f:
        f.write("""
        <html>
        <head>
            <title>Analysis Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f4f4f9;
                    color: #333;
                }
                h1 {
                    color: #004085;
                    text-align: center;
                }
                h2 {
                    color: #007bff;
                    border-bottom: 2px solid #007bff;
                    padding-bottom: 5px;
                }
                h3 {
                    color: #0056b3;
                    margin-top: 20px;
                }
                .container {
                    background: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                }
                .variations {
                    background: #e9ecef;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }
                .simulated-answer {
                    background: #d6e9f9; /* Light blue */
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }
                .api-response {
                    background: #fff3cd; /* Light yellow */
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }
                .audit {
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }
                .audit-high {
                    background: #d4edda; /* Green for scores 95+ */
                }
                .audit-low {
                    background: #f8d7da; /* Red for scores 94 and below */
                }
                .variation-list {
                    list-style-type: none;
                    padding: 0;
                }
                .variation-list li {
                    margin: 5px 0;
                }
                .api-container {
                    margin-top: 10px;
                }
            </style>
        </head>
        <body>
        """)

        f.write(f"<h1>Analysis Report for {os.path.basename(file_path)}</h1>")

        for idx, (original_question, variation_set, result) in enumerate(zip(original_questions, variations, results), 1):
            f.write('<div class="container">')

            # Original Question
            f.write(f"<h2>Original Question {idx}:</h2>")
            f.write(f"<p><strong>{original_question}</strong></p>")

            # Variations Section
            f.write(f'<div class="variations"><h3>Question Variations:</h3><ul class="variation-list">')
            for var_idx, variation in enumerate(variation_set, 1):
                variation = re.sub(r'^\d+\.\s*', '', variation) 
                f.write(f"<li><strong>Variation {var_idx}:</strong> {variation}</li>")
            f.write("</ul></div>")

            # Simulated Answer (Light Blue)
            f.write(f'<div class="simulated-answer"><h3>Simulated Answer:</h3><p>{result["answer"]}</p></div>')

            # API Response Section (Light Yellow)
            f.write(f'<div class="api-response"><h3>API Responses:</h3>')
            for var_idx, api_response in enumerate(result.get("api_responses", []), 1):
                variation = re.sub(r'^\d+\.\s*', '', variation_set[var_idx-1])
                f.write(f'<div class="api-container">')
                f.write(f"<p><strong>Variation {var_idx}:</strong> {variation}</p>")
                f.write(f"<p><strong>Chat ID:</strong> {api_response.get('chat_id', 'N/A')}</p>")
                f.write(f"<p><strong>Answer:</strong> {api_response.get('answer', 'No answer provided')}</p>")
                f.write('</div>')
            f.write("</div>")

            audit_class = "audit-high" if result["score"] >= 95 else "audit-low"

            # Auditor Evaluation
            f.write(f'<div class="audit {audit_class}"><h3># Agent: QC Auditor Agent</h3>')
            f.write(f"<p><strong>Final Score: {result['score']}</strong></p>")
            f.write(f"<p>{result['explanation']}</p></div>")

            f.write("</div>") 

        f.write("</body></html>")

    print(f"Enhanced HTML report generated: {html_file}")

def analyze_questions(pdf_files):
    agents = QuestionAnalysisAgents()
    qc_agent = agents.qc_testing_agent()
    auditor_agent = agents.qc_auditor_agent()
    paraphrasing_agent = agents.question_paraphrasing_agent()

    for file_path in pdf_files:
        if os.path.exists(file_path):
            print(f"\nProcessing file: {file_path}")
            try:
                questions, answers = extract_questions_and_answers_from_pdf(file_path)

                if not questions:
                    print(f"No questions found in the PDF file: {file_path}")
                    continue

                results = []
                question_variations_list = []

                for idx, (question, answer) in enumerate(zip(questions, answers), 1):
                    print(f"\n--- Processing Question {idx} ---")
                    print(f"Original Q{idx}: {question}")

                    # Generate 5 variations of the question
                    question_variations = generate_question_variations(paraphrasing_agent, question)
                    print(f"Generated Variations: {question_variations}")

                    if not question_variations:
                        print(f"Skipping question {idx} as no variations were generated.")
                        continue

                    question_variations_list.append(question_variations)

                    # QC Testing Agent
                    qc_task = Task(
                        description=f"Provide a simulated answer strictly based on the question: {question}",
                        agent=qc_agent,
                        expected_output="A simulated response strictly based on the question."
                    )
                    qc_result = qc_agent.execute_task(qc_task)
                    print(f"Simulated Answer: {qc_result}")
                    
                    # Make API requests for each question variation
                    print(f"\n--- Processing Question {idx} API Requests ---")
                    # print api url and referer
                    print(f"API URL: {os.getenv('CHL_API_URL')}")
                    print(f"API Referer: {os.getenv('CHL_API_REFERER')}")
                    api_responses = []
                    for variation in question_variations:
                        variation_clean = re.sub(r'^\d+\.\s*', '', variation).strip()
                        print(f"Making API request for: {variation_clean}")
                        api_response = make_api_request(variation_clean)
                        api_responses.append(api_response)
                        print(f"API Response: {json.dumps(api_response, indent=2)}")

                    # QC Auditor Agent
                    audit_task = Task(
                        description=(
                            f"Evaluate the response based on the question: {question}\n"
                            f"Simulated Answer: {qc_result}\n"
                            "Return a numerical score (0-100) with justification."
                        ),
                        agent=auditor_agent,
                        expected_output="A score between 0 and 100 with justification."
                    )
                    audit_result = auditor_agent.execute_task(audit_task)
                    print(f"Audit Result: {audit_result}")

                    # Extract score and justification
                    score_match = re.search(r'\b(\d{1,3})\b', audit_result)
                    if score_match:
                        score = int(score_match.group(1))
                        print(f"Score Extracted: {score}")
                    else:
                        score = 50 
                        print(f"No valid score found for Q{idx}. Defaulting to score: {score}")

                    explanation = audit_result.strip() 

                    results.append({
                        "question": question,
                        "answer": qc_result,
                        "score": score,
                        "explanation": explanation,
                        "api_responses": api_responses  # Add API responses to results
                    })

                if results and question_variations_list:
                    create_html_report(file_path, questions, question_variations_list, results)
                else:
                    print(f"Skipping report generation for {file_path} due to missing data.")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")


if __name__ == "__main__":
    pdf_files = [
        "pdfs/Chl_chatbot_test_questions_accessibility.pdf",
        "pdfs/Chl_chatbot_test_questions_app_development.pdf",
        "pdfs/Chl_chatbot_test_questions_maintenance_support.pdf",
        "pdfs/Chl_chatbot_test_questions_marketing_site.pdf",
        "pdfs/Chl_chatbot_test_questions_motion_graphics.pdf",
        "pdfs/Chl_chatbot_test_questions_seo.pdf",
    ]

    analyze_questions(pdf_files)
