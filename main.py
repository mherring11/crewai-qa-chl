import os
import re
import pandas as pd
from PyPDF2 import PdfReader
from crewai import Agent, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

print("Environment variables loaded.")
print(f"OPENAI_API_KEY: {openai_api_key}")


class QuestionAnalysisAgents:
    def __init__(self):
        print("Initializing Analysis Agents...")
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")  # Using GPT-4
        print("GPT-4 model initialized.")

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


def extract_questions_and_answers_from_pdf(file_path):
    """
    Extracts questions and answers from a PDF file.
    Assumes questions start with "Question:" and answers start with "Answer:".
    """
    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() for page in reader.pages)
        
        # Use regex to match questions and answers
        question_pattern = r"(?<=Question:).*?(?=\nAnswer:)"
        answer_pattern = r"(?<=Answer:).*?(?=\nQuestion:|$)"
        
        questions = re.findall(question_pattern, text, re.DOTALL)
        answers = re.findall(answer_pattern, text, re.DOTALL)
        
        # Clean up whitespace
        questions = [q.strip() for q in questions]
        answers = [a.strip() for a in answers]

        return questions, answers
    except Exception as e:
        print(f"Error reading PDF file: {file_path} | {e}")
        return [], []


def create_html_report(file_path, questions, results):
    """
    Creates an HTML report for the analysis results.
    """
    html_file = f"{os.path.splitext(file_path)[0]}_report.html"
    with open(html_file, "w") as f:
        f.write("<html><head><title>Analysis Report</title></head><body>")
        f.write(f"<h1>Analysis Report for {os.path.basename(file_path)}</h1>")
        for idx, (question, result) in enumerate(zip(questions, results), 1):
            f.write(f"<h2>Question {idx}</h2>")
            f.write(f"<p><strong>{question}</strong></p>")
            f.write(f"<p><strong>Simulated Answer:</strong> {result['answer']}</p>")
            f.write(f"<p><strong># Agent: QC Auditor Agent</strong></p>")
            f.write(f"<p><strong>## Final Answer:</strong> {result['score']}<br>{result['explanation']}</p>")
        f.write("</body></html>")
    print(f"HTML report generated: {html_file}")


def analyze_questions(pdf_files):
    agents = QuestionAnalysisAgents()
    qc_agent = agents.qc_testing_agent()
    auditor_agent = agents.qc_auditor_agent()

    for file_path in pdf_files:
        if os.path.exists(file_path):
            print(f"\nProcessing file: {file_path}")
            try:
                questions, answers = extract_questions_and_answers_from_pdf(file_path)

                if not questions:
                    print(f"No questions found in the PDF file: {file_path}")
                    continue

                results = []
                for idx, (question, answer) in enumerate(zip(questions, answers), 1):
                    print(f"\n--- Processing Question {idx} ---")
                    print(f"Q{idx}: {question}")
                    print(f"A{idx}: {answer}\n")

                    # QC Testing Agent
                    qc_task = Task(
                        description=f"Provide a simulated answer strictly based on the question: {question}",
                        agent=qc_agent,
                        expected_output="A simulated response strictly based on the question."
                    )
                    qc_result = qc_agent.execute_task(qc_task)

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

                    # Extract score and justification
                    score_match = re.search(r'\b(\d{1,3})\b', audit_result)
                    if score_match:
                        score = int(score_match.group(1))
                        print(f"Score Extracted: {score}")
                    else:
                        score = 50  # Default score if no valid score is found
                        print(f"No valid score found for Q{idx}. Defaulting to score: {score}")

                    explanation = audit_result.strip()  # Full response from QC Auditor Agent

                    results.append({
                        "question": question,
                        "answer": qc_result,
                        "score": score,
                        "explanation": explanation
                    })

                # Generate HTML report
                create_html_report(file_path, questions, results)

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
