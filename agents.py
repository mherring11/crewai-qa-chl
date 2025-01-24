import os
import re
import logging
import matplotlib.pyplot as plt
from crewai import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools.pdf_reader import PDFReader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    logging.error("OPENAI_API_KEY not found in environment.")
else:
    logging.info("Environment variables loaded successfully.")


class QuestionAnalysisAgents:
    def __init__(self):
        logging.info("Initializing Analysis Agents...")
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")  # Using GPT-4
        logging.info("GPT-4 model initialized.")

    def qc_testing_agent(self):
        return Agent(
            role="QC Testing Agent",
            goal="Simulate answers for the questions provided.",
            backstory="Analyzes the provided questions and generates simulated answers.",
            llm=self.llm,
            verbose=True
        )

    def qc_auditor_agent(self):
        return Agent(
            role="QC Auditor Agent",
            goal="Evaluate the accuracy of answers and provide a score.",
            backstory="Reads the simulated answer and rates it based on relevance, accuracy, and clarity.",
            llm=self.llm,
            verbose=True
        )

def extract_score_from_file(filepath):
    """Extracts the score from the auditor agent section of the analysis file."""
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            match = re.search(r'Score:\s*(\d+)', content)
            if match:
                return int(match.group(1))
            else:
                logging.warning(f"Score not found in {filepath}")
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    except Exception as e:
        logging.error(f"Error reading score from {filepath}: {e}")
    return None

def analyze_questions(pdf_files):
    agents = QuestionAnalysisAgents()
    qc_agent = agents.qc_testing_agent()
    auditor_agent = agents.qc_auditor_agent()
    
    summary = {}

    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            logging.info(f"File exists: {pdf_path}")
            try:
                text = PDFReader.read_pdf(pdf_path)

                output_file = f"{os.path.splitext(pdf_path)[0]}_analysis.txt"
                with open(output_file, 'w') as f:
                    logging.info(f"Processing {pdf_path}")

                    qc_result = qc_agent.execute_task(text)
                    f.write(f"--- QC Testing Agent ({pdf_path}) ---\n")
                    f.write(f"Questions and Answers:\n{text}\n")
                    f.write(f"Simulated Response:\n{qc_result}\n")

                    audit_input = f"Evaluate the following response and provide a score between 0 and 100: {qc_result}"
                    audit_result = auditor_agent.execute_task(audit_input)
                    f.write("\n--- Auditor Agent ---\n")
                    f.write(f"Evaluation and Score:\n{audit_result}\n")

                    score = extract_score_from_file(output_file)
                    summary[pdf_path] = score if score is not None else "Score not found"

            except Exception as e:
                logging.error(f"Error processing {pdf_path}: {e}")
                summary[pdf_path] = "Error in processing"
        else:
            logging.warning(f"File not found: {pdf_path}")
            summary[pdf_path] = "File not found"

    logging.info("\n--- Summary of Scores ---")
    for pdf, score in summary.items():
        logging.info(f"{pdf}: {score}")

    generate_graph(summary)

def generate_graph(summary):
    pdf_files = [pdf for pdf, score in summary.items() if isinstance(score, int)]
    scores = [score for score in summary.values() if isinstance(score, int)]
    
    if not scores:
        logging.warning("No valid scores found to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.barh(pdf_files, scores, color='skyblue')
    plt.xlabel('Score')
    plt.title('Summary of Scores for Each PDF')
    plt.tight_layout()
    plt.show()

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