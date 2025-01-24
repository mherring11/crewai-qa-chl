import pdfplumber
import logging

class PDFReader:
    @staticmethod
    def read_pdf(file_path):
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += (page.extract_text() or "").strip() + "\n"
                return text
        except FileNotFoundError:
            logging.error(f"PDF file not found: {file_path}")
        except Exception as e:
            logging.error(f"Error reading PDF {file_path}: {e}")
        return ""
