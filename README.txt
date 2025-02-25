# Question Analysis Tool with CHL Chatbot Integration

## Overview
This tool analyzes questions from PDF files, generates variations using AI, simulates answers, 
tests them against a chatbot API endpoint, and generates a comprehensive HTML report of the results.

## Requirements
- Python 3.11.6
- OpenAI API key

## Setup Instructions

### 1. Python Environment Setup

```
# Update and install python
brew update
brew install python@3.11

# Make sure Python 3.11 is in your PATH
brew link python@3.11

# Create a virtual environment
python3.11 -m venv crewai-qc-chl-env-py311

# Activate the virtual environment
source crewai-qc-chl-env-py311/bin/activate

```


### 2. Install Dependencies
```
# Install all required packages
pip install -r requirements.txt
```

### 3. Set Up project specific configuration
Create a file named `.env` in the root directory with your OpenAI API key:
```
# OpenAI API Key (Required)
OPENAI_API_KEY=your_openai_api_key_here

# API Endpoint Configuration (Optional - defaults are already set in the code)
CHL_API_URL=https://wordpresss-install.com/wp-json/chl-smart-search-chatbot/v1/ask
CHL_API_REFERER=https://wordpressinstall.com 
```

### 4. Prepare PDF Files
Place your PDF files in a 'pdfs' directory. The PDFs should have content formatted with:
- "Question:" prefix for questions
- "Answer:" prefix for answers

### 5. Running the Script
```
# Run the main script
python main.py
```

## What the Script Does
1. Extracts questions and answers from PDF files
2. Generates variations of each question using GPT-4
3. Creates simulated answers using an AI agent
4. Tests each question variation against the CHL chatbot API
5. Evaluates the quality of answers
6. Generates an HTML report with all results

## Output
HTML reports will be generated in the same directory as the input PDF files, 
with filenames matching the pattern: `{original_pdf_name}_report.html`

## Troubleshooting

### Common Issues:
1. **API Key Error**: Ensure your OpenAI API key is correctly set in the `.env` file
2. **Module Not Found**: Make sure all dependencies are installed using `pip install -r requirements.txt`
3. **PDF Reading Error**: Ensure PDF files exist in the 'pdfs' directory and have proper format
4. **API Request Error**: Check your internet connection and the API endpoint URL

### For Additional Help:
If you encounter any issues not covered here, please check the error messages 
for specific details or contact the development team. 