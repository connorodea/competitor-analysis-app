from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import openai
import re
from fpdf import FPDF
import os
import logging
import json
from typing import Optional, Tuple
from textblob import TextBlob
import nltk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    filename='competitor_analysis.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Initialize FastAPI
app = FastAPI(title="Competitor Analysis API")

# Set OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set Crunchbase API key from environment variable
CRUNCHBASE_API_KEY = os.getenv('CRUNCHBASE_API_KEY')

# Define report templates
REPORT_TEMPLATES = {
    'standard': [
        'Company Overview',
        'Key Offerings',
        'Industry Impact',
        'Notable Collaborations and Partnerships',
        'Competitive Advantage',
        'Challenges and Risks',
        'Financial Performance and Funding',
        'Future Prospects and Strategic Direction',
        'SWOT Analysis',
        'Competitive Scoring'
    ],
    # Additional templates can be added here
}

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Pydantic models for request and response validation
class ReportRequest(BaseModel):
    url: str
    output_format: str = 'pdf'  # 'pdf' or 'txt'
    template_name: str = 'standard'

class ReportResponse(BaseModel):
    company_name: str
    report_path: str
    message: str

# API Endpoints
@app.post("/generate_report", response_model=ReportResponse)
async def generate_report_endpoint(request: ReportRequest):
    report, company_name = generate_report(
        request.url,
        template_name=request.template_name
    )
    if report and company_name:
        try:
            report_path = save_report(
                report,
                company_name,
                output_format=request.output_format
            )
            return ReportResponse(
                company_name=company_name,
                report_path=report_path,
                message="Report generated successfully."
            )
        except Exception as e:
            logging.error(f"Error saving report: {e}")
            raise HTTPException(status_code=500, detail="Error saving report.")
    else:
        raise HTTPException(status_code=500, detail="Failed to generate report.")

# Functions for analysis
def fetch_website_content(url: str) -> Optional[str]:
    """
    Fetches and cleans main content from the provided URL.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        logging.info(f"Successfully fetched content from {url}")
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    main_content = []

    # Extract text from main tags with noise filtering
    for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4']):
        text = tag.get_text().strip()
        if len(text.split()) > 3 and not re.match(r'^\s*$', text):
            main_content.append(text)

    full_text = " ".join(main_content)
    return re.sub(r'\s+', ' ', full_text)

def fetch_additional_data(company_name: str) -> dict:
    """
    Fetches additional company data from Crunchbase API.
    """
    data = {}
    if CRUNCHBASE_API_KEY:
        # Fetch data from Crunchbase API
        try:
            api_url = f"https://api.crunchbase.com/api/v4/entities/organizations/{company_name}"
            params = {
                "field_ids": ["identifier", "description", "funding_total", "num_employees_enum"],
                "user_key": CRUNCHBASE_API_KEY
            }
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            logging.info(f"Fetched additional data for {company_name} from Crunchbase")
        except requests.RequestException as e:
            logging.error(f"Error fetching data from Crunchbase: {e}")
    else:
        logging.warning("Crunchbase API key not provided. Skipping data enrichment.")

    return data

def sentiment_analysis(content: str) -> str:
    """
    Performs sentiment analysis on the content.
    """
    blob = TextBlob(content)
    sentiment = blob.sentiment.polarity  # Returns a value between -1.0 (negative) and 1.0 (positive)
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

def generate_swot_analysis(content: str, company_name: str) -> str:
    """
    Generates a SWOT analysis using OpenAI's API.
    """
    prompt = f"""
    Based on the information about {company_name}, provide a SWOT analysis with the following sections:

    1. Strengths
    2. Weaknesses
    3. Opportunities
    4. Threats

    Content: {content}
    """

    try:
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=prompt,
            max_tokens=500,
            temperature=0.5
        )
        logging.info(f"Generated SWOT analysis for {company_name}")
        return response['choices'][0]['text'].strip()
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API error during SWOT analysis: {e}")
        return ""

def calculate_competitive_score(additional_data: dict) -> int:
    """
    Calculates a competitive score based on predefined criteria.
    """
    score = 0
    try:
        # Example scoring criteria (this can be expanded)
        funding = additional_data.get('data', {}).get('properties', {}).get('funding_total', 0)
        if funding:
            funding = int(funding)
            if funding > 10000000:
                score += 30
            elif funding > 1000000:
                score += 20
            else:
                score += 10

        employees = additional_data.get('data', {}).get('properties', {}).get('num_employees_enum', '')
        if employees:
            if employees in ['11-50', '51-100']:
                score += 10
            elif employees in ['101-250', '251-500']:
                score += 20
            else:
                score += 30

        logging.info(f"Calculated competitive score: {score}")
    except Exception as e:
        logging.error(f"Error calculating competitive score: {e}")

    return score

def generate_analysis_prompt(content: str, company_name: str, additional_data: dict, template: list) -> str:
    """
    Prepares the structured prompt for OpenAI to generate a competitor analysis.
    """
    prompt_sections = "\n".join([f"{i+1}. {section}" for i, section in enumerate(template)])
    prompt = f"""
    Based on the content provided about {company_name}, along with additional data, generate a competitor analysis report with these sections:

    {prompt_sections}

    Website Content:
    {content}

    Additional Data:
    {json.dumps(additional_data, indent=2)}
    """
    return prompt

def analyze_content(content: str, company_name: str, additional_data: dict, template: list) -> Optional[str]:
    """
    Generates a competitor analysis report based on content, using OpenAI's language model.
    """
    prompt = generate_analysis_prompt(content, company_name, additional_data, template)
    try:
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=prompt,
            max_tokens=3000,
            temperature=0.5
        )
        logging.info(f"Generated analysis for {company_name}")
        return response['choices'][0]['text'].strip()
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return None

def generate_report(url: str, template_name: str = 'standard') -> Tuple[Optional[str], Optional[str]]:
    """
    Generates a competitor analysis report for a given company URL.
    """
    logging.info(f"Starting report generation for {url}")
    content = fetch_website_content(url)
    if not content:
        logging.error("Failed to retrieve content for analysis.")
        return None, None

    company_name = extract_company_name(url)
    additional_data = fetch_additional_data(company_name)

    # Sentiment Analysis
    sentiment = sentiment_analysis(content)
    additional_data['sentiment'] = sentiment

    # SWOT Analysis
    swot_analysis = generate_swot_analysis(content, company_name)

    # Competitive Scoring
    competitive_score = calculate_competitive_score(additional_data)
    additional_data['competitive_score'] = competitive_score

    # Choose report template
    template = REPORT_TEMPLATES.get(template_name, REPORT_TEMPLATES['standard'])

    report = analyze_content(content, company_name, additional_data, template)

    # Append SWOT Analysis and Competitive Scoring
    if report:
        report += f"\n\nSWOT Analysis:\n{swot_analysis}"
        report += f"\n\nCompetitive Score: {competitive_score}/100"
        logging.info(f"Finalized report for {company_name}")
    else:
        logging.error(f"Failed to generate report for {company_name}")

    return (report, company_name) if report else (None, None)

def extract_company_name(url: str) -> str:
    """
    Extracts the company name from the URL.
    """
    domain = url.split("//")[-1].split("/")[0]
    company_name = domain.replace('www.', '').split('.')[0]
    company_name = company_name.replace('-', ' ').title()
    return company_name

def save_report(report: str, company_name: str, output_dir: str = "reports", format: str = "pdf") -> str:
    """
    Saves the generated report to a specified format (PDF or text).
    Returns the path to the saved report.
    """
    os.makedirs(output_dir, exist_ok=True)

    if format == "txt":
        filename = os.path.join(output_dir, f"{company_name}_analysis_report.txt")
        with open(filename, "w", encoding='utf-8') as file:
            file.write(report)
        logging.info(f"Report saved as {filename}")
    elif format == "pdf":
        filename = os.path.join(output_dir, f"{company_name}_analysis_report.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"{company_name} Competitor Analysis Report", ln=True, align='C')
        pdf.ln(10)

        # Content
        sections = re.split(r'(?=\d+\.\s)', report)
        for section in sections:
            match = re.match(r'(\d+\.\s)(.*)', section.strip())
            if match:
                header = match.group(2).split('\n')[0]
                content = section.replace(match.group(1) + header, '').strip()
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, header, ln=True)
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 8, content)
                pdf.ln(5)
            else:
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 8, section.strip())
                pdf.ln(5)

        pdf.output(filename)
        logging.info(f"Report saved as {filename}")
    else:
        logging.error("Unsupported format. Please choose 'txt' or 'pdf'.")
        raise ValueError("Unsupported format. Please choose 'txt' or 'pdf'.")

    return filename

# Uncomment this section if you want to run the script without the API
# if __name__ == "__main__":
#     # Example usage
#     urls = [
#         'https://scale.com/',
#         'https://anothercompany.com/'
#     ]
#     for url in urls:
#         report, company_name = generate_report(url)
#         if report and company_name:
#             save_report(report, company_name, output_format="pdf")
#         else:
#             logging.error(f"Failed to generate report for {url}")
