# Smart Job Application Agent

An intelligent automation system using Python, Streamlit, and LangChain frameworks to streamline job applications. Integrated Google Gemini LLM APIs for resume-job matching analysis and professional email generation. Implemented web scraping techniques with BeautifulSoup, Gmail API integration for automated email handling, and PDF processing capabilities.

## Features

- 📄 **Resume Processing**: PDF upload and text extraction for analysis
- 🔍 **Job Description Extraction**: Web scraping from job posting URLs
- 🤖 **AI-Powered Matching**: Resume-job compatibility analysis using Google Gemini LLM
- ✍️ **Professional Email Generation**: Natural language processing for personalized application emails
- 📧 **Gmail Integration**: Automated email sending/drafting with PDF attachments
- 📊 **Match Analysis**: Detailed compatibility scoring and improvement suggestions

## Tech Stack

- **Languages**: Python
- **Frameworks**: Streamlit, LangChain
- **LLM APIs**: Google Gemini AI
- **Libraries**: BeautifulSoup, PyPDF, Requests
- **APIs**: Gmail API, Google Auth
- **Techniques**: Web scraping, NLP, ML, PDF processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Antriksh999/Job-Mail-and-analysis-agent.git
cd Job-Mail-and-analysis-agent
```

2. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create `.env` file with your Google API keys
   - Add Gmail credentials JSON file

5. Run the application:
```bash
streamlit run agent-4.py
```

## Usage

1. **Upload Resume**: Upload your PDF resume
2. **Job Description**: Paste job URL or description manually
3. **Analysis**: Get AI-powered resume-job matching analysis
4. **Email Generation**: Generate professional application emails
5. **Send/Draft**: Automatically send or create Gmail drafts with resume attachment

## Configuration

- Set `GOOGLE_API_KEY` in `.env` file
- Add Gmail `credentials.json` for email functionality
- Customize prompts and settings in the main script

## Contributing

Feel free to contribute by opening issues or submitting pull requests.

## License

This project is open source and available under the MIT License.