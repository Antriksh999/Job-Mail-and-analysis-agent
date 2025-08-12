import streamlit as st
import os
import json
import pypdf
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Google Auth imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# LangChain imports
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import build_resource_service
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Resume processing imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Search imports
from langchain_community.utilities import SerpAPIWrapper
try:
    from langchain.tools import Tool
except Exception:
    from langchain_core.tools import Tool

# === Configuration ===
CONFIG_FILE = "resume_config.json"
DB_PATH = "./chroma_resume"
HISTORY_FILE = "email_history.json"
TOKEN_FILE = "token.json"
CREDS_FILE = "credentials.json"
SCOPES = ["https://mail.google.com/"]

load_dotenv()

# Only set environment variables if they exist in .env file
google_api_key = os.getenv("GOOGLE_API_KEY")
serpapi_api_key = os.getenv("SERP_API_KEY")  # Note: using SERP_API_KEY to match .env file

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
if serpapi_api_key:
    os.environ["SERPAPI_API_KEY"] = serpapi_api_key

# === Utility Functions ===
def load_email_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            st.warning(f"Email history file corrupted, creating new one.")
            if os.path.exists(HISTORY_FILE):
                os.rename(HISTORY_FILE, f"{HISTORY_FILE}.backup")
            return []
    return []

def save_email_history(new_entry):
    try:
        history = load_email_history()
        history.append(new_entry)
        history = history[-10:]
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save email history: {e}")

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as pdf_file:
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

def validate_url(url):
    """Simple URL validation and correction"""
    if not url:
        return None
    
    url = url.strip()
    if not url:
        return None
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Basic URL pattern check
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if url_pattern.match(url):
        return url
    else:
        return None

def scrape_job_description(url):
    """Scrape job description from URL with site-specific strategies and JS handling"""
    try:
        # Check if this might be a JavaScript-heavy site
        if any(site in url.lower() for site in ['naukri.com', 'linkedin.com', 'indeed.com']):
            st.info("Detected JavaScript-heavy job site. The extracted content might be limited.")
            st.info("💡 **Tip**: For best results, copy the job description manually and paste it in the text area.")
        
        # Enhanced headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
        }
        
        # Make request with session for better handling
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=15, allow_redirects=True)
        response.raise_for_status()
        
        # Debug: Show response status
        st.info(f"Successfully fetched URL (Status: {response.status_code})")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check if page is JavaScript-heavy
        script_count = len(soup.find_all('script'))
        if script_count > 15:
            st.warning(f"⚠️ This page has {script_count} script tags and likely uses JavaScript rendering. Manual input is recommended.")
        
        # Generic extraction strategy only
        job_text = ""
        
        # Strategy 2: Generic job description selectors
        if not job_text:
            job_selectors = [
                # Job description specific
                '[class*="job-description"]', '[id*="job-description"]', '[class*="jobdescription"]',
                '[class*="job-detail"]', '[class*="job-content"]', '[class*="description"]',
                '[data-testid*="job"]', '[data-test*="job"]', '[class*="posting"]',
                
                # Content areas
                'main article', 'main section', '[role="main"]', '.main-content',
                '[class*="content"]', '[class*="details"]', '[class*="info"]',
                
                # Broader fallbacks
                'main', 'article', '.container', '.content'
            ]
            
            for selector in job_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        text = element.get_text(strip=True)
                        if len(text) > 300 and has_job_keywords(text):  # More content + job keywords
                            job_text = text
                            st.info(f"Found content using selector: {selector}")
                            break
                    if job_text:
                        break
        
        # Strategy 3: Try to extract from meta tags or structured data
        if not job_text:
            job_text = extract_from_meta_tags(soup)
        
        # Strategy 4: Extract from body if nothing else works
        if not job_text:
            st.warning("Using fallback extraction method...")
            job_text = extract_from_body(soup)
        
        # Clean and validate the extracted text
        if job_text:
            cleaned_text = clean_job_text(job_text)
            
            if len(cleaned_text) < 100:
                st.warning(f"Retrieved text seems too short ({len(cleaned_text)} chars). For JavaScript-heavy sites like Naukri, manual input works better.")
                return cleaned_text if cleaned_text else None
            
            # Final validation
            if not has_job_keywords(cleaned_text):
                st.warning("The extracted text doesn't seem to contain typical job description content. Please use manual input.")
                return cleaned_text  # Return anyway, let user decide
            
            st.success(f"Successfully extracted {len(cleaned_text)} characters from the job posting")
            return cleaned_text
        else:
            st.error("No meaningful content found on the page. Please copy the job description manually.")
            return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error while fetching job description: {e}")
        return None
    except Exception as e:
        st.error(f"Error parsing job description: {e}")
        return None

def extract_from_meta_tags(soup):
    """Try to extract job info from meta tags"""
    job_text = ""
    
    # Check meta description
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        content = meta_desc.get('content')
        if len(content) > 100 and has_job_keywords(content):
            job_text += content + " "
    
    # Check Open Graph description
    og_desc = soup.find('meta', attrs={'property': 'og:description'})
    if og_desc and og_desc.get('content'):
        content = og_desc.get('content')
        if len(content) > 100 and has_job_keywords(content):
            job_text += content + " "
    
    # Check for JSON-LD structured data
    json_scripts = soup.find_all('script', type='application/ld+json')
    for script in json_scripts:
        try:
            import json
            data = json.loads(script.string)
            if isinstance(data, dict) and 'description' in data:
                desc = data['description']
                if isinstance(desc, str) and len(desc) > 100:
                    job_text += desc + " "
        except:
            continue
    
    return job_text.strip()

def extract_from_body(soup):
    """Fallback extraction from body content"""
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'form', 'button']):
        element.decompose()
    
    # Get text from body
    body = soup.find('body') or soup
    return body.get_text()

def has_job_keywords(text):
    """Check if text contains job-related keywords"""
    job_keywords = [
        'job', 'position', 'role', 'responsibilities', 'requirements', 'qualifications',
        'experience', 'skills', 'company', 'work', 'employment', 'candidate', 'team',
        'salary', 'benefits', 'apply', 'hiring', 'career', 'opportunity', 'developer',
        'engineer', 'manager', 'analyst', 'intern', 'graduate', 'senior', 'junior'
    ]
    
    text_lower = text.lower()
    keyword_count = sum(1 for keyword in job_keywords if keyword in text_lower)
    return keyword_count >= 3

def clean_job_text(text):
    """Clean and process job description text"""
    # Split into lines and filter
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Remove very short lines and navigation elements
    filtered_lines = []
    for line in lines:
        if len(line) < 15:  # Skip very short lines
            continue
        
        # Skip navigation/UI elements
        skip_phrases = [
            'cookie', 'privacy', 'terms', 'copyright', 'sign in', 'log in', 'register',
            'subscribe', 'newsletter', 'social media', 'follow us', 'contact us',
            'about us', 'careers', 'help', 'support', 'menu', 'navigation', 'search',
            'filter', 'sort by', 'page', 'next', 'previous', 'home', 'back to top',
            'share', 'print', 'save', 'bookmark', 'report', 'flag'
        ]
        
        line_lower = line.lower()
        if not any(phrase in line_lower for phrase in skip_phrases):
            filtered_lines.append(line)
    
    # Join lines
    cleaned_text = ' '.join(filtered_lines)
    
    # Clean up spacing and characters
    import re
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Limit length
    if len(cleaned_text) > 6000:
        cleaned_text = cleaned_text[:6000] + "..."
    
    return cleaned_text

class JobApplicationAgent:
    """
    Job Application Agent that handles:
    1. Resume processing and storage for attachment
    2. Job description extraction from URLs
    3. Resume-job matching analysis
    4. Professional email generation
    5. Email sending/drafting with PDF attachments
    
    Workflow:
    - generate_professional_email() creates email content
    - attach_and_send_email() handles PDF attachment and sending
    - _send_email_with_attachment() is the internal Gmail agent function
    """
    def __init__(self):
        self.setup_environment()
        self.gmail_tools = None
        self.resume_text = None
        self.resume_file_path = None
        self.job_description = None
        
    def setup_environment(self):
        """Setup API keys and authentication"""
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.serpapi_key = os.getenv("SERP_API_KEY")  # Note: using SERP_API_KEY to match .env file
        
        # Only set environment variables if they exist
        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key
        if self.serpapi_key:
            os.environ["SERPAPI_API_KEY"] = self.serpapi_key
    
    def setup_gmail_auth(self):
        """Setup Gmail authentication"""
        try:
            creds = None
            if os.path.exists(TOKEN_FILE):
                creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if os.path.exists(CREDS_FILE):
                        flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
                        creds = flow.run_local_server(port=0)
                    else:
                        return False, "Gmail credentials.json file not found."
                with open(TOKEN_FILE, "w") as token:
                    token.write(creds.to_json())
            
            api_resource = build_resource_service(credentials=creds)
            toolkit = GmailToolkit(api_resource=api_resource)
            self.gmail_tools = toolkit.get_tools()
            return True, "Gmail connected successfully!"
        except Exception as e:
            return False, f"Failed to setup Gmail: {e}"
    
    def process_resume(self, uploaded_file):
        """Process uploaded resume and save file for attachment"""
        try:
            # Save uploaded file permanently for email attachment
            self.resume_file_path = f"resume_{uploaded_file.name}"
            with open(self.resume_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text for analysis only
            self.resume_text = extract_text_from_pdf(self.resume_file_path)
            
            if not self.resume_text:
                return False, "Could not extract text from PDF"
                
            return True, f"Resume processed successfully! ({len(self.resume_text)} characters)"
        except Exception as e:
            return False, f"Error processing resume: {e}"
    
    def fetch_job_description(self, job_url):
        """Fetch job description from URL with enhanced error handling"""
        try:
            if not job_url or not job_url.strip():
                return False, "Please provide a valid job URL"
            
            # Validate and fix URL format
            validated_url = validate_url(job_url)
            if not validated_url:
                return False, "Invalid URL format. Please check the URL and try again."
            
            st.info(f"Attempting to fetch job description from: {validated_url}")
            
            self.job_description = scrape_job_description(validated_url)
            
            if self.job_description and len(self.job_description.strip()) > 50:
                # Show preview of extracted content for debugging
                preview = self.job_description[:500] + "..." if len(self.job_description) > 500 else self.job_description
                with st.expander("📝 Preview of extracted job description (click to expand)"):
                    st.text_area("Extracted content:", value=preview, height=150, disabled=True)
                return True, f"Job description fetched successfully! ({len(self.job_description)} characters)"
            else:
                return False, f"Could not fetch meaningful job description from URL. Try using the manual input option below."
                
        except Exception as e:
            return False, f"Error fetching job description: {str(e)}"
    
    def analyze_job_match(self):
        """Analyze resume against job description with improved error handling"""
        try:
            if not self.resume_text or not self.job_description:
                return False, "Resume or job description not available"
            
            # Setup LLM with more conservative settings
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3
            )
            
            # Simplified analysis prompt without agent framework
            analysis_prompt = f"""
            You are a professional resume analyst. Please analyze this resume against the job description.
            
            JOB DESCRIPTION:
            {self.job_description[:3000]}
            
            RESUME CONTENT:
            {self.resume_text[:3000]}
            
            Please provide a clear analysis with the following sections:
            
            1. MATCH PERCENTAGE: Give an estimated percentage (0-100%) of how well the resume matches the job requirements.
            
            2. MISSING KEYWORDS: List 5-10 important keywords from the job description that are missing in the resume.
            
            3. IMPROVEMENT SUGGESTIONS: Provide 3-5 specific, actionable suggestions to improve the resume for this job.
            
            4. OVERALL ASSESSMENT: Summarize whether this candidate would be a good fit and why.
            
            Keep your response clear, professional, and structured. If the job description seems unclear or corrupted, focus on what you can understand and mention any limitations.
            """
            
            try:
                response = llm.invoke(analysis_prompt)
                analysis_result = response.content
                
                # Basic validation of the response
                if len(analysis_result) < 100:
                    return False, "Analysis response was too short. Please try again."
                
                return True, analysis_result
                
            except Exception as llm_error:
                # Fallback: Basic text analysis without LLM
                st.warning("LLM analysis failed, providing basic text comparison...")
                
                basic_analysis = self._basic_text_analysis()
                return True, basic_analysis
                
        except Exception as e:
            return False, f"Error in job analysis: {str(e)}"
    
    def _basic_text_analysis(self):
        """Fallback basic analysis if LLM fails"""
        try:
            job_words = set(self.job_description.lower().split())
            resume_words = set(self.resume_text.lower().split())
            
            # Calculate basic overlap
            common_words = job_words.intersection(resume_words)
            match_percentage = min(100, int((len(common_words) / len(job_words)) * 100)) if job_words else 0
            
            # Find potential missing keywords (simplified)
            important_words = [word for word in job_words if len(word) > 4 and word.isalpha()]
            missing_words = [word for word in important_words[:20] if word not in resume_words]
            
            basic_result = f"""
            BASIC TEXT ANALYSIS (LLM analysis failed):
            
            MATCH PERCENTAGE: Approximately {match_percentage}%
            
            MISSING KEYWORDS: {', '.join(missing_words[:10]) if missing_words else 'None identified'}
            
            IMPROVEMENT SUGGESTIONS:
            - Consider adding more keywords from the job description
            - Review the job requirements and highlight matching experience
            - Customize your resume to better align with this specific role
            
            OVERALL ASSESSMENT: Basic text comparison completed. For detailed analysis, please ensure the job description is clear and try again.
            
            Note: This is a simplified analysis due to technical limitations. For best results, ensure both resume and job description are clearly formatted.
            """
            
            return basic_result
            
        except Exception:
            return "Unable to perform analysis due to technical issues. Please verify your inputs and try again."
    
    def generate_professional_email(self, recipient_email):
        """Generate professional email content with PDF attachment"""
        try:
            if not self.resume_text or not self.job_description:
                return False, "Resume or job description not available"
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3
            )
            
            # Extract candidate name from resume for signature
            candidate_name = "Applicant"  # Default name
            # Try to extract a name from the resume text
            import re
            name_match = re.search(r'([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})', self.resume_text[:500])
            if name_match:
                candidate_name = name_match.group(1)
                
            # Get date in proper format
            from datetime import datetime
            current_date = datetime.now().strftime("%B %d, %Y")
            
            # Extract job title from job description
            job_title = "position"
            job_title_match = re.search(r'([A-Z][a-zA-Z\s]+(?:Developer|Engineer|Scientist|Intern|Analyst|Manager|Specialist))', self.job_description[:1000])
            if job_title_match:
                job_title = job_title_match.group(1)
            
            # Email generation prompt - let AI think naturally
            email_prompt = f"""
            You are writing a professional job application email. Think about this naturally and write a good email.
            
            JOB DETAILS:
            {self.job_description[:1500]}
            
            CANDIDATE RESUME:
            {self.resume_text[:1500]}
            
            Write a professional job application email. Be natural and authentic. The resume will be attached as a PDF file.
            Email to the hiring manager highlighting my relevant skills.
            End with either "Sincerely," "Best regards," or similar, followed by {candidate_name} on the next line.
            
            Start your response with:
            Subject: [your subject line]
            
            Then write the email body naturally.
            """
            
            response = llm.invoke(email_prompt)
            email_content = response.content
            
            # Parse subject and body with better error handling
            try:
                lines = email_content.split('\n')
                subject = ""
                body_lines = []
                
                # Look for subject line
                for i, line in enumerate(lines):
                    if line.lower().startswith('subject:'):
                        subject = line.split(':', 1)[1].strip()
                        # Start body from next non-empty line
                        for j in range(i + 1, len(lines)):
                            if lines[j].strip():
                                body_lines = lines[j:]
                                break
                        break
                
                # If no subject found, use first line that looks like a subject
                if not subject:
                    for line in lines[:3]:
                        if any(keyword in line.lower() for keyword in ['application', 'position', 'role', 'job']):
                            subject = line.strip()
                            break
                    
                    if not subject:
                        subject = f"Job Application - {candidate_name}"
                    
                    # Use all content as body
                    body_lines = lines
                
                # Join body lines
                body = '\n'.join(body_lines).strip()
                
                # Clean up subject
                subject = subject.strip('"').strip("'").strip()
                
                # Don't include resume content in body - it will be attached as PDF
                
                # Format the body with proper business letter formatting
                formatted_body = self._format_business_letter(body)
                
                return True, {"subject": subject, "body": formatted_body}
                
            except Exception as parse_error:
                # If parsing fails, let AI write naturally
                subject = f"Job Application - {candidate_name}"
                
                simple_body = f"""Dear Hiring Team,

I am interested in applying for the position at your company.

My background and experience are outlined in the attached resume. I would appreciate the opportunity to discuss my qualifications with you.

Thank you for your consideration.

Best regards,
{candidate_name}"""
                
                formatted_body = self._format_business_letter(simple_body)
                return True, {"subject": subject, "body": formatted_body}
                
        except Exception as e:
            return False, f"Error generating email: {e}"
    
    def _format_business_letter(self, body_text):
        """Format email body with proper business letter structure"""
        import re
        
        # Normalize line breaks
        body_text = re.sub(r'\r\n', '\n', body_text)
        body_text = re.sub(r'\r', '\n', body_text)
        
        # Ensure proper greeting format
        body_text = re.sub(r'^(Dear [^,\n]+)[,]?\s*\n?', r'\1,\n\n', body_text)
        
        # Format text into proper paragraphs
        formatted_body = self._format_text_paragraphs(body_text)
        
        # Ensure proper closing format with applicant name
        formatted_body = re.sub(r'\n*(Sincerely,|Best regards,|Warm regards,)\s*\n*([A-Za-z\s]+)$', r'\n\n\1\n\2', formatted_body)
        
        return formatted_body
    
    def _format_text_paragraphs(self, text):
        """Format text into proper paragraphs"""
        # Split into paragraphs and clean up
        paragraphs = []
        lines = text.split('\n')
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(line)
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Join paragraphs with double line breaks
        return '\n\n'.join(paragraphs)
    
    def send_email(self, recipient_email, email_content, action="draft"):
        """Send or draft email using Gmail with PDF attachment and plain text format"""
        try:
            if not self.gmail_tools:
                return False, "Gmail not connected"
            
            if not self.resume_file_path or not os.path.exists(self.resume_file_path):
                return False, "Resume file not found for attachment"
            
            # Use the separate attachment function
            return self._send_email_with_attachment(recipient_email, email_content, self.resume_file_path, action)
            
        except Exception as e:
            return False, f"Error {action}ing email: {e}"
    
    def _send_email_with_attachment(self, recipient_email, email_content, attachment_path, action="draft"):
        """Internal function to handle email with PDF attachment using direct Gmail API"""
        try:
            # Get Gmail credentials
            creds = None
            if os.path.exists(TOKEN_FILE):
                creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            
            if not creds or not creds.valid:
                return False, "Gmail credentials not available"
            
            # Build Gmail service directly
            from googleapiclient.discovery import build
            service = build('gmail', 'v1', credentials=creds)
            
            # Create email with attachment
            import base64
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.mime.base import MIMEBase
            from email import encoders
            
            # Create message
            msg = MIMEMultipart()
            msg['to'] = recipient_email
            msg['subject'] = email_content['subject']
            
            # Add body
            msg.attach(MIMEText(email_content['body'], 'plain'))
            
            # Add attachment
            if os.path.exists(attachment_path):
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                filename = os.path.basename(attachment_path)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}',
                )
                msg.attach(part)
            
            # Convert to bytes
            raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            
            # Send or draft
            if action == "draft":
                # Create draft
                draft = service.users().drafts().create(
                    userId='me',
                    body={'message': {'raw': raw_message}}
                ).execute()
                return True, f"Draft created successfully with ID: {draft.get('id')}"
            else:
                # Send email
                message = service.users().messages().send(
                    userId='me',
                    body={'raw': raw_message}
                ).execute()
                return True, f"Email sent successfully with ID: {message.get('id')}"
            
        except Exception as e:
            return False, f"Error in Gmail API: {e}"
    
    def attach_and_send_email(self, recipient_email, subject, body, attachment_path, action="draft"):
        """General function to attach any file and send/draft email"""
        try:
            if not self.gmail_tools:
                return False, "Gmail not connected"
            
            if not os.path.exists(attachment_path):
                return False, f"Attachment file not found: {attachment_path}"
            
            # Create email content structure
            email_content = {
                "subject": subject,
                "body": body
            }
            
            # Use the internal attachment function
            return self._send_email_with_attachment(recipient_email, email_content, attachment_path, action)
            
        except Exception as e:
            return False, f"Error in attach_and_send_email: {e}"
    
def main():
    st.set_page_config(
        page_title="Smart Job Application Agent", 
        page_icon="🚀", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Smart Job Application Agent</h1>
        <p>Upload Resume → Paste Job URL → Enter Email → Get Professional Results!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agent
    agent = JobApplicationAgent()
    
    # API Keys check
    if not agent.google_api_key or not agent.serpapi_key:
        st.error("⚠️ Please set GOOGLE_API_KEY and SERPAPI_API_KEY in your .env file")
        st.stop()
    
    # Gmail setup
    gmail_status, gmail_msg = agent.setup_gmail_auth()
    if gmail_status:
        st.success(f"✅ {gmail_msg}")
    else:
        st.warning(f"⚠️ {gmail_msg}")
    
    # Main input form
    with st.form("job_application_form", clear_on_submit=False):
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        # Three main inputs
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("### 📄 Upload Resume")
            uploaded_file = st.file_uploader("Choose PDF file", type="pdf", help="Upload your resume in PDF format")
        
        with col2:
            st.markdown("### 🔗 Job Post URL")
            job_url = st.text_input("Job posting URL", placeholder="https://company.com/job-posting", help="Paste the full URL of the job posting")
            
            # Add option for manual job description input
            st.markdown("**OR**")
            st.info("💡 **For JavaScript-heavy sites**: Copy the job description from the website and paste it here for best results!")
            manual_job_desc = st.text_area("Paste Job Description Manually", 
                                         placeholder="Copy the complete job description from any job site and paste here...", 
                                         height=120, 
                                         help="This is the recommended approach for sites that load content dynamically")
        
        with col3:
            st.markdown("### 📧 Recipient Email")
            recipient_email = st.text_input("Recruiter's email", placeholder="recruiter@company.com", help="Enter the recruiter's email address")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        col_analyze, col_draft, col_send = st.columns([1, 1, 1])
        
        with col_analyze:
            analyze_clicked = st.form_submit_button("🔍 Analyze Match", use_container_width=True, type="secondary")
        
        with col_draft:
            draft_clicked = st.form_submit_button("📝 Create Draft", use_container_width=True, type="secondary")
        
        with col_send:
            send_clicked = st.form_submit_button("📨 Send Email", use_container_width=True, type="primary")
    
    # Process actions
    if analyze_clicked or draft_clicked or send_clicked:
        # Validate inputs
        if not uploaded_file:
            st.error("❌ Please upload your resume")
            st.stop()
        
        if not job_url and not manual_job_desc:
            st.error("❌ Please provide either a job posting URL or paste the job description manually")
            st.stop()
        
        if (draft_clicked or send_clicked) and not recipient_email:
            st.error("❌ Please provide the recipient's email address")
            st.stop()
        
        if (draft_clicked or send_clicked) and not gmail_status:
            st.error("❌ Gmail is not connected. Please add credentials.json file.")
            st.stop()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Process Resume
            status_text.text("📄 Processing resume...")
            progress_bar.progress(20)
            
            resume_success, resume_msg = agent.process_resume(uploaded_file)
            if not resume_success:
                st.error(f"❌ {resume_msg}")
                st.stop()
            
            # Step 2: Fetch Job Description
            status_text.text("🔗 Fetching job description...")
            progress_bar.progress(40)
            
            # Use manual job description if provided, otherwise scrape URL
            if manual_job_desc and manual_job_desc.strip():
                agent.job_description = manual_job_desc.strip()
                job_success = True
                job_msg = f"Manual job description loaded successfully! ({len(agent.job_description)} characters)"
                st.info("✅ Using manually provided job description")
            elif job_url:
                job_success, job_msg = agent.fetch_job_description(job_url)
            else:
                job_success = False
                job_msg = "No job description provided"
            
            if not job_success:
                st.error(f"❌ {job_msg}")
                st.stop()
            
            # Step 3: Analyze Match (always do this)
            status_text.text("🔍 Analyzing job match...")
            progress_bar.progress(60)
            
            analysis_success, analysis_result = agent.analyze_job_match()
            if not analysis_success:
                st.error(f"❌ {analysis_result}")
                st.stop()
            
            # Display analysis results
            st.markdown("### 📊 Job Match Analysis")
            st.markdown(f'<div class="success-box">{analysis_result}</div>', unsafe_allow_html=True)
            
            # Step 4: Email actions (if requested)
            if draft_clicked or send_clicked:
                status_text.text("✍️ Generating professional email...")
                progress_bar.progress(80)
                
                email_success, email_content = agent.generate_professional_email(recipient_email)
                if not email_success:
                    st.error(f"❌ {email_content}")
                    st.stop()
                
                # Display generated email
                st.markdown("### 📧 Generated Email")
                st.markdown(f"**Subject:** {email_content['subject']}")
                st.markdown("**Body:**")
                st.text_area("Email content", value=email_content['body'], height=200, disabled=True)
                
                # Now use the attachment function to send/draft
                action = "draft" if draft_clicked else "send"
                status_text.text(f"📤 {'Creating draft' if draft_clicked else 'Sending email'} with PDF attachment...")
                progress_bar.progress(100)
                
                # Use the dedicated attachment function
                send_success, send_result = agent.attach_and_send_email(
                    recipient_email, 
                    email_content['subject'], 
                    email_content['body'], 
                    agent.resume_file_path, 
                    action
                )
                
                if send_success:
                    success_msg = f"✅ Email {'drafted' if draft_clicked else 'sent'} successfully with PDF attachment!"
                    st.markdown(f'<div class="success-box">{success_msg}<br><small>{send_result}</small></div>', unsafe_allow_html=True)
                else:
                    st.error(f"❌ {send_result}")
            else:
                progress_bar.progress(100)
            
            status_text.text("✅ Process completed!")
            
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")
    
    # Email History
    if st.checkbox("📊 Show Email History"):
        history = load_email_history()
        if history:
            st.markdown("### 📜 Recent Email History")
            for i, email in enumerate(reversed(history[-5:]), 1):
                if isinstance(email, dict):
                    with st.expander(f"Email {i}: {email.get('subject', 'No Subject')} ({email.get('action', 'unknown')})"):
                        st.write(f"**To:** {email.get('recipient', 'Unknown')}")
                        st.write(f"**Subject:** {email.get('subject', 'No Subject')}")
                        st.write(f"**Action:** {email.get('action', 'Unknown')}")
                        st.write(f"**Body:** {email.get('body', 'No content')[:200]}...")
        else:
            st.info("No email history found.")

if __name__ == "__main__":
    main()