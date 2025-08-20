import streamlit as st
import os
import pypdf
import json
from datetime import datetime

# Google Auth imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI

# === Configuration ===
TOKEN_FILE = "token.json"
CREDS_FILE = "credentials.json"
SCOPES = ["https://mail.google.com/"]


# === Session-based Storage Functions ===
def init_session_history():
    """Initialize session-based activity history"""
    if 'activity_history' not in st.session_state:
        st.session_state.activity_history = []

def add_to_session_history(entry):
    """Add entry to session-based history"""
    if 'activity_history' not in st.session_state:
        st.session_state.activity_history = []
    
    # Add timestamp
    entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add to beginning of list (most recent first)
    st.session_state.activity_history.insert(0, entry)
    
    # Keep only last 10 entries to manage memory
    if len(st.session_state.activity_history) > 10:
        st.session_state.activity_history = st.session_state.activity_history[:10]

def clear_session_data():
    """Clear all session data including persistent inputs"""
    keys_to_clear = ['activity_history', 'analysis_result', 'email_content', 'credentials_content',
                     'job_desc_text', 'recipient_email_text', 'uploaded_file_data', 'uploaded_file_name']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Also remove credentials.json file if it exists
    if os.path.exists("credentials.json"):
        try:
            os.remove("credentials.json")
        except:
            pass  # Ignore errors if file can't be removed

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as pdf_file:
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

class JobApplicationAgent:
    """
    Job Application Agent that handles:
    1. Resume processing and storage for attachment
    2. Manual job description input only
    3. Resume-job matching analysis
    4. Professional email generation
    5. Email sending/drafting with PDF attachments
    
    Workflow:
    - generate_professional_email() creates email content
    - attach_and_send_email() handles PDF attachment and sending
    - _send_email_with_attachment() is the internal Gmail agent function
    """
    def __init__(self, google_api_key=None, sender_email=None):
        self.google_api_key = google_api_key
        self.sender_email = sender_email
        self.setup_environment()
        self.gmail_tools = None
        self.gmail_credentials = None  # Store session-based Gmail credentials
        self.resume_text = None
        self.resume_file_path = None
        self.job_description = None
        
    def setup_environment(self):
        """Setup API keys and authentication from session"""
        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key
    
    def setup_gmail_auth(self):
        """Setup Gmail authentication using session-based credentials"""
        try:
            # Check if credentials are available in session
            if 'credentials_content' not in st.session_state:
                return False, "Gmail credentials not uploaded. Please upload credentials.json file."
            
            # Use session-based credentials
            creds = None
            
            # Check for existing token in session
            if 'gmail_token' in st.session_state:
                try:
                    creds = Credentials.from_authorized_user_info(st.session_state.gmail_token, SCOPES)
                except Exception:
                    # Token invalid, will need to re-authenticate
                    pass
            
            # If no valid credentials, authenticate using uploaded credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    # Create flow using session credentials
                    flow = InstalledAppFlow.from_client_config(st.session_state.credentials_content, SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Store token in session (not file)
                st.session_state.gmail_token = json.loads(creds.to_json())
            
            # Store credentials for use in email functions
            self.gmail_credentials = creds
            return True, "Gmail connected successfully!"
            
        except Exception as e:
            return False, f"Failed to setup Gmail: {str(e)}"
    
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
    
    def set_job_description(self, job_description):
        """Set job description manually"""
        try:
            if not job_description or not job_description.strip():
                return False, "Please provide a job description"
            
            self.job_description = job_description.strip()
            return True, f"Job description set successfully! ({len(self.job_description)} characters)"
        except Exception as e:
            return False, f"Error setting job description: {e}"
    
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
    
    def _convert_to_html_format(self, plain_text):
        """Convert plain text email to HTML format for better display"""
        import html
        
        # Escape HTML characters
        html_content = html.escape(plain_text)
        
        # Convert line breaks to HTML
        # Double line breaks become paragraph breaks
        html_content = html_content.replace('\n\n', '</p><p>')
        # Single line breaks become <br> tags
        html_content = html_content.replace('\n', '<br>')
        
        # Wrap in HTML structure with professional styling
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            line-height: 1.6;
            color: #333333;
            margin: 20px;
            background-color: #ffffff;
        }}
        p {{
            margin: 16px 0;
            text-align: left;
        }}
        .greeting {{
            margin-bottom: 20px;
        }}
        .signature {{
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <p>{html_content}</p>
</body>
</html>"""
        
        return html_body
    
    def send_email(self, recipient_email, email_content, action="draft"):
        """Send or draft email using Gmail with PDF attachment and plain text format"""
        try:
            if not self.gmail_credentials:
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
            # Use session-based Gmail credentials
            creds = self.gmail_credentials
            if not creds or not creds.valid:
                return False, "Gmail credentials not available. Please reconnect to Gmail."
            
            # Build Gmail service directly
            from googleapiclient.discovery import build
            service = build('gmail', 'v1', credentials=creds)
            
            # Create email with attachment
            import base64
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.mime.base import MIMEBase
            from email import encoders
            
            # Create main message container
            msg = MIMEMultipart()
            msg['to'] = recipient_email
            msg['subject'] = email_content['subject']
            
            # Create alternative container for text/html versions
            msg_alternative = MIMEMultipart('alternative')
            
            # Add plain text version (fallback)
            msg_alternative.attach(MIMEText(email_content['body'], 'plain'))
            
            # Add HTML version (preferred)
            html_body = self._convert_to_html_format(email_content['body'])
            msg_alternative.attach(MIMEText(html_body, 'html'))
            
            # Attach the alternative part to main message
            msg.attach(msg_alternative)
            
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
            if not self.gmail_credentials:
                return False, "Gmail not connected"
            if not os.path.exists(attachment_path):
                return False, f"Attachment file not found: {attachment_path}"
            # Create email content structure
            email_content = {
                "subject": subject,
                "body": body
            }
            return self._send_email_with_attachment(recipient_email, email_content, attachment_path, action)
        except Exception as e:
            return False, f"Error in attach_and_send_email: {e}"
    
def main():
    st.set_page_config(
        page_title="Smart Job Application Agent", 
        page_icon="🚀", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 1rem;
        height: 150px;
    }
    .status-box {
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: .1rem 2rem;
        margin: 0.1rem 0rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    .status-box h4 {
        color: #000000 !important;
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
    }
    .status-box h3 {
        color: #000000 !important;
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
    }
    .status-box p {
        color: #000000 !important;
        margin: 0.25rem 0 0 0;
        font-size: 0.85rem;
    }
    .status-active {
        background: #e3f2fd;
        border: 2px solid #1976d2;
    }
    .status-active h4, .status-active p {
        color: #1976d2 !important;
    }
    .status-success {
        background: #e8f5e8;
        border: 2px solid #4caf50;
    }
    .status-success h3, .status-success h4, .status-success p {
        color: #2e7d32 !important;
    }
    .status-error {
        background: #ffebee;
        border: 2px solid #f44336;
    }
    .status-error h3, .status-error h4, .status-error p {
        color: #c62828 !important;
    }
    .input-section {
        background: transparent !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: none !important;
        box-shadow: none !important;
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
    
    /* Custom styling for sidebar warning box */
    .sidebar .stAlert > div {
        font-size: 0.75rem !important;
        padding: 0.5rem !important;
        margin: 0.25rem 0 !important;
        line-height: 1.2 !important;
    }
    
    /* Alternative: Target warning specifically */
    div[data-testid="stAlert"] div {
        font-size: 0.75rem !important;
        padding: 0.5rem 0.75rem !important;
        margin: 0.25rem 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h4>🚀 Smart Job Application Agent</h4>
        <p>Upload Resume → Paste Job Description → Enter Email → Get Professional Results!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API Key, Credentials, and History
    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        google_api_key = st.text_input(
            "Google API Key", 
            type="password", 
            placeholder="Enter your Google Gemini API key here...",
            help="Get your API key from Google AI Studio",
            key="api_key_input"
        )
        
        # Credentials file upload for Gmail
        st.markdown("### 📁 Gmail Credentials")
        credentials_file = st.file_uploader(
            "Upload credentials.json",
            type="json",
            help="Upload your Google Cloud Platform credentials JSON file for Gmail access",
            key="credentials_upload"
        )
        
        # Store credentials in session and save to file
        if credentials_file is not None:
            try:
                import json
                credentials_content = json.load(credentials_file)
                
                # Save to credentials.json file
                with open("credentials.json", "w") as f:
                    json.dump(credentials_content, f, indent=2)
                
                # Also store in session for reference
                st.session_state.credentials_content = credentials_content
                st.markdown('<div style="background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 0.25rem 0.5rem; border-radius: 3px; font-size: 0.7rem; margin: 0.1rem 0; line-height: 1.2;">✅ Credentials uploaded and saved successfully!</div>', unsafe_allow_html=True)
                st.markdown('<div style="background: #cce5ff; border: 1px solid #99ccff; color: #004085; padding: 0.25rem 0.5rem; border-radius: 3px; font-size: 0.7rem; margin: 0.1rem 0; line-height: 1.2;">📧 Ready for Gmail authentication</div>', unsafe_allow_html=True)
                
            except json.JSONDecodeError:
                st.markdown('<div style="background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 0.25rem 0.5rem; border-radius: 3px; font-size: 0.7rem; margin: 0.1rem 0; line-height: 1.2;">❌ Invalid JSON file. Please upload a valid credentials.json file.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div style="background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 0.25rem 0.5rem; border-radius: 3px; font-size: 0.7rem; margin: 0.1rem 0; line-height: 1.2;">❌ Error saving credentials: {e}</div>', unsafe_allow_html=True)
        
        if not google_api_key:
            st.markdown('<div style="background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 0.25rem 0.5rem; border-radius: 3px; font-size: 0.7rem; margin: 0.1rem 0; line-height: 1.2;">⚠️ Please enter your Google API key.</div>', unsafe_allow_html=True)
            st.stop()
        
        # Check if credentials.json exists
        if not os.path.exists("credentials.json"):
            st.markdown('<div style="background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 0.5rem; border-radius: 4px; font-size: 0.75rem; margin: 0.25rem 0;">Please upload your credentials.json file.</div>', unsafe_allow_html=True)
            st.stop()
        
        # Initialize agent with API key only (email comes from Gmail OAuth)
        agent = JobApplicationAgent(google_api_key=google_api_key, sender_email=None)
        
        # Gmail setup
        gmail_status, gmail_msg = agent.setup_gmail_auth()
        if gmail_status:
            st.markdown('<div style="background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 0.25rem 0.5rem; border-radius: 3px; font-size: 0.7rem; margin: 0.1rem 0; line-height: 1.2;">✅ Gmail connected successfully!</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 0.25rem 0.5rem; border-radius: 3px; font-size: 0.7rem; margin: 0.1rem 0; line-height: 1.2;">❌ Gmail connection failed: {gmail_msg}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Email History with slider
        st.markdown("### � Activity History")
        # history = load_email_history()  # REMOVED - session only
        init_session_history()
        if st.session_state.activity_history:  # Changed to session
            # Display session history directly (no authentication needed)
            history = st.session_state.activity_history  # Use session history
            st.markdown(f"*Showing all {len(history)} session activities*")
            
            # Show all activities with enhanced display
            for i, entry in enumerate(reversed(history), 1):
                if isinstance(entry, dict):
                    # Determine entry type
                    entry_type = entry.get('type', 'email')
                    timestamp = entry.get('timestamp', 'Unknown time')
                    
                    if entry_type == 'analysis':
                        # Analysis Card
                        st.markdown(f"""
                        <div style="border: 1px solid #4CAF50; border-radius: 6px; padding: 8px; margin: 5px 0; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);">
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <span style="font-size: 16px; margin-right: 8px;">🔍</span>
                                <div>
                                    <h5 style="margin: 0; color: #2e7d32; font-size: 0.8rem;">Analysis #{i}</h5>
                                    <small style="color: #666; font-size: 0.65rem;">{timestamp}</small>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Expandable content for analysis
                        with st.expander(f"📊 View Full Analysis Results", expanded=False):
                            st.markdown("#### Analysis Details")
                            st.write(f"**Status:** {entry.get('status', 'Completed')}")
                            
                            if entry.get('job_title'):
                                st.write(f"**Job Title:** {entry.get('job_title')}")
                            
                            if entry.get('match_score'):
                                st.write(f"**Match Score:** {entry.get('match_score')}")
                            
                            if entry.get('summary'):
                                st.markdown("#### Full Analysis Summary")
                                st.text_area("", value=entry.get('summary'), height=300, disabled=True, key=f"analysis_summary_{i}")
                                
                            if entry.get('user_email'):
                                st.write(f"**User:** {entry.get('user_email')}")
                    else:
                        # Email Card
                        action = entry.get('action', 'Unknown').title()
                        recipient = entry.get('recipient', 'Unknown')
                        
                        st.markdown(f"""
                        <div style="border: 1px solid #2196F3; border-radius: 6px; padding: 8px; margin: 5px 0; background: linear-gradient(135deg, #f8f9fa 0%, #e3f2fd 100%);">
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <span style="font-size: 16px; margin-right: 8px;">📧</span>
                                <div>
                                    <h5 style="margin: 0; color: #1565c0; font-size: 0.8rem;">Email {action} #{i}</h5>
                                    <small style="color: #666; font-size: 0.65rem;">{timestamp}</small>
                                </div>
                            </div>
                            <p style="margin: 3px 0; color: #333; font-size: 0.7rem;"><strong>To:</strong> {recipient}</p>
                            <p style="margin: 3px 0; color: #333; font-size: 0.7rem;"><strong>Subject:</strong> {entry.get('subject', 'No Subject')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Expandable content for email
                        with st.expander(f"📝 View Full Email Content", expanded=False):
                            st.markdown("#### Email Details")
                            st.write(f"**To:** {entry.get('recipient', 'Unknown')}")
                            st.write(f"**Subject:** {entry.get('subject', 'No Subject')}")
                            st.write(f"**Action:** {entry.get('action', 'Unknown').title()}")
                            st.write(f"**Status:** {entry.get('status', 'Unknown')}")
                            
                            if entry.get('body'):
                                st.markdown("#### Full Email Body")
                                st.text_area("", value=entry.get('body'), height=400, disabled=True, key=f"email_body_{i}")
                                
                            if entry.get('user_email'):
                                st.write(f"**Sender:** {entry.get('user_email')}")
                else:
                    st.info("No activity history found yet.")
        
        # Session Management
        st.markdown("---")
        st.markdown("### 🧹 Session Management")
        if st.button("Clear All Session Data", type="secondary", help="Clear inputs, history, and all session data"):
            clear_session_data()
            st.success("Session data cleared!")
            st.rerun()
    
    # Main input interface - persistent until browser closes
    # Status Display Box
    st.markdown("""
    <div class="status-box">
        <h4>🎯 Ready to Process Your Application</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # Three main inputs with session state persistence
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 📄 Upload Resume")
        uploaded_file = st.file_uploader("Choose PDF file", type="pdf", help="Upload your resume in PDF format", key="resume_upload")
        # Store file in session for persistence
        if uploaded_file is not None:
            st.session_state.uploaded_file_data = uploaded_file.getbuffer()
            st.session_state.uploaded_file_name = uploaded_file.name
    
    with col2:
        st.markdown("### 📝 Job Description")
        # Initialize session state for job description
        if 'job_desc_text' not in st.session_state:
            st.session_state.job_desc_text = ""
        
        manual_job_desc = st.text_area("Paste Job Description", 
                                     value=st.session_state.job_desc_text,
                                     placeholder="Copy the complete job description from any job site and paste here...", 
                                     height=100, 
                                     help="Copy the full job description text from the job posting",
                                     key="job_desc_input")
        # Update session when changed
        if manual_job_desc != st.session_state.job_desc_text:
            st.session_state.job_desc_text = manual_job_desc
    
    with col3:
        st.markdown("### 📧 Recipient Email")
        # Initialize session state for recipient email
        if 'recipient_email_text' not in st.session_state:
            st.session_state.recipient_email_text = ""
        
        recipient_email = st.text_input("Recruiter's email", 
                                       value=st.session_state.recipient_email_text,
                                       placeholder="recruiter@company.com", 
                                       help="Enter the recruiter's email address", 
                                       key="email_input")
        # Update session when changed
        if recipient_email != st.session_state.recipient_email_text:
            st.session_state.recipient_email_text = recipient_email
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons (not in form for persistence)
    col_analyze, col_draft, col_send = st.columns([1, 1, 1])
    
    with col_analyze:
        analyze_clicked = st.button("🔍 Analyze Match", use_container_width=True, type="secondary")
    
    with col_draft:
        draft_clicked = st.button("📝 Create Draft", use_container_width=True, type="secondary")
    
    with col_send:
        send_clicked = st.button("📨 Send Email", use_container_width=True, type="primary")
    
    # Process actions
    if analyze_clicked or draft_clicked or send_clicked:
        # Determine which task is being performed
        current_task = 'analyze' if analyze_clicked else ('draft' if draft_clicked else 'send')
        
        # Update status display
        task_descriptions = {
            'analyze': {'emoji': '🔍', 'title': 'Analyzing Job Match', 'desc': 'Comparing your resume with the job requirements to find matches and suggestions'},
            'draft': {'emoji': '📝', 'title': 'Creating Email Draft', 'desc': 'Generating professional email draft with your resume attached'},
            'send': {'emoji': '📨', 'title': 'Sending Email', 'desc': 'Sending professional email with resume attachment to the recruiter'}
        }
        
        task_info = task_descriptions[current_task]
        
        # Create a status container that we can update
        status_container = st.empty()
        
        # Show current task status
        status_container.markdown(f"""
        <div class="status-box status-active">
            <h4>{task_info['emoji']} {task_info['title']} ...</h4>
            <p>{task_info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for results
        if 'analysis_result' not in st.session_state:
            st.session_state.analysis_result = None
        if 'email_content' not in st.session_state:
            st.session_state.email_content = None
        
        # Validate inputs (check both current upload and session data)
        if not uploaded_file and 'uploaded_file_data' not in st.session_state:
            st.error("❌ Please upload your resume")
            st.stop()
        
        if not manual_job_desc:
            st.error("❌ Please paste the job description in the text area")
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
            # Step 1: Process Resume (handle both current upload and session data)
            status_text.text("📄 Processing resume...")
            progress_bar.progress(20)
            
            if uploaded_file:
                # Fresh upload
                resume_success, resume_msg = agent.process_resume(uploaded_file)
            elif 'uploaded_file_data' in st.session_state:
                # Use session data - create temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(st.session_state.uploaded_file_data)
                    temp_file.flush()
                    
                    # Create a file-like object
                    class SessionFile:
                        def __init__(self, path, name):
                            self.name = name
                            self._path = path
                        def getbuffer(self):
                            with open(self._path, 'rb') as f:
                                return f.read()
                    
                    session_file = SessionFile(temp_file.name, st.session_state.uploaded_file_name)
                    resume_success, resume_msg = agent.process_resume(session_file)
            else:
                resume_success, resume_msg = False, "No resume found"
            if not resume_success:
                st.error(f"❌ {resume_msg}")
                st.stop()
            
            # Step 2: Set Job Description
            status_text.text("📝 Processing job description...")
            progress_bar.progress(40)
            
            job_success, job_msg = agent.set_job_description(manual_job_desc)
            if not job_success:
                st.error(f"❌ {job_msg}")
                st.stop()
            
            st.info("✅ Job description loaded successfully")
            
            # Step 3: Analyze Match (always run, but only show for analyze_clicked)
            status_text.text("🔍 Analyzing job match...")
            progress_bar.progress(60)
            
            analysis_success, analysis_result = agent.analyze_job_match()
            if not analysis_success:
                st.error(f"❌ {analysis_result}")
                st.stop()
            
            # Only store and show analysis for analyze button, run silently for draft/send
            if analyze_clicked:
                # Store analysis result in session state for display
                st.session_state.analysis_result = analysis_result
                st.session_state.show_analysis = True  # Flag to show analysis results
                
                # Save analysis to session history
                add_to_session_history({
                    "type": "analysis",
                    "status": "completed",
                    "summary": analysis_result,
                    "job_title": manual_job_desc[:100] if len(manual_job_desc) > 100 else manual_job_desc,
                    "match_score": "Analysis completed"
                })
            else:
                # For draft/send, analysis runs in background but results are not displayed
                st.session_state.show_analysis = False
            
            # Step 4: Email actions (if requested)
            if draft_clicked or send_clicked:
                status_text.text("✍️ Generating professional email...")
                progress_bar.progress(80)
                
                email_success, email_content = agent.generate_professional_email(recipient_email)
                if not email_success:
                    st.error(f"❌ {email_content}")
                    st.stop()
                
                # Store email content in session state
                st.session_state.email_content = email_content
                
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
                    
                    # Save to session history
                    add_to_session_history({
                        "type": "email",
                        "recipient": recipient_email,
                        "subject": email_content['subject'],
                        "body": email_content['body'],
                        "action": action,
                        "status": "success"
                    })
                    

                else:
                    st.error(f"❌ {send_result}")
            else:
                progress_bar.progress(100)
            
            status_text.text("✅ Process completed!")
            
            # Show completion status in the same status container
            task_success = {
                'analyze': {'emoji': '✅', 'title': 'Job Match Analysis Completed', 'desc': 'Resume analysis and job matching completed successfully'},
                'draft': {'emoji': '✅', 'title': 'Email Draft Created', 'desc': 'Professional email draft has been created and saved to Gmail'},
                'send': {'emoji': '✅', 'title': 'Email Sent Successfully', 'desc': 'Your professional email with resume has been sent to the recruiter'}
            }
            
            success_info = task_success[current_task]
            
            status_container.markdown(f"""
            <div class="status-box status-success">
                <h4>{success_info['emoji']} {success_info['title']}</h4>
                <p>{success_info['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")
            
            # Show error status in the same container
            status_container.markdown(f"""
            <div class="status-box status-error">
                <h4>❌ Task Failed</h4>
                <p>An error occurred while processing your request. Please check your inputs and try again.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display persistent results
    # Only show analysis if it was explicitly requested (analyze button clicked)
    if st.session_state.get('analysis_result') and st.session_state.get('show_analysis', False):
        st.markdown("### 🔍 Job Match Analysis")
        st.markdown(f'<div class="success-box">{st.session_state.analysis_result}</div>', unsafe_allow_html=True)
    
    # Always show email content if available (for draft/send operations)
    if st.session_state.get('email_content'):
        st.markdown("### 📧 Generated Email")
        st.markdown(f"**Subject:** {st.session_state.email_content['subject']}")
        st.markdown("**Body:**")
        st.text_area("Email content", value=st.session_state.email_content['body'], height=200, disabled=True, key="email_preview")

if __name__ == "__main__":
    main()
