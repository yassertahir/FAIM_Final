import streamlit as st
import time
import os
import json
import pandas as pd
import numpy as np
import re
from openai import OpenAI
from utils import create_assistant, create_thread, create_message, get_response

# Set page configuration
st.set_page_config(page_title="VC Assistant", layout="wide")

# Initialize API client
OPENAI_API_KEY = st.secrets["openai_api_key"]
client = OpenAI(api_key=OPENAI_API_KEY)

# File to store IDs - use absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_FILE = os.path.join(BASE_DIR, "assistant_data.json")

# Initialize session state variables
if "page" not in st.session_state:
    st.session_state.page = "upload"  # Options: "upload", "report", "valuation"

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "file_metadata" not in st.session_state:
    st.session_state.file_metadata = []

if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

if "report_content" not in st.session_state:
    st.session_state.report_content = ""

if "checklist_values" not in st.session_state:
    # Initialize with default values: [weight, score]
    st.session_state.checklist_values = {
        "founders_team": [30, 75],
        "idea": [20, 70],
        "market": [20, 70],
        "product_ip": [15, 65],
        "execution": [15, 40]
    }

if "scorecard_values" not in st.session_state:
    # Initialize with default values: [weight, multiplier]
    st.session_state.scorecard_values = {
        "team_strength": [24, 1.4],
        "opportunity_size": [22, 1.2],
        "product_service": [20, 0.7],
        "competition": [16, 1.3],
        "marketing_sales": [12, 0.9],
        "need_funding": [6, 1.0]
    }

if "perfect_valuation" not in st.session_state:
    st.session_state.perfect_valuation = 10000000  # $10M default

if "median_valuation" not in st.session_state:
    st.session_state.median_valuation = 8000000  # $8M default

# Function to navigate between pages
def change_page(page):
    st.session_state.page = page

# Function to calculate the checklist valuation
def calculate_checklist():
    total = 0
    for key, value in st.session_state.checklist_values.items():
        weight = value[0] / 100  # Convert percentage to decimal
        score = value[1] / 100   # Convert percentage to decimal
        total += weight * score * st.session_state.perfect_valuation
    return total

# Function to calculate the scorecard valuation
def calculate_scorecard():
    total = 0
    for key, value in st.session_state.scorecard_values.items():
        weight = value[0] / 100  # Convert percentage to decimal
        multiplier = value[1]    # Premium/discount multiplier
        total += weight * multiplier * st.session_state.median_valuation
    return total

# Function to load or create assistant and thread
def get_or_create_assistant_and_thread():
    # Default values
    assistant_id = None
    thread_id = None
    
    # Check if storage file exists
    try:
        if os.path.exists(STORAGE_FILE):
            with open(STORAGE_FILE, "r") as f:
                data = json.load(f)
                
                # Verify the assistant still exists
                try:
                    assistant = client.beta.assistants.retrieve(data.get("assistant_id", ""))
                    assistant_id = data["assistant_id"]
                    st.sidebar.info(f"Using existing assistant: {assistant_id}")
                except Exception as e:
                    st.sidebar.warning(f"Assistant retrieval error: {e}")
                    assistant_id = None
                    
                # Verify the thread still exists
                try:
                    thread = client.beta.threads.retrieve(data.get("thread_id", ""))
                    thread_id = data["thread_id"]
                    st.sidebar.info(f"Using existing thread: {thread_id}")
                except Exception as e:
                    st.sidebar.warning(f"Thread retrieval error: {e}")
                    thread_id = None
    except Exception as e:
        st.sidebar.warning(f"Storage file error: {e}")
        assistant_id = None
        thread_id = None
    
    # Create new assistant if needed
    if assistant_id is None:
        try:
            assistant_id = create_assistant(client)
            st.sidebar.success(f"Created new assistant: {assistant_id}")
        except Exception as e:
            st.sidebar.error(f"Failed to create assistant: {e}")
            return None, None
    
    # Create new thread if needed
    if thread_id is None:
        try:
            thread_id = create_thread(client)
            st.sidebar.success(f"Created new thread: {thread_id}")
        except Exception as e:
            st.sidebar.error(f"Failed to create thread: {e}")
            return assistant_id, None
    
    # Save IDs
    try:
        os.makedirs(os.path.dirname(STORAGE_FILE), exist_ok=True)
        with open(STORAGE_FILE, "w") as f:
            json.dump({
                "assistant_id": assistant_id,
                "thread_id": thread_id
            }, f)
        st.sidebar.success("Assistant data saved successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to save assistant data: {e}")
    
    return assistant_id, thread_id

# Function to wait for active runs
def wait_for_active_runs(client, thread_id, max_wait_seconds=60):
    """Check for and wait for any active runs to complete with timeout"""
    start_time = time.time()
    
    while True:
        # Check if we've waited too long
        elapsed_time = time.time() - start_time
        if elapsed_time > max_wait_seconds:
            st.sidebar.warning(f"Timed out after waiting {max_wait_seconds} seconds for run to complete")
            return False
        
        # List runs for the thread
        runs = client.beta.threads.runs.list(thread_id=thread_id)
        
        # Filter for active runs
        active_runs = [run for run in runs.data if run.status in ["queued", "in_progress"]]
        requires_action_runs = [run for run in runs.data if run.status == "requires_action"]
        
        # No runs at all, we can continue
        if not active_runs and not requires_action_runs:
            return True
            
        # If there are runs requiring action, don't wait for them here
        if requires_action_runs:
            st.sidebar.info("Run requires action - continuing with function calling...")
            return True
        
        # Only wait for queued or in_progress runs
        if active_runs:
            status = active_runs[0].status
            st.sidebar.info(f"Waiting for active run to complete ({status})... {int(elapsed_time)}s elapsed")
            time.sleep(1)

# Get or create assistant and thread
assistant_id, thread_id = get_or_create_assistant_and_thread()
st.session_state.assistant_id = assistant_id
st.session_state.thread_id = thread_id

# Sidebar content
with st.sidebar:
    st.title("VC Assistant")
    
    # Navigation
    st.subheader("Navigation")
    st.button("Upload Files", on_click=change_page, args=["upload"], disabled=st.session_state.page=="upload")
    st.button("View Report", on_click=change_page, args=["report"], disabled=not st.session_state.report_generated or st.session_state.page=="report")
    st.button("Valuation Tools", on_click=change_page, args=["valuation"], disabled=not st.session_state.report_generated or st.session_state.page=="valuation")
    
    # Create a new conversation button
    if st.button("Start New Conversation"):
        # Create a new thread
        new_thread_id = create_thread(client)
        
        # Update the storage file
        try:
            with open(STORAGE_FILE, "r") as f:
                data = json.load(f)
            
            data["thread_id"] = new_thread_id
            
            with open(STORAGE_FILE, "w") as f:
                json.dump(data, f)
            
            # Update session state
            st.session_state.thread_id = new_thread_id
            st.session_state.processed_files = set()
            st.session_state.file_metadata = []
            st.session_state.report_generated = False
            st.session_state.report_content = ""
            change_page("upload")
            
            st.sidebar.success("Started a new conversation!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error starting new conversation: {e}")

# Upload Page
if st.session_state.page == "upload":
    st.title("VC Assistant - File Upload")
    
    # Add explanatory text
    st.markdown("""
    ## Upload Your Startup Documents
    
    The VC Assistant will analyze your startup documents and provide a comprehensive evaluation, 
    including valuations using the **Checklist Method** and **Scorecard Method**.
    
    **Recommended documents to upload:**
    - Business plan or pitch deck
    - Financial projections
    - Team CVs or bios
    - Market research data
    - Product specifications
    - Any other relevant startup documentation
    """)
    
    # Create two columns
    upload_col, info_col = st.columns([2, 1])
    
    with upload_col:
        # File uploader with enhanced UI
        uploaded_files = st.file_uploader(
            "Upload files...", 
            accept_multiple_files=True,
            type=["pdf", "csv", "xlsx", "docx", "txt"]
        )
        
        # Show what files are ready to be analyzed with more visual representation
        if uploaded_files:
            st.subheader("Files ready for analysis:")
            file_info = []
            for uploaded_file in uploaded_files:
                status = "✅ Processed" if uploaded_file.name in st.session_state.processed_files else "⏳ Ready to process"
                file_type = uploaded_file.name.split('.')[-1].upper()
                
                # Icons based on file type
                if file_type == "PDF":
                    icon = "📕"
                elif file_type in ["CSV", "XLSX"]:
                    icon = "📊"
                elif file_type == "DOCX":
                    icon = "📝"
                else:
                    icon = "📄"
                
                file_info.append({
                    "Icon": icon,
                    "File Name": uploaded_file.name,
                    "Type": file_type,
                    "Status": status
                })
            
            # Display as a nice table
            st.table(pd.DataFrame(file_info))
    
    with info_col:
        # Show information about the valuation methods
        st.markdown("""
        ### Valuation Methods
        
        #### Checklist Method
        Evaluates key areas with weighted scores:
        - Founders & Team (30%)
        - Idea (20%)
        - Market (20%)
        - Product & IP (15%)
        - Execution (15%)
        
        #### Scorecard Method
        Compares against median valuation:
        - Team Strength (24%)
        - Opportunity Size (22%)
        - Product/Service (20%)
        - Competition (16%)
        - Marketing & Sales (12%)
        - Need for Funding (6%)
        """)
        
        # Add a note about the process
        st.info("After uploading, click 'Analyze All Files' to generate a comprehensive report and valuation analysis.")
    
    # Add a button to process all uploaded files
    if uploaded_files and any(file.name not in st.session_state.processed_files for file in uploaded_files):
        if st.button("Analyze All Files"):
            file_ids = []
            file_names = []
            
            # First, upload all files to OpenAI
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    # Build a full path based on the current file's directory
                    temp_file_path = os.path.join(BASE_DIR, f"{uploaded_file.name}")
                    
                    # Write the file to that path
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Upload file to OpenAI
                        file = client.files.create(
                            file=open(temp_file_path, "rb"),
                            purpose="assistants"
                        )
                        
                        # Add to our tracking lists
                        file_ids.append({
                            "file_id": file.id, 
                            "file_name": uploaded_file.name
                        })
                        file_names.append(uploaded_file.name)
                        st.sidebar.success(f"File uploaded: {uploaded_file.name}")
                        
                        # Clean up temp file
                        os.remove(temp_file_path)
                        
                        # Mark as processed
                        st.session_state.processed_files.add(uploaded_file.name)
                        
                        # Store file metadata for future context
                        st.session_state.file_metadata.append({
                            "file_name": uploaded_file.name,
                            "file_id": file.id,
                            "upload_time": time.time()
                        })
                        
                    except Exception as e:
                        st.sidebar.error(f"Error uploading {uploaded_file.name}: {e}")
            
            # If we have files to process
            if file_ids:
                # Send a system message instructing the assistant to remember document contents
                system_message = client.beta.threads.messages.create(
                    thread_id=st.session_state.thread_id,
                    role="user",
                    content=[{
                        "type": "text", 
                        "text": "I've uploaded important documents about a startup proposal. Please remember the contents of these documents throughout our entire conversation, even after multiple exchanges. When I ask questions later, refer back to these documents for context."
                    }]
                )
                
                # Wait for any active runs
                wait_for_active_runs(client, st.session_state.thread_id)
                
                # Separate CSV files from other files for appropriate handling
                csv_files = [f for f in file_ids if f["file_name"].lower().endswith('.csv')]
                non_csv_files = [f for f in file_ids if not f["file_name"].lower().endswith('.csv')]
                
                # Create attachments list for file_search-compatible files
                attachments = []
                for file_info in non_csv_files:
                    attachments.append({
                        "file_id": file_info["file_id"],
                        "tools": [{"type": "file_search"}]
                    })
                
                # Create attachments list for CSV files that need code_interpreter
                csv_attachments = []
                for file_info in csv_files:
                    csv_attachments.append({
                        "file_id": file_info["file_id"],
                        "tools": [{"type": "code_interpreter"}]
                    })
                
                # Create a message with all attachments
                message_text = f"I've uploaded {len(file_ids)} files for analysis: {', '.join(file_names)}. "
                
                if csv_files:
                    message_text += f"The following are CSV files that need code_interpreter: {', '.join(f['file_name'] for f in csv_files)}. "
                
                message_text += "Please analyze all files together for a comprehensive evaluation using both Checklist Method and Scorecard Method."
                
                # Create message with all attachments
                message = client.beta.threads.messages.create(
                    thread_id=st.session_state.thread_id,
                    role="user",
                    content=[{
                        "type": "text",
                        "text": message_text
                    }],
                    attachments=attachments + csv_attachments  # Combine all attachments
                )
                
                # Wait for any active runs that message might create
                wait_for_active_runs(client, st.session_state.thread_id)
                
                # Show progress
                with st.status("Analyzing files...", expanded=True) as status:
                    st.write("Analyzing uploaded files...")
                    
                    has_csv = any(file.endswith('.csv') for file in file_names)
                    has_pdf = any(file.endswith('.pdf') for file in file_names)
                    
                    # Build dynamic instructions
                    instructions = "Please analyze all the uploaded files together to provide a comprehensive evaluation. "

                    if has_csv:
                        instructions += "For CSV files, use the code_interpreter tool to analyze the data. "

                    if has_pdf and has_csv:
                        instructions += "Use both the business plan in the PDF and analyze the competitive data in the CSV files. "

                    instructions += """
                    Please provide a comprehensive evaluation of the attached startup proposal. 
                    Follow the exact format below to ensure the valuation parameters can be automatically extracted:

                    Structure your response with these sections:
                    1. Executive Summary
                    2. Detailed Analysis (strengths, weaknesses, opportunities)
                    3. Team Assessment
                    4. Market Analysis
                    5. Product Evaluation
                    6. Checklist Method Results
                    7. Scorecard Method Results
                    8. Final Recommendation

                    For the Checklist Method section, include a table exactly in this format:
                    
                    ## Checklist Method Results
                    
                    | Key Measures | Weight | Company Score | Perfect Valuation | Checklist Valuation |
                    | ------------ | ------ | ------------- | ----------------- | ------------------- |
                    | Founders & Team | 30% | XX% | $10,000,000 | $X,XXX,XXX |
                    | Idea | 20% | XX% | $10,000,000 | $X,XXX,XXX |
                    | Market | 20% | XX% | $10,000,000 | $X,XXX,XXX |
                    | Product & IP | 15% | XX% | $10,000,000 | $X,XXX,XXX |
                    | Execution | 15% | XX% | $10,000,000 | $X,XXX,XXX |
                    | **Total** | **100%** | | | **$X,XXX,XXX** |
                    
                    Replace XX% with your actual scores. Each score should be between 0-100%.

                    For the Scorecard Method section, include a table exactly in this format:
                    
                    ## Scorecard Method Results
                    
                    | Key Measures | Weight | Premium/Discount | Pre-money Median | Scorecard Valuation |
                    | ------------ | ------ | ---------------- | ---------------- | ------------------- |
                    | Team Strength | 24% | X.X | $8,000,000 | $X,XXX,XXX |
                    | Opportunity Size | 22% | X.X | $8,000,000 | $X,XXX,XXX |
                    | Product/Service | 20% | X.X | $8,000,000 | $X,XXX,XXX |
                    | Competition | 16% | X.X | $8,000,000 | $X,XXX,XXX |
                    | Marketing & Sales | 12% | X.X | $8,000,000 | $X,XXX,XXX |
                    | Need for Funding | 6% | X.X | $8,000,000 | $X,XXX,XXX |
                    | **Total** | **100%** | | | **$X,XXX,XXX** |
                    
                    Replace X.X with your actual multipliers, where 1.0 is average, above 1.0 is premium, below 1.0 is discount.
                    
                    Also include a summary section with valuation results:
                    
                    ## Valuation Summary
                    
                    - Checklist Method Valuation: $X,XXX,XXX
                    - Scorecard Method Valuation: $X,XXX,XXX
                    - Average Valuation: $X,XXX,XXX
                    
                    Be sure to follow this format exactly so the scores and multipliers can be automatically extracted.
                    """

                    # Create the run with our instructions
                    run = client.beta.threads.runs.create(
                        thread_id=st.session_state.thread_id,
                        assistant_id=st.session_state.assistant_id,
                        instructions=instructions
                    )
                    
                    # Wait for the run to complete
                    wait_for_active_runs(client, st.session_state.thread_id, max_wait_seconds=120)
                    
                    # Get the response
                    response_content = get_response(client, st.session_state.thread_id, st.session_state.assistant_id)
                    
                    # Store the report content
                    st.session_state.report_content = response_content
                    st.session_state.report_generated = True
                    
                    # Try to extract values from the report
                    try:
                        import re
                        
                        # Extract Checklist Method scores
                        # Look for patterns like "Founders & Team: 75%" or "Founders & Team (75%)"
                        founders_pattern = r"Founders\s*&\s*Team[:\(]\s*(\d+)%"
                        idea_pattern = r"Idea[:\(]\s*(\d+)%"
                        market_pattern = r"Market[:\(]\s*(\d+)%"
                        product_pattern = r"Product\s*(?:&|and)\s*IP[:\(]\s*(\d+)%"
                        execution_pattern = r"Execution[:\(]\s*(\d+)%"
                        
                        # Search for matches
                        if match := re.search(founders_pattern, response_content):
                            st.session_state.checklist_values["founders_team"][1] = int(match.group(1))
                        
                        if match := re.search(idea_pattern, response_content):
                            st.session_state.checklist_values["idea"][1] = int(match.group(1))
                            
                        if match := re.search(market_pattern, response_content):
                            st.session_state.checklist_values["market"][1] = int(match.group(1))
                            
                        if match := re.search(product_pattern, response_content):
                            st.session_state.checklist_values["product_ip"][1] = int(match.group(1))
                            
                        if match := re.search(execution_pattern, response_content):
                            st.session_state.checklist_values["execution"][1] = int(match.group(1))
                        
                        # Extract Scorecard Method multipliers
                        # Look for patterns like "Team Strength: 1.4x" or "Team Strength (1.4x)"
                        team_pattern = r"Team\s*Strength[:\(]\s*([\d\.]+)x?"
                        opportunity_pattern = r"Opportunity\s*Size[:\(]\s*([\d\.]+)x?"
                        product_srv_pattern = r"Product\/Service[:\(]\s*([\d\.]+)x?"
                        competition_pattern = r"Competition[:\(]\s*([\d\.]+)x?"
                        marketing_pattern = r"Marketing\s*(?:&|and)\s*Sales[:\(]\s*([\d\.]+)x?"
                        funding_pattern = r"Need\s*(?:for)?\s*Funding[:\(]\s*([\d\.]+)x?"
                        
                        # Search for matches
                        if match := re.search(team_pattern, response_content):
                            st.session_state.scorecard_values["team_strength"][1] = float(match.group(1))
                            
                        if match := re.search(opportunity_pattern, response_content):
                            st.session_state.scorecard_values["opportunity_size"][1] = float(match.group(1))
                            
                        if match := re.search(product_srv_pattern, response_content):
                            st.session_state.scorecard_values["product_service"][1] = float(match.group(1))
                            
                        if match := re.search(competition_pattern, response_content):
                            st.session_state.scorecard_values["competition"][1] = float(match.group(1))
                            
                        if match := re.search(marketing_pattern, response_content):
                            st.session_state.scorecard_values["marketing_sales"][1] = float(match.group(1))
                            
                        if match := re.search(funding_pattern, response_content):
                            st.session_state.scorecard_values["need_funding"][1] = float(match.group(1))
                            
                    except Exception as e:
                        st.warning(f"Some values could not be automatically extracted from the report. You can adjust them manually in the Valuation Tools page.")
                    
                    status.update(label="Analysis complete!", state="complete", expanded=False)
                
                # Navigate to report page
                change_page("report")
                st.rerun()

# Report Page
elif st.session_state.page == "report":
    st.title("VC Assistant - Startup Analysis Report")
    
    if st.session_state.report_content:
        # Add a progress indicator and download button at the top
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### Analysis Complete")
            st.progress(100)
        
        with col2:
            if st.button("📊 Proceed to Valuation", type="primary"):
                change_page("valuation")
                st.rerun()
        
        # Add tabs for different sections of the report
        tab1, tab2, tab3, tab4 = st.tabs(["Executive Summary", "Detailed Analysis", "Valuation Data", "Full Report"])
        
        with tab1:
            # Extract and display executive summary
            report_content = st.session_state.report_content
            if "Executive Summary" in report_content:
                summary_start = report_content.find("Executive Summary")
                next_heading = re.search(r"\n##? ", report_content[summary_start+1:])
                
                if next_heading:
                    summary_end = summary_start + next_heading.start() + 1
                    summary = report_content[summary_start:summary_end]
                else:
                    summary = "Executive summary not found in the expected format."
                
                st.markdown(summary)
            else:
                # Try an alternative approach
                first_paragraph = report_content.split('\n\n')[0]
                if len(first_paragraph) > 100:
                    st.markdown("### Executive Summary")
                    st.markdown(first_paragraph)
                else:
                    st.markdown("Executive summary not found. Please view the full report.")
        
        with tab2:
            # Extract and display strengths and weaknesses
            strengths_section = ""
            weaknesses_section = ""
            
            if "Strengths" in report_content:
                strengths_start = report_content.find("Strengths")
                next_heading = re.search(r"\n##? ", report_content[strengths_start+1:])
                
                if next_heading:
                    strengths_end = strengths_start + next_heading.start() + 1
                    strengths_section = report_content[strengths_start:strengths_end]
            
            if "Areas for Improvement" in report_content or "Weaknesses" in report_content:
                section_title = "Areas for Improvement" if "Areas for Improvement" in report_content else "Weaknesses"
                weaknesses_start = report_content.find(section_title)
                next_heading = re.search(r"\n##? ", report_content[weaknesses_start+1:])
                
                if next_heading:
                    weaknesses_end = weaknesses_start + next_heading.start() + 1
                    weaknesses_section = report_content[weaknesses_start:weaknesses_end]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(strengths_section if strengths_section else "### Strengths\nStrengths section not found.")
            
            with col2:
                st.markdown(weaknesses_section if weaknesses_section else "### Areas for Improvement\nAreas for improvement not found.")
                
            # Team and Market Analysis
            team_section = ""
            market_section = ""
            
            if "Team Assessment" in report_content:
                team_start = report_content.find("Team Assessment")
                next_heading = re.search(r"\n##? ", report_content[team_start+1:])
                
                if next_heading:
                    team_end = team_start + next_heading.start() + 1
                    team_section = report_content[team_start:team_end]
            
            if "Market Analysis" in report_content:
                market_start = report_content.find("Market Analysis")
                next_heading = re.search(r"\n##? ", report_content[market_start+1:])
                
                if next_heading:
                    market_end = market_start + next_heading.start() + 1
                    market_section = report_content[market_start:market_end]
            
            st.markdown("---")
            
            if team_section:
                st.markdown(team_section)
            
            if market_section:
                st.markdown(market_section)
        
        with tab3:
            # Extract and display valuation data
            checklist_data = None
            scorecard_data = None
            
            # Try to extract valuation tables
            if "Checklist Method Results" in report_content:
                checklist_start = report_content.find("Checklist Method Results")
                next_heading = re.search(r"\n##? ", report_content[checklist_start+1:])
                
                if next_heading:
                    checklist_end = checklist_start + next_heading.start() + 1
                    checklist_data = report_content[checklist_start:checklist_end]
            
            if "Scorecard Method Results" in report_content:
                scorecard_start = report_content.find("Scorecard Method Results")
                next_heading = re.search(r"\n##? ", report_content[scorecard_start+1:])
                
                if next_heading:
                    scorecard_end = scorecard_start + next_heading.start() + 1
                    scorecard_data = report_content[scorecard_start:scorecard_end]
            
            # Display valuation data
            st.markdown("### Valuation Data")
            st.markdown("This data is used as the starting point for the valuation tools. You can adjust these values in the Valuation Tools section.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if checklist_data:
                    st.markdown(checklist_data)
                else:
                    st.markdown("### Checklist Method Results\nNo checklist data found.")
            
            with col2:
                if scorecard_data:
                    st.markdown(scorecard_data)
                else:
                    st.markdown("### Scorecard Method Results\nNo scorecard data found.")
            
            st.markdown("---")
            st.markdown("### Proceed to Interactive Valuation Tools")
            if st.button("Open Valuation Tools", type="primary"):
                change_page("valuation")
                st.rerun()
        
        with tab4:
            # Display full report
            st.markdown(report_content)
    else:
        # No report yet
        st.warning("No report has been generated. Please upload and analyze files first.")
        
        # Add an illustration of the process
        st.markdown("""
        ### Startup Evaluation Process
        
        1. **Upload Files** - Submit your business plan, financial projections, and team information
        2. **AI Analysis** - Our VC Assistant will analyze all documents
        3. **Review Report** - Examine the detailed analysis and findings
        4. **Interactive Valuation** - Use the valuation tools to explore different scenarios
        """)
        
        # Add a progress tracker
        st.error("● Upload Files → ○ AI Analysis → ○ Review Report → ○ Interactive Valuation")
        
        if st.button("Go to File Upload", type="primary"):
            change_page("upload")
            st.rerun()

# Valuation Page
elif st.session_state.page == "valuation":
    st.title("VC Assistant - Valuation Tools")
    
    # Create two columns for the two valuation methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Checklist Method")
        
        # Perfect valuation input
        st.session_state.perfect_valuation = st.number_input(
            "Perfect Valuation ($)",
            min_value=1000000,
            max_value=100000000,
            value=st.session_state.perfect_valuation,
            step=1000000,
            format="%d"
        )
        
        st.subheader("Adjust Weights and Scores")
        
        # Create sliders for each factor
        st.write("**Founders & Team**")
        weight_founders = st.slider(
            "Weight (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.checklist_values["founders_team"][0],
            key="weight_founders"
        )
        score_founders = st.slider(
            "Score (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.checklist_values["founders_team"][1],
            key="score_founders"
        )
        st.session_state.checklist_values["founders_team"] = [weight_founders, score_founders]
        
        st.write("**Idea**")
        weight_idea = st.slider(
            "Weight (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.checklist_values["idea"][0],
            key="weight_idea"
        )
        score_idea = st.slider(
            "Score (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.checklist_values["idea"][1],
            key="score_idea"
        )
        st.session_state.checklist_values["idea"] = [weight_idea, score_idea]
        
        st.write("**Market**")
        weight_market = st.slider(
            "Weight (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.checklist_values["market"][0],
            key="weight_market"
        )
        score_market = st.slider(
            "Score (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.checklist_values["market"][1],
            key="score_market"
        )
        st.session_state.checklist_values["market"] = [weight_market, score_market]
        
        st.write("**Product & IP**")
        weight_product = st.slider(
            "Weight (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.checklist_values["product_ip"][0],
            key="weight_product"
        )
        score_product = st.slider(
            "Score (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.checklist_values["product_ip"][1],
            key="score_product"
        )
        st.session_state.checklist_values["product_ip"] = [weight_product, score_product]
        
        st.write("**Execution**")
        weight_execution = st.slider(
            "Weight (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.checklist_values["execution"][0],
            key="weight_execution"
        )
        score_execution = st.slider(
            "Score (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.checklist_values["execution"][1],
            key="score_execution"
        )
        st.session_state.checklist_values["execution"] = [weight_execution, score_execution]
        
        # Calculate total weight to show warning if not 100%
        total_weight_checklist = sum(value[0] for value in st.session_state.checklist_values.values())
        if total_weight_checklist != 100:
            st.warning(f"Total weight is {total_weight_checklist}%. It should be 100%.")
        
        # Calculate and display valuation
        checklist_valuation = calculate_checklist()
        st.subheader("Checklist Valuation")
        st.write(f"${checklist_valuation:,.2f}")
        
        # Create a table showing the calculation
        checklist_data = []
        for key, value in st.session_state.checklist_values.items():
            weight = value[0] / 100
            score = value[1] / 100
            component_value = weight * score * st.session_state.perfect_valuation
            
            # Format the key for display
            display_key = key.replace("_", " ").title()
            
            checklist_data.append({
                "Key Measures": display_key,
                "Weight": f"{value[0]}%",
                "Company Score": f"{value[1]}%",
                "Perfect Valuation": f"${st.session_state.perfect_valuation:,.0f}",
                "Checklist Valuation": f"${component_value:,.0f}"
            })
        
        # Add total row
        checklist_data.append({
            "Key Measures": "Total",
            "Weight": f"{total_weight_checklist}%",
            "Company Score": "",
            "Perfect Valuation": f"${st.session_state.perfect_valuation:,.0f}",
            "Checklist Valuation": f"${checklist_valuation:,.0f}"
        })
        
        # Display the table
        st.table(pd.DataFrame(checklist_data))
    
    with col2:
        st.header("Scorecard Method")
        
        # Median valuation input
        st.session_state.median_valuation = st.number_input(
            "Median Pre-money Valuation ($)",
            min_value=1000000,
            max_value=100000000,
            value=st.session_state.median_valuation,
            step=1000000,
            format="%d"
        )
        
        st.subheader("Adjust Weights and Multipliers")
        
        # Create sliders for each factor
        st.write("**Team Strength**")
        weight_team = st.slider(
            "Weight (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.scorecard_values["team_strength"][0],
            key="weight_team"
        )
        multiplier_team = st.slider(
            "Premium/Discount", 
            min_value=0.1, 
            max_value=3.0, 
            value=float(st.session_state.scorecard_values["team_strength"][1]),
            step=0.1,
            key="multiplier_team"
        )
        st.session_state.scorecard_values["team_strength"] = [weight_team, multiplier_team]
        
        st.write("**Opportunity Size**")
        weight_opportunity = st.slider(
            "Weight (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.scorecard_values["opportunity_size"][0],
            key="weight_opportunity"
        )
        multiplier_opportunity = st.slider(
            "Premium/Discount", 
            min_value=0.1, 
            max_value=3.0, 
            value=float(st.session_state.scorecard_values["opportunity_size"][1]),
            step=0.1,
            key="multiplier_opportunity"
        )
        st.session_state.scorecard_values["opportunity_size"] = [weight_opportunity, multiplier_opportunity]
        
        st.write("**Product/Service**")
        weight_product_service = st.slider(
            "Weight (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.scorecard_values["product_service"][0],
            key="weight_product_service"
        )
        multiplier_product_service = st.slider(
            "Premium/Discount", 
            min_value=0.1, 
            max_value=3.0, 
            value=float(st.session_state.scorecard_values["product_service"][1]),
            step=0.1,
            key="multiplier_product_service"
        )
        st.session_state.scorecard_values["product_service"] = [weight_product_service, multiplier_product_service]
        
        st.write("**Competition**")
        weight_competition = st.slider(
            "Weight (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.scorecard_values["competition"][0],
            key="weight_competition"
        )
        multiplier_competition = st.slider(
            "Premium/Discount", 
            min_value=0.1, 
            max_value=3.0, 
            value=float(st.session_state.scorecard_values["competition"][1]),
            step=0.1,
            key="multiplier_competition"
        )
        st.session_state.scorecard_values["competition"] = [weight_competition, multiplier_competition]
        
        st.write("**Marketing & Sales**")
        weight_marketing = st.slider(
            "Weight (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.scorecard_values["marketing_sales"][0],
            key="weight_marketing"
        )
        multiplier_marketing = st.slider(
            "Premium/Discount", 
            min_value=0.1, 
            max_value=3.0, 
            value=float(st.session_state.scorecard_values["marketing_sales"][1]),
            step=0.1,
            key="multiplier_marketing"
        )
        st.session_state.scorecard_values["marketing_sales"] = [weight_marketing, multiplier_marketing]
        
        st.write("**Need for Funding**")
        weight_funding = st.slider(
            "Weight (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.scorecard_values["need_funding"][0],
            key="weight_funding"
        )
        multiplier_funding = st.slider(
            "Premium/Discount", 
            min_value=0.1, 
            max_value=3.0, 
            value=float(st.session_state.scorecard_values["need_funding"][1]),
            step=0.1,
            key="multiplier_funding"
        )
        st.session_state.scorecard_values["need_funding"] = [weight_funding, multiplier_funding]
        
        # Calculate total weight to show warning if not 100%
        total_weight_scorecard = sum(value[0] for value in st.session_state.scorecard_values.values())
        if total_weight_scorecard != 100:
            st.warning(f"Total weight is {total_weight_scorecard}%. It should be 100%.")
        
        # Calculate and display valuation
        scorecard_valuation = calculate_scorecard()
        st.subheader("Scorecard Valuation")
        st.write(f"${scorecard_valuation:,.2f}")
        
        # Create a table showing the calculation
        scorecard_data = []
        for key, value in st.session_state.scorecard_values.items():
            weight = value[0] / 100
            multiplier = value[1]
            component_value = weight * multiplier * st.session_state.median_valuation
            
            # Format the key for display
            display_key = key.replace("_", " ").title()
            
            scorecard_data.append({
                "Key Measures": display_key,
                "Weight": f"{value[0]}%",
                "Premium/Discount": f"{value[1]:.1f}",
                "Pre-money Median": f"${st.session_state.median_valuation:,.0f}",
                "Scorecard Valuation": f"${component_value:,.0f}"
            })
        
        # Add total row
        scorecard_data.append({
            "Key Measures": "Total",
            "Weight": f"{total_weight_scorecard}%",
            "Premium/Discount": "",
            "Pre-money Median": f"${st.session_state.median_valuation:,.0f}",
            "Scorecard Valuation": f"${scorecard_valuation:,.0f}"
        })
        
        # Display the table
        st.table(pd.DataFrame(scorecard_data))
    
    # Summary section at the bottom with enhanced visual presentation
    st.markdown("---")
    st.header("Valuation Summary")
    
    # Calculate average valuation
    average_valuation = (checklist_valuation + scorecard_valuation) / 2
    
    # Create a more visually appealing summary with metrics and comparison
    summary_col1, summary_col2 = st.columns([3, 1])
    
    with summary_col1:
        # Create columns for summary metrics with more visual appeal
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Checklist Method", f"${checklist_valuation:,.0f}")
        
        with metric_col2:
            st.metric("Scorecard Method", f"${scorecard_valuation:,.0f}")
        
        with metric_col3:
            # Show difference vs average
            diff_percentage = ((checklist_valuation - scorecard_valuation) / scorecard_valuation * 100 
                              if scorecard_valuation > 0 else 0)
            
            st.metric(
                "Average Valuation", 
                f"${average_valuation:,.0f}",
                f"{diff_percentage:.1f}% diff" if abs(diff_percentage) > 5 else None,
                delta_color="off" if abs(diff_percentage) <= 5 else "normal"
            )
        
        # Add a bar chart comparing the valuations
        st.markdown("### Valuation Comparison")
        chart_data = pd.DataFrame({
            'Method': ['Checklist', 'Scorecard', 'Average'],
            'Valuation': [checklist_valuation, scorecard_valuation, average_valuation]
        })
        
        st.bar_chart(
            chart_data, 
            x='Method', 
            y='Valuation',
            use_container_width=True
        )
    
    with summary_col2:
        # Add contextual information
        st.markdown("### Valuation Context")
        
        # Calculate the difference between methods
        difference = abs(checklist_valuation - scorecard_valuation)
        difference_percent = (difference / average_valuation) * 100 if average_valuation > 0 else 0
        
        if difference_percent < 10:
            st.success(f"The valuation methods are in close agreement (within {difference_percent:.1f}%).")
        elif difference_percent < 25:
            st.info(f"The valuation methods show moderate variance ({difference_percent:.1f}%).")
        else:
            st.warning(f"The valuation methods show significant variance ({difference_percent:.1f}%).")
            
        # Add recommendations
        st.markdown("#### Recommendations")
        if difference_percent > 25:
            st.markdown("- Consider re-evaluating factors with large differences")
            st.markdown("- Gather additional market data")
            st.markdown("- Consult other valuation methods")
        else:
            st.markdown("- This valuation appears well-supported")
            st.markdown("- Consider using the average as your baseline")
    
    # Option to save or share the valuation results
    st.markdown("---")
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button("Export Valuation Report (PDF)", type="primary"):
            # This would generate a PDF in a real application
            st.success("Valuation report exported!")
            st.info("In a complete implementation, this would generate a downloadable PDF report with charts and tables.")
    
    with export_col2:
        if st.button("Back to Analysis Report"):
            change_page("report")
            st.rerun()
    
    # Add explanatory notes about the valuation methods
    with st.expander("About the Valuation Methods"):
        st.markdown("""
        ### Checklist Method
        The Checklist Method evaluates the startup against a perfect hypothetical startup worth the "Perfect Valuation" amount. 
        Each factor is assigned a weight and a score, and the final valuation is calculated as: 
        `sum(weight × score × perfect valuation)` for all factors.
        
        ### Scorecard Method
        The Scorecard Method compares the startup against the median pre-money valuation of similar startups in the region. 
        Each factor is assigned a weight and a premium/discount multiplier, and the final valuation is calculated as:
        `sum(weight × multiplier × median valuation)` for all factors.
        
        A multiplier greater than 1.0 represents a premium compared to average startups, while a multiplier less than 1.0 represents a discount.
        """)
