import streamlit as st
import streamlit as st
import time
import os
import re
from openai import OpenAI
from utils import create_assistant, create_thread, create_message, get_response
import json
import speech_recognition as sr
import numpy as np
import pandas as pd
import copy

# Set page configuration
st.set_page_config(page_title="VC Assistant", layout="wide")

# Initialize API client
OPENAI_API_KEY = st.secrets["openai_api_key"]
client = OpenAI(api_key=OPENAI_API_KEY)

# File to store IDs - use absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_FILE = os.path.join(BASE_DIR, "assistant_data.json")

# Show the file path in debug mode
# Store debug mode setting in session state for consistent access
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
    
st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", st.session_state.debug_mode)
if st.session_state.debug_mode:
    st.sidebar.info(f"Storage file location: {STORAGE_FILE}")

# Initialize state variables if they don't exist
if 'current_view' not in st.session_state:
    st.session_state.current_view = "upload"  # Options: upload, report, valuation

if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = None

if 'valuation_data' not in st.session_state:
    # Default values for valuation parameters
    st.session_state.valuation_data = {
        # Checklist method
        "checklist": {
            "perfect_valuation": 10000000,
            "founders_team": {"weight": 0.30, "score": 0.75},
            "idea": {"weight": 0.20, "score": 0.70},
            "market": {"weight": 0.20, "score": 0.70},
            "product_ip": {"weight": 0.15, "score": 0.65},
            "execution": {"weight": 0.15, "score": 0.40}
        },
        # Scorecard method
        "scorecard": {
            "median_valuation": 8000000,
            "team_strength": {"weight": 0.24, "multiplier": 1.4},
            "opportunity_size": {"weight": 0.22, "multiplier": 1.2},
            "product_service": {"weight": 0.20, "multiplier": 0.7},
            "competition": {"weight": 0.16, "multiplier": 1.3},
            "marketing_sales": {"weight": 0.12, "multiplier": 0.9},
            "need_funding": {"weight": 0.06, "multiplier": 1.0}
        }
    }

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
                    # Use the existing debug mode state from the app
                    if st.session_state.get("debug_mode", False):
                        st.sidebar.info(f"Using existing assistant: {assistant_id}")
                except Exception as e:
                    st.sidebar.warning(f"Assistant retrieval error: {e}")
                    assistant_id = None
                    
                # Verify the thread still exists
                try:
                    thread = client.beta.threads.retrieve(data.get("thread_id", ""))
                    thread_id = data["thread_id"]
                    # Use the existing debug mode state from the app
                    if st.session_state.get("debug_mode", False):
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
            st.warning(f"Timed out after waiting {max_wait_seconds} seconds for run to complete")
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
            st.info("Run requires action - continuing with function calling...")
            return True
        
        # Only wait for queued or in_progress runs
        if active_runs:
            status = active_runs[0].status
            st.info(f"Waiting for active run to complete ({status})... {int(elapsed_time)}s elapsed")
            time.sleep(1)

# Function to calculate checklist valuation
def calculate_checklist_valuation(valuation_data):
    checklist = valuation_data["checklist"]
    perfect_val = checklist["perfect_valuation"]
    
    founders_val = perfect_val * checklist["founders_team"]["weight"] * checklist["founders_team"]["score"]
    idea_val = perfect_val * checklist["idea"]["weight"] * checklist["idea"]["score"]
    market_val = perfect_val * checklist["market"]["weight"] * checklist["market"]["score"]
    product_val = perfect_val * checklist["product_ip"]["weight"] * checklist["product_ip"]["score"]
    execution_val = perfect_val * checklist["execution"]["weight"] * checklist["execution"]["score"]
    
    total_val = founders_val + idea_val + market_val + product_val + execution_val
    
    return {
        "total": total_val,
        "components": {
            "founders_team": founders_val,
            "idea": idea_val,
            "market": market_val,
            "product_ip": product_val,
            "execution": execution_val
        }
    }

# Function to calculate scorecard valuation
def calculate_scorecard_valuation(valuation_data):
    scorecard = valuation_data["scorecard"]
    median_val = scorecard["median_valuation"]
    
    team_val = median_val * scorecard["team_strength"]["weight"] * scorecard["team_strength"]["multiplier"]
    opportunity_val = median_val * scorecard["opportunity_size"]["weight"] * scorecard["opportunity_size"]["multiplier"]
    product_val = median_val * scorecard["product_service"]["weight"] * scorecard["product_service"]["multiplier"]
    competition_val = median_val * scorecard["competition"]["weight"] * scorecard["competition"]["multiplier"]
    marketing_val = median_val * scorecard["marketing_sales"]["weight"] * scorecard["marketing_sales"]["multiplier"]
    funding_val = median_val * scorecard["need_funding"]["weight"] * scorecard["need_funding"]["multiplier"]
    
    total_val = team_val + opportunity_val + product_val + competition_val + marketing_val + funding_val
    
    return {
        "total": total_val,
        "components": {
            "team_strength": team_val,
            "opportunity_size": opportunity_val,
            "product_service": product_val,
            "competition": competition_val,
            "marketing_sales": marketing_val,
            "need_funding": funding_val
        }
    }

# Function to extract valuation parameters from AI analysis
def extract_valuation_params(analysis_text):
    """Extract valuation parameters from the AI analysis text.
    
    This function parses the VALUATION CRITERIA SCORES section in the analysis 
    to extract numerical values for checklist scores and scorecard multipliers.
    """
    # Default values if extraction fails - use deep copy to avoid reference issues
    valuation_data = copy.deepcopy(st.session_state.valuation_data)
    
    # Print a sample of the AI text for debugging
    if st.session_state.get("debug_mode", False):
        # Show a snippet of the text we're analyzing (last 1000 chars to avoid overwhelming the UI but show more context)
        if len(analysis_text) > 1000:
            st.sidebar.write(f"AI Text snippet (end):\n{analysis_text[-1000:]}")
        else:
            st.sidebar.write(f"AI Text snippet:\n{analysis_text}")
    
    try:
        # First attempt to find and extract organized sections in the output
        section_found = False
        valuation_section = ""
        
        # Check for various possible section headers with flexible casing
        section_patterns = [
            r"(?:VALUATION CRITERIA SCORES|Valuation Criteria Scores|VALUATION CRITERIA|Valuation Criteria):?",
            r"\d+\.?\s*(?:VALUATION CRITERIA SCORES|Valuation Criteria Scores|VALUATION CRITERIA|Valuation Criteria):?"
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, analysis_text)
            if match:
                section_found = True
                section_start = match.end()
                # Try to find where the section ends - look for the next numbered section or double newline
                next_section = re.search(r"\n\n\d+\.|\n\n[A-Z]+\s+[A-Z]+:|$", analysis_text[section_start:])
                if next_section:
                    valuation_section = analysis_text[section_start:section_start + next_section.start()]
                else:
                    # Take the rest of the text if no clear ending is found
                    valuation_section = analysis_text[section_start:]
                
                if st.session_state.get("debug_mode", False):
                    st.sidebar.write(f"Found valuation section using pattern: {pattern}")
                    st.sidebar.write(f"Extracted valuation section: {valuation_section}")
                break
        
        if section_found:
            # Try to extract checklist and scorecard sections from the valuation section
            
            # CHECKLIST SECTION EXTRACTION
            checklist_section = ""
            checklist_patterns = [
                r"(?:CHECKLIST METHOD SCORES|Checklist Method Scores|CHECKLIST METHOD|Checklist Method):?\s*\n",
                r"(?:CHECKLIST SCORES|Checklist Scores|Checklist):?\s*\n"
            ]
            
            for pattern in checklist_patterns:
                match = re.search(pattern, valuation_section)
                if match:
                    start_idx = match.end()
                    # Find where checklist section ends - either at scorecard section or end of text
                    scorecard_match = re.search(r"(?:SCORECARD METHOD MULTIPLIERS|Scorecard Method Multipliers|SCORECARD METHOD|Scorecard Method):?", valuation_section[start_idx:])
                    if scorecard_match:
                        checklist_section = valuation_section[start_idx:start_idx + scorecard_match.start()]
                    else:
                        checklist_section = valuation_section[start_idx:]
                    
                    if st.session_state.get("debug_mode", False):
                        st.sidebar.write(f"Found checklist section using pattern: {pattern}")
                        st.sidebar.write(f"Extracted checklist section: {checklist_section}")
                    break
            
            # SCORECARD SECTION EXTRACTION
            scorecard_section = ""
            scorecard_patterns = [
                r"(?:SCORECARD METHOD MULTIPLIERS|Scorecard Method Multipliers|SCORECARD METHOD|Scorecard Method):?\s*\n", 
                r"(?:SCORECARD MULTIPLIERS|Scorecard Multipliers|Scorecard):?\s*\n"
            ]
            
            for pattern in scorecard_patterns:
                match = re.search(pattern, valuation_section)
                if match:
                    scorecard_section = valuation_section[match.end():]
                    
                    if st.session_state.get("debug_mode", False):
                        st.sidebar.write(f"Found scorecard section using pattern: {pattern}")
                        st.sidebar.write(f"Extracted scorecard section: {scorecard_section}")
                    break
        
        # Process checklist parameters if section was found
        if section_found and checklist_section:
            # Extract individual checklist scores
            extract_checklist_scores_from_section(checklist_section, valuation_data)
        
        # Process scorecard parameters if section was found
        if section_found and scorecard_section:
            # Extract individual scorecard multipliers
            extract_scorecard_multipliers_from_section(scorecard_section, valuation_data)
        
        # If section-based extraction didn't work or is incomplete, try the fallback approach
        if not section_found or not checklist_section or not scorecard_section:
            if st.session_state.get("debug_mode", False):
                st.sidebar.write("Using fallback extraction method for valuation parameters")
            
            # Scan the entire text for patterns
            extract_parameters_with_fallback(analysis_text, valuation_data)
        
    except Exception as e:
        st.warning(f"Error extracting valuation parameters: {str(e)}")
        import traceback
        if st.session_state.get("debug_mode", False):
            st.sidebar.write(f"Exception traceback: {traceback.format_exc()}")
    
    # Debug info about the final values
    if st.session_state.get("debug_mode", False):
        st.sidebar.write("---")
        st.sidebar.markdown("### Final extracted valuation parameters:")
        st.sidebar.write(f"Founders & Team score: {valuation_data['checklist']['founders_team']['score']:.2f}")
        st.sidebar.write(f"Idea score: {valuation_data['checklist']['idea']['score']:.2f}")
        st.sidebar.write(f"Market score: {valuation_data['checklist']['market']['score']:.2f}")
        st.sidebar.write(f"Product score: {valuation_data['checklist']['product_ip']['score']:.2f}")
        st.sidebar.write(f"Execution score: {valuation_data['checklist']['execution']['score']:.2f}")
        st.sidebar.write("---")
        st.sidebar.write(f"Team multiplier: {valuation_data['scorecard']['team_strength']['multiplier']:.1f}x")
        st.sidebar.write(f"Opportunity multiplier: {valuation_data['scorecard']['opportunity_size']['multiplier']:.1f}x")
        st.sidebar.write(f"Product multiplier: {valuation_data['scorecard']['product_service']['multiplier']:.1f}x")
        st.sidebar.write(f"Competition multiplier: {valuation_data['scorecard']['competition']['multiplier']:.1f}x")
        st.sidebar.write(f"Marketing multiplier: {valuation_data['scorecard']['marketing_sales']['multiplier']:.1f}x")
        st.sidebar.write(f"Funding multiplier: {valuation_data['scorecard']['need_funding']['multiplier']:.1f}x")
    
    return valuation_data

# Helper function to extract checklist scores from a section
def extract_checklist_scores_from_section(section, valuation_data):
    """Extract checklist scores from a section of text."""
    if st.session_state.get("debug_mode", False):
        st.sidebar.write("Extracting checklist scores from section")
    
    # Parameters to extract with corresponding regex patterns and data keys
    checklist_params = [
        {
            "name": "Founders & Team",
            "patterns": [
                r"Founders\s*&?\s*Team:?\s*(\d+)%",
                r"Founders\s*&?\s*Team:?\s*(\d+)\s*percent",
                r"Founders\s*&?\s*Team:?\s*(\d+)"
            ],
            "data_key": "founders_team"
        },
        {
            "name": "Idea",
            "patterns": [
                r"Idea:?\s*(\d+)%",
                r"Idea:?\s*(\d+)\s*percent",
                r"Idea:?\s*(\d+)"
            ],
            "data_key": "idea"
        },
        {
            "name": "Market Size",
            "patterns": [
                r"Market\s*Size:?\s*(\d+)%",
                r"Market\s*Size:?\s*(\d+)\s*percent",
                r"Market\s*Size:?\s*(\d+)"
            ],
            "data_key": "market"
        },
        {
            "name": "Product & IP",
            "patterns": [
                r"Product\s*&?\s*IP:?\s*(\d+)%",
                r"Product\s*&?\s*IP:?\s*(\d+)\s*percent", 
                r"Product\s*&?\s*IP:?\s*(\d+)"
            ],
            "data_key": "product_ip"
        },
        {
            "name": "Execution Potential",
            "patterns": [
                r"Execution\s*(?:Potential)?:?\s*(\d+)%",
                r"Execution\s*(?:Potential)?:?\s*(\d+)\s*percent",
                r"Execution\s*(?:Potential)?:?\s*(\d+)"
            ],
            "data_key": "execution"
        }
    ]
    
    # Extract each parameter
    for param in checklist_params:
        for pattern in param["patterns"]:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                score = float(match.group(1)) / 100
                valuation_data["checklist"][param["data_key"]]["score"] = score
                if st.session_state.get("debug_mode", False):
                    st.sidebar.write(f"Found {param['name']} score: {score:.2f} with pattern: {pattern}")
                break

# Helper function to extract scorecard multipliers from a section
def extract_scorecard_multipliers_from_section(section, valuation_data):
    """Extract scorecard multipliers from a section of text."""
    if st.session_state.get("debug_mode", False):
        st.sidebar.write("Extracting scorecard multipliers from section")
    
    # Parameters to extract with corresponding regex patterns and data keys
    scorecard_params = [
        {
            "name": "Team Strength",
            "patterns": [
                r"Team\s*Strength:?\s*([\d\.]+)x",
                r"Team\s*Strength:?\s*([\d\.]+)\s*times",
                r"Team\s*Strength:?\s*([\d\.]+)"
            ],
            "data_key": "team_strength"
        },
        {
            "name": "Opportunity Size",
            "patterns": [
                r"Opportunity\s*Size:?\s*([\d\.]+)x",
                r"Opportunity\s*Size:?\s*([\d\.]+)\s*times",
                r"Opportunity\s*Size:?\s*([\d\.]+)"
            ],
            "data_key": "opportunity_size"
        },
        {
            "name": "Product/Service",
            "patterns": [
                r"Product(?:/Service)?:?\s*([\d\.]+)x",
                r"Product(?:/Service)?:?\s*([\d\.]+)\s*times",
                r"Product(?:/Service)?:?\s*([\d\.]+)"
            ],
            "data_key": "product_service"
        },
        {
            "name": "Competition",
            "patterns": [
                r"Competition:?\s*([\d\.]+)x",
                r"Competition:?\s*([\d\.]+)\s*times",
                r"Competition:?\s*([\d\.]+)"
            ],
            "data_key": "competition"
        },
        {
            "name": "Marketing & Sales",
            "patterns": [
                r"Marketing\s*&?\s*Sales:?\s*([\d\.]+)x",
                r"Marketing\s*&?\s*Sales:?\s*([\d\.]+)\s*times",
                r"Marketing\s*&?\s*Sales:?\s*([\d\.]+)"
            ],
            "data_key": "marketing_sales"
        },
        {
            "name": "Need for Funding",
            "patterns": [
                r"Need\s*(?:for)?\s*Funding:?\s*([\d\.]+)x",
                r"Need\s*(?:for)?\s*Funding:?\s*([\d\.]+)\s*times",
                r"Need\s*(?:for)?\s*Funding:?\s*([\d\.]+)"
            ],
            "data_key": "need_funding"
        }
    ]
    
    # Extract each parameter
    for param in scorecard_params:
        for pattern in param["patterns"]:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                multiplier = float(match.group(1))
                valuation_data["scorecard"][param["data_key"]]["multiplier"] = multiplier
                if st.session_state.get("debug_mode", False):
                    st.sidebar.write(f"Found {param['name']} multiplier: {multiplier:.1f}x with pattern: {pattern}")
                break

# Helper function for fallback extraction
def extract_parameters_with_fallback(full_text, valuation_data):
    """Scan the entire text for valuation parameters as a fallback method."""
    if st.session_state.get("debug_mode", False):
        st.sidebar.write("Using direct text scan for valuation parameters")
    
    # Extract checklist scores across the entire text
    # Looking for patterns like "Founders & Team: 75%" or just numbers near key phrases
    
    # More comprehensive checklist parameter patterns
    checklist_params = [
        {
            "name": "Founders & Team",
            "patterns": [
                r"Founders\s*&?\s*Team:?\s*(\d+)%",
                r"Founders\s*&?\s*Team:?\s*(\d+)\s*percent",
                r"Founders\s*&?\s*Team(?:\s*score)?:?\s*(\d+)",
                r"Founders\s*(?:score|rating):?\s*(\d+)"
            ],
            "data_key": "founders_team"
        },
        {
            "name": "Idea",
            "patterns": [
                r"Idea:?\s*(\d+)%",
                r"Idea:?\s*(\d+)\s*percent",
                r"Idea(?:\s*score)?:?\s*(\d+)",
                r"Concept(?:\s*score)?:?\s*(\d+)"
            ],
            "data_key": "idea"
        },
        {
            "name": "Market Size",
            "patterns": [
                r"Market\s*Size:?\s*(\d+)%",
                r"Market\s*Size:?\s*(\d+)\s*percent",
                r"Market(?:\s*score)?:?\s*(\d+)",
                r"Market\s*Size(?:\s*score)?:?\s*(\d+)"
            ],
            "data_key": "market"
        },
        {
            "name": "Product & IP",
            "patterns": [
                r"Product\s*&?\s*IP:?\s*(\d+)%",
                r"Product\s*&?\s*IP:?\s*(\d+)\s*percent",
                r"Product(?:\s*score)?:?\s*(\d+)",
                r"Product\s*&?\s*IP(?:\s*score)?:?\s*(\d+)",
                r"Intellectual\s*Property(?:\s*score)?:?\s*(\d+)"
            ],
            "data_key": "product_ip"
        },
        {
            "name": "Execution Potential",
            "patterns": [
                r"Execution\s*(?:Potential)?:?\s*(\d+)%",
                r"Execution\s*(?:Potential)?:?\s*(\d+)\s*percent",
                r"Execution(?:\s*score)?:?\s*(\d+)",
                r"Execution\s*Potential(?:\s*score)?:?\s*(\d+)",
                r"Execution\s*Risk(?:\s*score)?:?\s*(\d+)"
            ],
            "data_key": "execution"
        }
    ]
    
    # More comprehensive scorecard parameter patterns
    scorecard_params = [
        {
            "name": "Team Strength",
            "patterns": [
                r"Team\s*Strength:?\s*([\d\.]+)x",
                r"Team\s*Strength:?\s*([\d\.]+)\s*times",
                r"Team(?:\s*multiplier)?:?\s*([\d\.]+)",
                r"Team\s*Strength(?:\s*multiplier)?:?\s*([\d\.]+)"
            ],
            "data_key": "team_strength"
        },
        {
            "name": "Opportunity Size",
            "patterns": [
                r"Opportunity\s*Size:?\s*([\d\.]+)x",
                r"Opportunity\s*Size:?\s*([\d\.]+)\s*times",
                r"Opportunity(?:\s*multiplier)?:?\s*([\d\.]+)",
                r"Opportunity\s*Size(?:\s*multiplier)?:?\s*([\d\.]+)",
                r"Market\s*Opportunity(?:\s*multiplier)?:?\s*([\d\.]+)"
            ],
            "data_key": "opportunity_size"
        },
        {
            "name": "Product/Service",
            "patterns": [
                r"Product(?:/Service)?:?\s*([\d\.]+)x",
                r"Product(?:/Service)?:?\s*([\d\.]+)\s*times",
                r"Product(?:\s*multiplier)?:?\s*([\d\.]+)",
                r"Product/Service(?:\s*multiplier)?:?\s*([\d\.]+)",
                r"Service(?:\s*multiplier)?:?\s*([\d\.]+)"
            ],
            "data_key": "product_service"
        },
        {
            "name": "Competition",
            "patterns": [
                r"Competition:?\s*([\d\.]+)x",
                r"Competition:?\s*([\d\.]+)\s*times",
                r"Competition(?:\s*multiplier)?:?\s*([\d\.]+)",
                r"Competitive\s*Environment(?:\s*multiplier)?:?\s*([\d\.]+)"
            ],
            "data_key": "competition"
        },
        {
            "name": "Marketing & Sales",
            "patterns": [
                r"Marketing\s*&?\s*Sales:?\s*([\d\.]+)x",
                r"Marketing\s*&?\s*Sales:?\s*([\d\.]+)\s*times",
                r"Marketing(?:\s*multiplier)?:?\s*([\d\.]+)",
                r"Marketing\s*&?\s*Sales(?:\s*multiplier)?:?\s*([\d\.]+)",
                r"Sales(?:\s*multiplier)?:?\s*([\d\.]+)"
            ],
            "data_key": "marketing_sales"
        },
        {
            "name": "Need for Funding",
            "patterns": [
                r"Need\s*(?:for)?\s*Funding:?\s*([\d\.]+)x",
                r"Need\s*(?:for)?\s*Funding:?\s*([\d\.]+)\s*times",
                r"Funding(?:\s*multiplier)?:?\s*([\d\.]+)",
                r"Need\s*for\s*Funding(?:\s*multiplier)?:?\s*([\d\.]+)",
                r"Funding\s*Need(?:\s*multiplier)?:?\s*([\d\.]+)"
            ],
            "data_key": "need_funding"
        }
    ]
    
    # Extract checklist scores
    for param in checklist_params:
        for pattern in param["patterns"]:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                score = float(match.group(1)) / 100
                valuation_data["checklist"][param["data_key"]]["score"] = score
                if st.session_state.get("debug_mode", False):
                    st.sidebar.write(f"Fallback found {param['name']} score: {score:.2f} with pattern: {pattern}")
                break
    
    # Extract scorecard multipliers
    for param in scorecard_params:
        for pattern in param["patterns"]:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                multiplier = float(match.group(1))
                valuation_data["scorecard"][param["data_key"]]["multiplier"] = multiplier
                if st.session_state.get("debug_mode", False):
                    st.sidebar.write(f"Fallback found {param['name']} multiplier: {multiplier:.1f}x with pattern: {pattern}")
                break
                
                # Market Size
                if "Market Size:" in checklist_section:
                    match = re.search(r"Market Size:\s*(\d+)%", checklist_section)
                    if match:
                        score = float(match.group(1)) / 100
                        valuation_data["checklist"]["market"]["score"] = score
                
                # Product & IP
                if "Product & IP:" in checklist_section:
                    match = re.search(r"Product & IP:\s*(\d+)%", checklist_section)
                    if match:
                        score = float(match.group(1)) / 100
                        valuation_data["checklist"]["product_ip"]["score"] = score
                
                # Execution Potential
                if "Execution Potential:" in checklist_section:
                    match = re.search(r"Execution Potential:\s*(\d+)%", checklist_section)
                    if match:
                        score = float(match.group(1)) / 100
                        valuation_data["checklist"]["execution"]["score"] = score
            
            # Extract Scorecard Method Multipliers
            if "SCORECARD METHOD MULTIPLIERS:" in valuation_section:
                scorecard_section = valuation_section.split("SCORECARD METHOD MULTIPLIERS:")[1]
                
                # Team Strength
                if "Team Strength:" in scorecard_section:
                    match = re.search(r"Team Strength:\s*([\d\.]+)x", scorecard_section)
                    if match:
                        multiplier = float(match.group(1))
                        valuation_data["scorecard"]["team_strength"]["multiplier"] = multiplier
                
                # Opportunity Size
                if "Opportunity Size:" in scorecard_section:
                    match = re.search(r"Opportunity Size:\s*([\d\.]+)x", scorecard_section)
                    if match:
                        multiplier = float(match.group(1))
                        valuation_data["scorecard"]["opportunity_size"]["multiplier"] = multiplier
                
                # Product/Service
                if "Product/Service:" in scorecard_section:
                    match = re.search(r"Product/Service:\s*([\d\.]+)x", scorecard_section)
                    if match:
                        multiplier = float(match.group(1))
                        valuation_data["scorecard"]["product_service"]["multiplier"] = multiplier
                
                # Competition
                if "Competition:" in scorecard_section:
                    match = re.search(r"Competition:\s*([\d\.]+)x", scorecard_section)
                    if match:
                        multiplier = float(match.group(1))
                        valuation_data["scorecard"]["competition"]["multiplier"] = multiplier
                
                # Marketing & Sales
                if "Marketing & Sales:" in scorecard_section:
                    match = re.search(r"Marketing & Sales:\s*([\d\.]+)x", scorecard_section)
                    if match:
                        multiplier = float(match.group(1))
                        valuation_data["scorecard"]["marketing_sales"]["multiplier"] = multiplier
                
                # Need for Funding
                if "Need for Funding:" in scorecard_section:
                    match = re.search(r"Need for Funding:\s*([\d\.]+)x", scorecard_section)
                    if match:
                        multiplier = float(match.group(1))
                        valuation_data["scorecard"]["need_funding"]["multiplier"] = multiplier
        
        # Fallback to less structured extraction if the section isn't found
        else:
            # Checklist extraction
            if "founders" in analysis_text.lower() and "team" in analysis_text.lower():
                score = 0.75  # Default
                # Try to extract a percentage
                if "founders & team" in analysis_text.lower():
                    content_after = analysis_text.lower().split("founders & team")[1].split("\n")[0]
                    if "%" in content_after:
                        try:
                            percentage = float(content_after.split("%")[0].strip().split(" ")[-1]) / 100
                            if 0 <= percentage <= 1:
                                score = percentage
                        except:
                            pass
                valuation_data["checklist"]["founders_team"]["score"] = score
            
            # Scorecard extraction
            if "team strength" in analysis_text.lower():
                content_after = analysis_text.lower().split("team strength")[1].split("\n")[0]
                if "premium" in content_after or "discount" in content_after:
                    multiplier = 1.4  # Default
                    try:
                        # Try to extract multiplier value
                        for word in ["premium", "discount", "multiplier", "factor"]:
                            if word in content_after:
                                number_part = content_after.split(word)[1].strip().split(" ")[0]
                                if number_part.startswith("of"):
                                    number_part = number_part[2:]
                                multiplier = float(number_part.replace("x", ""))
                                break
                        valuation_data["scorecard"]["team_strength"]["multiplier"] = multiplier
                    except:
                        pass
        
    except Exception as e:
        st.warning(f"Error extracting valuation parameters: {e}")
    
    return valuation_data

# Get or create assistant and thread
assistant_id, thread_id = get_or_create_assistant_and_thread()
st.session_state.assistant_id = assistant_id
st.session_state.thread_id = thread_id

# Function to reset for a new startup evaluation
def start_new_evaluation():
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
        st.session_state.current_view = "upload"
        st.session_state.ai_analysis = None
        
        # Clear AI predicted values
        if 'ai_predicted_values' in st.session_state:
            del st.session_state.ai_predicted_values
        
        # Reset valuation data to defaults
        st.session_state.valuation_data = {
            # Checklist method
            "checklist": {
                "perfect_valuation": 10000000,
                "founders_team": {"weight": 0.30, "score": 0.75},
                "idea": {"weight": 0.20, "score": 0.70},
                "market": {"weight": 0.20, "score": 0.70},
                "product_ip": {"weight": 0.15, "score": 0.65},
                "execution": {"weight": 0.15, "score": 0.40}
            },
            # Scorecard method
            "scorecard": {
                "median_valuation": 8000000,
                "team_strength": {"weight": 0.24, "multiplier": 1.4},
                "opportunity_size": {"weight": 0.22, "multiplier": 1.2},
                "product_service": {"weight": 0.20, "multiplier": 0.7},
                "competition": {"weight": 0.16, "multiplier": 1.3},
                "marketing_sales": {"weight": 0.12, "multiplier": 0.9},
                "need_funding": {"weight": 0.06, "multiplier": 1.0}
            }
        }
        
        st.success("Started a new evaluation!")
        st.rerun()
    except Exception as e:
        st.error(f"Error starting new evaluation: {e}")

# Create a set for processed files
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Navigation UI
with st.sidebar:
    st.title("VC Assistant")
    
    # Add a prominent button to start a new evaluation
    if st.button("Start New Evaluation", type="primary"):
        start_new_evaluation()
    
    st.divider()

    # Navigation buttons
    st.subheader("Navigation")
    if st.button("📤 Upload Documents", disabled=st.session_state.current_view=="upload"):
        st.session_state.current_view = "upload"
        st.rerun()
    
    report_disabled = st.session_state.ai_analysis is None
    if st.button("📝 View Report", disabled=report_disabled):
        st.session_state.current_view = "report"
        st.rerun()
    
    valuation_disabled = st.session_state.ai_analysis is None
    if st.button("💰 Valuation Models", disabled=valuation_disabled):
        st.session_state.current_view = "valuation"
        st.rerun()

# UPLOAD VIEW
if st.session_state.current_view == "upload":
    st.title("Upload Startup Documents")
    st.markdown("""
    This assistant evaluates startup proposals like a venture capitalist. 
    Upload your business plan, pitch deck, financial projections, or team CVs to get feedback and a valuation.
    """)
    
    # Allow multiple files
    uploaded_files = st.file_uploader(
        "Upload files...", 
        accept_multiple_files=True,
        type=["pdf", "csv", "xlsx", "docx", "txt"]
    )
    
    # Show what files are ready to be analyzed
    if uploaded_files:
        st.write("Files ready for analysis:")
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.processed_files:
                st.write(f"✅ {uploaded_file.name} (already processed)")
            else:
                st.write(f"📄 {uploaded_file.name}")
    
    # Add a button to process all uploaded files
    if uploaded_files and any(file.name not in st.session_state.processed_files for file in uploaded_files):
        if st.button("Analyze All Files", type="primary"):
            file_ids = []
            file_names = []
            
            # First, upload all files to OpenAI
            with st.status("Uploading files...") as status:
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
                            status.update(label=f"Uploaded: {uploaded_file.name}")
                            
                            # Clean up temp file
                            os.remove(temp_file_path)
                            
                            # Mark as processed
                            st.session_state.processed_files.add(uploaded_file.name)
                            
                        except Exception as e:
                            status.update(label=f"Error uploading {uploaded_file.name}: {e}", state="error")
                
                status.update(label="All files uploaded successfully!", state="complete")
            
            # If we have files to process
            if file_ids:
                # Send a system message instructing the assistant to remember document contents
                system_message = client.beta.threads.messages.create(
                    thread_id=st.session_state.thread_id,
                    role="user",
                    content=[{
                        "type": "text", 
                        "text": "I've uploaded important documents about a startup proposal. Please remember the contents of these documents throughout our entire conversation."
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
                
                # Create a message with all attachments - focus on report with valuation criteria scores
                message_text = f"""I've uploaded {len(file_ids)} files for analysis: {', '.join(file_names)}. 
                Please analyze all files together for a comprehensive evaluation of this startup.

                Follow the structure below and ensure your bullet points are expanded and thoroughly explained:

                1. Summary of the proposal
                2. Strengths (detailed bullet points/paragraphs)
                3. Areas for improvement (detailed bullet points/paragraphs)
                4. Team assessment (including a detailed table of team members if CVs are provided)
                5. Competitive analysis (with names or descriptions of similar startups)
                6. Overall score (1-10)
                7. Final recommendation
                8. Valuation Criteria Scores (REQUIRED - use this exact format):

                VALUATION CRITERIA SCORES:
                
                CHECKLIST METHOD SCORES:
                - Founders & Team: [SCORE]% (e.g. 75%)
                - Idea: [SCORE]% (e.g. 65%)
                - Market Size: [SCORE]% (e.g. 80%)
                - Product & IP: [SCORE]% (e.g. 70%)
                - Execution Potential: [SCORE]% (e.g. 60%)
                
                SCORECARD METHOD MULTIPLIERS:
                - Team Strength: [MULTIPLIER]x (e.g. 1.2x)
                - Opportunity Size: [MULTIPLIER]x (e.g. 1.5x)
                - Product/Service: [MULTIPLIER]x (e.g. 0.9x)
                - Competition: [MULTIPLIER]x (e.g. 1.1x)
                - Marketing & Sales: [MULTIPLIER]x (e.g. 0.8x)
                - Need for Funding: [MULTIPLIER]x (e.g. 1.0x)
                
                Replace the examples with your actual assessments based on the pitchdeck. The valuation criteria scores must be included in your response with this exact format.
                """
                
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
                
                # Show analysis in progress
                with st.status("Analyzing startup proposal...") as status:
                    # Build dynamic instructions
                    instructions = """Please analyze all the uploaded files to provide a comprehensive evaluation of the startup proposal.
                    Use your VC evaluation framework and follow the structure requested in the message.
                    
                    Be specific about the values you assign to each valuation metric and explain your reasoning in detail.
                    
                    For the Checklist Method:
                    - Founders & Team (30%): Assign a percentage score based on the team's experience and fit
                    - Idea (20%): Assess the quality of the idea and problem-solution fit
                    - Market Size (20%): Evaluate the TAM, SAM, and SOM
                    - Product & IP (15%): Assess product uniqueness and IP protection
                    - Execution Potential (15%): Evaluate the team's ability to execute the plan
                    
                    For the Scorecard Method:
                    - Team Strength (24%): Assign a multiplier (premium or discount)
                    - Opportunity Size (22%): Assess market potential with a multiplier
                    - Product/Service (20%): Evaluate product quality with a multiplier
                    - Competition (16%): Assess competitive landscape with a multiplier
                    - Marketing & Sales (12%): Evaluate go-to-market strategy with a multiplier
                    - Need for Funding (6%): Assess funding timing with a multiplier
                    """

                    # Create a run with specific instructions
                    run = client.beta.threads.runs.create(
                        thread_id=st.session_state.thread_id,
                        assistant_id=st.session_state.assistant_id,
                        instructions=instructions
                    )
                    
                    status.update(label="AI is analyzing the startup proposal...", state="running")
                    
                    # Wait for the run to complete
                    wait_for_active_runs(client, st.session_state.thread_id)
                    
                    # Get the response
                    response_content = get_response(client, st.session_state.thread_id, st.session_state.assistant_id)
                    
                    # Store analysis in session state
                    st.session_state.ai_analysis = response_content
                    
                    # Extract valuation parameters from the analysis
                    st.session_state.valuation_data = extract_valuation_params(response_content)
                    
                    # Store AI predicted values separately
                    st.session_state.ai_predicted_values = st.session_state.valuation_data.copy()
                    
                    status.update(label="Analysis complete!", state="complete")
                
                # Change view to report
                st.session_state.current_view = "report"
                st.rerun()
    
    # Show instruction if no files
    if not uploaded_files:
        st.info("Please upload files to begin the startup evaluation.")

# REPORT VIEW
elif st.session_state.current_view == "report":
    if st.session_state.ai_analysis:
        st.title("Startup Evaluation Report")
        
        # Display the AI analysis
        st.markdown(st.session_state.ai_analysis)
        
        # Create a divider
        st.divider()
        
        # Add valuation parameter sliders directly on the report page
        st.subheader("Valuation Base Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Perfect valuation for Checklist Method
            perfect_val = st.number_input(
                "Perfect Valuation (Checklist Method) ($)",
                min_value=1000000,
                max_value=100000000,
                value=int(st.session_state.valuation_data["checklist"]["perfect_valuation"]),
                step=1000000,
            )
            st.session_state.valuation_data["checklist"]["perfect_valuation"] = perfect_val
        
        with col2:
            # Median valuation for Scorecard Method
            median_val = st.number_input(
                "Median Industry Valuation (Scorecard Method) ($)",
                min_value=1000000,
                max_value=100000000,
                value=int(st.session_state.valuation_data["scorecard"]["median_valuation"]),
                step=1000000,
            )
            st.session_state.valuation_data["scorecard"]["median_valuation"] = median_val
            
        # Store a copy of the AI predictions if not already stored
        if 'ai_predicted_values' not in st.session_state:
            st.session_state.ai_predicted_values = st.session_state.valuation_data.copy()
        
        # Add a button to proceed to valuation
        if st.button("Proceed to Valuation Models ➡️", type="primary"):
            st.session_state.current_view = 'valuation'
            st.rerun()
    else:
        st.warning("No analysis available. Please upload documents first.")
        st.session_state.current_view = "upload"
        st.rerun()

# VALUATION VIEW
elif st.session_state.current_view == "valuation":
    st.title("Valuation Models")
    
    if st.session_state.ai_analysis is None:
        st.warning("No analysis available. Please upload documents first.")
        st.session_state.current_view = "upload"
        st.rerun()
    
    # Display valuation overview
    st.markdown("""
    ## Valuation Overview
    
    This section provides two different methods to calculate the valuation of the startup:
    
    1. **Checklist Method**: Evaluates specific areas with percentage scores against a perfect valuation
    2. **Scorecard Method**: Applies premium/discount multipliers to a median industry valuation
    
    The sliders are pre-set with AI-predicted values based on the pitchdeck quality, but you can adjust them.
    """)
            
    # Add a reset button to restore AI predictions
    if st.button("Reset to AI-Predicted Values"):
        st.session_state.valuation_data = st.session_state.ai_predicted_values.copy()
        st.rerun()
    
    # Create tabs for the two valuation methods
    tab1, tab2 = st.tabs(["Checklist Method", "Scorecard Method"])
    
    with tab1:
        st.header("Checklist Method")
        st.markdown("""
        This method evaluates 5 key areas with weighted scores, then multiplies by a perfect valuation.
        Adjust the sliders to see how different factors affect the valuation.
        """)
        
        # Perfect valuation input
        perfect_val = st.number_input(
            "Perfect Valuation ($)",
            min_value=1000000,
            max_value=100000000,
            value=int(st.session_state.valuation_data["checklist"]["perfect_valuation"]),
            step=1000000,
            key="checklist_perfect_val"
        )
        st.session_state.valuation_data["checklist"]["perfect_valuation"] = perfect_val
        
        # Sliders for each factor
        st.subheader("Founders & Team (30%)")
        # Show the AI-predicted value first
        ai_founders_score = st.session_state.ai_predicted_values["checklist"]["founders_team"]["score"]
        st.caption(f"AI-predicted value: {ai_founders_score:.0%}")
        
        # Then show the slider for adjustment
        founders_score = st.slider(
            "Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.valuation_data["checklist"]["founders_team"]["score"],
            step=0.05,
            format="%.0f%%",
            key="founders_score"
        )
        st.session_state.valuation_data["checklist"]["founders_team"]["score"] = founders_score
        
        st.subheader("Idea (20%)")
        # Show AI-predicted value
        ai_idea_score = st.session_state.ai_predicted_values["checklist"]["idea"]["score"]
        st.caption(f"AI-predicted value: {ai_idea_score:.0%}")
        
        idea_score = st.slider(
            "Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.valuation_data["checklist"]["idea"]["score"],
            step=0.05,
            format="%.0f%%",
            key="idea_score"
        )
        st.session_state.valuation_data["checklist"]["idea"]["score"] = idea_score
        
        st.subheader("Market Size (20%)")
        # Show AI-predicted value
        ai_market_score = st.session_state.ai_predicted_values["checklist"]["market"]["score"]
        st.caption(f"AI-predicted value: {ai_market_score:.0%}")
        
        market_score = st.slider(
            "Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.valuation_data["checklist"]["market"]["score"],
            step=0.05,
            format="%.0f%%",
            key="market_score"
        )
        st.session_state.valuation_data["checklist"]["market"]["score"] = market_score
        
        st.subheader("Product and IP (15%)")
        # Show AI-predicted value
        ai_product_score = st.session_state.ai_predicted_values["checklist"]["product_ip"]["score"]
        st.caption(f"AI-predicted value: {ai_product_score:.0%}")
        
        product_score = st.slider(
            "Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.valuation_data["checklist"]["product_ip"]["score"],
            step=0.05,
            format="%.0f%%",
            key="product_score"
        )
        st.session_state.valuation_data["checklist"]["product_ip"]["score"] = product_score
        
        st.subheader("Execution (15%)")
        # Show AI-predicted value
        ai_execution_score = st.session_state.ai_predicted_values["checklist"]["execution"]["score"]
        st.caption(f"AI-predicted value: {ai_execution_score:.0%}")
        
        execution_score = st.slider(
            "Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.valuation_data["checklist"]["execution"]["score"],
            step=0.05,
            format="%.0f%%",
            key="execution_score"
        )
        st.session_state.valuation_data["checklist"]["execution"]["score"] = execution_score
        
        # Calculate valuation
        checklist_val = calculate_checklist_valuation(st.session_state.valuation_data)
        
        # Display results in a table
        st.subheader("Checklist Valuation Results")
        checklist_df = pd.DataFrame({
            "Key Measures": ["Founders & Team", "Idea", "Market", "Product and IP", "Execution", "TOTAL"],
            "Weight": ["30%", "20%", "20%", "15%", "15%", "100%"],
            "Score": [f"{founders_score:.0%}", f"{idea_score:.0%}", f"{market_score:.0%}", 
                     f"{product_score:.0%}", f"{execution_score:.0%}", ""],
            "Perfect Valuation": [f"${perfect_val:,.0f}", f"${perfect_val:,.0f}", 
                                 f"${perfect_val:,.0f}", f"${perfect_val:,.0f}", 
                                 f"${perfect_val:,.0f}", ""],
            "Checklist Valuation": [
                f"${checklist_val['components']['founders_team']:,.0f}",
                f"${checklist_val['components']['idea']:,.0f}",
                f"${checklist_val['components']['market']:,.0f}",
                f"${checklist_val['components']['product_ip']:,.0f}",
                f"${checklist_val['components']['execution']:,.0f}",
                f"${checklist_val['total']:,.0f}"
            ]
        })
        st.table(checklist_df)
        
        # Summary
        st.success(f"Checklist Method Valuation: ${checklist_val['total']:,.0f}")
    
    with tab2:
        st.header("Scorecard Method")
        st.markdown("""
        This method applies premium or discount multipliers to each weighted factor, then multiplies by a median valuation.
        Adjust the sliders to see how different factors affect the valuation.
        """)
        
        # Median valuation input
        median_val = st.number_input(
            "Median Pre-money Valuation ($)",
            min_value=1000000,
            max_value=50000000,
            value=int(st.session_state.valuation_data["scorecard"]["median_valuation"]),
            step=1000000,
            key="scorecard_median_val"
        )
        st.session_state.valuation_data["scorecard"]["median_valuation"] = median_val
        
        # Sliders for each factor multiplier
        st.subheader("Team Strength (24%)")
        # Show AI-predicted value
        ai_team_multiplier = st.session_state.ai_predicted_values["scorecard"]["team_strength"]["multiplier"]
        st.caption(f"AI-predicted value: {ai_team_multiplier:.1f}x")
        
        team_multiplier = st.slider(
            "Premium/Discount",
            min_value=0.1,
            max_value=3.0,
            value=st.session_state.valuation_data["scorecard"]["team_strength"]["multiplier"],
            step=0.1,
            format="%.1fx",
            key="team_multiplier"
        )
        st.session_state.valuation_data["scorecard"]["team_strength"]["multiplier"] = team_multiplier
        
        st.subheader("Opportunity Size (22%)")
        # Show AI-predicted value
        ai_opportunity_multiplier = st.session_state.ai_predicted_values["scorecard"]["opportunity_size"]["multiplier"]
        st.caption(f"AI-predicted value: {ai_opportunity_multiplier:.1f}x")
        
        opportunity_multiplier = st.slider(
            "Premium/Discount",
            min_value=0.1,
            max_value=3.0,
            value=st.session_state.valuation_data["scorecard"]["opportunity_size"]["multiplier"],
            step=0.1,
            format="%.1fx",
            key="opportunity_multiplier"
        )
        st.session_state.valuation_data["scorecard"]["opportunity_size"]["multiplier"] = opportunity_multiplier
        
        st.subheader("Product/Service (20%)")
        # Show AI-predicted value
        ai_product_multiplier = st.session_state.ai_predicted_values["scorecard"]["product_service"]["multiplier"]
        st.caption(f"AI-predicted value: {ai_product_multiplier:.1f}x")
        
        product_multiplier = st.slider(
            "Premium/Discount",
            min_value=0.1,
            max_value=3.0,
            value=st.session_state.valuation_data["scorecard"]["product_service"]["multiplier"],
            step=0.1,
            format="%.1fx",
            key="product_multiplier"
        )
        st.session_state.valuation_data["scorecard"]["product_service"]["multiplier"] = product_multiplier
        
        st.subheader("Competition (16%)")
        # Show AI-predicted value
        ai_competition_multiplier = st.session_state.ai_predicted_values["scorecard"]["competition"]["multiplier"]
        st.caption(f"AI-predicted value: {ai_competition_multiplier:.1f}x")
        
        competition_multiplier = st.slider(
            "Premium/Discount",
            min_value=0.1,
            max_value=3.0,
            value=st.session_state.valuation_data["scorecard"]["competition"]["multiplier"],
            step=0.1,
            format="%.1fx",
            key="competition_multiplier"
        )
        st.session_state.valuation_data["scorecard"]["competition"]["multiplier"] = competition_multiplier
        
        st.subheader("Marketing & Sales (12%)")
        # Show AI-predicted value
        ai_marketing_multiplier = st.session_state.ai_predicted_values["scorecard"]["marketing_sales"]["multiplier"]
        st.caption(f"AI-predicted value: {ai_marketing_multiplier:.1f}x")
        
        marketing_multiplier = st.slider(
            "Premium/Discount",
            min_value=0.1,
            max_value=3.0,
            value=st.session_state.valuation_data["scorecard"]["marketing_sales"]["multiplier"],
            step=0.1,
            format="%.1fx",
            key="marketing_multiplier"
        )
        st.session_state.valuation_data["scorecard"]["marketing_sales"]["multiplier"] = marketing_multiplier
        
        st.subheader("Need for Funding (6%)")
        # Show AI-predicted value
        ai_funding_multiplier = st.session_state.ai_predicted_values["scorecard"]["need_funding"]["multiplier"]
        st.caption(f"AI-predicted value: {ai_funding_multiplier:.1f}x")
        
        funding_multiplier = st.slider(
            "Premium/Discount",
            min_value=0.1,
            max_value=3.0,
            value=st.session_state.valuation_data["scorecard"]["need_funding"]["multiplier"],
            step=0.1,
            format="%.1fx",
            key="funding_multiplier"
        )
        st.session_state.valuation_data["scorecard"]["need_funding"]["multiplier"] = funding_multiplier
        
        # Calculate valuation
        scorecard_val = calculate_scorecard_valuation(st.session_state.valuation_data)
        
        # Display results in a table
        st.subheader("Scorecard Valuation Results")
        scorecard_df = pd.DataFrame({
            "Key Measures": ["Team Strength", "Opportunity Size", "Product/Service", 
                           "Competition", "Marketing & Sales", "Need for Funding", "TOTAL"],
            "Weight": ["24%", "22%", "20%", "16%", "12%", "6%", "100%"],
            "Premium/Discount": [
                f"{team_multiplier:.1f}x", 
                f"{opportunity_multiplier:.1f}x", 
                f"{product_multiplier:.1f}x", 
                f"{competition_multiplier:.1f}x", 
                f"{marketing_multiplier:.1f}x", 
                f"{funding_multiplier:.1f}x", 
                ""
            ],
            "Pre-money Median": [
                f"${median_val:,.0f}", 
                f"${median_val:,.0f}", 
                f"${median_val:,.0f}", 
                f"${median_val:,.0f}", 
                f"${median_val:,.0f}", 
                f"${median_val:,.0f}", 
                ""
            ],
            "Scorecard Valuation": [
                f"${scorecard_val['components']['team_strength']:,.0f}",
                f"${scorecard_val['components']['opportunity_size']:,.0f}",
                f"${scorecard_val['components']['product_service']:,.0f}",
                f"${scorecard_val['components']['competition']:,.0f}",
                f"${scorecard_val['components']['marketing_sales']:,.0f}",
                f"${scorecard_val['components']['need_funding']:,.0f}",
                f"${scorecard_val['total']:,.0f}"
            ]
        })
        st.table(scorecard_df)
        
        # Summary
        st.success(f"Scorecard Method Valuation: ${scorecard_val['total']:,.0f}")
    
    # Comparison of both methods
    st.header("Valuation Comparison")
    
    # Calculate both valuations based on user adjustments
    checklist_val = calculate_checklist_valuation(st.session_state.valuation_data)
    scorecard_val = calculate_scorecard_valuation(st.session_state.valuation_data)
    
    # Calculate valuations based on AI predictions
    ai_checklist_val = calculate_checklist_valuation(st.session_state.ai_predicted_values)
    ai_scorecard_val = calculate_scorecard_valuation(st.session_state.ai_predicted_values)
    
    # Create and display comparison metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("AI-Predicted Valuations")
        st.metric("Checklist Method", f"${ai_checklist_val['total']:,.0f}")
        st.metric("Scorecard Method", f"${ai_scorecard_val['total']:,.0f}")
        ai_average_val = (ai_checklist_val['total'] + ai_scorecard_val['total']) / 2
        st.metric("Average", f"${ai_average_val:,.0f}")
    
    with col2:
        st.subheader("Your Adjusted Valuations")
        st.metric("Checklist Method", f"${checklist_val['total']:,.0f}", 
                 delta=f"${checklist_val['total'] - ai_checklist_val['total']:,.0f}")
        st.metric("Scorecard Method", f"${scorecard_val['total']:,.0f}",
                 delta=f"${scorecard_val['total'] - ai_scorecard_val['total']:,.0f}")
        average_val = (checklist_val['total'] + scorecard_val['total']) / 2
        st.metric("Average", f"${average_val:,.0f}", 
                 delta=f"${average_val - ai_average_val:,.0f}")
    
    # Create and display a bar chart comparing all valuations
    comparison_data = pd.DataFrame({
        'Method': ['Checklist (AI)', 'Checklist (Adjusted)', 
                  'Scorecard (AI)', 'Scorecard (Adjusted)',
                  'Average (AI)', 'Average (Adjusted)'],
        'Valuation ($)': [ai_checklist_val['total'], checklist_val['total'],
                         ai_scorecard_val['total'], scorecard_val['total'],
                         ai_average_val, average_val]
    })
    
    st.bar_chart(comparison_data.set_index('Method'))
    
    # Button to view report
    st.button("⬅️ Back to Report", on_click=lambda: setattr(st.session_state, 'current_view', 'report'))
