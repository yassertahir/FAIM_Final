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
import re
from ml_predictor import predict_valuation, process_pitchbook_data, CategoricalImputer, check_pitchbook_data, get_required_features

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
        },
        # ML prediction method
        "ml_prediction": {
            "is_available": False,
            "predicted_value": 0,
            "confidence_score": 0,
            "features_present": 0
        },
        # Region settings
        "region": {
            "selected": "First World / Developed",
            "scaling_factors": {
                "First World / Developed": 1.0,                # Default multiplier (no adjustment)
                "MENA (Middle East & North Africa)": 0.8,      # 20% reduction
                "South Asia (India, Pakistan, etc.)": 0.65,    # 35% reduction
                "Southeast Asia": 0.7,                         # 30% reduction
                "Latin America": 0.75,                         # 25% reduction
                "Africa (excl. North Africa)": 0.6,            # 40% reduction
                "Eastern Europe": 0.85                         # 15% reduction
            }
        }
    }


# Add this helper function to format currency values in millions
def format_in_millions(value):
    """Format currency values in millions."""
    # The value is already in millions, so just format it
    return f"${value:.2f} million"

# Add this to calculate region-adjusted ML prediction
def calculate_ml_prediction_with_region(valuation_data):
    """Calculate ML prediction with regional adjustment applied"""
    # Get the base prediction (already in millions)
    base_prediction = valuation_data["ml_prediction"]["predicted_value"]
    
    # Get the regional scaling factor
    region_name = valuation_data["region"]["selected"]
    region_factor = valuation_data["region"]["scaling_factors"][region_name]
    
    # Apply regional scaling
    adjusted_prediction = base_prediction * region_factor
    
    return {
        "base": base_prediction,
        "adjusted": adjusted_prediction,
        "region_name": region_name,
        "region_factor": region_factor
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
    
    # Apply regional scaling factor
    region_name = valuation_data["region"]["selected"]
    region_factor = valuation_data["region"]["scaling_factors"][region_name]
    
    founders_val = perfect_val * checklist["founders_team"]["weight"] * checklist["founders_team"]["score"]
    idea_val = perfect_val * checklist["idea"]["weight"] * checklist["idea"]["score"]
    market_val = perfect_val * checklist["market"]["weight"] * checklist["market"]["score"]
    product_val = perfect_val * checklist["product_ip"]["weight"] * checklist["product_ip"]["score"]
    execution_val = perfect_val * checklist["execution"]["weight"] * checklist["execution"]["score"]
    
    # Apply scaling to the total
    total_val = (founders_val + idea_val + market_val + product_val + execution_val) * region_factor
    
    # Scale individual components for consistency
    return {
        "total": total_val,
        "components": {
            "founders_team": founders_val * region_factor,
            "idea": idea_val * region_factor,
            "market": market_val * region_factor,
            "product_ip": product_val * region_factor,
            "execution": execution_val * region_factor
        }
    }

# Function to calculate scorecard valuation
def calculate_scorecard_valuation(valuation_data):
    scorecard = valuation_data["scorecard"]
    median_val = scorecard["median_valuation"]
    
    # Apply regional scaling factor
    region_name = valuation_data["region"]["selected"]
    region_factor = valuation_data["region"]["scaling_factors"][region_name]
    
    team_val = median_val * scorecard["team_strength"]["weight"] * scorecard["team_strength"]["multiplier"]
    opportunity_val = median_val * scorecard["opportunity_size"]["weight"] * scorecard["opportunity_size"]["multiplier"]
    product_val = median_val * scorecard["product_service"]["weight"] * scorecard["product_service"]["multiplier"]
    competition_val = median_val * scorecard["competition"]["weight"] * scorecard["competition"]["multiplier"]
    marketing_val = median_val * scorecard["marketing_sales"]["weight"] * scorecard["marketing_sales"]["multiplier"]
    funding_val = median_val * scorecard["need_funding"]["weight"] * scorecard["need_funding"]["multiplier"]
    
    # Apply scaling to the total
    total_val = (team_val + opportunity_val + product_val + competition_val + marketing_val + funding_val) * region_factor
    
    # Scale individual components for consistency
    return {
        "total": total_val,
        "components": {
            "team_strength": team_val * region_factor,
            "opportunity_size": opportunity_val * region_factor,
            "product_service": product_val * region_factor,
            "competition": competition_val * region_factor,
            "marketing_sales": marketing_val * region_factor,
            "need_funding": funding_val * region_factor
        }
    }

# Function to extract valuation parameters from AI analysis
def extract_valuation_params(analysis_text):
    """Extract valuation parameters from the AI analysis text.
    
    This function parses the VALUATION CRITERIA SCORES section in the analysis 
    to extract numerical values for checklist scores and scorecard multipliers.
    It also predicts the startup's region based on content.
    """
    import copy
    
    # Default values if extraction fails - use deep copy to avoid reference issues
    valuation_data = copy.deepcopy(st.session_state.valuation_data)
    
    # Check if we're using ML prediction - if so, we only need to extract the region
    ml_prediction_available = st.session_state.get('pitchbook_data_available', False) and st.session_state.get('valuation_data', {}).get('ml_prediction', {}).get('is_available', False)
    
    # Predict region from the analysis text
    predicted_region = predict_region_from_text(analysis_text)
    
    # Update region data with prediction but preserve scaling factors
    region_data = copy.deepcopy(st.session_state.valuation_data["region"])
    region_data["selected"] = predicted_region
    valuation_data["region"] = region_data
    
    # Print a sample of the AI text for debugging
    if st.session_state.get("debug_mode", False):
        # Show a snippet of the text we're analyzing (last 1000 chars to avoid overwhelming the UI but show more context)
        if len(analysis_text) > 1000:
            st.sidebar.write(f"AI Text snippet (end):\n{analysis_text[-1000:]}")
        else:
            st.sidebar.write(f"AI Text snippet:\n{analysis_text}")
    
    # If using ML prediction, we only needed to extract region, so we can return early
    if ml_prediction_available:
        if st.session_state.get("debug_mode", False):
            st.sidebar.write("Using ML prediction - skipping checklist and scorecard extraction")
        return valuation_data
    
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
    
    # First look for explicit region information
    region_pattern = r"REGION:?\s*([A-Za-z0-9\s&\(\)/,-]+)"
    region_match = re.search(region_pattern, full_text)
    if region_match:
        region_name = region_match.group(1).strip()
        # Check if the extracted region is in our predefined list
        if region_name in valuation_data["region"]["scaling_factors"]:
            valuation_data["region"]["selected"] = region_name
            if st.session_state.get("debug_mode", False):
                st.sidebar.write(f"Found explicit region declaration: {region_name}")
    # If no explicit region found, keep the predicted region that was already set
    
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
        
        # Reset pitchbook data
        st.session_state.pitchbook_data_available = False
        st.session_state.pitchbook_data_df = None
        st.session_state.pitchbook_data_quality = {"is_valid": False, "message": "No data uploaded"}
        
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
            },
            # ML prediction method
            "ml_prediction": {
                "is_available": False,
                "predicted_value": 0,
                "confidence_score": 0,
                "features_present": 0
            },
            # Region settings
            "region": {
                "selected": "First World / Developed",
                "scaling_factors": {
                    "First World / Developed": 1.0,                # Default multiplier (no adjustment)
                    "MENA (Middle East & North Africa)": 0.8,      # 20% reduction
                    "South Asia (India, Pakistan, etc.)": 0.65,    # 35% reduction
                    "Southeast Asia": 0.7,                         # 30% reduction
                    "Latin America": 0.75,                         # 25% reduction
                    "Africa (excl. North Africa)": 0.6,            # 40% reduction
                    "Eastern Europe": 0.85                         # 15% reduction
                }
            }
        }
        
        st.success("Started a new evaluation!")
        st.rerun()
    except Exception as e:
        st.error(f"Error starting new evaluation: {e}")

# Function to predict region from text content
def predict_region_from_text(text):
    """
    Predicts the startup's region based on text content analysis.
    Uses keyword matching to identify region-specific references.
    """
    # Dictionary of regions and their associated keywords
    region_keywords = {
        "MENA (Middle East & North Africa)": [
            "mena", "middle east", "north africa", "dubai", "egypt", "uae", "saudi", 
            "qatar", "bahrain", "kuwait", "oman", "morocco", "tunisia", "algeria", 
            "jordan", "lebanon", "riyadh", "cairo", "gcc"
        ],
        "South Asia (India, Pakistan, etc.)": [
            "india", "pakistan", "bangladesh", "sri lanka", "nepal", "mumbai", 
            "delhi", "karachi", "bangalore", "hyderabad", "lahore", "kolkata", "chennai"
        ],
        "Southeast Asia": [
            "southeast asia", "singapore", "malaysia", "indonesia", "thailand", 
            "philippines", "vietnam", "bangkok", "jakarta", "manila", "kuala lumpur", 
            "hanoi", "asean"
        ],
        "Latin America": [
            "latin america", "brazil", "mexico", "argentina", "chile", "colombia", 
            "peru", "sao paulo", "buenos aires", "mexico city", "santiago", "bogota", "lima"
        ],
        "Africa (excl. North Africa)": [
            "africa", "sub-saharan", "nigeria", "kenya", "south africa", "ghana", 
            "ethiopia", "tanzania", "lagos", "nairobi", "johannesburg", "accra", 
            "addis ababa"
        ],
        "Eastern Europe": [
            "eastern europe", "russia", "poland", "ukraine", "romania", "czech", 
            "hungary", "bulgaria", "moscow", "warsaw", "kyiv", "prague"
        ]
    }
    
    text = text.lower()
    
    # Count mentions of each region's keywords
    region_counts = {region: 0 for region in region_keywords}
    for region, keywords in region_keywords.items():
        for keyword in keywords:
            count = text.count(keyword)
            region_counts[region] += count
    
    # Find the region with the most keyword mentions
    max_count = 0
    predicted_region = "First World / Developed"  # Default
    
    for region, count in region_counts.items():
        if count > max_count:
            max_count = count
            predicted_region = region
    
    # Only return a specific region if we have enough evidence (at least 3 mentions)
    if max_count >= 3:
        return predicted_region
    else:
        return "First World / Developed"  # Default to developed if insufficient evidence

# Create a set for processed files
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Initialize pitchbook data tracking
if "pitchbook_data_available" not in st.session_state:
    st.session_state.pitchbook_data_available = False

if "pitchbook_data_df" not in st.session_state:
    st.session_state.pitchbook_data_df = None

if "pitchbook_data_quality" not in st.session_state:
    st.session_state.pitchbook_data_quality = {"is_valid": False, "message": "No data uploaded"}

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
    
    # Create tabs for different document types
    tab1, tab2 = st.tabs(["Pitch Documents", "Pitchbook Data (Optional)"])
    
    with tab1:
        # Region information (auto-predicted)
        st.subheader("Region Detection")
        st.info("""
        The system will automatically detect the startup's region from your documents.
        Region-specific market conditions will be applied to the valuation calculations.
        You can override the detected region in the valuation page if needed.
        """)
        
        # Allow multiple files
        uploaded_files = st.file_uploader(
            "Upload pitch documents...", 
            accept_multiple_files=True,
            type=["pdf", "xlsx", "docx", "txt"]
        )
        
    with tab2:
        st.subheader("Pitchbook Data")
        st.info("""
        If you have access to Pitchbook data for this company, you can upload a CSV file here.
        This will enable advanced AI-powered valuation prediction using machine learning.
        """)
        
        # Add a button to check if pitchbook data is available
        if st.button("Check Pitchbook Data Availability"):
            st.info("Checking if pitchbook data is available for this company...")
            # For demonstration purposes, we'll just show a message
            st.success("Pitchbook data is available for this company. You can upload it below.")
        
        # Allow CSV file upload for pitchbook data
        pitchbook_file = st.file_uploader(
            "Upload pitchbook CSV data...", 
            accept_multiple_files=False,
            type=["csv"]
        )
        
        # Process pitchbook data if uploaded
        if pitchbook_file:
            try:
                # Load the CSV data
                df = pd.read_csv(pitchbook_file)
                
                # Check if the data has sufficient columns for prediction
                is_valid, message = check_pitchbook_data(df)
                
                # Store results
                st.session_state.pitchbook_data_df = df
                st.session_state.pitchbook_data_quality = {"is_valid": is_valid, "message": message}
                st.session_state.pitchbook_data_available = is_valid
                
                # Update ML prediction availability
                st.session_state.valuation_data["ml_prediction"]["is_available"] = is_valid
                
                # Display results
                if is_valid:
                    st.success(f"✅ Pitchbook data loaded successfully: {message}")
                    
                    # Preview the data
                    st.write("Data Preview:")
                    st.dataframe(df.head(5))
                    
                    # If valid, run the prediction model
                    with st.spinner("Running AI prediction model..."):
                        prediction = predict_valuation(df)
                        # Store the prediction
                        st.session_state.valuation_data["ml_prediction"]["predicted_value"] = float(prediction[0])
                        # Calculate feature coverage as a confidence proxy
                        features_info = get_required_features()
                        total_features = len(features_info['numerical_features']) + len(features_info['categorical_features'])
                        features_present = len([col for col in df.columns if col in features_info['numerical_features'] + features_info['categorical_features']])
                        confidence = min(100, (features_present / total_features) * 100)
                        st.session_state.valuation_data["ml_prediction"]["confidence_score"] = confidence
                        st.session_state.valuation_data["ml_prediction"]["features_present"] = features_present
                        
                        # Show the prediction
                        st.metric("AI-Predicted Valuation", f"${prediction[0]:,.2f}")
                        st.progress(confidence/100, text=f"Confidence: {confidence:.1f}% ({features_present}/{total_features} features present)")
                        
                        # Add an informative message about what this means
                        st.info("""
                        ℹ️ **ML-Based Valuation Active**
                        
                        When you analyze your startup documents, the system will now use this machine learning prediction 
                        instead of asking the AI assistant to generate checklist and scorecard values. This provides a more 
                        data-driven valuation based on real market data from Pitchbook.
                        """)
                else:
                    st.error(f"❌ {message}")
                    st.info("Please provide a CSV file with sufficient pitchbook data. Required fields include: Deal Size, Pre-money Valuation, Primary Industry Sector, Deal Type.")
                    
            except Exception as e:
                st.error(f"Error processing the pitchbook data: {str(e)}")
                st.session_state.pitchbook_data_available = False
                st.session_state.valuation_data["ml_prediction"]["is_available"] = False
    
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
                # Check if ML prediction is available from pitchbook data
                ml_prediction_available = st.session_state.get('pitchbook_data_available', False) and st.session_state.get('valuation_data', {}).get('ml_prediction', {}).get('is_available', False)
                
                # Customize message based on ML prediction availability
                if ml_prediction_available:
                    # Message when ML prediction is available - no need for checklist and scorecard valuations
                    message_text = f"""I've uploaded {len(file_ids)} files for analysis: {', '.join(file_names)}. 
                    Please analyze all files together for a comprehensive evaluation of this startup.

                    IMPORTANT: Identify the geographic region where this startup operates (e.g., MENA, South Asia, Southeast Asia, Latin America, Africa, Eastern Europe, or First World/Developed).

                    ALSO IMPORTANT: We have ML-based valuation data for this startup from Pitchbook data. Skip the checklist and scorecard valuations - they are not needed.

                    Follow the structure below and ensure your bullet points are expanded and thoroughly explained:

                    1. Summary of the proposal
                    2. Strengths (detailed bullet points/paragraphs)
                    3. Areas for improvement (detailed bullet points/paragraphs)
                    4. Team assessment (including a detailed table of team members if CVs are provided)
                    5. Competitive analysis (with names or descriptions of similar startups)
                    6. Regional market analysis (clearly specify which region the startup operates in)
                    7. Overall score (1-10)
                    8. Final recommendation
                    9. Region Information (REQUIRED):

                    REGION: [REGION_NAME]

                    (Please note: Valuation will be calculated using Machine Learning predictions from Pitchbook data)
                    """
                else:
                    # Original message when no ML prediction is available
                    message_text = f"""I've uploaded {len(file_ids)} files for analysis: {', '.join(file_names)}. 
                    Please analyze all files together for a comprehensive evaluation of this startup.

                    IMPORTANT: Identify the geographic region where this startup operates (e.g., MENA, South Asia, Southeast Asia, Latin America, Africa, Eastern Europe, or First World/Developed).

                    Follow the structure below and ensure your bullet points are expanded and thoroughly explained:

                    1. Summary of the proposal
                    2. Strengths (detailed bullet points/paragraphs)
                    3. Areas for improvement (detailed bullet points/paragraphs)
                    4. Team assessment (including a detailed table of team members if CVs are provided)
                    5. Competitive analysis (with names or descriptions of similar startups)
                    6. Regional market analysis (clearly specify which region the startup operates in)
                    7. Overall score (1-10)
                    8. Final recommendation
                    9. Valuation Criteria Scores (REQUIRED - use this exact format):

                    VALUATION CRITERIA SCORES:
                    
                    CHECKLIST METHOD SCORES:
                    - Founders & Team: [SCORE]% (e.g. 75%)
                    - Idea: [SCORE]% (e.g. 65%)
                    - Market Size: [SCORE]% (e.g. 80%)
                    - Product & IP: [SCORE]% (e.g. 70%)
                    - Execution Potential: [SCORE]% (e.g. 60%)
                    
                    REGION: [REGION_NAME]
                    
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
                    # Check if ML prediction is available
                    ml_prediction_available = st.session_state.get('pitchbook_data_available', False) and st.session_state.get('valuation_data', {}).get('ml_prediction', {}).get('is_available', False)
                    
                    # Build dynamic instructions with region detection
                    if ml_prediction_available:
                        instructions = """Please analyze all the uploaded files to provide a comprehensive evaluation of the startup proposal.
                        Use your VC evaluation framework and follow the structure requested in the message.
                        
                        IMPORTANT: Identify which geographic region this startup operates in based on the documents (MENA, South Asia, 
                        Southeast Asia, Latin America, Africa, Eastern Europe, or First World/Developed). This detection 
                        is critical as it will affect the valuation calculation. Include this determination in your analysis.
                        
                        In the Regional Market Analysis section, explicitly state which region the startup operates in 
                        and explain the evidence for this determination.
                        
                        IMPORTANT: We have ML-based valuation data for this startup from Pitchbook data. 
                        DO NOT include any Checklist Method scores or Scorecard Method multipliers in your analysis. 
                        The startup's valuation will be calculated using our machine learning model instead of traditional methods.
                        Simply focus on providing an insightful analysis of the startup's prospects, team, market, and competition,
                        and clearly identify the region of operation.
                        """
                    else:
                        instructions = """Please analyze all the uploaded files to provide a comprehensive evaluation of the startup proposal.
                        Use your VC evaluation framework and follow the structure requested in the message.
                        
                        IMPORTANT: Identify which geographic region this startup operates in based on the documents (MENA, South Asia, 
                        Southeast Asia, Latin America, Africa, Eastern Europe, or First World/Developed). This detection 
                        is critical as it will affect the valuation calculation. Include this determination in your analysis.
                        
                        In the Regional Market Analysis section, explicitly state which region the startup operates in 
                        and explain the evidence for this determination.
                        
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
                    
                    # Store AI predicted values separately with deep copy
                    import copy
                    st.session_state.ai_predicted_values = copy.deepcopy(st.session_state.valuation_data)
                    
                    # Debug log extracted values
                    if st.session_state.get("debug_mode", False):
                        st.sidebar.write("---")
                        st.sidebar.write("AI predicted values stored:")
                        st.sidebar.write(f"Founders score: {st.session_state.ai_predicted_values['checklist']['founders_team']['score']:.2f}")
                        st.sidebar.write(f"Team multiplier: {st.session_state.ai_predicted_values['scorecard']['team_strength']['multiplier']:.1f}x")
                    
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
        
        # Check if ML prediction is available
        ml_prediction_available = st.session_state.get('pitchbook_data_available', False) and st.session_state.get('valuation_data', {}).get('ml_prediction', {}).get('is_available', False)
        
        # If ML prediction is available, display a notice at the top
        if ml_prediction_available:
            with st.expander("ML-based Valuation Information", expanded=True):
                ml_prediction_val = st.session_state.valuation_data["ml_prediction"]["predicted_value"]
                ml_confidence = st.session_state.valuation_data["ml_prediction"]["confidence_score"]
                features_present = st.session_state.valuation_data["ml_prediction"]["features_present"]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.info("📊 Machine Learning Valuation Active: This startup has been valued using our advanced machine learning model trained on Pitchbook data. The traditional checklist and scorecard valuation methods have been bypassed in favor of this more data-driven approach.")
                    
                with col2:
                    st.metric("ML-Predicted Valuation", format_in_millions(ml_prediction_val))
                    st.progress(ml_confidence/100, text=f"Confidence: {ml_confidence:.1f}%")
                
                feature_info = get_required_features()
                total_features = len(feature_info['numerical_features']) + len(feature_info['categorical_features'])
                st.caption(f"Based on {features_present}/{total_features} features from Pitchbook data")
        
        # Display the AI analysis
        st.markdown(st.session_state.ai_analysis)
        
        # Create a divider
        st.divider()
        
        # Only show valuation parameter sliders if ML prediction is not available
        if not ml_prediction_available:
            st.subheader("Valuation Base Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Perfect valuation for Checklist Method as a slider
                perfect_val = st.slider(
                    "Perfect Valuation (Checklist Method) ($)",
                    min_value=1000000,
                    max_value=100000000,
                    value=int(st.session_state.valuation_data["checklist"]["perfect_valuation"]),
                    step=500000,
                    format="$%d"
                )
                st.session_state.valuation_data["checklist"]["perfect_valuation"] = perfect_val
            
            with col2:
                # Median valuation for Scorecard Method as a slider
                median_val = st.slider(
                    "Median Industry Valuation (Scorecard Method) ($)",
                    min_value=1000000,
                    max_value=100000000,
                    value=int(st.session_state.valuation_data["scorecard"]["median_valuation"]),
                    step=500000,
                    format="$%d"
                )
                st.session_state.valuation_data["scorecard"]["median_valuation"] = median_val
            
        # Store a deep copy of the AI predictions if not already stored
        if 'ai_predicted_values' not in st.session_state:
            import copy
            st.session_state.ai_predicted_values = copy.deepcopy(st.session_state.valuation_data)
            
            # Debug info
            if st.session_state.get("debug_mode", False):
                st.sidebar.write("Created AI predicted values in report view")
                st.sidebar.write(f"Founders score: {st.session_state.ai_predicted_values['checklist']['founders_team']['score']:.2f}")
                st.sidebar.write(f"Team multiplier: {st.session_state.ai_predicted_values['scorecard']['team_strength']['multiplier']:.1f}x")
        
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
    
    # Display valuation overview with region information
    region_name = st.session_state.valuation_data["region"]["selected"]
    region_factor = st.session_state.valuation_data["region"]["scaling_factors"][region_name]
    
    # Check if ML prediction is available
    ml_prediction_available = st.session_state.valuation_data["ml_prediction"]["is_available"]
    
    if ml_prediction_available:
        st.markdown(f"""
        ## ML-Based Valuation Overview
        
        This startup is being valued using our machine learning model trained on Pitchbook data.
        The model incorporates regional market conditions and industry-specific factors to generate a data-driven valuation.
        
        ### Regional Adjustment
        
        **Auto-Detected Region: {region_name}**  
        **Regional Scaling Factor: {region_factor:.2f}x** *(ML predictions are adjusted by this factor to account for regional market conditions)*
        
        *The region was automatically detected from your documents. You can change it below if needed.*
        """)
    else:
        st.markdown(f"""
        ## Valuation Overview
        
        This section provides two different methods to calculate the valuation of the startup:
        
        1. **Checklist Method**: Evaluates specific areas with percentage scores against a perfect valuation
        2. **Scorecard Method**: Applies premium/discount multipliers to a median industry valuation
        
        The sliders are pre-set with AI-predicted values based on the pitchdeck quality, but you can adjust them.
        
        ### Regional Adjustment
        
        **Auto-Detected Region: {region_name}**  
        **Regional Scaling Factor: {region_factor:.2f}x** *(valuations are scaled by this factor to account for regional market conditions)*
        
        *The region was automatically detected from your documents. You can change it below if needed.*
        """)
            
    # Add controls for region selection and reset to AI values
    col1, col2 = st.columns(2)
    
    with col1:
        # Region selector
        region_options = list(st.session_state.valuation_data["region"]["scaling_factors"].keys())
        selected_region = st.selectbox(
            "Select Region:",
            options=region_options,
            index=region_options.index(st.session_state.valuation_data["region"]["selected"]),
            key="valuation_region_selector"
        )
        
        # Update the selected region
        if selected_region != st.session_state.valuation_data["region"]["selected"]:
            st.session_state.valuation_data["region"]["selected"] = selected_region
            
            # If using ML prediction, inform the user that regional adjustment affects ML valuation
            if ml_prediction_available:
                st.info(f"The ML-based valuation will be adjusted based on the {selected_region} regional market conditions.")
            
            st.rerun()
    
    with col2:
        if ml_prediction_available:
            # For ML prediction, offer button to recalculate with regional adjustment
            if st.button("Apply Regional Adjustment to ML Prediction"):
                # No need to reset values since we're only using ML prediction
                st.success(f"ML prediction adjusted for {selected_region} regional market conditions.")
                st.rerun()
        else:
            # For traditional methods, add reset button to restore AI predictions
            if st.button("Reset to AI-Predicted Values"):
                # Deep copy to ensure we don't have reference issues
                import copy
                current_region = st.session_state.valuation_data["region"]["selected"]
                st.session_state.valuation_data = copy.deepcopy(st.session_state.ai_predicted_values)
                
                # Debug info
                if st.session_state.get("debug_mode", False):
                    st.sidebar.write("Reset to AI predicted values")
                    st.sidebar.write(f"Founders score: {st.session_state.valuation_data['checklist']['founders_team']['score']:.2f}")
                    st.sidebar.write(f"Team multiplier: {st.session_state.valuation_data['scorecard']['team_strength']['multiplier']:.1f}x")
                st.rerun()
    
    # Check if ML prediction is available
    ml_prediction_available = st.session_state.valuation_data["ml_prediction"]["is_available"]
    
    # Show different view based on whether ML prediction is available
    if ml_prediction_available:
        st.success("Using Machine Learning prediction for valuation instead of traditional methods")
        
        # Display ML prediction details
        ml_prediction_val = st.session_state.valuation_data["ml_prediction"]["predicted_value"]
        ml_confidence = st.session_state.valuation_data["ml_prediction"]["confidence_score"]
        features_present = st.session_state.valuation_data["ml_prediction"]["features_present"]
        
        st.metric("ML-Predicted Valuation", format_in_millions(ml_prediction_val))
        st.progress(ml_confidence/100, text=f"Confidence: {ml_confidence:.1f}%")
        
        feature_info = get_required_features()
        total_features = len(feature_info['numerical_features']) + len(feature_info['categorical_features'])
        st.info(f"This valuation is based on {features_present}/{total_features} features from Pitchbook data")
        
        st.markdown("""
        ### Machine Learning Valuation Methodology
        
        This startup has been valued using our advanced machine learning model trained on Pitchbook data.
        The model analyzes various factors such as:
        
        - Deal size and structure
        - Industry sector performance
        - Growth metrics
        - Investment stage
        - Market conditions
        
        The traditional checklist and scorecard valuation methods have been bypassed in favor of this more data-driven approach.
        """)
    else:
        # Create tabs for the two valuation methods
        tab1, tab2 = st.tabs(["Checklist Method", "Scorecard Method"])
        
        with tab1:
            st.header("Checklist Method")
            st.markdown("""
            This method evaluates 5 key areas with weighted scores, then multiplies by a perfect valuation
            """)
            
            # Perfect valuation input as a slider
            perfect_val = st.slider(
                "Perfect Valuation ($)",
                min_value=1000000,
                max_value=100000000,
                value=int(st.session_state.valuation_data["checklist"]["perfect_valuation"]),
                step=500000,
                format="$%d",
                key="checklist_perfect_val"
            )
            st.session_state.valuation_data["checklist"]["perfect_valuation"] = perfect_val
            
            # Create a more compact layout with columns for factor sliders
            st.write("### Adjust Scores for Each Factor")
            
            # First row: Founders & Team and Idea
            col1, col2 = st.columns(2)
            
            with col1:
                # Founders & Team
                st.write("**Founders & Team (30%)**")
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
            
            with col2:
                # Idea
                st.write("**Idea (20%)**")
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
            
            # Second row: Market Size and Product and IP
            col1, col2 = st.columns(2)
            
            with col1:
                # Market Size
                st.write("**Market Size (20%)**")
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
            
            with col2:
                # Product and IP
                st.write("**Product and IP (15%)**")
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
            
            # Third row: Execution only
            st.write("**Execution (15%)**")
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
        
            # Median valuation input as a slider
            median_val = st.slider(
                "Median Pre-money Valuation ($)",
                min_value=1000000,
                max_value=50000000,
                value=int(st.session_state.valuation_data["scorecard"]["median_valuation"]),
                step=500000,
                format="$%d",
                key="scorecard_median_val"
            )
            st.session_state.valuation_data["scorecard"]["median_valuation"] = median_val
            
            # Create a more compact layout with columns for factor sliders
            st.write("### Adjust Premium/Discount for Each Factor")
            
            # First row: Team Strength and Opportunity Size
            col1, col2 = st.columns(2)
            
            with col1:
                # Team Strength
                st.write("**Team Strength (24%)**")
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
            
            with col2:
                # Opportunity Size
                st.write("**Opportunity Size (22%)**")
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
            
            # Second row: Product/Service and Competition
            col1, col2 = st.columns(2)
            
            with col1:
                # Product/Service
                st.write("**Product/Service (20%)**")
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
            
            with col2:
                # Competition
                st.write("**Competition (16%)**")
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
            
            # Third row: Marketing & Sales and Need for Funding
            col1, col2 = st.columns(2)
            
            with col1:
                # Marketing & Sales
                st.write("**Marketing & Sales (12%)**")
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
            
            with col2:
                # Need for Funding
                st.write("**Need for Funding (6%)**")
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
    
    # Valuation results section
    st.header("Valuation Results")
    
    # Get ML prediction if available
    ml_prediction_available = st.session_state.valuation_data["ml_prediction"]["is_available"]
    
    # Show different views based on whether ML prediction is available
    if ml_prediction_available:
        # Display just the ML prediction with detailed information
        ml_prediction_details = calculate_ml_prediction_with_region(st.session_state.valuation_data)
        ml_prediction_base = ml_prediction_details["base"]
        ml_prediction_adjusted = ml_prediction_details["adjusted"]
        region_name = ml_prediction_details["region_name"]
        region_factor = ml_prediction_details["region_factor"]
        
        # Get confidence metrics
        ml_confidence = st.session_state.valuation_data["ml_prediction"]["confidence_score"]
        features_present = st.session_state.valuation_data["ml_prediction"]["features_present"]
        feature_info = get_required_features()
        total_features = len(feature_info['numerical_features']) + len(feature_info['categorical_features'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("ML-Based Valuation Details")
            st.info("""
            **Pitchbook ML Model Valuation**
            
            This startup's valuation is generated by our machine learning model trained on real Pitchbook data.
            Traditional valuation methods (Checklist and Scorecard) have been bypassed in favor of this data-driven approach.
            """)
            
            st.markdown(f"""
            ### Technical Information
            - **Features Used**: {features_present} out of {total_features} available features
            - **Confidence Score**: {ml_confidence:.1f}% (based on feature coverage and data quality)
            - **Regional Market**: {region_name} (scaling factor: {region_factor:.2f}x)
            """)
        
        with col2:
            # Show both base and adjusted valuations
            st.metric("Base ML Valuation", format_in_millions(ml_prediction_base))
            st.metric("Region-Adjusted Valuation", 
                    format_in_millions(ml_prediction_adjusted),
                    delta=f"{'+' if region_factor > 1 else ''}{(region_factor - 1) * 100:.0f}%")
            st.progress(ml_confidence/100, text=f"Confidence: {ml_confidence:.1f}%")
            
    else:
        # Display traditional valuation methods comparison
        # Calculate both valuations based on user adjustments
        checklist_val = calculate_checklist_valuation(st.session_state.valuation_data)
        scorecard_val = calculate_scorecard_valuation(st.session_state.valuation_data)
        
        # Calculate valuations based on AI predictions
        ai_checklist_val = calculate_checklist_valuation(st.session_state.ai_predicted_values)
        ai_scorecard_val = calculate_scorecard_valuation(st.session_state.ai_predicted_values)
        
        # Create columns for displaying traditional valuations
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
            """It's typically more accurate than traditional methods since it's based on actual market data 
            rather than subjective assessments.
            """
            
            # Show a notice if ML prediction is significantly different
            if 'ml_prediction' in st.session_state.valuation_data and st.session_state.valuation_data['ml_prediction'].get('is_available'):
                # Get ML prediction value
                ml_prediction_val = st.session_state.valuation_data['ml_prediction']['predicted_value']
                
                # Show a notice if ML prediction is significantly different
                threshold = 0.2  # 20% difference
                avg_traditional = (checklist_val['total'] + scorecard_val['total']) / 2
                
                # Convert ML prediction to same scale as traditional methods (from millions to full value)
                ml_prediction_full = ml_prediction_val * 1000000
                diff_percentage = abs(ml_prediction_full - avg_traditional) / avg_traditional
                
                if diff_percentage > threshold:
                    st.info(f"⚠️ ML prediction (${ml_prediction_val:.2f}M) differs by {diff_percentage:.1%} from traditional methods. Consider reviewing the valuation.")
    # Create and display a bar chart for valuations
    # if ml_prediction_available:
    #     # When ML prediction is available, just show that
    #     methods = ['ML Prediction (Pitchbook AI)']
    #     values = [ml_prediction_val]  # Value is already in millions
        
    #     # Add reference to average regional valuation for context
    #     region_name = st.session_state.valuation_data["region"]["selected"]
    #     region_factor = st.session_state.valuation_data["region"]["scaling_factors"][region_name]
    #     regional_avg = st.session_state.valuation_data["scorecard"]["median_valuation"] * region_factor / 1000000  # Convert to millions
        
    #     methods.append(f'Regional Avg ({region_name})')
    #     values.append(regional_avg)
        
    #     comparison_data = pd.DataFrame({
    #         'Method': methods,
    #         'Valuation ($ millions)': values  # Updated label to indicate millions
    #     })
    # else:
    if not ml_prediction_available:
        # When using traditional methods, show all valuation approaches
        methods = ['Checklist (AI)', 'Checklist (Adjusted)', 
                  'Scorecard (AI)', 'Scorecard (Adjusted)',
                  'Average (AI)', 'Average (Adjusted)']
        
        values = [ai_checklist_val['total'], checklist_val['total'],
                ai_scorecard_val['total'], scorecard_val['total'],
                ai_average_val, average_val]
        
        comparison_data = pd.DataFrame({
            'Method': methods,
            'Valuation ($)': values
        })
    
        st.bar_chart(comparison_data.set_index('Method'))
    
    # Regional scaling factors adjustment section
    with st.expander("Adjust Regional Scaling Factors"):
        st.write("These scaling factors adjust valuations based on regional market conditions.")
        st.write("A factor of 1.0 means no adjustment, while lower values reduce the valuation.")
        
        # Store the current factors to check for changes
        old_factors = st.session_state.valuation_data["region"]["scaling_factors"].copy()
        new_factors = {}
        
        # Create sliders for each region
        for region, factor in st.session_state.valuation_data["region"]["scaling_factors"].items():
            new_factors[region] = st.slider(
                f"{region}",
                min_value=0.1,
                max_value=2.0,
                value=factor,
                step=0.05,
                format="%.2fx",
                key=f"region_factor_{region}"
            )
        
        # Check if any factors changed and update
        factors_changed = False
        for region, factor in new_factors.items():
            if factor != old_factors[region]:
                factors_changed = True
        
        if factors_changed:
            st.session_state.valuation_data["region"]["scaling_factors"] = new_factors
            st.warning("Regional scaling factors have been updated. The changes will be reflected in all valuations.")
    
    # Button to view report
    st.button("⬅️ Back to Report", on_click=lambda: setattr(st.session_state, 'current_view', 'report'))
