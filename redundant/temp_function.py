# Function to extract valuation parameters from AI analysis
def extract_valuation_params(analysis_text):
    """Extract valuation parameters from the AI analysis text.
    
    This function parses the VALUATION CRITERIA SCORES section in the analysis 
    to extract numerical values for checklist scores and scorecard multipliers.
    """
    import copy
    import re
    
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
