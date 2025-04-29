"""
Schema definitions for the package name element extractor.
"""

# --- Name Element Categories --- 
CATEGORIES = {
    'A': 'Architecture',
    'S': 'Model size',
    'D': 'Dataset',
    'C': 'Dataset characteristic',
    'V': 'Model version',
    'F': 'Reuse method',
    'L': 'Language',
    'T': 'Task or Application Goal',
    'R': 'Training process',
    'N': 'Number of layers',
    'H': 'Number of heads',
    'P': 'Number of parameters',
    'O': 'Other'
}

# --- JSON Schema for response format ---
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "packageAnalysis": {
            "type": "array",
            "description": "Analysis of each package name's elements",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The simplified package name (after the slash)"
                    },
                    "componentMapping": {
                        "type": "array",
                        "description": "Mapping of name components to their categories",
                        "items": {
                            "type": "object",
                            "properties": {
                                "component": {
                                    "type": "string",
                                    "description": "The specific component of the model name"
                                },
                                "category": {
                                    "type": "string",
                                    "description": "Category code (A, S, D, etc.)"
                                }
                            },
                            "required": ["component", "category"]
                        }
                    }
                },
                "required": ["name", "componentMapping"]
            }
        }
    },
    "required": ["packageAnalysis"]
}

