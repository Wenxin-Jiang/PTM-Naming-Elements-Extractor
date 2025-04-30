"""
Schema definitions for the package name element extractor.
"""


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
                            "required": ["component", "category"],
                            "additionalProperties": False  # Disallow extra fields in component mapping
                        }
                    }
                },
                "required": ["name", "componentMapping"],
                "additionalProperties": False  # Disallow extra fields in package analysis items
            }
        }
    },
    "required": ["packageAnalysis"],
    "additionalProperties": False  # Disallow extra fields at the top level
}

