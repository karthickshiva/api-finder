

# API Finder

API Finder is a Python tool that uses natural language processing to help users find relevant API endpoints from OpenAPI specifications. It employs semantic search to match user queries with the most appropriate API endpoints.

## Features

- Semantic search using sentence transformers
- Support for OpenAPI 3.0 specifications
- Confidence scoring for matched endpoints
- Returns multiple matching endpoints ranked by relevance
- Handles endpoint parameters and descriptions

## Requirements

- Python 3.6+
- sentence-transformers
- scikit-learn
- numpy

## Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/api-finder.git
cd api-finder
```

2. Install the required packages
```bash
pip install sentence-transformers scikit-learn numpy
```

## Usage

```python
from api_finder import OpenAPISearcher

# Initialize the searcher with your OpenAPI specification
searcher = OpenAPISearcher('path/to/your/openapi_spec.json')

# Search for relevant endpoints
results = searcher.find_matching_api("How do I get a list of all users?")

# Process results
for match in results:
    print(f"Method: {match['method']}")
    print(f"Path: {match['path']}")
    print(f"Confidence: {match['confidence']}")
    print(f"Description: {match['description']}")
```

## Example Output

```
Query: How do I get a list of all users?

Match 1 (Confidence: 0.785):
Method: GET
Path: /users
Description: GET /users List all users Returns a list of users in the system Parameter 'limit' (query): Maximum number of users to return
```

## How It Works

1. The tool loads an OpenAPI specification and processes each endpoint
2. Endpoint descriptions are converted into embeddings using the `all-MiniLM-L6-v2` model
3. When a query is received, it's converted to the same embedding space
4. Cosine similarity is used to find the most relevant endpoints
5. Results are returned with confidence scores

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.