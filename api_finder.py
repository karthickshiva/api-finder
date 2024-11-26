from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class APIEndpoint:
    path: str
    method: str
    description: str
    parameters: List[Dict[str, Any]]
    embedding: np.ndarray = None

class OpenAPISearcher:
    def __init__(self, openapi_spec_path: str):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.endpoints = self._process_openapi_spec(openapi_spec_path)

    def _process_openapi_spec(self, spec_path: str) -> List[APIEndpoint]:
        with open(spec_path) as f:
            spec = json.load(f)

        endpoints = []
        for path, methods in spec['paths'].items():
            for method, details in methods.items():
                description = self._create_endpoint_description(path, method, details)
                embedding = self.model.encode(description)
                endpoint = APIEndpoint(
                    path=path,
                    method=method.upper(),
                    description=description,
                    parameters=details.get('parameters', []),
                    embedding=embedding
                )
                endpoints.append(endpoint)
        return endpoints

    def _create_endpoint_description(self, path: str, method: str, details: Dict) -> str:
        parts = [
            f"{method.upper()} {path}",
            details.get('summary', ''),
            details.get('description', '')
        ]
        params = details.get('parameters', [])
        if params:
            param_desc = [f"Parameter '{p['name']}' ({p['in']}): {p.get('description', '')}" 
                         for p in params]
            parts.extend(param_desc)
        return ' '.join(filter(None, parts))

    def find_matching_api(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(query)
        similarities = []
        for endpoint in self.endpoints:
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                endpoint.embedding.reshape(1, -1)
            )[0][0]
            similarities.append((similarity, endpoint))
        
        similarities.sort(reverse=True)
        top_matches = similarities[:top_k]
        
        results = []
        for similarity, endpoint in top_matches:
            results.append({
                'path': endpoint.path,
                'method': endpoint.method,
                'description': endpoint.description,
                'confidence': float(similarity),
                'parameters': endpoint.parameters
            })
        return results

if __name__ == "__main__":
    SAMPLE_SPEC = {
        "openapi": "3.0.0",
        "paths": {
            "/users": {
                "get": {
                    "summary": "List all users",
                    "description": "Returns a list of users in the system",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "Maximum number of users to return"
                        }
                    ]
                },
                "post": {
                    "summary": "Create a new user",
                    "description": "Creates a new user in the system"
                }
            },
            "/users/{id}": {
                "get": {
                    "summary": "Get user details",
                    "description": "Returns details of a specific user",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "description": "The user ID"
                        }
                    ]
                }
            }
        }
    }

    with open('sample_spec.json', 'w') as f:
        json.dump(SAMPLE_SPEC, f)

    searcher = OpenAPISearcher('sample_spec.json')

    queries = [
        "How do I get a list of all users?",
        "I want to create a new user",
        "Get details about a specific user"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        matches = searcher.find_matching_api(query)
        for i, match in enumerate(matches, 1):
            print(f"\nMatch {i} (Confidence: {match['confidence']:.3f}):")
            print(f"Method: {match['method']}")
            print(f"Path: {match['path']}")
            print(f"Description: {match['description']}")
