from flask import Flask, request, jsonify
import mysql.connector
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Initialize Flask app
app = Flask(__name__)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'dev',
    'password': 'dev',
    'database': 'similaritydetection'
}

# Helper function to generate embedding
def generate_embedding(text):
    return model.encode(text).tolist()  # Convert to list for JSON compatibility

# Helper function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.route('/api/generate_embeddings', methods=['POST'])
def generate_embeddings():
    """Generate embeddings for all projects in the database and store them."""
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)

    # Fetch all project records
    cursor.execute("SELECT id, title, abstract FROM Project")
    projects = cursor.fetchall()

    # Generate and update embeddings for each project
    for project in projects:
        text = project['title'] + " " + project['abstract']
        embedding = generate_embedding(text)
        embedding_json = json.dumps(embedding)

        # Update project record with embedding
        cursor.execute("UPDATE Project SET embedding = %s WHERE id = %s", (embedding_json, project['id']))
    
    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({"message": "Embeddings generated and stored for all projects"}), 200

@app.route('/api/find_similar_projects', methods=['POST'])
def find_similar_projects():
    """Find top 3 similar projects based on user input text."""
    user_text = request.json.get('text')
    if not user_text:
        return jsonify({"error": "Text input is required"}), 400

    # Generate embedding for the user input text
    user_embedding = generate_embedding(user_text)

    # Connect to the database and fetch all project embeddings
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT id, title, abstract, embedding FROM Project")
    projects = cursor.fetchall()

    # Calculate cosine similarity with each project embedding
    similarities = []
    for project in projects:
        project_embedding = json.loads(project['embedding'])
        similarity = cosine_similarity(user_embedding, project_embedding)
        similarities.append((project['id'], project['title'], project['abstract'], similarity))

    # Sort projects by similarity score in descending order and get top 3
    top_projects = sorted(similarities, key=lambda x: x[3], reverse=True)[:3]
    
    # Format the response
    result = [
        {"id": proj[0], "title": proj[1], "abstract": proj[2], "similarity": proj[3]}
        for proj in top_projects
    ]

    cursor.close()
    connection.close()

    return jsonify(result), 200

if __name__ == '__main__':
    app.run(port=5001)
