from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import json
import os
import requests
import uuid
from datetime import datetime
import re

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'dev_user',
    'password': 'secure_password',
    'database': 'project_similarity'
}

# Helper function to generate embedding
def generate_embedding(text):
    return model.encode(text).tolist()

# Helper function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Helper function to categorize similarity
def categorize_similarity(similarity_score):
    if similarity_score >= 0.8:
        return "High"
    elif similarity_score >= 0.5:
        return "Medium"
    else:
        return "Low"


@app.route('/api/load_data', methods=['POST'])
def load_data():
    """Load original data from a CSV file into the database."""
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    try:
        # Read data from a CSV file (update the path to your CSV file)
        df = pd.read_csv('data/dataset.csv', encoding='ISO-8859-1')

        # Insert the data into the Project table
        insert_query = "INSERT INTO project_iomp (id, title, abstract) VALUES (%s, %s, %s)"
        data = df[['id', 'title', 'abstract']].values.tolist()
        
        # Clear the existing data (optional)
        cursor.execute("TRUNCATE TABLE Project")
        
        # Insert the new data
        cursor.executemany(insert_query, data)
        connection.commit()
        return jsonify({"message": "Original data loaded successfully"}), 200

    except Exception as e:
        connection.rollback()
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

@app.route('/api/generate_embeddings', methods=['POST'])
def generate_embeddings():
    """Generate embeddings for all projects in the database."""
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute("SELECT id, title, abstract FROM project_iomp")
        projects = cursor.fetchall()

        for project in projects:
            combined_text = project['title'] + " " + project['abstract']
            embedding = generate_embedding(combined_text)
            embedding_json = json.dumps(embedding)
            cursor.execute("UPDATE project_iomp SET embedding = %s WHERE id = %s", (embedding_json, project['id']))

        connection.commit()
        return jsonify({"message": "Embeddings generated successfully"}), 200

    except Exception as e:
        connection.rollback()
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

@app.route('/api/find_similar_projects', methods=['POST'])
def find_similar_projects():
    """Find top 10 similar projects based on user input, categorize similarity, and check for gibberish input."""
    user_text = request.json.get('text')
    user_abstract = request.json.get('abstract')
    print('step1')

    if not user_text or not user_abstract:
        return jsonify({"error": "Both title and abstract input are required"}), 400

    # Generate embedding for the combined user input (title + abstract)
    combined_input = user_text + " " + user_abstract
    user_embedding = generate_embedding(combined_input)

    # Connect to the database and fetch all project embeddings
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT id, title, abstract, embedding FROM project_iomp")
    projects = cursor.fetchall()

    similarities = []
    max_similarity = 0

    for project in projects:
        project_embedding = json.loads(project['embedding'])
        similarity = cosine_similarity(user_embedding, project_embedding)

        # Update max similarity score
        max_similarity = max(max_similarity, similarity)

        # Check if both title and abstract are identical
        is_identical = (
            user_text.strip().lower() == project['title'].strip().lower() and
            user_abstract.strip().lower() == project['abstract'].strip().lower()
        )
        similarity_category = categorize_similarity(similarity)

        similarities.append({
            "id": project['id'],
            "title": project['title'],
            "abstract": project['abstract'],
            "similarity": similarity,
            "similarity_category": "Identical" if is_identical else similarity_category,
            "warning": "Identical Title and Abstract" if is_identical else None
        })

    print('step 2')

    # Set a threshold to detect gibberish input
    GIBBERISH_THRESHOLD = 0.2
    # if max_similarity < GIBBERISH_THRESHOLD:
    #     return jsonify([]), 200

    # Filter out low similarity projects (optional)
    similarities = [item for item in similarities if item['similarity'] > 0.1]

    # Sort projects by similarity score in descending order and place identical entries at the top
    similarities = sorted(similarities, key=lambda x: (-int(x['similarity_category'] == "Identical"), -x['similarity']))[:3]

    print('step 3')
    search_guid = check_with_llm(similarities, user_text)
    print('step 4')
    search_results = get_matching_data(search_guid)
    cursor.close()
    connection.close()
    print('step 5')
    return jsonify(search_results), 200

def get_matching_data(search_guid):
    print(f'Getting matching data for {search_guid}')
    # SQL query
    query = """
    SELECT 
        project_iomp.id,
        project_iomp.title,
        project_iomp.abstract,
        user_session.search_guid,
        user_session.matching_score,
        user_session.matching_comments
    FROM user_session
    INNER JOIN project_iomp ON user_session.matched_project_id = project_iomp.id
    WHERE user_session.search_guid = %s
    """

    try:
        # Establish the database connection
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)

        # Execute the query with the provided search GUID
        cursor.execute(query, (search_guid,))

        # Fetch all the results
        results = cursor.fetchall()

        # Return the results as a list of dictionaries
        return results

    except mysql.connector.Error as e:
        print(f"Error fetching data: {e}")
        return []

    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def check_with_llm(projects, user_abstract):
    results = []
    guid = str(uuid.uuid4())

    for idx, project in enumerate(projects):
        print(f"Processing Project {idx + 1}/{len(projects)}")
        # Call the interact_with_llm method for each project abstract
        comparison_result = interact_with_llm(project['abstract'], user_abstract)
        store_matching_info_to_db(guid, project, comparison_result, user_abstract)

    return guid

def store_matching_info_to_db(guid, project, result_data, user_abstract):
    print(f'Storing the matching informaiton')

     # Extract information from result_data
    matched_project_id = project['id']
    matching_score = result_data.get("similarity_score", None)
    matching_comments = result_data.get("comments", "")

    try:
        # Establish the database connection
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)

        # SQL insert query
        insert_query = """
        INSERT INTO user_session (search_guid, user_abstract, matched_project_id, matching_score, matching_comments, created_dt)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        # Prepare the data for insertion
        created_dt = datetime.now().date()
        data = (guid, user_abstract, matched_project_id, matching_score, json.dumps(matching_comments), created_dt)

        print(f'Executing the Insert with {guid} :: {user_abstract} ::{matched_project_id} :: {matching_score} :: {json.dumps(matching_comments)} :: {created_dt}')

        # Execute the insert query
        cursor.execute(insert_query, data)

        # Commit the transaction
        connection.commit()

        return "Record successfully inserted into user_session table."

    except mysql.connector.Error as e:
        return f"Error inserting record: {e}"

    except Exception as e:
        # Catch any unexpected errors
        print(f"An unexpected error occurred: {e}")
        return f"Unexpected error: {e}"

    finally:
        # Close the database connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def interact_with_llm(project_abstract, user_abstract):
    """
    Interacts with the LLaMA API, sends a prompt, and extracts JSON response.
    Returns an empty dictionary if any error occurs.
    """
    prompt = generate_prompt(project_abstract, user_abstract)
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 256,
        "stop": None,
        "stream": False
    }

    try:
        # Send POST request to the LLaMA API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        # Extract the 'response' field from the API response
        data = response.json()
        llm_response = data.get('response', '')

        # Extract JSON data from the response text
        completion = extract_json_from_response(llm_response)
        print("Extracted JSON:", completion)

        return completion

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {}

    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from API response: {e}")
        return {}

    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}

def extract_json_from_response(response_text):
    """
    Extracts JSON data from the given response text.
    Returns an empty dictionary if no valid JSON is found.
    """
    try:
        # Regular expression to find the JSON object in the response
        json_pattern = r"\{[\s\S]*\}"
        match = re.search(json_pattern, response_text)

        if match:
            # Extract the JSON string
            json_str = match.group(0).strip()

            # Check if the extracted string is non-empty
            if not json_str:
                print("Extracted JSON string is empty.")
                return {}

            # Parse the JSON string into a Python dictionary
            json_data = json.loads(json_str)
            return json_data

        else:
            print("No JSON found in the response.")
            return {}

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to decode JSON: {e}")
        return {}

    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}


def generate_prompt(project_abstract: str, user_abstract: str) -> str:
    """
    Generates a system prompt for LLaMA by appending the project abstract and user abstract,
    prefixed with 'ABSTRACT:', using a predefined system prompt for plagiarism review.

    Args:
        project_abstract (str): The abstract for the project.
        user_abstract (str): The user-provided abstract.

    Returns:
        str: The combined prompt ready for LLaMA input.
    """
    # Hard-coded system prompt
    system_prompt = (
        "I am a lecturer reviewing two student project abstracts to ensure their ideas are original "
        "and have not been previously carried out. Please perform the following tasks:\n\n"
        "Calculate the Percentage of Similarity: Analyze the two abstracts and determine the percentage "
        "of similarity between them, and provide the following information in JSON format with the following details:\n"
        "{\n"
        "    similarity_score: <<Calculated similarity score percentage from 0 to 100% (100% being exactly equal)>>,\n"
        "    comments: <<Bullet list of why the project is considered similar / dissimilar>>\n"
        "}"
    )

    # Format the abstracts
    formatted_project_abstract = f"ABSTRACT:\n{project_abstract.strip()}\n"
    formatted_user_abstract = f"ABSTRACT:\n{user_abstract.strip()}\n"

    # Combine the system prompt with the abstracts
    combined_prompt = (
        f"{system_prompt}\n\n"
        f"{formatted_project_abstract}\n"
        f"{formatted_user_abstract}"
    )

    return combined_prompt

if __name__ == '__main__':
    app.run(port=5001)