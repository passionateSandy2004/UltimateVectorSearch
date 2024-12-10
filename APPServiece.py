from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai

# Configure Generative AI
genai.configure(api_key="AIzaSyAnLVmAm9r4ZkiCW-TXCz8HAaff-IfvWn0")
gen_model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory storage for user embeddings and data
user_data = {}

def make_embedding(user_name, string_list):
    """
    Takes a username and a list of strings, computes embeddings, 
    and stores them in the global user_data dictionary.
    """
    embeddings = model.encode(string_list, convert_to_tensor=True)
    user_data[user_name] = {
        "string_list": string_list,
        "embeddings": embeddings
    }

def pre_process_data(data):
    """
    Prepares data by combining key-value pairs into strings for embedding storage.
    """
    processed_data = []
    for item in data:
        string_representation = " ".join(f"{key}: {value}" for key, value in item.items())
        processed_data.append(string_representation)
    return processed_data

@app.route('/load', methods=['POST'])
def load_data():
    """
    API endpoint to load JSON data for a user.
    Expects JSON payload with 'user_name' and 'data' fields.
    """
    data = request.json
    user_name = data.get("user_name")
    user_data_input = data.get("data")
    
    if not user_name or not user_data_input:
        return jsonify({"error": "user_name and data are required"}), 400
    
    try:
        processed_data = pre_process_data(user_data_input)
        make_embedding(user_name, processed_data)
        return jsonify({"message": f"Data loaded successfully for user: {user_name}"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load data: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    API endpoint to perform a chat-based semantic search.
    Expects JSON payload with 'user_name' and 'query'.
    """
    data = request.json
    user_name = data.get("user_name")
    query = data.get("query")
    
    if not user_name or not query:
        return jsonify({"error": "user_name and query are required"}), 400
    
    if user_name not in user_data:
        return jsonify({"error": f"No data found for user: {user_name}"}), 404

    try:
        # Step 1: Generate refined semantic search questions
        prompt = (
            f"The user has a very large database and wants to perform semantic search."
            f" Based on the query '{query}', provide refined semantic search questions or"
            f" key phrases separated by commas. Do not include any unrelated words."
        )
        gen_response = gen_model.generate_content(prompt)
        refined_queries = gen_response.text.split(",")
        
        # Step 2: Perform semantic search
        combined_results = ""
        indexes = []
        for refined_query in refined_queries:
            query_embedding = model.encode(refined_query.strip(), convert_to_tensor=True)
            embeddings = user_data[user_name]["embeddings"]
            cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)
            most_similar_index = torch.argmax(cosine_scores).item()
            most_similar_string = user_data[user_name]["string_list"][most_similar_index]
            
            if most_similar_string not in combined_results:
                combined_results += f"{most_similar_string} "
                indexes.append(most_similar_index)
        
        # Step 3: Generate the final response
        final_prompt = f"Answer the question '{query}' based on the email data: {combined_results}"
        final_response = gen_model.generate_content(final_prompt)
        return jsonify({
            "query": query,
            "response": final_response.text,
            "matched_indexes": indexes,
            "matched_data": [user_data[user_name]["string_list"][i] for i in indexes]
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to process chat: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
