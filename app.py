from flask import Flask, render_template, jsonify, request, redirect, url_for
from pymongo import MongoClient
from bson import ObjectId
import requests
import os
import json
from ollama import Client
import base64
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb+srv://f87study:admin1234@cluster0.fqatder.mongodb.net/Desert")
db = client.get_database()

# Ollama configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "chef"

# Initialize Ollama client for recipe generation
recipe_client = Client(host='http://localhost:11434')
recipe_model = 'ingredients-chef'

# Initialize YOLO model with the default YOLOv8 model
model = YOLO('yolov8n.pt')  # Using the default YOLOv8 nano model

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="g7A7DM8HhOu5hpCY4jll"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/deserts')
def get_deserts():
    deserts = list(db.Desert.find())
    # Convert ObjectId to string for JSON serialization
    for desert in deserts:
        desert['_id'] = str(desert['_id'])
    return jsonify(deserts)

@app.route('/add-recipe', methods=['GET', 'POST'])
def add_recipe():
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        img_url = request.form.get('img_url')
        ingredients = request.form.get('ingredients').split('\n')
        recipe = request.form.get('recipe')
        
        # Create new recipe document
        new_recipe = {
            'name': name,
            'img': img_url,
            'ingredients': [ing.strip() for ing in ingredients if ing.strip()],
            'recipe': recipe
        }
        
        # Insert into MongoDB
        db.Desert.insert_one(new_recipe)
        
        return redirect(url_for('index'))
    
    return render_template('add_recipe.html')

@app.route('/edit-recipe/<recipe_id>', methods=['GET', 'POST'])
def edit_recipe(recipe_id):
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        img_url = request.form.get('img_url')
        ingredients = request.form.get('ingredients').split('\n')
        recipe = request.form.get('recipe')
        
        # Update recipe document
        db.Desert.update_one(
            {'_id': ObjectId(recipe_id)},
            {
                '$set': {
                    'name': name,
                    'img': img_url,
                    'ingredients': [ing.strip() for ing in ingredients if ing.strip()],
                    'recipe': recipe
                }
            }
        )
        
        return redirect(url_for('index'))
    
    # Get the recipe for editing
    recipe = db.Desert.find_one({'_id': ObjectId(recipe_id)})
    if recipe:
        recipe['_id'] = str(recipe['_id'])
        return render_template('edit_recipe.html', recipe=recipe)
    return redirect(url_for('index'))

@app.route('/delete-recipe/<recipe_id>', methods=['POST'])
def delete_recipe(recipe_id):
    db.Desert.delete_one({'_id': ObjectId(recipe_id)})
    return redirect(url_for('index'))

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/detect')
def detect_page():
    return render_template('detect.html')

@app.route('/api/detect', methods=['POST'])
def detect_ingredients():
    print("Received detection request")
    if 'image' not in request.files:
        print("No image file in request")
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No selected file'}), 400
    
    # Check file size (limit to 5MB)
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    if file_size > 5 * 1024 * 1024:
        return jsonify({'error': 'File size too large. Please upload an image smaller than 5MB.'}), 400
    
    print(f"Processing file: {file.filename} (size: {file_size} bytes)")
    
    try:
        # Save the file temporarily
        temp_path = 'temp_image.jpg'
        file.save(temp_path)
        
        # Run Roboflow detection
        print("Running Roboflow detection...")
        result = CLIENT.infer(temp_path, model_id="vegetables-el4g6/1")
        
        # Process results
        ingredients = []
        if 'predictions' in result:
            for prediction in result['predictions']:
                class_name = prediction['class'].lower()
                confidence = prediction['confidence']
                
                ingredients.append({
                    'name': class_name,
                    'confidence': confidence
                })
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print(f"Detection complete. Found {len(ingredients)} ingredients")
        return jsonify({'ingredients': ingredients})
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': error_msg}), 500

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    prompt = data.get('prompt', '')
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-recipe', methods=['POST'])
def generate_recipe():
    try:
        data = request.get_json()
        ingredients = data.get('ingredients', [])
        
        if not ingredients:
            return jsonify({'error': 'No ingredients provided'}), 400
        
        # Format ingredients for the prompt
        ingredients_text = "\n".join([
            f"- {ing['name']} (confidence: {ing['confidence']:.2f})"
            for ing in ingredients
        ])
        
        # Generate recipe using Ollama
        response = recipe_client.generate(
            model=recipe_model,
            prompt=f"""Based on the following detected ingredients and their confidence scores, create a recipe:

{ingredients_text}

Please create a recipe that primarily uses the high-confidence ingredients and suggests substitutions for low-confidence ones."""
        )
        
        return jsonify({'recipe': response['response']})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)
