from flask import Flask, render_template, jsonify, request, redirect, url_for
from pymongo import MongoClient
from bson import ObjectId
import requests
from ultralytics import YOLO
import os
import json
from ollama import Client

app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb+srv://f87study:admin1234@cluster0.fqatder.mongodb.net/Desert")
db = client.get_database()

# Initialize YOLO model
yolo_model = YOLO('yolov8n.pt')

# Ollama configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama port
OLLAMA_MODEL = "chef"  # Using our custom chef model

# Initialize Ollama client for recipe generation
recipe_client = Client(host='http://localhost:11434')
recipe_model = 'ingredients-chef'

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
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded file temporarily
    temp_path = 'temp_image.jpg'
    try:
        file.save(temp_path)
        
        # Run YOLO detection with timeout
        try:
            results = yolo_model(temp_path)
        except Exception as e:
            return jsonify({'error': f'Error during detection: {str(e)}'}), 500
        
        # Process results
        ingredients = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                
                # Filter for food-related classes (you may need to adjust this based on your YOLO model)
                if class_name.lower() in ['apple', 'banana', 'orange', 'carrot', 'broccoli', 'tomato', 'potato', 'onion', 'garlic', 'chicken', 'beef', 'fish', 'egg', 'bread', 'cheese', 'milk', 'butter', 'rice', 'pasta']:
                    ingredients.append({
                        'name': class_name,
                        'confidence': confidence
                    })
        
        return jsonify({'ingredients': ingredients})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

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
    app.run(debug=True)
