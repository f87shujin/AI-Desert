from flask import Flask, render_template, jsonify, request, redirect, url_for
from pymongo import MongoClient
from bson import ObjectId
import requests

app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb+srv://f87study:admin1234@cluster0.fqatder.mongodb.net/Desert")
db = client.get_database()

# Ollama configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama port
OLLAMA_MODEL = "chef"  # Using our custom chef model

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

if __name__ == '__main__':
    app.run(debug=True)
