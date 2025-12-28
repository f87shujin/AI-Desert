from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from pymongo import MongoClient
from bson import ObjectId
import requests
import os
import json
import base64
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
from dotenv import load_dotenv
import secrets
from datetime import datetime
import random

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))

# MongoDB connection
client = MongoClient(os.getenv('MONGODB_URI'))
db = client.get_database()

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# Google Custom Search API configuration
GOOGLE_API_KEYS = [os.getenv('GOOGLE_API_KEY_1'), os.getenv('GOOGLE_API_KEY_2')]
GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="g7A7DM8HhOu5hpCY4jll"
)

def fetch_recipe_image(recipe_name):
    """Fetch a random recipe image from Google Custom Search API"""
    # Blacklisted domains that often have loading issues
    BLACKLISTED_DOMAINS = [
        'facebook.com', 'fb.com', 'fbcdn.net',
        'instagram.com', 'cdninstagram.com',
        'twitter.com', 'twimg.com',
        'linkedin.com', 'licdn.com'
    ]
    
    def is_valid_image_url(url):
        """Check if image URL is from a valid source"""
        url_lower = url.lower()
        # Check if URL is from blacklisted domain
        for domain in BLACKLISTED_DOMAINS:
            if domain in url_lower:
                return False
        # Prefer common image formats
        return any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.webp']) or 'image' in url_lower
    
    try:
        # Filter out None values from API keys
        valid_keys = [key for key in GOOGLE_API_KEYS if key]
        
        print(f"DEBUG: GOOGLE_API_KEYS loaded: {len(valid_keys)} valid keys")
        print(f"DEBUG: GOOGLE_SEARCH_ENGINE_ID: {GOOGLE_SEARCH_ENGINE_ID}")
        
        if not valid_keys:
            print("ERROR: No valid Google API keys found")
            return "https://via.placeholder.com/400x300?text=Recipe+Image"
        
        if not GOOGLE_SEARCH_ENGINE_ID:
            print("ERROR: Google Search Engine ID not found")
            return "https://via.placeholder.com/400x300?text=Recipe+Image"
        
        # Use a random API key from the list
        api_key = random.choice(valid_keys)
        
        # Make request to Google Custom Search API
        params = {
            'key': api_key,
            'cx': GOOGLE_SEARCH_ENGINE_ID,
            'q': f"{recipe_name} recipe food",
            'searchType': 'image',
            'num': 10,  # Get 10 results
            'imgSize': 'large',
            'safe': 'active'
        }
        
        print(f"Fetching image for: {recipe_name}")
        print(f"Request URL: {GOOGLE_SEARCH_URL}")
        response = requests.get(GOOGLE_SEARCH_URL, params=params, timeout=10)
        print(f"Response status: {response.status_code}")
        
        response.raise_for_status()
        data = response.json()
        
        print(f"Response keys: {data.keys()}")
        
        # Get valid images from the results
        if 'items' in data and len(data['items']) > 0:
            # Filter valid image URLs
            valid_images = [item['link'] for item in data['items'] if is_valid_image_url(item['link'])]
            
            if valid_images:
                image_url = random.choice(valid_images)
                print(f"SUCCESS: Found image: {image_url}")
                return image_url
            else:
                print("WARNING: No valid images after filtering")
                return "https://via.placeholder.com/400x300?text=Recipe+Image"
        else:
            print(f"WARNING: No images found. Response: {data}")
            return "https://via.placeholder.com/400x300?text=Recipe+Image"
    except requests.exceptions.RequestException as e:
        print(f"REQUEST ERROR: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response text: {e.response.text}")
        return "https://via.placeholder.com/400x300?text=Recipe+Image"
    except Exception as e:
        print(f"GENERAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return "https://via.placeholder.com/400x300?text=Recipe+Image"

def get_session_id():
    """Get or create session ID for the current user"""
    if 'user_id' not in session:
        session['user_id'] = secrets.token_hex(16)
    return session['user_id']

@app.route('/')
def index():
    # Initialize session for user
    get_session_id()
    return render_template('index.html')

@app.route('/api/deserts')
def get_deserts():
    user_id = get_session_id()
    # Fetch only recipes belonging to the current user
    deserts = list(db.Desert.find({'user_id': user_id}))
    # Convert ObjectId to string for JSON serialization
    for desert in deserts:
        desert['_id'] = str(desert['_id'])
    return jsonify(deserts)

@app.route('/add-recipe', methods=['GET', 'POST'])
def add_recipe():
    user_id = get_session_id()
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        img_url = request.form.get('img_url')
        ingredients = request.form.get('ingredients').split('\n')
        recipe = request.form.get('recipe')
        
        # Create new recipe document with user_id
        new_recipe = {
            'name': name,
            'img': img_url,
            'ingredients': [ing.strip() for ing in ingredients if ing.strip()],
            'recipe': recipe,
            'user_id': user_id,
            'created_at': datetime.utcnow()
        }
        
        # Insert into MongoDB
        db.Desert.insert_one(new_recipe)
        
        return redirect(url_for('index'))
    
    return render_template('add_recipe.html')

@app.route('/edit-recipe/<recipe_id>', methods=['GET', 'POST'])
def edit_recipe(recipe_id):
    user_id = get_session_id()
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        img_url = request.form.get('img_url')
        ingredients = request.form.get('ingredients').split('\n')
        recipe = request.form.get('recipe')
        
        # Update recipe document only if it belongs to the current user
        db.Desert.update_one(
            {'_id': ObjectId(recipe_id), 'user_id': user_id},
            {
                '$set': {
                    'name': name,
                    'img': img_url,
                    'ingredients': [ing.strip() for ing in ingredients if ing.strip()],
                    'recipe': recipe,
                    'updated_at': datetime.utcnow()
                }
            }
        )
        
        return redirect(url_for('index'))
    
    # Get the recipe for editing, only if it belongs to the current user
    recipe = db.Desert.find_one({'_id': ObjectId(recipe_id), 'user_id': user_id})
    if recipe:
        recipe['_id'] = str(recipe['_id'])
        return render_template('edit_recipe.html', recipe=recipe)
    return redirect(url_for('index'))

@app.route('/delete-recipe/<recipe_id>', methods=['POST'])
def delete_recipe(recipe_id):
    user_id = get_session_id()
    # Delete only if recipe belongs to the current user
    db.Desert.delete_one({'_id': ObjectId(recipe_id), 'user_id': user_id})
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
        
        # Read the image for visualization
        img = cv2.imread(temp_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Run Roboflow detection
        print("Running Roboflow detection...")
        result = CLIENT.infer(temp_path, model_id="vegetables-el4g6/1")
        
        # Process results and draw boxes
        ingredients = []
        if 'predictions' in result:
            for prediction in result['predictions']:
                class_name = prediction['class'].lower()
                confidence = prediction['confidence']
                
                # Get bounding box coordinates
                x = int(prediction['x'])
                y = int(prediction['y'])
                width = int(prediction['width'])
                height = int(prediction['height'])
                
                # Calculate box coordinates
                x1 = x - width // 2
                y1 = y - height // 2
                x2 = x + width // 2
                y2 = y + height // 2
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Prepare label text
                label = f"{class_name}: {confidence:.2f}"
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw label background
                cv2.rectangle(img, (x1, y1 - text_height - 4), (x1 + text_width, y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                ingredients.append({
                    'name': class_name,
                    'confidence': confidence
                })
        
        # Save the annotated image
        annotated_path = 'annotated_image.jpg'
        cv2.imwrite(annotated_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Convert annotated image to base64
        with open(annotated_path, 'rb') as img_file:
            annotated_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up temp files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(annotated_path):
            os.remove(annotated_path)
        
        print(f"Detection complete. Found {len(ingredients)} ingredients")
        return jsonify({
            'ingredients': ingredients,
            'annotated_image': annotated_image_base64
        })
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        # Clean up temp files if they exist
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        if 'annotated_path' in locals() and os.path.exists(annotated_path):
            os.remove(annotated_path)
        return jsonify({'error': error_msg}), 500

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    prompt = data.get('prompt', '')
    chat_history = data.get('history', [])  # Get conversation history
    
    try:
        # Build messages with context
        messages = [
            {"role": "system", "content": """You are an expert professional chef and culinary advisor with years of experience. 
You help users with:
- Cooking techniques and tips
- Recipe suggestions and modifications
- Ingredient substitutions
- Kitchen equipment advice
- Food safety and storage
- Dietary accommodations

Provide detailed, practical, and friendly advice. When discussing recipes, be specific about measurements, temperatures, and timing."""}
        ]
        
        # Add conversation history (limit to last 10 messages for context)
        for msg in chat_history[-10:]:
            messages.append(msg)
        
        # Add current user prompt
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000
            }
        )
        response.raise_for_status()
        result = response.json()
        return jsonify({
            "response": result['choices'][0]['message']['content']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-recipe-from-detect', methods=['POST'])
def generate_recipe_from_detect():
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
        
        # Generate recipe using DeepSeek API
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": """You are a creative professional chef who creates delicious recipes based on available ingredients.
Your recipes should be:
- Practical and easy to follow
- Include specific measurements and cooking times
- Use proper cooking terminology
- Include helpful tips and techniques
- Consider flavor combinations and textures"""},
                    {"role": "user", "content": f"""Based on the following detected ingredients, create a complete recipe:

{ingredients_text}

Please:
1. Suggest a creative dish name
2. List all ingredients with measurements (use high-confidence ingredients as the base)
3. Provide step-by-step cooking instructions with temperatures and timing
4. Include any additional common pantry items needed
5. Add serving suggestions and optional garnishes
6. Mention possible substitutions for low-confidence ingredients

Format your response in a clear, cookbook-style format."""}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
        )
        response.raise_for_status()
        result = response.json()
        
        return jsonify({'recipe': result['choices'][0]['message']['content']})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-recipe-from-name', methods=['POST'])
def generate_recipe_from_name():
    try:
        data = request.get_json()
        dish_name = data.get('dish_name')
        
        if not dish_name:
            return jsonify({'error': 'Dish name is required'}), 400
        
        # Generate recipe using DeepSeek API
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": """You are a professional chef who creates detailed, structured recipes. 
You MUST respond with ONLY a valid JSON object, no additional text before or after.
Your recipes should be authentic, practical, and include proper measurements and techniques."""},
                    {"role": "user", "content": f"""Create a detailed recipe for {dish_name}. 

Respond with ONLY a JSON object in this EXACT format (no other text):

{{
    "name": "Proper name for the {dish_name}",
    "ingredients": [
        "1 cup all-purpose flour",
        "2 large eggs",
        "1/2 teaspoon salt"
    ],
    "recipe": "Step 1: Preheat oven to 350°F (175°C).\\nStep 2: Mix dry ingredients in a large bowl.\\nStep 3: Continue with detailed instructions..."
}}

Requirements:
- Use specific measurements (cups, tablespoons, grams, etc.)
- Include temperatures and cooking times
- Write clear, numbered step-by-step instructions
- Add helpful cooking tips in the instructions
- Each ingredient must be a complete string with quantity and name
- Separate steps with \\n for line breaks"""}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract the JSON part from the response
        response_text = result['choices'][0]['message']['content'].strip()
        # Find the first { and last } to extract the JSON
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("Invalid response format from AI")
        
        json_str = response_text[start:end]
        recipe_data = json.loads(json_str)
        
        # Convert ingredients to strings if they're objects
        ingredients = []
        for ingredient in recipe_data.get('ingredients', []):
            if isinstance(ingredient, dict):
                # If it's an object, try to format it as a string
                quantity = ingredient.get('quantity', '')
                name = ingredient.get('name', '')
                ingredients.append(f"{quantity} {name}".strip())
            else:
                # If it's already a string, use it as is
                ingredients.append(str(ingredient))
        
        # Fetch image from Google
        image_url = fetch_recipe_image(recipe_data.get('name', dish_name))
        
        return jsonify({
            'name': recipe_data.get('name', dish_name),
            'ingredients': ingredients,
            'recipe': recipe_data.get('recipe', ''),
            'image_url': image_url
        })
    
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Failed to parse AI response: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)
