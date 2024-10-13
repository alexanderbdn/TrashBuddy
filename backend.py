from flask import Flask, request, render_template
import os
import clip
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from PIL import Image

app = Flask(__name__)
# Directory where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a dictionary to map materials to recycling categories
material_to_category = {
    'plastic': 'recycle',
    'plastic bottle': 'recycle',
    'metal': 'recycle',
    'glass': 'recycle',
    'glass bottle': 'recycle',
    'paper': 'recycle',
    'foil': 'recycle',
    'metal food container': 'recycle',
    'plastic food container': 'recycle',
    'cardboard': 'recycle',
    'aluminum can': 'recycle',
    'tin can': 'recycle',
    'magazines': 'recycle',
    'newspaper': 'recycle',
    'plastic plate': 'recycle',
    'carton': 'recycle',
    'plastic bag': 'recycle',
    'organic waste': 'nature',
    'paper plate': 'nature',
    'food': 'nature',
    'wood': 'nature',
    'plants': 'nature',
    'fruits': 'nature',
    'banana': 'nature',
    'vegetables': 'nature',
    'coffee grounds': 'nature',
    'tea leaves': 'nature',
    'leaves': 'nature',
    'grass': 'nature',
    'compostable packaging': 'nature',
    'electronics': 'trash',
    'battery': 'trash',
    'light bulb': 'trash',
    'ceramic': 'trash',
    'diapers': 'trash',
    'styrofoam': 'trash',
    'tissue paper': 'trash',
    'disposable utensils': 'trash',
    'cigarette butts': 'trash',
    'rubber': 'trash',
    'latex gloves': 'trash',
    'medical waste': 'trash',
    'hazardous waste': 'trash',
    'broken glass': 'trash',
    'human': 'none',
    'animal': 'none'
}

# Initialize the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)  # Load the CLIP model

# List of materials to identify
materials = list(material_to_category.keys())

@app.route('/classify_image', methods=['GET', 'POST'])
def classify_waste():
    image_path = "./uploads/upload1.jpeg"
    # Load and preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Tokenize the text labels (materials)
    text = clip.tokenize(materials).to(device)

    # Forward pass through the model to get image and text features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Compute similarities between the image and text descriptions
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # Get the most likely material
    predicted_material_index = probs.argmax()
    predicted_material = materials[predicted_material_index]
    predicted_probability = probs[predicted_material_index]

    # Determine if it should go in recycle, trash, or nature
    disposal_category = material_to_category[predicted_material]

    # Print the result
    print(f"Predicted material: {predicted_material} ({predicted_probability:.2f})")
    print(f"Disposal category: {disposal_category}")
    
    return predicted_material, disposal_category


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save file to the uploads directory
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'upload1.jpeg')
            file.save(filepath)
            out_a, out_b = classify_waste() 
            #return f'The material is {out_a} and its recycling category is {out_b}'
            if out_b == "recycle":
                return render_template('recycling.html', material=out_a)
            elif out_b == "nature":
                return render_template('nature.html', material=out_a)
            elif out_b == "trash":
                return render_template('trash.html', material=out_a)
            elif out_b == "none":
                return render_template('none.html', material=out_a)
            

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=5500)
    
