
'''
import os
import easyocr
from fuzzywuzzy import fuzz
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import warnings
from werkzeug.utils import secure_filename

# Suppress the fuzzywuzzy warning about python-Levenshtein
warnings.filterwarnings("ignore", message="Using slow pure-python SequenceMatcher")

# Load bad words from a file
def load_bad_words(filepath="bad_words.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

BAD_WORDS = load_bad_words()

# Initialize OCR reader
reader = easyocr.Reader(['en', 'ar'])

# Define the model
class SafeImageClassifier(nn.Module):
    def __init__(self):
        super(SafeImageClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 classes: safe/unsafe

    def forward(self, x):
        return self.model(x)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize Flask app
app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize and load the model
model = SafeImageClassifier()
state_dict = torch.load("model\enhanced_model_checkpoint.pth", map_location="cpu")
model.model.load_state_dict(state_dict)
model.eval()

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# HTML template with improved feedback display
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
    <head>
        <title>Image Safety Analyzer</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            form { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .details { margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            pre { white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>Image Safety Analyzer</h1>
        <p>Upload multiple images to analyze them for inappropriate content.</p>
        
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="images" accept="image/*" multiple required>
            <button type="submit">Analyze Images</button>
        </form>
        
        {% if result %}
            <div class="result {{ 'success' if result.status == 'success' else 'error' }}">
                <h3>{{ result.message }}</h3>
                {% if result.details %}
                    <div class="details">
                        <h4>Analysis Details:</h4>
                        <pre>{{ result.details | safe }}</pre>
                    </div>
                {% endif %}
            </div>
        {% endif %}
    </body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_TEMPLATE, result=None)

@app.route("/upload", methods=["POST"])
def upload_images():
    if 'images' not in request.files:
        return render_template_string(
            INDEX_TEMPLATE, 
            result={"status": "error", "message": "No images provided"}
        )
    
    image_files = request.files.getlist('images')
    
    if not image_files:
        return render_template_string(
            INDEX_TEMPLATE, 
            result={"status": "error", "message": "No images uploaded"}
        )
    
    results = []
    for image_file in image_files:
        if not allowed_file(image_file.filename):
            results.append({"filename": image_file.filename, "status": "error", "message": "Invalid file type"})
            continue

        try:
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(UPLOAD_DIR, filename)
            image_file.save(image_path)

            result = analyze_image(image_path)
            
            ocr_result = result.get("ocr_result", {})
            image_result = result.get("image_result", {})

            has_bad_words = ocr_result.get("has_bad_words", False)
            is_unsafe_image = image_result.get("is_unsafe", False)
            
            result_status = "accepted" if not (has_bad_words or is_unsafe_image) else "rejected"
            results.append({
                "filename": filename,
                "status": result_status,
                "ocr_result": ocr_result,
                "image_result": image_result
            })

        except Exception as e:
            results.append({
                "filename": image_file.filename,
                "status": "error",
                "message": str(e)
            })

    return render_template_string(
        INDEX_TEMPLATE, 
        result={"status": "success", "message": f"Analysis complete for {len(image_files)} images.", "details": results}
    )

def analyze_image(image_path):
    try:
        # OCR
        results = reader.readtext(image_path)
        text = " ".join([res[1] for res in results])
        bad_words_found = [word for word in BAD_WORDS if fuzz.partial_ratio(word.lower(), text.lower()) > 80]

        # Image Safety
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        ocr_result = {
            "has_bad_words": bool(bad_words_found),
            "message": "✅ No bad words found." if not bad_words_found else f"🚫 Bad words detected: {', '.join(bad_words_found)}",
            "detected_words": bad_words_found,
            "full_text": text
        }

        image_result = {
            "is_unsafe": prediction == 1,
            "message": "✅ Image is visually clean." if prediction == 0 else "🚫 Unsafe image.",
            "confidence": output[0, prediction].item() * 100
        }

        return {"ocr_result": ocr_result, "image_result": image_result}

    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

# --- Run the app ---
if __name__ == "__main__":
    app.run(debug=True)


from flask import Flask, request, jsonify, render_template_string, send_from_directory
import os
from werkzeug.utils import secure_filename
import easyocr
from fuzzywuzzy import fuzz
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'ims/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize models
reader = easyocr.Reader(['en', 'ar'])
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Bad words list (same as before)
BAD_WORDS = [
    "fuck", "shit", "ass", "bitch", "idiot", "moron", "nigger", "cunt", "whore", "slut",
    "bastard", "dick", "pussy", "damn", "hell", "crap", "douche", "fag", "retard", "screw",
    "fuck you", "son of a bitch", "motherfucker", "asshole", "dumbass", "shithead", "cock", "wanker",
    "عرص", "كس", "طيز", "زب", "كلب", "عاهر", "قحبة", "كفر", "ملحد", "شرموطة", "عاهره",
    "منيوك", "منيوكة", "زبالة", "خول", "دعارة", "فاجر", "فاسق", "فاحشة", "ممحونة", "ممحون",
    "ابن الكلب", "ابن العاهرة", "ابن الشرموطة", "يا خول", "يا عاهر", "يا كلب", "يا زبالة"
]

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_text_and_check(image_path, bad_words, similarity_threshold=80):
    try:
        results = reader.readtext(image_path, detail=1)
        extracted_text = " ".join([res[1] for res in results])
        
        detected_bad_words = []
        for bad_word in bad_words:
            if fuzz.partial_ratio(bad_word.lower(), extracted_text.lower()) > similarity_threshold:
                detected_bad_words.append(bad_word)
        
        return {
            'text_detected': extracted_text,
            'bad_words_detected': detected_bad_words,
            'text_analysis_success': True
        }
    except Exception as e:
        return {
            'text_analysis_success': False,
            'error': str(e)
        }

def classify_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        with torch.no_grad():
            out = resnet(batch_t)
        
        # Load ImageNet classes
        imagenet_classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        classes = []
        try:
            import urllib.request
            with urllib.request.urlopen(imagenet_classes_url) as f:
                classes = [line.strip() for line in f.read().decode('utf-8').splitlines()]
        except:
            classes = [str(i) for i in range(1000)]
        
        # Get top predictions
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        top_results = [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
        
        # Check for inappropriate content
        inappropriate_classes = []
        for class_name, confidence in top_results:
            class_name_lower = class_name.lower()
            if any(bad_word in class_name_lower for bad_word in ['naked', 'underwear', 'bikini', 'gun', 'knife', 'weapon']) and confidence > 20:
                inappropriate_classes.append(f"{class_name} ({confidence:.1f}%)")
        
        return {
            'inappropriate': bool(inappropriate_classes),
            'inappropriate_classes': inappropriate_classes,
            'top_predictions': top_results
        }
    except Exception as e:
        return {'error': str(e)}

# HTML template (same as before)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>T-Shirt Content Moderation</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .results { margin-top: 20px; }
        .result-card { 
            border: 1px solid #ddd; 
            padding: 15px; 
            margin-bottom: 15px; 
            border-radius: 5px; 
            background: #f9f9f9;
        }
        .rejected { border-left: 5px solid #ff5252; }
        .accepted { border-left: 5px solid #4caf50; }
        .thumbnail { max-width: 200px; max-height: 200px; }
        .bad-word { color: #ff5252; font-weight: bold; }
        .compact { font-size: 14px; }
    </style>
</head>
<body>
    <h1>T-Shirt Content Moderation</h1>
    <form method="post" action="/" enctype="multipart/form-data">
        <input type="file" name="files" multiple>
        <input type="submit" value="Check Images">
    </form>
    
    <div class="results">
        {% for result in results %}
        <div class="result-card {{ 'rejected' if result.overall_status == 'rejected' else 'accepted' }}">
            <h3>{{ result.filename }}</h3>
            <img src="{{ result.image_url }}" class="thumbnail">
            <div class="compact">
                <p><strong>Status:</strong> 
                    <span class="{{ 'bad-word' if result.overall_status == 'rejected' }}">
                        {{ result.overall_status|upper }}
                    </span>
                </p>
                {% if result.text_analysis.bad_words_detected %}
                <p><strong>Bad Words:</strong> 
                    {% for word in result.text_analysis.bad_words_detected %}
                    <span class="bad-word">{{ word }}</span>{% if not loop.last %}, {% endif %}
                    {% endfor %}
                </p>
                {% endif %}
                <p><strong>Detected Text:</strong> {{ result.text_analysis.text_detected[:50] }}{% if result.text_analysis.text_detected|length > 50 %}...{% endif %}</p>
                {% if result.image_classification.inappropriate_classes %}
                <p><strong>Inappropriate Content:</strong> {{ result.image_classification.inappropriate_classes|join(', ') }}</p>
                {% endif %}
            </div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        results = []
        for file in request.files.getlist('files'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                text_results = detect_text_and_check(filepath, BAD_WORDS)
                image_results = classify_image(filepath)
                
                status = 'accepted'
                if text_results.get('bad_words_detected') or image_results.get('inappropriate'):
                    status = 'rejected'
                
                results.append({
                    'filename': filename,
                    'image_url': f"/uploads/{filename}",
                    'overall_status': status,
                    'text_analysis': text_results,
                    'image_classification': image_results
                })
        
        return render_template_string(HTML_TEMPLATE, results=results)
    
    return render_template_string(HTML_TEMPLATE, results=[])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
    
'''

from flask import Flask, request, jsonify
import easyocr
from fuzzywuzzy import fuzz
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# Initialize OCR reader
reader = easyocr.Reader(['en', 'ar'])

# Initialize image classification model
model = models.resnet50(pretrained=True)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Bad words list (expanded from your original list)
BAD_WORDS = [
    "fuck", "shit", "ass", "bitch", "idiot", "moron", "nigger", "cunt", "whore", "slut",
    "bastard", "dick", "pussy", "damn", "hell", "crap", "douche", "fag", "retard", "screw",
    "fuck you", "son of a bitch", "motherfucker", "asshole", "dumbass", "shithead", "cock", "wanker",
    "عرص", "كس", "طيز", "زب", "كلب", "عاهر", "قحبة", "كفر", "ملحد", "شرموطة", "عاهره",
    "منيوك", "منيوكة", "زبالة", "خول", "دعارة", "فاجر", "فاسق", "فاحشة", "ممحونة", "ممحون",
    "ابن الكلب", "ابن العاهرة", "ابن الشرموطة", "يا خول", "يا عاهر", "يا كلب", "يا زبالة"
]

def detect_text_and_check(image_data, bad_words, similarity_threshold=80):
    """Detects text in image and checks for bad words"""
    try:
        # Convert image data to numpy array for EasyOCR
        import numpy as np
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        results = reader.readtext(image_np, detail=1)
        extracted_text = " ".join([res[1] for res in results])

        detected_bad_words = [
            bad_word for bad_word in bad_words
            if fuzz.partial_ratio(bad_word.lower(), extracted_text.lower()) > similarity_threshold
        ]

        return {
            "success": True,
            "detected_bad_words": detected_bad_words,
            "has_bad_words": len(detected_bad_words) > 0
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def classify_image(image_data):
    """Classifies image content and checks for inappropriate content"""
    try:
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_tensor = transform(img)
        batch_tensor = torch.unsqueeze(img_tensor, 0)
        
        with torch.no_grad():
            output = model(batch_tensor)
        
        # Get probabilities and class indices
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, indices = torch.sort(output, descending=True)
        
        # Load ImageNet classes
        try:
            import urllib.request
            classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            with urllib.request.urlopen(classes_url) as f:
                classes = [line.strip() for line in f.read().decode('utf-8').splitlines()]
        except:
            # Fallback if unable to load classes
            classes = [str(i) for i in range(1000)]
        
        # Get top 5 predictions
        top_results = []
        for idx in indices[0][:5]:
            class_name = classes[idx]
            confidence = probabilities[idx].item() * 100
            top_results.append({"class": class_name, "confidence": confidence})
        
        # Check for inappropriate content keywords in top predictions
        inappropriate_keywords = ['naked', 'underwear', 'bikini', 'gun', 'weapon']
        inappropriate_classes = []
        
        for result in top_results:
            class_name = result["class"].lower()
            confidence = result["confidence"]
            if any(keyword in class_name for keyword in inappropriate_keywords) and confidence > 20:
                inappropriate_classes.append({
                    "class": result["class"],
                    "confidence": confidence
                })
        
        return {
            "success": True,
            "has_inappropriate_content": len(inappropriate_classes) > 0
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]

    # Validate image format
    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        return jsonify({"error": "Unsupported file format"}), 400

    # Read image data directly into memory
    image_data = image.read()
    
    # Perform text detection and check for bad words
    text_result = detect_text_and_check(image_data, BAD_WORDS)
    
    # Perform image classification
    image_result = classify_image(image_data)
    
    # Create simplified output format
    result = {
        "overall_status": "rejected" if (
            text_result.get("has_bad_words", False) or 
            image_result.get("has_inappropriate_content", False)
        ) else "accepted",
        "text_analysis_success": text_result.get("success", False)
    }
    
    # Only add bad_words_detected if there are any
    if text_result.get("has_bad_words", False):
        result["bad_words_detected"] = text_result.get("detected_bad_words", [])
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


