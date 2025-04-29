from flask import Flask, render_template, request, send_file
import requests

app = Flask(__name__)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Crowd Counting Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 20px;
        }
        img {
            max-width: 80%;
            height: auto;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Crowd Counting</h1>
    <form action="/upload" method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br><br>
        <button type="submit">Upload Image</button>
    </form>
   
    <h2>Processed Image</h2>
    <img src="{{ image_url }}" alt="Processed Image">
    
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/upload', methods=['POST'])
def upload_image():
    # Get file from form
    file = request.files['file']
    if not file:
        return "No file selected!", 400

    # Send file to the API server
    url = 'http://127.0.0.1:5000/upload'  # Replace with your API server URL
    files = {'file': (file.filename, file.read(), file.content_type)}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        # Save the returned processed image locally
        output_file = 'processed_image.jpg'
        with open(output_file, 'wb') as f:
            f.write(response.content)

        # Display processed image
        return send_file(output_file, mimetype='image/jpeg')

    else:
        return f"Error: {response.text}", response.status_code

if __name__ == '__main__':
    app.run(debug=True ,  host= '127.0.0.1', port=8000)

