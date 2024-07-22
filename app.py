from flask import Flask, request, jsonify
import face_recognition
import io
from PIL import Image

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Both image1 and image2 are required'}), 400

    image1_file = request.files['image1']
    image2_file = request.files['image2']

    image1 = face_recognition.load_image_file(io.BytesIO(image1_file.read()))
    image2 = face_recognition.load_image_file(io.BytesIO(image2_file.read()))

    image1_encodings = face_recognition.face_encodings(image1)
    image2_encodings = face_recognition.face_encodings(image2)

    if len(image1_encodings) == 0 or len(image2_encodings) == 0:
        return jsonify({'error': 'No faces found in one or both images'}), 400

    image1_encoding = image1_encodings[0]
    image2_encoding = image2_encodings[0]

    results = face_recognition.compare_faces([image1_encoding], image2_encoding)
    distance = face_recognition.face_distance([image1_encoding], image2_encoding)[0]

    return jsonify({'are_same': bool(results[0]), 'distance': float(distance)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000,debug=True)
