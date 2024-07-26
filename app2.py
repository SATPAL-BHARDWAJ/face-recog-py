from flask import Flask, request, jsonify
import face_recognition
import io
from PIL import Image
import base64
import json

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def compare_faces():
    if 'users' not in request.form or 'new_image' not in request.files:
        return jsonify({'error': 'Users array and new_image are required'}), 400

    users = json.loads(request.form['users'])
    new_image_file = request.files['new_image']

    # Read the new image
    try:
        new_image = face_recognition.load_image_file(new_image_file)
    except Exception as e:
        return jsonify({'error': f'Failed to load new image: {str(e)}'}), 400
    
    new_image_encodings = face_recognition.face_encodings(new_image)

    if len(new_image_encodings) == 0:
        return jsonify({'error': 'No face found in the new image'}), 400

    new_image_encoding = new_image_encodings[0]

    # Iterate over users to find a match
    for user in users:
        user_id = user.get('id')
        user_encoding = user.get('photo')

        if user_id is None or user_encoding is None:
            continue

        user_encoding = json.loads(user_encoding)  # Assuming encoding is stored as JSON string
        if len(user_encoding) != 128:
            return jsonify({'error': f'User encoding length is not 128: {len(user_encoding)}'}), 400
                
        try:
            results = face_recognition.compare_faces([user_encoding], new_image_encoding)
            distance = face_recognition.face_distance([user_encoding], new_image_encoding)[0]
        except Exception as e:
            return jsonify({'error': f'Error comparing faces: {str(e)}'}), 400

        if results[0]:
            return jsonify({'match': True, 'user_id': user_id, 'distance': float(distance)})

    return jsonify({'match': False, 'encoding': new_image_encoding.tolist()}), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
