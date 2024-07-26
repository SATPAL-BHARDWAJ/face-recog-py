from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import tempfile

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Both image1 and image2 are required'}), 400

    image1_file = request.files['image1']
    image2_file = request.files['image2']

    # Save the uploaded images to temporary files
    with tempfile.NamedTemporaryFile(delete=False) as temp_image1, \
         tempfile.NamedTemporaryFile(delete=False) as temp_image2:
        image1_path = temp_image1.name
        image2_path = temp_image2.name
        image1_file.save(image1_path)
        image2_file.save(image2_path)

    try:
        result = DeepFace.verify(
            img1_path=image1_path,
            img2_path=image2_path,
            model_name="Facenet512",
            distance_metric="euclidean_l2",
            detector_backend="retinaface",
            align=False
        ) 
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files
        os.remove(image1_path)
        os.remove(image2_path)

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
