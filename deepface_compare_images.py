from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import tempfile
import pandas as pd

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def compare_faces():
    if 'image' not in request.files:
        return jsonify({'error': 'image is required'}), 400

    image_file = request.files['image']

    # Save the uploaded images to temporary files
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        image_path = temp_image.name
        image_file.save(image_path)

    try:
        
        results = DeepFace.find(
            img_path=image_path,
            model_name="Facenet512",
            distance_metric="euclidean_l2",
            detector_backend="retinaface",
            align=True,
            db_path="./uploads"
        )
        
        # Initialize a list to store results
        all_results = []

        # Check if results is a list of DataFrames
        if isinstance(results, list) and all(isinstance(df, pd.DataFrame) for df in results):
            for df in results:
                # Convert each DataFrame to a list of dictionaries
                results_list = df.to_dict(orient='records')
                all_results.extend(results_list)  # Combine all results
        else:
            # Handle cases where results might not be a list of DataFrames
            return jsonify({'error': 'Unexpected format of results'}), 500
         
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files
        os.remove(image_path)

    return jsonify({'result': all_results})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
