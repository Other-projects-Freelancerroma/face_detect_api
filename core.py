from flask import Flask, request, jsonify
import os
from deepface import DeepFace
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
REFERENCE_DIR = './references'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)

RECOGNITION_THRESHOLD = 0.4


@app.route('/recognize', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({"error": "Изображение не передано"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Файл не выбран"}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    try:
        DeepFace.extract_faces(
            img_path=input_path,
            detector_backend="opencv",
            enforce_detection=True,
            align=False
        )

        reference_files = [
            f for f in os.listdir(REFERENCE_DIR)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not reference_files:
            os.remove(input_path)
            return jsonify({
                "status": "error",
                "message": "Нет референсных фотографий для сравнения"
            })

        found_person = None
        best_distance = float('inf')
        best_match_name = None

        for ref_file in reference_files:
            ref_path = os.path.join(REFERENCE_DIR, ref_file)

            try:
                result = DeepFace.verify(
                    img1_path=input_path,
                    img2_path=ref_path,
                    model_name="VGG-Face",
                    distance_metric="cosine",
                    detector_backend="opencv",
                    align=True,
                    enforce_detection=False,
                )

                distance = result['distance']
                person_name = os.path.splitext(ref_file)[0]

                if distance < RECOGNITION_THRESHOLD:
                    found_person = person_name
                    best_distance = distance
                    break

                if distance < best_distance:
                    best_distance = distance
                    best_match_name = person_name

            except Exception:
                continue

        os.remove(input_path)

        if found_person:
            return jsonify({
                "status": "recognized",
                "person": found_person,
                "distance": round(best_distance, 4)
            })
        else:
            return jsonify({
                "status": "not_recognized",
                "message": "Лицо обнаружено, но совпадений не найдено",
                "best_match": best_match_name,
                "best_distance": (
                    round(best_distance, 4) if best_match_name else None
                ),
            })

    except ValueError as ve:
        if "Face could not be detected" in str(ve):
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({
                "status": "no_face",
                "message": "На изображении не обнаружено лицо"
            })
        else:
            raise

    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        return jsonify({
            "status": "error",
            "message": f"Ошибка обработки: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
