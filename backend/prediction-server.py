from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
from preprocessimage import hair_removal, image_filtering, normalize_image
import numpy as np
loaded_model = tf.keras.models.load_model(
    'D:\Desktop\College files\Final Year Project\FINAL PROJECT\FINAL PROJECT\CODE\PYTHON\MODELS\skin_cancer_hybrid.h5')

app = Flask(__name__)
cors = CORS(app)


def top_3_numbers(arr):
    arr = arr[0]
    sorted_arr = sorted(enumerate(arr), key=lambda x: x[1], reverse=True)[:3]
    return [(idx, val) for idx, val in sorted_arr]


def get_prediction(file_name):
    try:
        cv2_img = cv2.imread(file_name)
        remove_hair = hair_removal(cv2_img)
        image_normalization = normalize_image(remove_hair)
        filter_image = image_filtering(image_normalization)
        preprocessed_file_name = "pre" + file_name
        cv2.imwrite(preprocessed_file_name, filter_image)
        img = image.load_img(preprocessed_file_name, target_size=(224, 224))
        test_image = image.img_to_array(img)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0
        predictions = loaded_model.predict(test_image)

        if os.path.exists(file_name):
            os.remove(file_name)
        if os.path.exists(preprocessed_file_name):
            os.remove(preprocessed_file_name)

        print('File deleted.')

        top_three = top_3_numbers(predictions[0])

        return (top_three)
    except Exception as e:
        print(e)


@app.route('/predict_cancer', methods=['POST'])
def myendpoint():
    try:
        data = request.form
        print(data)
        file = request.files['file']
        file.save(file.filename)
        predictions = get_prediction(file.filename)

        disease = []
        confidence = []

        for i in predictions:
            print(i)
            if(i[0] == 0):
                disease.append("Actinic keratoses")
                confidence.append(i[1] * 100)
                # return jsonify({"status": "true", "cancer_type": "Actinic keratoses"})
            elif(i[0] == 1):
                disease.append("Basal cell carcinoma")
                confidence.append(i[1] * 100)
                # return jsonify({"status": "true", "cancer_type": "Basal cell carcinoma"})
            elif(i[0] == 2):
                disease.append("Benign keratosis-like lesions")
                confidence.append(i[1] * 100)
                # return jsonify({"status": "true", "cancer_type": "Benign keratosis-like lesions"})
            elif(i[0] == 3):
                disease.append("Dermatofibroma")
                confidence.append(i[1] * 100)
                # return jsonify({"status": "true", "cancer_type": "Dermatofibroma"})
            elif(i[0] == 4):
                disease.append("Melanocytic nevi")
                confidence.append(i[1] * 100)
                # return jsonify({"status": "true", "cancer_type": "Melanocytic nevi"})
            elif(i[0] == 5):
                disease.append("Vascular lesions")
                confidence.append(i[1] * 100)
                # return jsonify({"status": "true", "cancer_type": "Vascular lesions"})
            elif(i[0] == 6):
                disease.append("Melanoma")
                confidence.append(i[1] * 100)
                # return jsonify({"status": "true", "cancer_type": "Melanoma"})

        return jsonify({"status": "true", "cancer_type": disease, "conf": confidence})

    except Exception as e:
        print(e)
        return jsonify({"status": "false"})


if __name__ == '__main__':
    app.run(debug=True)
