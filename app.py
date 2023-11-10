from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name)

# 모델 로드 (VGG16)
model = tf.keras.applications.VGG16(weights='imagenet')


# 모델 예측 함수
def predict_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((224, 224))  # 모델의 입력 크기에 맞게 이미지 크기 조정
    img = np.array(img)
    img = tf.keras.applications.vgg16.preprocess_input(img)  # 이미지 전처리
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가
    prediction = model.predict(img)
    return tf.keras.applications.vgg16.decode_predictions(prediction)


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="이미지를 업로드하세요")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="파일을 선택하세요")

        if file:
            image_data = file.read()
            prediction = predict_image(image_data)
            # 상위 3개 예측 결과 표시
            top3_predictions = prediction[0][:3]
            return render_template('index.html', prediction=top3_predictions)

    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
