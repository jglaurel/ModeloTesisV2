import os
# Forzar CPU en Render (evita errores CUDA)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from tensorflow.keras.models import Model
from keras.applications.resnet50 import preprocess_input
from keras.utils import img_to_array
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import matplotlib.cm as cm
import base64
import cv2
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

app = FastAPI()

# Cargar modelo
model = load_model("resnet50_v21.h5", compile=False)
IMG_SIZE = (224, 224)
class_labels = {0: "NORMAL", 1: "NEUMONIA"}

app = FastAPI()

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"mensaje": "API para predecir neumonía con ResNet50"}

# Cargar el modelo entrenado
model = load_model("resnet50_v21.h5", compile=False)
IMG_SIZE = (224, 224)
class_labels = {0: "NORMAL", 1: "NEUMONIA"}

@app.get("/")
def home():
    return {"mensaje": "API para diagnóstico con ResNet50"}

@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)[0][0]
    clase = 1 if pred >= 0.5 else 0

    return {
        "clase_predicha": class_labels[clase],
        "confianza": float(pred if clase == 1 else 1 - pred)
    }

#endpoint de gradcam
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    target_layer = model.get_layer(last_conv_layer_name)

    grad_model = Model(
        inputs=[model.inputs],
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    if tf.reduce_max(heatmap) != 0:
        heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy(), predictions

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Usamos la capa de resnet50 conv5_block3_out para Grad-CAM
    heatmap, predictions = make_gradcam_heatmap(img_array, model, "conv5_block3_out")

    # Superponer heatmap
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    # Convertir a base64
    _, buffer = cv2.imencode(".png", superimposed_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "mensaje": "Grad-CAM generado",
        "prediccion": float(predictions[0][0]),
        "imagen_gradcam": img_base64
    }
