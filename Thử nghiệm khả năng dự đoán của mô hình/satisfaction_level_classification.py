# -*- coding: UTF-8 -*-
import sys
from pathlib import Path
import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input

sys.path.append(str(Path(__file__).resolve().parent))

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import non_max_suppression_face, scale_coords
from utils.torch_utils import select_device

satisfaction_level = ['Neutral', 'Satisfied', 'Unsatisfied']

def load_models(yolo_weights, keras_model_path, device):
    yolo_model = attempt_load(yolo_weights, map_location=device)
    keras_model = load_model(keras_model_path)
    return yolo_model, keras_model

def preprocess_face(face_crop, target_size=(224, 224)):
    face_resized = cv2.resize(face_crop, target_size)
    face_input = preprocess_input(face_resized.astype(np.float32))
    return np.expand_dims(face_input, axis=0)

def detect_faces_and_satisfaction_level(yolo_model, keras_model, source_input, device):
    imgsz = 640
    conf_thres = 0.6
    iou_thres = 0.5

    webcam = source_input.isnumeric()
    dataset = LoadStreams(source_input, img_size=imgsz) if webcam else LoadImages(source_input, img_size=imgsz)

    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = yolo_model(img)[0]
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy() if webcam else im0s.copy()

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for d in det:
                    x1, y1, x2, y2 = map(int, d[:4])
                    face_crop = im0[y1:y2, x1:x2]
                    try:
                        face_input = preprocess_face(face_crop)
                        preds = keras_model.predict(face_input, verbose=0)
                        label = satisfaction_level[np.argmax(preds)]
                        prob = np.max(preds)
                        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(im0, f"{label} ({prob:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    except Exception as e:
                        print("Error:", e)
            else:
                cv2.putText(im0, "No face", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.namedWindow("Satisfaction level detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Satisfaction level detection", 900, 600)
            cv2.imshow("Satisfaction level detection", im0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == '__main__':
    yolo_weights = 'path'
    vgg19_model_path = 'path'
    source_input = '0' 

    device = select_device('')
    yolo_model, keras_model = load_models(yolo_weights, vgg19_model_path, device)
    detect_faces_and_satisfaction_level(yolo_model, keras_model, source_input, device)