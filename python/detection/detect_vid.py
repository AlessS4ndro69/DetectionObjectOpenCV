import cv2
import time
import numpy as np
import os
import ffmpeg

import firebase_admin
from firebase_admin import firestore,  storage
from firebase_admin import credentials

from flask import Flask, jsonify, request

cred = credentials.Certificate('./service-account-file.json')
bucket_name = "video-search-1bb50.appspot.com"
pathVideo = ""


app = Flask(__name__)

@app.route('/api/progress', methods=['GET'])
def progress():
    return jsonify({'message': 'Processing video...'})

@app.route('/api/greet', methods=['GET'])
def greet():
    return jsonify({'message': 'Hello, welcome to the API!'})

def download_video_from_storage(bucket_name, video_name, local_path):
    
    print("descargando video para procesar")
    app = firebase_admin.initialize_app(cred,{"storageBucket": bucket_name})
    db = firestore.client()

    print(bucket_name)
    print(video_name)
    print(local_path)

    """Descargar video desde Google Cloud Storage"""
    # Crea el directorio si no existe
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    bucket = storage.bucket()
    print("llamando a blob")
    blob = bucket.blob(video_name)
    print("llamando a download_to_filename")
    try:
        blob.download_to_filename(filename=local_path)
    except Exception as e:
        print(f"Error al descargar el archivo: {e}")

    print("borrando instancia sdk firebase")

    firebase_admin.delete_app(app)


@app.route('/process_video', methods=['GET'])
def process_video():
     # Obtener nombres del bucket y del video desde los parámetros de la URL
    
    print("iniciando proceso...")
    bucket_name = request.args.get('bucket_name')
    video_name = request.args.get('video_name')
    

    if not bucket_name or not video_name:
        return jsonify({'error': 'Falta el nombre del bucket o del video en la URL'})

    # Ruta local para descargar el video
    local_video_path = f'./descargas/{video_name}'
    global pathVideo
    pathVideo = local_video_path

    try:
        # Descargar el video desde Google Cloud Storage
        download_video_from_storage(bucket_name, video_name, local_video_path)

        # Aquí puedes realizar el procesamiento del video utilizando OpenCV u otros métodos
        # Por ejemplo, podrías cargar el video con cv2.VideoCapture(local_video_path)

        # Placeholder para el resultado del procesamiento
        result = {'message': 'Procesamiento del video completado'}
    except Exception as e:
        return jsonify({'error': f'Error al descargar el video: {str(e)}'})


    

    return get_objects()



def add_register_to_firestore(collection, data):
    cred = credentials.Certificate('./service-account-file.json')
    app = firebase_admin.initialize_app(cred)
    db = firestore.client()

    coll_ref = db.collection(collection)
    docs = coll_ref.stream()

    # Agrega el documento a la colección
    update_time , doc_ref = coll_ref.add(data)

    # Imprime el ID del documento recién creado
    print(f'Documento agregado con ID: {doc_ref.id}')
    

# load the COCO class names
@app.route('/get_objects',methods=['GET'])
def get_objects():
    with open('../../input/object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')

    # get a different color array for each of the classes
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    # load the DNN model
    model = cv2.dnn.readNet(model='../../input/frozen_inference_graph.pb',
                            config='../../input/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                            framework='TensorFlow')

    # capture the video
    #cap = cv2.VideoCapture('../../input/1.mp4')
    global pathVideo
    cap = cv2.VideoCapture(pathVideo)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # get the video frames' width and height for proper saving of 
    print("fps: ",fps)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # create the `VideoWriter()` object
    out = cv2.VideoWriter('../../outputs/video_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))

    # detect objects in each frame of the video

    data = {'timestamps': {}}
    video_path = '../../input/video_1.mp4'
    
    frame_number = 0
    timetotal = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            image = frame
            image_height, image_width, _ = image.shape
            # create blob from image
            blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                                        swapRB=True)
            # start time to calculate FPS
            start = time.time()
            model.setInput(blob)
            output = model.forward()        
            # end time after detection
            end = time.time()
            # calculate the FPS for current frame detection
            fps = 1 / (end-start)
            #print(fps)

            #frame_timestamp = frame_timestamps[frame_number]

            # loop over each of the detections
            for detection in output[0, 0, :, :]:
                # extract the confidence of the detection
                confidence = detection[2]
                # draw bounding boxes only if the detection confidence is above...
                # ... a certain threshold, else skip 
                if confidence > .4:
                    # get the class id
                    class_id = detection[1]
                    # map the class id to the class 
                    class_name = class_names[int(class_id)-1]
                    color = COLORS[int(class_id)]
                    # get the bounding box coordinates
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    # get the bounding box width and height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    # draw a rectangle around each detected object
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                    # put the class name text on the detected object
                    cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    t = time.time()
                    seconds = t-timetotal
                    if class_name in data['timestamps']:
                        data['timestamps'][class_name].append(seconds)
                    else:
                        data['timestamps'][class_name] = [seconds]
                    
                    # put the FPS text on top of the frame
                    cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
                    
                    cv2.putText(image, f"time {(seconds):.2f} seconds", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
                    print("iterando")
            
            cv2.imshow('image', image)
            out.write(image)
            frame_number += 1

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    #cap.release()
    cv2.destroyAllWindows()
    print(data)
    add_register_to_firestore("coleccionprueba",data)

    return jsonify(data)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
