import cv2
import numpy as np
import openvino as ov
import sys

import threading
from queue import Queue, Empty

# thread Queue에 대한 접근 제어
input_lock = threading.Lock()
output_lock = threading.Lock()

# 실행 옵션 설정
if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("At least 1 argv needed : [device(CPU or GPU)] [video_path(default webcam 0)]")
    sys.exit(0)
    
if len(sys.argv) == 3:
    video_path = sys.argv[2]

select_device = sys.argv[1]

# Load the OpenVINO model
def load_model():
    model_xml_path = "model_yoloxS/openvino/openvino.xml"
    model_bin_path = "model_yoloxS/openvino/openvino.bin"
    
    core = ov.Core()
    model = core.read_model(model=model_xml_path, weights=model_bin_path)
    
    # static shape convert(GPU가속 동적모델 지원X)
    if select_device == 'GPU':
        model.reshape([1, 3, 416, 416]) 
        
    compiled_model = core.compile_model(model=model, device_name=select_device)

    return compiled_model

# webcam or video input
def init_camera(buf_size=3):
    global video_path
    if len(sys.argv) != 3:
        video_path = 1
        
    cap = cv2.VideoCapture(video_path)
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, buf_size)  # 버퍼 크기 설정
    #cap.set(cv2.CAP_PROP_POS_MSEC, timeout) # (ms) timeout
    if not cap.isOpened():
        print("Error with Camera")
        
    return cap

# Set the dimensions explicitly
def get_shape(compiled_model):
    input_layer = compiled_model.input(0)
    
    # dynamic shape
    if select_device == 'CPU':
        partial_shape = input_layer.get_partial_shape()
        _, _, H, W = partial_shape
        h, w = H.get_length(), W.get_length()
    # static shape
    else:
        _, _, h, w = input_layer.shape
    
    return h, w

# Preprocess the frame to match the input requirements of the model
def preprocess(frame, W, H):
    resized_frame = cv2.resize(frame, (W, H))
    input_frame = np.expand_dims(resized_frame.transpose(2, 0, 1), axis=0)

    return resized_frame, input_frame

# 사이즈 비율에 맞게 재조정
def adjust_ratio(frame, resized_frame):
    (real_y, real_x), (resized_y, resized_x) = frame.shape[:2], resized_frame.shape[:2]
    ratio_x, ratio_y = (real_x / resized_x), (real_y / resized_y)
    
    return ratio_x, ratio_y

# 인식박스 만들기
def creat_boxes(frame, x_min, y_min, x_max, y_max, text, color):
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# 사고 detect
def detect_accident(input_queue, output_queue):    
    # 레이블 정의
    labels = {
        "0": {"name": "fall", "color": (255, 0, 0)}, # 넘어진 사람
        "1": {"name": "moderate", "color": (0, 255, 0)}, # 경미한 사고
        "2": {"name": "severe", "color": (0, 0, 255)} # 중대한 사고
    }
    compiled_model = load_model()
    
    # Define the output layers
    output_layer_labels = compiled_model.output("labels")
    output_layer_boxes = compiled_model.output("boxes")
    
    H, W = get_shape(compiled_model)
    print("before thread loop")
    while True:
        with input_lock:
            # cam frame input
            frame = input_queue.get()
        
        # 종료
        if frame is None:
            break
    
        resized_frame, input_frame = preprocess(frame, W, H)
    
        # 화면비율을 통한 좌표수정
        ratio_x, ratio_y = adjust_ratio(frame, resized_frame)
        
        # Perform inference
        results = compiled_model([input_frame])[output_layer_boxes] # 좌표
        class_ids = compiled_model([input_frame])[output_layer_labels] # class id
        
        valid_detections = np.where(results[0][:, 4] > 0.8)[0]
        print("running thread")
        for i in valid_detections:
            print(f"xy: {results[0][i]}")
            print(f"class_id: {str(class_ids[0][i])}")
            """x_min, y_min, x_max, y_max, score = results[0][i] # 좌표, 신뢰도 추출
            class_id = str(class_ids[0][i])  # class id 추출

            #좌상단, 우하단 좌표
            x_min = int(max(x_min * ratio_x, 10))
            y_min = int(y_min * ratio_y)
            x_max = int(x_max * ratio_x)
            y_max = int(y_max * ratio_y)

            # 인식한 객체에 대한 레이블 정보 가져오기
            label_name = labels[class_id]["name"]
            color = labels[class_id]["color"]

            st = f"{label_name}  {score:.2f}"
            print(st)
            creat_boxes(frame, x_min, y_min, x_max, y_max, st, color)"""
        
        with output_lock:
            output_queue.put(('Webcam Object Detection', frame))
            print("output thread frame")
        


def main():
    fps = 15 # 영상 FPS 조절
    
    input_queue = Queue(maxsize=3) # 영상 input queue
    output_queue = Queue() # 후처리 영상 output queue
    
    detection_thread = threading.Thread(
        target=detect_accident, 
        args=(input_queue, output_queue))
    detection_thread.start()

    #time.sleep(0.1) # load_model time waiting
    cap = init_camera()

    while True:
        ret, frame = cap.read()
        if (not ret): #and (output_queue.empty() and input_queue.empty()):
            break
        
        if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
            break
        
        print("input frame")
        input_queue.put(frame, block=True, timeout=100)
        
        print("응애")
        # get frame from queue
        try:
            with output_lock:
                data = output_queue.get_nowait()
        except Empty:
            continue
            
        name, det_frame = data
        
        cv2.imshow(name, det_frame)
        
        output_queue.task_done()
        
    input_queue.put(None)
    detection_thread.join()
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()