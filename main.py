import cv2
import multiprocessing as mp
from paddleocr import PaddleOCR
from skimage.metrics import structural_similarity as ssim
import time
from ultralytics import YOLO
import numpy as np
# Function to load frames from video
def load_frames(video_path, frame_queue,imgsz):
    cap = cv2.VideoCapture(video_path)
    fr = 0
    n = 3
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        elif fr%n == 0:
            
            frame_queue.put(frame)
        fr += 1
    print(fr)
    cap.release()
    frame_queue.put(None)  # Signal that video loading is done

# Function to run inference on frames]


def run_inference(frame_queue,model_l,imgsz,model_ocr):
    last = None
    num_pass = 0
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        h,w,c = frame.shape
        resized_frame = cv2.resize(frame,(imgsz,imgsz))
        r = model_l.predict(source = resized_frame,imgsz = imgsz,verbose = False, conf = 0.65,max_det = 1)
        for result in r:
            if len(result.boxes.xyxyn): 
                x1 = int(w*result.boxes.xyxyn[0][0].item())
                y1 = int(h*result.boxes.xyxyn[0][1].item())
                x2 = int(w*result.boxes.xyxyn[0][2].item())
                y2 = int(h*result.boxes.xyxyn[0][3].item())
                plate = frame[y1:y2,x1:x2]

                if not last == None:
                    if (x1 > last):
                         num_pass = 0
                last = x1
                if num_pass < 3:
                    ocr_r = model_ocr(plate)

                    num_pass += 1
# Dummy inference function for demonstration


def main(video_path,model_path_l,imgsz,model_path_ocr):
    model_l = YOLO(model_path_l)
    model_ocr = PaddleOCR(lang = 'en',show_log = False,enable_mlkdnn = True,rec_model_dir = '/home/kkabir/Desktop/sparseTry/ppocr4ret.onnx',det_model_dir = '/home/kkabir/Desktop/sparseTry/ppocr3det.onnx', use_onnx = True)
    frame_queue = mp.Queue(maxsize=1000)  # Adjust maxsize as needed
    # Create threads for loading frames and running inference
    loader_thread = mp.Process(target=load_frames, args=(video_path, frame_queue,imgsz))
    inference_thread = mp.Process(target=run_inference, args=(frame_queue,model_l,imgsz,model_ocr))
    t = time.time()
    # Start the threads
    loader_thread.start()
    inference_thread.start()
    # Wait for both threads to complete
    loader_thread.join()
    inference_thread.join()
    print(time.time()-t)


main('/home/kkabir/Desktop/sparseTry/resized_cars3.mp4','/home/kkabir/Desktop/sparseTry/models/256_w_metadata.onnx',256,'/home/kkabir/Desktop/sparseTry/models/256_OCR.onnx')
