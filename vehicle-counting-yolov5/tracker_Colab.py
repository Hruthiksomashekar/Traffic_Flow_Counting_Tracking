#################### Tracker #################################################
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


FILE = Path.cwd().resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Global counters and trackers
up_count = 0
down_count = 0
car_count = 0
truck_count = 0
bus_count = 0
tracker1 = []
tracker2 = []
dir_data = {}

# Function to determine movement direction based on Y-coordinates
def direction(id, y):
    global dir_data

    if id not in dir_data:
        dir_data[id] = y
    else:
        diff = dir_data[id] - y

        if diff < 0:
            return "South"
        else:
            return "North"

# Function to count objects based on direction and object class
def count_obj(box, w, h, id, direct, cls):
    global up_count, down_count, tracker1, tracker2, car_count, truck_count, bus_count
    cx, cy = (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2))

    # Only process objects that cross a certain line in the frame
    if cy <= int(h // 2):
        return

    if direct == "South":
        if cy > (h - 300):
            if id not in tracker1:
                down_count += 1
                tracker1.append(id)
                if cls == 2:  # Class 2 is Car
                    car_count += 1
                elif cls == 7:  # Class 7 is Truck
                    truck_count += 1
                elif cls == 5:  # Class 5 is Bus
                    bus_count += 1

    elif direct == "North":
        if cy < (h - 150):
            if id not in tracker2:
                up_count += 1
                tracker2.append(id)

def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok = \
        opt.output, opt.source, opt.weights, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    save_vid = True

    # Initialize DeepSort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # Half precision only supported on CUDA

    # Capture the input video resolution
    cap = cv2.VideoCapture(opt.source)  # Open the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the input video
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the input video

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to its own .txt file
    if not evaluate:
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make directory

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # Check image size

    # Set video writer to match input resolution
    fourcc = cv2.VideoWriter_fourcc(*opt.fourcc)  # Define codec for video writer
    vid_path, vid_writer = None, None

    try:
        # Main processing loop
        for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(LoadImages(source, img_size=imgsz, stride=stride)):
            t1 = time_sync()

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # Normalize
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t2 = time_sync()

            # Inference
            pred = model(img, augment=opt.augment, visualize=opt.visualize)
            t3 = time_sync()

            # Apply NMS (Non-Max Suppression)
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)

            # Process detections
            for i, det in enumerate(pred):
                im0 = im0s.copy()
                annotator = Annotator(im0, line_width=2, pil=not ascii)
                w, h = im0.shape[1],im0.shape[0]
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # Pass detections to DeepSort
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                    # Draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            # print(f"Img: {im0.shape}\n")
                            _dir =  direction(id,bboxes[1])
                            # Call count_obj to update counts of objects
                            count_obj(bboxes, frame_width, frame_height, id, "South", cls)

                            label = f'{id} {names[int(cls)]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(int(cls), True))

                im0 = annotator.result()
                # Right Lane Line
                cv2.line(im0,(500,h-300),(w,h-300),(0,0,255),thickness=3)

                thickness = 3 # font thickness
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 2.5
                    #cv2.putText(im0, "Outgoing Traffic:  "+str(up_count), (60, 150), font,
                    #   fontScale, (0,0,255), thickness, cv2.LINE_AA)

                cv2.putText(im0, "Incoming Traffic:  "+str(down_count), (700,150), font,
                fontScale, (0,255,0), thickness, cv2.LINE_AA)

                    # -- Uncomment the below lines to computer car and truck count --
                    # It is the count of both incoming and outgoing vehicles

                    #Objects
                cv2.putText(im0, "Cars:  "+str(car_count), (60, 250), font,
                1.5, (20,255,0), 3, cv2.LINE_AA)

                cv2.putText(im0, "Trucks:  "+str(truck_count), (60, 350), font,
                1.5, (20,255,0), 3, cv2.LINE_AA)

                cv2.putText(im0, "Busses:  "+str(bus_count), (60, 450), font,
                1.5, (20,255,0), 3, cv2.LINE_AA)

                # Show video
                #if show_vid:
                    #cv2_imshow(im0)

                # Define save_path to save the processed video
                save_path = str(Path(save_dir) / Path(path).name)

                # Save video results
                if save_vid:
                    if vid_path != save_path:  # New video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # Release previous video writer

                        # Set video writer to match input resolution
                        vid_writer = cv2.VideoWriter('final_output.mp4', fourcc, fps, (frame_width, frame_height))

                    vid_writer.write(im0)

    except KeyboardInterrupt:
        print("Process interrupted by user.")

    finally:
        # Ensure the video writer is released properly
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()
        print("Video writer released and process completed.")

    # Print results
    print(f'Results saved to {save_path}')
    print(f'Counts - Up: {up_count}, Down: {down_count}, Cars: {car_count}, Trucks: {truck_count}, Buses: {bus_count}')


if __name__ == '__main__':
    __author__ = '-'

    # Instead of using argparse, define opt variables directly
    class Opt:
      weights = ['yolov5s.pt']  # Your YOLO model path
      deep_sort_model = 'osnet_x0_25'  # Deep Sort model
      source = 'input.mp4'  # Input source
      output = 'inference/output'  # Output folder
      imgsz = [480, 640]  # Image size (height, width)
      conf_thres = 0.35  # Confidence threshold
      iou_thres = 0.5  # IOU threshold for NMS
      fourcc = 'XVID'  # Codec for video output
      device = 'cpu'  # CUDA device, use 'cpu' if no GPU
      show_vid = False  # Set to True if you want to show video
      save_vid = True  # Save video results
      save_txt = False  # Save MOT-compliant results in txt
      classes = None  # Filter classes
      agnostic_nms = False  # Class-agnostic NMS
      augment = False  # Augmented inference
      evaluate = False  # Evaluate mode
      config_deepsort = 'deep_sort/configs/deep_sort.yaml'  # DeepSort config
      half = False  # Use FP16 half precision inference
      visualize = False  # Visualize features
      max_det = 1000  # Maximum number of detections per image
      dnn = False  # Use OpenCV DNN for ONNX inference
      project = ROOT / 'runs/track'  # Save results to this project/name
      name = 'exp'  # Name of the current run
      exist_ok = True  # Allow existing project/name

    opt = Opt()

    # Run the detect function with the options defined in `opt`
    with torch.no_grad():
        detect(opt)
