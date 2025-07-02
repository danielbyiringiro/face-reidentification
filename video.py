import os
import cv2
import random
import warnings
import argparse
import logging
import numpy as np
import time
from datetime import datetime
import pyttsx3
import threading

from ultralytics import YOLO
from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox

warnings.filterwarnings("ignore")

# TTS and metadata globals
tts_engine = None
person_metadata = {}
threat_states = {} # Track threat status of detected persons
announced_persons = set()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Face Recognition from Webcam")
    parser.add_argument(
        "--det-weight",
        type=str,
        default="./weights/det_10g.onnx",
        help="Path to detection model"
    )
    parser.add_argument(
        "--rec-weight",
        type=str,
        default="./weights/w600k_r50.onnx",
        help="Path to recognition model"
    )
    parser.add_argument(
        "--similarity-thresh",
        type=float,
        default=0.4,
        help="Similarity threshold between faces"
    )
    parser.add_argument(
        "--confidence-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for face detection"
    )
    parser.add_argument(
        "--faces-dir",
        type=str,
        default="./faces",
        help="Path to faces stored dir"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="./assets/in_video.mp4",
        help="Path to video you want to process"
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=0,
        help="Maximum number of face detections from a frame"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=10,
        help="Interval (in frames) to update performance metrics"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="./person_metadata.json",
        help="Path to person metadata JSON file"
    )
    parser.add_argument(
        "--announcement-cooldown",
        type=int,
        default=30,
        help="Cooldown period in seconds between announcements for the same person"
    )
    parser.add_argument(
        "--gun-det-weight",
        type=str,
        default="./best.pt",
        help="Path to gun detection model weight"
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def init_tts():
    """Initialize text-to-speech engine"""
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        # Set speech rate (optional)
        tts_engine.setProperty('rate', 150)
        # Set volume (optional)
        tts_engine.setProperty('volume', 0.8)
        logging.info("Text-to-speech engine initialized")
    except Exception as e:
        logging.error(f"Failed to initialize TTS engine: {e}")
        tts_engine = None


def speak_async(text):
    """Speak text asynchronously to avoid blocking the main thread"""
    def speak():
        if tts_engine:
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as e:
                logging.error(f"TTS error: {e}")
    
    if tts_engine:
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()


def check_bbox_intersection(person_bbox, gun_bbox, overlap_thresh=0.1):
    """
    Check if a person's bounding box intersects with a gun's bounding box.

    Args:
        person_bbox (list): Bounding box of the person [x1, y1, x2, y2].
        gun_bbox (list): Bounding box of the gun [x1, y1, x2, y2].
        overlap_thresh (float): The threshold for intersection over union (IoU).

    Returns:
        bool: True if the bounding boxes intersect significantly, False otherwise.
    """
    px1, py1, px2, py2 = person_bbox
    gx1, gy1, gx2, gy2 = gun_bbox

    # Calculate intersection area
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    # Calculate person's bounding box area
    person_area = (px2 - px1) * (py2 - py1)
    if person_area == 0:
        return False

    # Check if intersection is significant
    overlap = inter_area / person_area
    return overlap > overlap_thresh

def build_targets(detector, recognizer, params: argparse.Namespace):
    """
    Build targets using face detection and recognition.

    Args:
        detector (SCRFD): Face detector model.
        recognizer (ArcFace): Face recognizer model.
        params (argparse.Namespace): Command line arguments.

    Returns:
        List[Tuple[np.ndarray, str]]: A list of tuples containing feature vectors and corresponding image names.
    """
    targets = []
    logging.info(f"Loading face targets from {params.faces_dir}")
    
    if not os.path.exists(params.faces_dir):
        os.makedirs(params.faces_dir)
        logging.info(f"Created faces directory: {params.faces_dir}")
    
    for filename in os.listdir(params.faces_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        name = os.path.splitext(filename)[0].split('_')[0]  # Remove timestamp if present
        image_path = os.path.join(params.faces_dir, filename)

        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Could not read image {image_path}. Skipping...")
            continue
            
        bboxes, kpss = detector.detect(image, max_num=1)

        if len(kpss) == 0:
            logging.warning(f"No face detected in {image_path}. Skipping...")
            continue

        embedding = recognizer(image, kpss[0])
        targets.append((embedding, name))
        logging.info(f"Added target: {name}")

    logging.info(f"Loaded {len(targets)} face targets")
    return targets


def frame_processor(
    frame,
    detector,
    recognizer,
    gun_detector,
    targets,
    colors,
    params
):
    """
    Process a video frame for face detection and recognition.

    Args:
        frame (np.ndarray): The video frame.
        detector (SCRFD): Face detector model.
        recognizer (ArcFace): Face recognizer model.
        targets (List[Tuple[np.ndarray, str]]): List of target feature vectors and names.
        colors (dict): Dictionary of colors for drawing bounding boxes.
        params (argparse.Namespace): Command line arguments.

    Returns:
        Tuple[np.ndarray, int, np.ndarray, np.ndarray]: The processed video frame, number of faces detected, bboxes, and keypoints.
    """
    global face_labeling_mode, selected_face_bbox, selected_face_kps, threat_states
    
    start_time_proc = time.time()
    
    bboxes_det, kpss_det = detector.detect(frame, params.max_num)
    num_faces = len(bboxes_det)
    
    gun_results_frame = None
    if gun_detector:
        gun_results_frame = gun_detector(frame, verbose=False)

    for bbox_data, kps_data in zip(bboxes_det, kpss_det):
        face_bbox_coords = bbox_data[:4].astype(int)
        embedding = recognizer(frame, kps_data)

        max_similarity = 0.0
        best_match_name = "Unknown"
        for target_embedding, known_name in targets:
            similarity = compute_similarity(target_embedding, embedding)
            if similarity > max_similarity and similarity > params.similarity_thresh:
                max_similarity = similarity
                best_match_name = known_name

        if best_match_name != "Unknown" and best_match_name not in threat_states:
            threat_states[best_match_name] = {
                'is_threat': False,
                'last_gun_seen_time': 0,
                'weapon_in_view': False,
                'sitrep_announced': False
            }

        face_x1, face_y1, face_x2, face_y2 = face_bbox_coords
        face_h = face_y2 - face_y1
        face_w = face_x2 - face_x1
        person_body_box = [
            max(0, face_x1 - face_w), 
            face_y1, 
            min(frame.shape[1], face_x2 + face_w), 
            min(frame.shape[0], face_y2 + 3 * face_h)
        ]

        person_has_gun = False
        if gun_detector and gun_results_frame and gun_results_frame[0].boxes:
            for gun_box_data in gun_results_frame[0].boxes.data:
                gun_confidence = gun_box_data[4].item()
                if gun_confidence >= 0.80:
                    detected_gun_bbox = gun_box_data[:4].cpu().numpy().astype(int)
                    cv2.rectangle(frame, (detected_gun_bbox[0], detected_gun_bbox[1]), (detected_gun_bbox[2], detected_gun_bbox[3]), (0, 0, 255), 2)
                    cv2.putText(frame, f"Firearm ({gun_confidence:.2f})", (detected_gun_bbox[0], detected_gun_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    if check_bbox_intersection(person_body_box, detected_gun_bbox):
                        person_has_gun = True
        
        if best_match_name != "Unknown":
            current_time = time.time()
            state = threat_states[best_match_name]

            if person_has_gun:
                # --- WEAPON IS VISIBLE ---
                if state['is_threat'] and not state['weapon_in_view']:
                    # This is the TRANSITION from NOT IN VIEW -> IN VIEW
                    speak_async("Threat's weapon is back in view.")
                
                # Update state for weapon being visible
                state['is_threat'] = True
                state['weapon_in_view'] = True
                state['last_gun_seen_time'] = current_time

                # Announce the initial threat details if not already done
                if not state['sitrep_announced']:
                    announcement = f"Threat identified as {best_match_name}."
                    speak_async(announcement)
                    state['sitrep_announced'] = True
            
            else: # person_has_gun is False
                # --- WEAPON IS NOT VISIBLE ---
                # Check if the person is a known threat and the weapon was previously in view
                if state['is_threat'] and state['weapon_in_view']:
                    # This is the TRANSITION from IN VIEW -> NOT IN VIEW
                    # Check if enough time has passed since it was last seen
                    if current_time - state['last_gun_seen_time'] > 5.0:
                        speak_async("Threat's weapon not in view.")
                        # Update the state to reflect the weapon is no longer in view
                        state['weapon_in_view'] = False
        
        
        display_name_on_box = best_match_name
        box_color = colors.get(best_match_name, (0, 255, 0))

        if best_match_name != "Unknown":
            if threat_states[best_match_name]['is_threat']:
                box_color = (0, 0, 255)
            draw_bbox_info(frame, face_bbox_coords, similarity=max_similarity, name=display_name_on_box, color=box_color)
        else: # Unknown person
            box_color = (0, 0, 255) if person_has_gun else (255, 0, 0)
            draw_bbox_info(frame, face_bbox_coords, similarity=0, name="Unknown", color=box_color)
    
    process_time_val = time.time() - start_time_proc
    return frame, num_faces, process_time_val, bboxes_det, kpss_det


def main():
    global face_labeling_mode, selected_face_bbox, selected_face_kps, current_frame
    
    params = parse_args()
    setup_logging(params.log_level)

    # Initialize TTS and load metadata
    init_tts()

    logging.info("Initializing models...")
    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    recognizer = ArcFace(params.rec_weight)
    # Initialize Gun Detector (YOLO)
    logging.info("Initializing Gun Detector (YOLO)...")
    try:
        gun_detector = YOLO(params.gun_det_weight)  # Load the YOLO model for gun detection
        logging.info("Gun Detector initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Gun Detector: {e}")
        gun_detector = None # Ensure gun_detector is defined even if loading fails
    logging.info("Models initialized successfully")

    targets = build_targets(detector, recognizer, params)
    colors = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _, name in targets}

    logging.info(f"Opening video (filepath: {params.video_path}) ...")
    cap = cv2.VideoCapture(params.video_path)

    # print(f"Video open: {cap.isOpened()}")
    
    if not cap.isOpened():
        logging.error(f"Could not access video with path {params.video_path}")
        raise Exception(f"Could not access video with path {params.video_path}")

    # Try to set a higher resolution for the webcam

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    outptput_filename = f'output/file_{time.time()}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outptput_filename, fourcc, fps, (width, height))
    
    logging.info(f"Video opened: {width}x{height} at {fps} FPS")
    logging.info(f"Output filename: {outptput_filename}")
    
    # Performance tracking variables
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    processing_times = []
    face_counts = []
    
    while True:
        # print("DEBUG: Attempting cap.read()", flush=True) # Commented out for now
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.error("Failed to grab frame from video after cap.read()")
            # print("DEBUG: cap.read() failed or returned empty frame.", flush=True) # Commented out for now
            break

        current_frame = frame.copy()
        processed_frame, num_faces, process_time, bboxes_fp, kpss_fp = frame_processor(
            frame, detector, recognizer, gun_detector, targets, colors, params
        )
        
        # Track performance metrics
        processing_times.append(process_time)
        face_counts.append(num_faces)
        
        # Update performance metrics
        frame_count += 1
        if frame_count % params.update_interval == 0:
            elapsed = time.time() - start_time
            fps_display = params.update_interval / elapsed if elapsed > 0 else 0 # Avoid division by zero
            
            # Reset timing
            start_time = time.time()
            
            # Calculate average processing time and face count
            if processing_times:
                avg_process_time = sum(processing_times) / len(processing_times)
                avg_faces = sum(face_counts) / len(face_counts)
                
                # Reset lists to avoid memory growth
                processing_times = []
                face_counts = []
        
        # Add performance overlay
        cv2.putText(
            processed_frame,
            f"FPS: {fps_display:.1f} | Faces: {num_faces} | Process: {process_time*1000:.1f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        out.write(processed_frame)
    logging.info("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Done")


if __name__ == "__main__":
    main()