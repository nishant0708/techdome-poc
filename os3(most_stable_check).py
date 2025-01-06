import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import threading
import queue
import time
from collections import defaultdict

# Deep learning libraries
from ultralytics import YOLO
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F

class PersonReIDExtractor:
    def __init__(self, device=None):
        """
        Initialize person Re-Identification feature extractor using ResNet50
        
        :param device: Computation device (cuda/cpu)
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet50 model
        self.model = resnet50(pretrained=True)
        
        # Remove the last classification layer to get feature extraction
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image):
        """
        Extract feature vector for a person image
        
        :param image: Numpy array representing person image
        :return: Feature vector
        """
        try:
            # Preprocess image
            pil_image = self.transform(image)
            
            # Add batch dimension and move to device
            input_tensor = pil_image.unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
                
                # Normalize features
                features = F.normalize(features.squeeze(), p=2, dim=0)
            
            return features.cpu().numpy()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
 
class CrossCameraTrackingSystem:
    def __init__(self, camera_sources, common_areas=None):
        """
        Initialize advanced multi-camera tracking system
        
        :param camera_sources: List of camera sources
        :param common_areas: Dictionary of common area polygons for each camera
        """
        # Object detection model
        self.detection_model = YOLO("yolov8l.pt",task='detect')
        
        # Camera sources and common areas
        self.camera_sources = camera_sources
        print(self.camera_sources,":checking")
        self.common_areas = common_areas or {}
        
        # Feature extractor
        self.feature_extractor = PersonReIDExtractor()
        
        # Dynamic tracking data
        self.entrance_queues = {f"cam_{i}": queue.Queue() for i in range(len(camera_sources))}
        self.camera_entry_log = {f"cam_{i}": set() for i in range(len(camera_sources))}
        
        # Global tracking data
        self.global_tracks = {}
        
        # Tracking parameters
        self.similarity_threshold = 0.8
        self.track_timeout = 300  # 5 minutes timeout for tracks
        
        # Synchronization
        self.tracking_lock = threading.Lock()
    
    def cosine_similarity(self, feat1, feat2):
        """
        Calculate cosine similarity between two feature vectors
        
        :param feat1: First feature vector
        :param feat2: Second feature vector
        :return: Cosine similarity score
        """
        # Ensure features are 1D numpy arrays
        feat1 = feat1.flatten()
        feat2 = feat2.flatten()
        
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        return np.dot(feat1, feat2) / (norm1 * norm2)
    
    
    
    def is_in_common_area(self, bbox, camera_id):
        """
        Check if a bounding box is in the common area for a specific camera
        
        :param bbox: Bounding box coordinates [x, y, w, h]
        :param camera_id: Camera identifier
        :return: Boolean indicating if bbox is in common area
        """
        if not self.common_areas or camera_id not in self.common_areas:
            return False
        
        # Convert bbox to center point
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Use OpenCV's pointPolygonTest to check if point is in polygon
        return cv2.pointPolygonTest(
            self.common_areas[camera_id], 
            (center_x, center_y), 
            False
        ) >= 0
    
    def match_global_track(self, current_feature, current_camera):
        """
        Match current detection with global tracks
        
        :param current_feature: Feature vector of current detection
        :param current_camera: Current camera identifier
        :return: Matched track ID or None
        """
        best_match = None
        best_similarity = self.similarity_threshold
        
        with self.tracking_lock:
            # Clean up expired tracks
            current_time = time.time()
            self.global_tracks = {
                track_id: track_info for track_id, track_info 
                in self.global_tracks.items() 
                if current_time - track_info['last_seen_time'] < self.track_timeout
            }
            
            # Find best match in global tracks
            for track_id, track_info in self.global_tracks.items():
                similarity = self.cosine_similarity(current_feature, track_info['feature'])
                
                if similarity > best_similarity:
                    best_match = track_id
                    best_similarity = similarity
        
        return best_match

    def can_enter_camera(self, current_camera, track_id):
        """
        Check if a track can enter the current camera
        
        :param current_camera: Current camera identifier
        :param track_id: Track identifier
        :return: Boolean indicating if entry is allowed
        """
        # Always allow entry for the first camera
        camera_indices = [f"cam_{i}" for i in range(len(self.camera_sources))]
        current_index = camera_indices.index(current_camera)
        
        if current_index == 0:
            return True
        
        # Check if track exists in global tracks
        return track_id in self.global_tracks
    
    def update_global_track(self, track_id, feature, camera_id):
        """
        Update or create global track
        
        :param track_id: Track identifier
        :param feature: Feature vector
        :param camera_id: Current camera identifier
        """
        with self.tracking_lock:
            if track_id not in self.global_tracks:
                # Create new global track
                self.global_tracks[track_id] = {
                    'feature': feature,
                    'last_seen_camera': camera_id,
                    'last_seen_time': time.time()
                }
            else:
                # Update existing track
                self.global_tracks[track_id]['feature'] = feature
                self.global_tracks[track_id]['last_seen_camera'] = camera_id
                self.global_tracks[track_id]['last_seen_time'] = time.time()
    
        
    
    def handle_entrance_queue(self, camera_id, new_person_feature):
        """
        Handle entrance queue for cross-camera tracking
        
        :param camera_id: Camera identifier
        :param new_person_feature: Feature vector of new person
        :return: Matched track ID or None
        """
        # Check entrance queues from other cameras
        matched_track = None
        camera_indices = [f"cam_{i}" for i in range(len(self.camera_sources))]
        current_index = camera_indices.index(camera_id)
        
        # Check previous camera's entrance queue
        if current_index > 0:
            prev_camera = camera_indices[current_index - 1]
            
            # Convert queue to list to safely iterate
            queue_list = list(self.entrance_queues[prev_camera].queue)
            
            for queue_index, queue_item in enumerate(queue_list):
                similarity = self.cosine_similarity(
                    new_person_feature, 
                    queue_item['feature']
                )
                
                if similarity > self.similarity_threshold:
                    matched_track = queue_item['track_id']
                    
                    # Remove matched item from queue using index
                    del self.entrance_queues[prev_camera].queue[queue_index]
                    break
        
        return matched_track
    
    def add_to_entrance_queue(self, camera_id, person_info):
        """
        Add person to entrance queue
        
        :param camera_id: Camera identifier
        :param person_info: Dictionary with person tracking information
        """
        self.entrance_queues[camera_id].put(person_info)
        self.camera_entry_log[camera_id].add(person_info['track_id'])
        
        # Limit queue size to prevent memory issues
        if self.entrance_queues[camera_id].qsize() > 50:
            try:
                removed_item = self.entrance_queues[camera_id].get_nowait()
                self.camera_entry_log[camera_id].discard(removed_item['track_id'])
            except queue.Empty:
                pass
    
    def start_tracking(self):
                """
                Start tracking across multiple camera streams
                """
                threads = []
                for camera_id, camera_source in enumerate(self.camera_sources):
                    camera_name = f"cam_{camera_id}"
                    thread = threading.Thread(
                        target=self.process_camera, 
                        args=(camera_name, camera_source)
                    )
                    thread.start()
                    threads.append(thread)
                
                for thread in threads:
                    thread.join()
    


    def process_camera(self, camera_id, camera_source):
        """
        Process video stream from a single camera with robust person tracking
        
        :param camera_id: Unique camera identifier
        :param camera_source: Camera source
        """
        # Initialize tracking variables
        track_counter = 0
        active_tracks = {}
        
        # Track history and persistence parameters
        track_history = {}
        max_disappear_frames = 30  # Allow track to persist for 30 frames without detection
        print(camera_source,":checking11")
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_source}")
            return
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (640, 480))
            
            # Draw common area if defined
            if camera_id in self.common_areas:
                cv2.polylines(frame, [self.common_areas[camera_id]], True, (0, 0, 255), 2)
            
            # Detect persons using YOLO
            results = self.detection_model(frame, conf=0.25, iou=0.4)
            
            # Prepare for tracking
            current_detections = []
            person_images = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    # Filter for persons
                    if conf > 0.3 and cls == 0:
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        current_detections.append(bbox)
                        
                        # Try to extract person image for feature matching
                        try:
                            person_img = self.extract_person_image(frame, [x1, y1, x2, y2])
                            person_images.append(person_img)
                        except Exception as e:
                            print(f"Error extracting person image: {e}")
                            person_images.append(None)
            
            # Track and identify persons
            new_active_tracks = {}
            matched_detection_indices = set()
            
            # First, handle existing tracks
            for track_id, track_info in list(active_tracks.items()):
                # Check if track has persisted beyond max disappear frames
                track_history[track_id] = track_history.get(track_id, 0) + 1
                
                if track_history[track_id] > max_disappear_frames:
                    # Remove long-disappeared tracks
                    del active_tracks[track_id]
                    del track_history[track_id]
                    continue
                
                # Try to match with current detections
                best_match_idx = None
                best_similarity = self.similarity_threshold
                
                for idx, (bbox, person_img) in enumerate(zip(current_detections, person_images)):
                    if idx in matched_detection_indices:
                        continue
                    
                    if person_img is None or person_img.size == 0:
                        continue
                    
                    current_feature = self.feature_extractor.extract_features(person_img)
                    if current_feature is None:
                        continue
                    
                    # Compare with stored track feature
                    similarity = self.cosine_similarity(current_feature, track_info['feature'])
                    
                    if similarity > best_similarity:
                        best_match_idx = idx
                        best_similarity = similarity
                
                if best_match_idx is not None:
                    # Update track with new detection
                    x1, y1, w, h = current_detections[best_match_idx]
                    x2, y2 = x1 + w, y1 + h
                    
                    person_img = person_images[best_match_idx]
                    current_feature = self.feature_extractor.extract_features(person_img)
                    
                    new_track_info = {
                        'bbox': current_detections[best_match_idx],
                        'feature': current_feature,
                        'last_seen': camera_id
                    }
                    
                    new_active_tracks[track_id] = new_track_info
                    matched_detection_indices.add(best_match_idx)
                    track_history[track_id] = 0  # Reset disappearance counter
                    
                    # Update global track
                    self.update_global_track(track_id, current_feature, camera_id)
                    
                    # Draw bounding boxes and IDs
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                else:
                    # Keep track in memory even if not currently detected
                    new_active_tracks[track_id] = track_info
            
            # Handle unmatched detections
            for idx, (bbox, person_img) in enumerate(zip(current_detections, person_images)):
                if idx in matched_detection_indices:
                    continue
                
                if person_img is None or person_img.size == 0:
                    continue
                
                current_feature = self.feature_extractor.extract_features(person_img)
                if current_feature is None:
                    continue
                
                # Determine track ID
                if camera_id == 'cam_0':
                    # First camera: create new track
                    track_id = f"track_{track_counter}"
                    track_counter += 1
                else:
                    # Subsequent cameras: try to match with global tracks
                    global_match = self.match_global_track(current_feature, camera_id)
                    
                    if global_match:
                        track_id = global_match
                    else:
                        # Skip if no valid track can be assigned
                        continue
                
                # Create new track
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                
                track_info = {
                    'bbox': bbox,
                    'feature': current_feature,
                    'last_seen': camera_id
                }
                
                new_active_tracks[track_id] = track_info
                track_history[track_id] = 0
                
                # Update global track
                self.update_global_track(track_id, current_feature, camera_id)
                
                # Draw bounding boxes and IDs
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Update active tracks
            active_tracks = new_active_tracks
            
            # Display frame
            cv2.imshow(f"Camera {camera_id}", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
     
        
    def _cleanup_old_tracks(self):
                """
                Remove old tracks that haven't been seen recently
                """
                current_time = time.time()
                with self.tracking_lock:
                    # Remove tracks older than track_timeout
                    self.global_tracks = {
                        track_id: track_info for track_id, track_info 
                        in self.global_tracks.items() 
                        if current_time - track_info.get('last_seen_time', 0) < self.track_timeout
                }

    def extract_person_image(self, frame, bbox):
                """
                Extract person image from frame based on bounding box
                
                :param frame: Full frame image
                :param bbox: Bounding box coordinates
                :return: Extracted person image
                """
                x1, y1, x2, y2 = map(int, bbox)
                person_img = frame[y1:y2, x1:x2]
                return person_img
    
    

class CommonAreaSelector:
    def __init__(self, camera_sources):
        """
        Initialize common area selector for multiple cameras
        :param camera_sources: List of camera sources
        """
        self.camera_sources = camera_sources
        
    def select_common_area(self, camera_source):
        """
        Interactively select common area for a camera      
        :param camera_source: Camera source (video file or camera index)
        :return: Selected polygon coordinates
        """
        print(camera_source,":checking11")
        # Open video capture

        cap = cv2.VideoCapture(camera_source)
        print("----CAP-----", cap)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_source}")
            return None
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            cap.release()
            return None
        
        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        original_frame = frame.copy()
        
        # Points for polygon selection
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            """
            Mouse callback for polygon selection
            """
            nonlocal frame, points
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add point to polygon
                points.append((x, y))
                
                # Draw point
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # If more than one point, draw lines between points
                if len(points) > 1:
                    cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)
                
                cv2.imshow("Select Common Area", frame)
        
        # Create window and set mouse callback
        cv2.namedWindow("Select Common Area")
        cv2.setMouseCallback("Select Common Area", mouse_callback)
        
        while True:
            cv2.imshow("Select Common Area", frame)
            key = cv2.waitKey(1) & 0xFF
            
            # Complete polygon selection
            if key == ord('c') and len(points) >= 3:
                # Draw final polygon
                cv2.polylines(frame, [np.array(points)], True, (0, 0, 255), 2)
                cv2.imshow("Select Common Area", frame)
                break
            
            # Reset selection
            elif key == ord('r'):
                frame = original_frame.copy()
                points = []
            
            # Exit without selecting
            elif key == 27:  # ESC key
                cv2.destroyAllWindows()
                cap.release()
                return None
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap.release()
        
        return np.array(points, np.int32)

def main():
    """
    Main function to set up and start multi-camera tracking
    """
    # Define camera sources (video files or camera indices)
    camera_sources = [
        # Uncomment and modify as needed
        #0,  # First camera (webcam index)
        # 1,  # Second camera (webcam index)
        # Or use video files
        #  "rtsp://admin:cctv%401212@192.168.0.221:554/ch1/sub/av_stream",
        # "rtsp://admin:cctv%401212@192.168.0.141:554/ch1/sub/av_stream", #Mukul sir outside
         #mukul sir 
        #"rtsp://admin:cctv%401212@192.168.0.140:554/ch1/sub/av_stream"
          "v1.mp4",
          "v2.mp4",
        ]
    
    
    # Interactive ROI selection
    common_area_selector = CommonAreaSelector(camera_sources)
    common_areas = {}
    
    # Select common areas for each camera
    for camera_id, camera_source in enumerate(camera_sources):
        print(f"\nSelect common area for Camera {camera_id}")
        print("Instructions:")
        print("- Click to add points to define polygon")
        print("- Press 'c' to complete selection")
        print("- Press 'r' to reset current selection")
        print("- Press 'ESC' to skip/cancel")
        
        # Attempt to select common area
        area = common_area_selector.select_common_area(camera_source)
        if area is not None:
            common_areas[f"cam_{camera_id}"] = area
    
    # Initialize and start multi-camera tracker
    tracker = CrossCameraTrackingSystem(camera_sources, common_areas)
    tracker.start_tracking()

if __name__ == "__main__":
    main()