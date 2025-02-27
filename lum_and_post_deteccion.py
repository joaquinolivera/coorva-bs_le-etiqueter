import requests
import json
import os
import time
import argparse
import yaml
from typing import List, Dict, Any, Optional
import base64
import torch
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostLightDetector:
    def __init__(self, model_path: str = 'best.pt', confidence_threshold: float = 0.3, target_size: int = 640):
        """
        Initialize with the custom trained model for posts and lights
        
        Args:
            model_path: Path to the trained YOLO model
            confidence_threshold: Minimum confidence level for detections
            target_size: Target size for image dimensions (default: 640)
        """
        logger.info(f"Initializing custom YOLO model from {model_path}...")
        self.confidence_threshold = confidence_threshold
        self.target_size = target_size
        try:
            # Use your custom trained model
            self.model = YOLO(model_path)
            logger.info("Custom YOLO model loaded successfully")
            
            # Print available classes for verification
            logger.info("Available classes in custom model:")
            for idx, class_name in self.model.names.items():
                logger.info(f"Class {idx}: {class_name}")
            
            logger.info(f"Model will resize images to target size: {target_size}px")
                
        except Exception as e:
            logger.error(f"Error loading custom YOLO model: {e}")
            raise

    def detect_objects(self, image_path: str, visualize: bool = False, target_size: int = 640) -> List[Dict]:
        """
        Detect posts (postacion) and lights (luminaria) in an image
        
        Args:
            image_path: Path to the image file
            visualize: Whether to save a visualization of the detections
            target_size: Target size for image dimensions (default: 640)
            
        Returns:
            List of detection dictionaries with coordinates and labels
        """
        logger.info(f"Processing image: {image_path}")
        try:
            # Read original image
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"Could not read image at {image_path}")
                
            # Get original dimensions
            orig_height, orig_width = original_image.shape[:2]
            logger.info(f"Original image dimensions: {orig_width}x{orig_height}")
            
            # Resize image to fixed square dimensions (keeping aspect ratio by padding)
            # Create a square black canvas
            square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            
            # Calculate scaling factor to fit image within target_size
            scale = min(target_size / orig_width, target_size / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # Resize the original image
            resized = cv2.resize(original_image, (new_width, new_height))
            
            # Calculate padding to center the image
            x_offset = (target_size - new_width) // 2
            y_offset = (target_size - new_height) // 2
            
            # Place the resized image on the square canvas
            square_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            logger.info(f"Resized image to {target_size}x{target_size} with padding, scale factor: {scale:.3f}")
            
            # Run inference on square image
            results = self.model(square_img)
            
            detections = []

            # Map your custom model's classes to Label Studio labels
            class_mapping = {
                'postacion': 'postacion',
                'luminaria': 'luminaria'
            }
            
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = result
                class_name = self.model.names[int(cls)]
                logger.info(f"Found {class_name} with confidence {float(conf):.2f}")
                
                if conf > self.confidence_threshold and class_name in class_mapping:
                    # Adjust coordinates to remove padding
                    adj_x1 = max(0, float(x1) - x_offset)
                    adj_y1 = max(0, float(y1) - y_offset)
                    adj_x2 = max(0, float(x2) - x_offset)
                    adj_y2 = max(0, float(y2) - y_offset)
                    
                    # Skip if the detection is completely in the padding
                    if adj_x2 <= 0 or adj_y2 <= 0 or adj_x1 >= new_width or adj_y1 >= new_height:
                        logger.info(f"Skipping detection in padding area")
                        continue
                    
                    # Clip to the bounds of the resized image
                    adj_x1 = min(max(0, adj_x1), new_width)
                    adj_y1 = min(max(0, adj_y1), new_height)
                    adj_x2 = min(max(0, adj_x2), new_width)
                    adj_y2 = min(max(0, adj_y2), new_height)
                    
                    # Convert back to original image coordinates
                    orig_x1 = adj_x1 / scale
                    orig_y1 = adj_y1 / scale
                    orig_x2 = adj_x2 / scale
                    orig_y2 = adj_y2 / scale
                    
                    # Convert to percentage for Label Studio
                    detection = {
                        "x": orig_x1 / orig_width * 100,
                        "y": orig_y1 / orig_height * 100,
                        "width": (orig_x2 - orig_x1) / orig_width * 100,
                        "height": (orig_y2 - orig_y1) / orig_height * 100,
                        "label": class_mapping[class_name],
                        "confidence": float(conf)
                    }
                    detections.append(detection)
                    logger.info(f"Added detection {detection['label']} with confidence {detection['confidence']:.2f}")
            
            # Create visualization if requested
            if visualize and detections:
                vis_image = original_image.copy()
                for detection in detections:
                    x1 = int(detection['x'] * orig_width / 100)
                    y1 = int(detection['y'] * orig_height / 100)
                    w = int(detection['width'] * orig_width / 100)
                    h = int(detection['height'] * orig_height / 100)
                    
                    color = (0, 255, 0) if detection['label'] == 'postacion' else (255, 0, 0)
                    cv2.rectangle(vis_image, (x1, y1), (x1+w, y1+h), color, 2)
                    cv2.putText(vis_image, f"{detection['label']} {detection['confidence']:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Save the visualization
                vis_dir = Path('visualizations')
                vis_dir.mkdir(exist_ok=True)
                vis_path = vis_dir / Path(image_path).name
                cv2.imwrite(str(vis_path), vis_image)
                logger.info(f"Saved visualization to {vis_path}")
                
                # Also save the square input image for debugging
                debug_dir = Path('debug_square_inputs')
                debug_dir.mkdir(exist_ok=True)
                debug_path = debug_dir / Path(image_path).name
                cv2.imwrite(str(debug_path), square_img)
                logger.info(f"Saved square input image to {debug_path}")
            
            return detections
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise

class LabelStudioAPI:
    def __init__(self, api_token: str, base_url: str = "http://localhost:8080", max_retries: int = 3):
        """
        Initialize Label Studio API client
        
        Args:
            api_token: Authentication token for Label Studio
            base_url: Base URL of the Label Studio API
            max_retries: Maximum number of retry attempts for API calls
        """
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/json"
        }
        self.max_retries = max_retries
        logger.info(f"Initialized Label Studio API client with base URL: {base_url}")
        
        # Test connection
        try:
            self._api_call_with_retry('get', f"{self.base_url}/api/projects")
            logger.info("Successfully connected to Label Studio API")
        except Exception as e:
            logger.error(f"Failed to connect to Label Studio API: {e}")
            raise

    def _api_call_with_retry(self, method: str, url: str, json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make API call with exponential backoff retry
        
        Args:
            method: HTTP method ('get' or 'post')
            url: API endpoint URL
            json_data: JSON payload for POST requests
            
        Returns:
            API response as dictionary
        """
        retry = 0
        backoff_factor = 1.5
        
        while retry < self.max_retries:
            try:
                if method.lower() == 'get':
                    response = requests.get(url, headers=self.headers)
                elif method.lower() == 'post':
                    response = requests.post(url, headers=self.headers, json=json_data)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                wait_time = backoff_factor ** retry
                logger.warning(f"API call to {url} failed: {e}. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                retry += 1
        
        raise Exception(f"API call to {url} failed after {self.max_retries} retries")

    def create_project(self, name: str, label_config: str) -> int:
        """
        Create a new project in Label Studio
        
        Args:
            name: Project name
            label_config: XML configuration for the labeling interface
            
        Returns:
            Project ID
        """
        url = f"{self.base_url}/api/projects"
        payload = {
            "title": name,
            "label_config": label_config
        }
        
        response = self._api_call_with_retry('post', url, payload)
        return response["id"]

    def upload_images(self, project_id: int, image_folder: str, batch_size: int = 10) -> List[Dict]:
        """
        Upload multiple images to a project using Label Studio's tasks endpoint
        
        Args:
            project_id: Label Studio project ID
            image_folder: Folder containing image files
            batch_size: Number of images to upload in each batch
            
        Returns:
            List of created tasks with IDs and file paths
        """
        url = f"{self.base_url}/api/tasks"
        tasks = []
        
        image_files = []
        for image_file in os.listdir(image_folder):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(image_file)
        
        # Process in batches but upload one by one (for compatibility)
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1) // batch_size} with {len(batch)} images")
            
            for image_file in batch:
                image_path = os.path.join(image_folder, image_file)
                logger.info(f"Uploading image: {image_file}")
                
                try:
                    with open(image_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    payload = {
                        "project": project_id,
                        "data": {
                            "image": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                    
                    # Upload individual image (compatible with all Label Studio versions)
                    response = self._api_call_with_retry('post', url, payload)
                    
                    if 'id' in response:
                        task_id = response['id']
                        tasks.append({
                            "id": task_id,
                            "path": image_path
                        })
                        logger.info(f"Successfully uploaded {image_file} with task ID {task_id}")
                    else:
                        logger.error(f"Unexpected response format: {response}")
                    
                except Exception as e:
                    logger.error(f"Error uploading {image_file}: {e}")
                    continue
        
        return tasks

    def create_annotation(self, task_id: int, bounding_boxes: List[Dict]) -> int:
        """
        Create an annotation with bounding boxes for a task
        
        Args:
            task_id: Label Studio task ID
            bounding_boxes: List of bounding box dictionaries
            
        Returns:
            Annotation ID
        """
        url = f"{self.base_url}/api/tasks/{task_id}/annotations"
        
        payload = {
            "result": [
                {
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image", 
                    "value": {
                        "x": bb["x"],
                        "y": bb["y"],
                        "width": bb["width"],
                        "height": bb["height"],
                        "rotation": 0,
                        "rectanglelabels": [bb["label"]]
                    }
                }
                for bb in bounding_boxes
            ]
        }
        
        response = self._api_call_with_retry('post', url, payload)
        return response["id"]
    
    def get_project_progress(self, project_id: int) -> Dict[str, Any]:
        """
        Get progress information for a project
        
        Args:
            project_id: Label Studio project ID
            
        Returns:
            Project statistics dictionary
        """
        url = f"{self.base_url}/api/projects/{project_id}"
        response = self._api_call_with_retry('get', url)
        return response

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Return default configuration
        return {
            "api_token": "e9e1e0c4e13a2b89ca8eeeb33062890139fb43b9",
            "base_url": "http://localhost:8080",
            "model_path": "best.pt",
            "base_folder": "../Camino_1",
            "project_name": "Luminaria y Postacion",
            "visualize_detections": False,
            "batch_size": 10,
            "num_workers": 4,
            "confidence_threshold": 0.3,
            "max_retries": 3
        }

def process_images_in_parallel(detector: PostLightDetector, 
                              ls: LabelStudioAPI, 
                              project_id: int, 
                              tasks: List[Dict], 
                              visualize: bool, 
                              max_workers: int) -> None:
    """
    Process multiple images in parallel
    
    Args:
        detector: The object detector instance
        ls: Label Studio API client
        project_id: Label Studio project ID
        tasks: List of tasks to process
        visualize: Whether to save visualization of detections
        max_workers: Maximum number of concurrent workers
    """
    def process_task(task):
        try:
            detections = detector.detect_objects(
                task["path"], 
                visualize=visualize,
                target_size=detector.target_size
            )
            if detections:
                annotation_id = ls.create_annotation(task["id"], detections)
                logger.info(f"Created annotation {annotation_id} for task {task['id']} with {len(detections)} objects")
                return True
            else:
                logger.info(f"No posts or lights detected in task {task['id']}")
                return False
        except Exception as e:
            logger.error(f"Error processing task {task['id']}: {e}")
            return False
    
    logger.info(f"Processing {len(tasks)} tasks with {max_workers} workers")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_task, tasks))
    
    successful = results.count(True)
    logger.info(f"Successfully processed {successful} out of {len(tasks)} tasks")

def save_progress(project_id: int, subfolder: str, tasks: List[Dict], completed: bool = False) -> None:
    """
    Save progress information to a file for potential resuming
    
    Args:
        project_id: Label Studio project ID
        subfolder: Current subfolder being processed
        tasks: List of tasks
        completed: Whether processing of this subfolder is complete
    """
    progress_dir = Path('progress')
    progress_dir.mkdir(exist_ok=True)
    
    progress_info = {
        "project_id": project_id,
        "subfolder": subfolder,
        "tasks": tasks,
        "completed": completed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(progress_dir / f"{Path(subfolder).name}_progress.json", 'w') as f:
        json.dump(progress_info, f, indent=2)
    
    logger.info(f"Saved progress information for {subfolder}")

def load_progress(subfolder: str) -> Dict[str, Any]:
    """
    Load saved progress information
    
    Args:
        subfolder: Subfolder to check progress for
        
    Returns:
        Progress information dictionary or None if not found
    """
    progress_file = Path('progress') / f"{Path(subfolder).name}_progress.json"
    
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            logger.info(f"Loaded progress information for {subfolder}")
            return progress
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
    
    return None

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Detect posts and lights in images and upload to Label Studio')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--confidence', type=float, help='Confidence threshold for detections')
    parser.add_argument('--visualize', action='store_true', help='Save visualization of detections')
    parser.add_argument('--batch-size', type=int, help='Batch size for processing')
    parser.add_argument('--workers', type=int, help='Number of worker threads')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    parser.add_argument('--subfolder', type=str, help='Process only this subfolder')
    parser.add_argument('--target-size', type=int, help='Target size for image dimensions')
    return parser.parse_args()

def main():
    try:
        logger.info("Starting Post and Light Detection process")
        
        # Parse arguments and load configuration
        args = parse_arguments()
        config = load_config(args.config)
        
        # Override config with command line arguments if provided
        confidence_threshold = args.confidence if args.confidence is not None else config.get('confidence_threshold', 0.3)
        visualize = args.visualize or config.get('visualize_detections', False)
        batch_size = args.batch_size if args.batch_size is not None else config.get('batch_size', 10)
        max_workers = args.workers if args.workers is not None else config.get('num_workers', 4)
        target_size = args.target_size if args.target_size is not None else config.get('target_size', 640)
        
        # Initialize the detector and API client
        detector = PostLightDetector(
            model_path=config.get('model_path', 'best.pt'),
            confidence_threshold=confidence_threshold,
            target_size=target_size
        )
        
        ls = LabelStudioAPI(
            api_token=config.get('api_token', 'e9e1e0c4e13a2b89ca8eeeb33062890139fb43b9'),
            base_url=config.get('base_url', 'http://localhost:8080'),
            max_retries=config.get('max_retries', 3)
        )
        
        # Get base folder path
        base_folder_path = config.get('base_folder', '../Camino_1')
        base_folder = Path(base_folder_path)
        logger.info(f"Base folder: {base_folder}")
        
        # XML configuration for Label Studio
        label_config = """
        <View>
          <Image name="image" value="$image"/>
          <RectangleLabels name="label" toName="image">
            <Label value="luminaria" background="blue"/>
            <Label value="postacion" background="yellow"/>
          </RectangleLabels>
        </View>
        """
        
        # Create or load project
        project_name = config.get('project_name', 'Luminaria y Postacion')
        if args.resume:
            # Look for any progress files to get the project ID
            progress_dir = Path('progress')
            if progress_dir.exists():
                progress_files = list(progress_dir.glob('*_progress.json'))
                if progress_files:
                    with open(progress_files[0], 'r') as f:
                        progress = json.load(f)
                    project_id = progress.get('project_id')
                    logger.info(f"Resuming with existing project ID: {project_id}")
                else:
                    logger.info("No progress files found, creating new project")
                    project_id = ls.create_project(project_name, label_config)
            else:
                logger.info("No progress directory found, creating new project")
                project_id = ls.create_project(project_name, label_config)
        else:
            # Create new project
            project_id = ls.create_project(project_name, label_config)
            logger.info(f"Created project {project_name} with ID {project_id}")
        
        # Get subfolders or specific subfolder if specified
        if args.subfolder:
            subfolder_path = base_folder / args.subfolder
            if subfolder_path.is_dir():
                subfolders = [subfolder_path]
            else:
                logger.error(f"Specified subfolder {args.subfolder} not found")
                return
        else:
            subfolders = [f for f in base_folder.iterdir() if f.is_dir()]
        
        logger.info(f"Found {len(subfolders)} subfolders to process")
        
        # Process each subfolder
        total_images = 0
        for subfolder in subfolders:
            logger.info(f"\nProcessing subfolder: {subfolder.name}")
            
            # Check if we have progress to resume
            if args.resume:
                progress = load_progress(str(subfolder))
                if progress and progress.get('completed', False):
                    logger.info(f"Subfolder {subfolder.name} already completed, skipping")
                    continue
            
            try:
                # Upload images
                tasks = ls.upload_images(project_id, str(subfolder), batch_size=batch_size)
                
                if tasks:
                    num_images = len(tasks)
                    total_images += num_images
                    logger.info(f"Uploaded {num_images} images from {subfolder.name}")
                    
                    # Save progress after upload
                    save_progress(project_id, str(subfolder), tasks)
                    
                    # Process images in parallel
                    process_images_in_parallel(
                        detector=detector,
                        ls=ls,
                        project_id=project_id,
                        tasks=tasks,
                        visualize=visualize,
                        max_workers=max_workers
                    )
                    
                    # Mark as completed
                    save_progress(project_id, str(subfolder), tasks, completed=True)
                else:
                    logger.warning(f"No tasks were created for subfolder {subfolder.name}")
            
            except Exception as e:
                logger.error(f"Error processing subfolder {subfolder.name}: {e}")
                continue
        
        # Get final project stats
        project_stats = ls.get_project_progress(project_id)
        logger.info(f"\nProcess completed. Processed {total_images} images across {len(subfolders)} subfolders")
        logger.info(f"Final project statistics: {project_stats.get('task_count', 0)} tasks, {project_stats.get('total_annotations_count', 0)} annotations")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()