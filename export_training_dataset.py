import os
import requests
import yaml
import json
import base64
import random
import re
from urllib.parse import urljoin, urlparse
import time

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_headers(api_token):
    """Create headers for API requests."""
    return {
        'Authorization': f'Token {api_token}'
    }

def get_project_id(base_url, headers, project_name):
    """Get project ID by project name."""
    response = requests.get(urljoin(base_url, 'api/projects/'), headers=headers)
    response.raise_for_status()
    
    data = response.json()
    
    # Try to find projects in different formats
    projects = []
    if isinstance(data, list):
        projects = data
    elif isinstance(data, dict):
        if 'results' in data:
            projects = data['results']
    
    print(f"Found {len(projects)} projects")
    
    # Find our project
    for project in projects:
        if project.get('title') == project_name:
            return project['id']
    
    raise ValueError(f"Project '{project_name}' not found")

def get_all_tasks_paginated(base_url, headers, project_id):
    """Get all tasks with pagination for your specific Label Studio instance."""
    all_tasks = []
    page = 1
    while True:
        url = urljoin(base_url, f'api/projects/{project_id}/tasks?page={page}')
        print(f"Fetching tasks page {page}...")
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tasks = response.json()
        
        if not tasks or not isinstance(tasks, list) or len(tasks) == 0:
            break  # No more tasks
            
        all_tasks.extend(tasks)
        print(f"Retrieved {len(tasks)} tasks from page {page} (total: {len(all_tasks)})")
        
        # If we received fewer than 50 tasks (default page size), we're at the last page
        if len(tasks) < 50:
            break
            
        page += 1
        time.sleep(0.5)  # Avoid rate limiting
    
    print(f"Retrieved a total of {len(all_tasks)} tasks")
    return all_tasks

def extract_labels_from_config(label_config):
    """Extract labels from Label Studio config XML."""
    labels = set()
    
    # Extract from RectangleLabels, PolygonLabels, etc.
    label_tags = re.findall(r'<Label\s+value="([^"]+)"', label_config)
    labels.update(label_tags)
    
    # Extract from Choice tags
    choice_tags = re.findall(r'<Choice\s+value="([^"]+)"', label_config)
    labels.update(choice_tags)
    
    return {label: idx for idx, label in enumerate(sorted(labels))}

def get_project_labels(base_url, headers, project_id):
    """Extract label configuration from project."""
    response = requests.get(urljoin(base_url, f'api/projects/{project_id}'), headers=headers)
    response.raise_for_status()
    
    project = response.json()
    
    # Try to extract from label_config
    labels = {}
    if 'label_config' in project:
        label_config = project['label_config']
        labels = extract_labels_from_config(label_config)
    
    # If no labels found, extract from annotations
    if not labels:
        tasks = get_all_tasks_paginated(base_url, headers, project_id)
        
        unique_labels = set()
        for task in tasks:
            if 'annotations' in task and task['annotations']:
                for annotation in task['annotations']:
                    if 'result' in annotation:
                        for result in annotation['result']:
                            if 'value' in result:
                                value = result['value']
                                for key in ['rectanglelabels', 'labels', 'choices']:
                                    if key in value and isinstance(value[key], list):
                                        unique_labels.update(value[key])
        
        if unique_labels:
            labels = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    print(f"Found labels: {labels}")
    return labels

def save_base64_image(data_uri, save_path):
    """Save a base64 image to a file."""
    try:
        # Extract the base64 part from the data URI
        if data_uri.startswith('data:'):
            # Format: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/...
            header, base64_data = data_uri.split(',', 1)
        else:
            # Assume it's just the base64 data
            base64_data = data_uri
        
        # Decode and save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(base64.b64decode(base64_data))
        
        return True
    except Exception as e:
        print(f"Error saving base64 image: {str(e)}")
        return False

def download_image(url, save_path, headers):
    """Download image from URL or save base64 image."""
    try:
        # Check if this is a base64 data URI
        if url.startswith('data:image'):
            return save_base64_image(url, save_path)
        
        # Regular URL download
        if not url.startswith('http'):
            url = urljoin(base_url, url)
        
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out_file.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return False

def convert_to_yolo_format(annotation, label_map):
    """Convert Label Studio annotation to YOLO format."""
    yolo_annotations = []
    
    if 'result' not in annotation:
        return yolo_annotations
    
    for result in annotation['result']:
        try:
            if 'value' not in result:
                continue
                
            value = result['value']
            
            # Check for rectangle coordinates
            if all(k in value for k in ['x', 'y', 'width', 'height']):
                # Find label
                label = None
                if 'rectanglelabels' in value and value['rectanglelabels']:
                    label = value['rectanglelabels'][0]
                elif 'labels' in value and value['labels']:
                    label = value['labels'][0]
                
                if label and label in label_map:
                    # Extract coordinates (normalize to 0-1 range)
                    x = float(value['x']) / 100.0
                    y = float(value['y']) / 100.0
                    width = float(value['width']) / 100.0
                    height = float(value['height']) / 100.0
                    
                    # Convert to YOLO format (center_x, center_y, width, height)
                    center_x = x + width / 2
                    center_y = y + height / 2
                    
                    # Get class ID
                    class_id = label_map[label]
                    
                    # Add annotation
                    yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        except Exception as e:
            print(f"Error processing annotation: {str(e)}")
    
    return yolo_annotations

def process_task_batch(tasks, images_dir, labels_dir, label_map, headers, description):
    """Process a batch of tasks and return count of processed tasks."""
    count = 0
    total = len(tasks)
    print(f"\nProcessing {total} {description} tasks...")
    
    for i, task in enumerate(tasks):
        if i % 50 == 0 or i == total - 1:
            print(f"Progress: {i+1}/{total} {description} tasks")
            
        task_id = task.get('id', i)
        
        if 'data' not in task:
            continue
            
        task_data = task['data']
        
        # Find image field
        image_url = None
        for field in ['image', 'img', 'picture']:
            if field in task_data:
                image_url = task_data[field]
                break
        
        if not image_url:
            # Check all fields for anything that might be an image
            for field, value in task_data.items():
                if isinstance(value, str) and (value.startswith('http') or value.startswith('data:image')):
                    image_url = value
                    break
        
        if not image_url:
            continue
        
        # Create a filename for the image
        if image_url.startswith('data:'):
            image_filename = f"task_{task_id}.jpg"
        else:
            # Extract filename from URL
            image_filename = os.path.basename(urlparse(image_url).path.split('?')[0])
            if not image_filename or '.' not in image_filename:
                image_filename = f"task_{task_id}.jpg"
        
        # Download/save image
        image_path = os.path.join(images_dir, image_filename)
        if download_image(image_url, image_path, headers):
            # Get the latest annotation
            if 'annotations' in task and task['annotations']:
                annotation = task['annotations'][-1]
                
                # Convert to YOLO format
                yolo_annotations = convert_to_yolo_format(annotation, label_map)
                
                # Save annotation
                base_filename = os.path.splitext(image_filename)[0]
                label_path = os.path.join(labels_dir, f"{base_filename}.txt")
                
                with open(label_path, 'w') as f:
                    if yolo_annotations:
                        f.write('\n'.join(yolo_annotations))
                    # Empty file if no annotations
                
                count += 1
    
    return count

def export_project_for_yolov8_training(config_path, export_dir, train_ratio=0.8):
    """Export project as YOLOv8 dataset ready for fine-tuning."""
    try:
        # Load configuration
        config = load_config(config_path)
        global base_url
        base_url = config['base_url']
        
        # Create API headers
        headers = get_headers(config['api_token'])
        
        print(f"Connecting to Label Studio at {base_url}")
        
        # Get project ID
        project_id = get_project_id(base_url, headers, config['project_name'])
        print(f"Found project ID: {project_id}")
        
        # Get project labels
        label_map = get_project_labels(base_url, headers, project_id)
        
        if not label_map:
            print("ERROR: No labels found. Cannot proceed with export.")
            return None
        
        # Create export directory structure
        train_images_dir = os.path.join(export_dir, 'train', 'images')
        train_labels_dir = os.path.join(export_dir, 'train', 'labels')
        val_images_dir = os.path.join(export_dir, 'val', 'images')
        val_labels_dir = os.path.join(export_dir, 'val', 'labels')
        
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        
        # Get all tasks with pagination
        all_tasks = get_all_tasks_paginated(base_url, headers, project_id)
        
        if not all_tasks:
            print("ERROR: No tasks found in project.")
            return None
        
        # Filter valid tasks (with image and annotations)
        valid_tasks = []
        for task in all_tasks:
            if 'data' in task and 'annotations' in task and task['annotations']:
                valid_tasks.append(task)
        
        print(f"Found {len(valid_tasks)} valid tasks with annotations out of {len(all_tasks)} total tasks")
        
        if not valid_tasks:
            print("ERROR: No valid tasks found with annotations.")
            return None
        
        # Shuffle and split tasks
        random.shuffle(valid_tasks)
        train_size = int(len(valid_tasks) * train_ratio)
        
        train_tasks = valid_tasks[:train_size]
        val_tasks = valid_tasks[train_size:]
        
        print(f"Split dataset into {len(train_tasks)} training and {len(val_tasks)} validation samples")
        
        # Process tasks in batches
        train_count = process_task_batch(train_tasks, train_images_dir, train_labels_dir, 
                                         label_map, headers, "training")
        
        val_count = process_task_batch(val_tasks, val_images_dir, val_labels_dir, 
                                       label_map, headers, "validation")
        
        # Create data.yaml
        names_dict = {idx: name for name, idx in label_map.items()}
        
        data_yaml = {
            'path': os.path.abspath(export_dir),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(label_map),
            'names': names_dict
        }
        
        data_yaml_path = os.path.join(export_dir, 'data.yaml')
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"\nExport summary:")
        print(f"- Training images: {train_count}")
        print(f"- Validation images: {val_count}")
        print(f"- Total images: {train_count + val_count}")
        print(f"- Classes: {', '.join(label_map.keys())}")
        print(f"- Dataset directory: {os.path.abspath(export_dir)}")
        
        if train_count == 0 and val_count == 0:
            print("\nWARNING: No images were exported. Please check the console output for errors.")
        else:
            print(f"\nTo train with YOLOv8: yolo train model=yolov8n.pt data={data_yaml_path}")
        
        return export_dir
        
    except Exception as e:
        print(f"Error during export: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    config_path = "config.yaml"  # Path to your config.yaml
    export_dir = "yolov8_dataset"  # Directory to export to
    
    export_project_for_yolov8_training(config_path, export_dir, train_ratio=0.8)