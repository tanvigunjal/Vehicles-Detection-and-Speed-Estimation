import cv2
import torch
import math
import os
import time

# load the model
def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model


def detect_cars(video_path, output_path, model):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize frame counter
    frame_counter = 0
    car_ids = {}
    id_counter = 0
    
    # Define positions of detection lines
    line1 = 1000
    line2 = 1400
    
    # Dictionary to track which cars have crossed the detection line
    down = {}
    counter_down = []
    speeds = []
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        
        # Break loop if no frame is returned
        if not ret:
            break

        # Increment frame counter
        frame_counter += 1

        # Apply Gaussian blur to the frame
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0) # 0 is the standard deviation. Calualted by the formula sqrt(2*sigma^2)

        # Model for object detection
        results = model(blurred_frame)

        # Extract scores, bounding boxes, and classes from detection results
        scores = results.xyxy[0][:, 4]
        bboxes = results.xyxy[0][:, :4]
        classes = results.xyxy[0][:, 5]

        # Perform non-maximum suppression to filter out overlapping bounding boxes
        keep = torch.ops.torchvision.nms(bboxes, scores, iou_threshold=0.5)
        bboxes = bboxes[keep]
        scores = scores[keep]

        # Iterate over detected objects
        for bbox, score, cls in zip(bboxes, scores, classes):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate centroid of the bounding box
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Calculate centroid coordinates
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            # Initialize car ID
            car_id = None
            
            # Iterate over existing car IDs
            for id, bbox in car_ids.items():
                dist = math.hypot(cx - bbox[0], cy - bbox[1])
                if dist < 35:
                    car_id = id
                    break

            # Assign new ID if no existing ID is found
            if car_id is None:
                car_id = id_counter
                id_counter += 1

            # Update car IDs dictionary with new centroid
            car_ids[car_id] = (cx, cy)
        
            # Display the car id
            cv2.putText(frame, str(car_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check if car has crossed the detection line
            if cx > line1 and cx < line2 and car_id not in down:
                down[car_id] = time.time()
                
            # Record cars that have crossed the line and calculate speed
            if car_id in down and cx > line2 and car_id not in counter_down:
                counter_down.append(car_id) 
                time_crossed = time.time() - down[car_id]
                if time_crossed > 0:
                    distance = abs(line2 - line1) * 0.1  # 1 pixel = 0.1 meters (assumption)
                    speed = distance / time_crossed
                    speed = speed * 3.6  # convert speed from meters per second to km/h
                    print("Car ID: ", car_id, "Speed: ", round(speed, 2), "km/h")
                    speeds.append(round(speed, 2))
                cv2.putText(frame, str(round(speed, 2)) + " km/h", (cx+30, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        # Draw detection lines
        cv2.line(frame, (1000, 0), (1000, height), (255, 0, 0), 2)  
        cv2.line(frame, (1400, 0), (1400, height), (255, 0, 0), 2)  

        # Display frame
        cv2.imshow('frame', frame)
        
        out.write(frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer objects
    cap.release()
    out.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Calculate average speed
    avg_speed = sum(speeds) / len(speeds) if speeds else 0

    result = {"Total Cars": len(car_ids), 
              "Moving Cars": len(counter_down),
              "Average speed in km/hr": round(avg_speed, 2)}

    return result


if __name__ == '__main__':
    # Define paths
    video_path = './data/20240318_092055.mp4'
    output_path = 'output.mp4'
    file_path = 'model'

    # Load the model
    model = load_model(file_path)

    # Detect cars in the video
    detect_cars(video_path, output_path, model)
