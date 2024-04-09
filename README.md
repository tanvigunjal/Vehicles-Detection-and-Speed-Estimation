# Vehicles Detection and Speed Estimation

## Abstract 
Most of the deep neural object detection/classification models such as R-CNN, Faster R-CNN and SSD require significantly higher computational power to achieve respectable FPS and Accu- racy. But, we need a different CNN-based neural architecture to cater the scenarios where the object detection task is to be executed in real-time rather than waiting for hours to process a 10 min video stream or to run the detector on limited/low-power hardware. Thus, in this project, we implement a real-time event-driven Vehicle Detection and Speed Estimation system using the PyTorch’s Yolov5 implementation. The model detects and estimates vehicles speed that cross the lines of interest projected in any axis within the video stream.

## Implementation

### Implemented Task

1. Vehicle Detection: We are using pretrained YOLOv5 model sourced from the PyTorch hub
2. Vehicle Tracking: assigning IDs to vehicles
3. Speed Calculation: We are calculating the distance traveled by the tracked vehicle per second in pixels, necessitating the conversion of pixels to meters using pixel per meter ratios to obtain distances in meters. Subsequently, we convert the distance from meters per second to kilometers per second.


### Technologies used
1. Python
2. PyTorch
3. OpenCV
4. FastAPI

### How to run the project?

1. Create virtual environment Command : python3 -m venv detect
2. Activate virtual environment command: source detect/bin/activate
3. Install requirements
command: pip install -r requirements.txt
4. Run the FASTAPI Server (in Dev mode): Execute the server.py script to start the server and deploy the FASTAPI application
command: python3 src/detection.py
5. AccesstheAPI:Once the server is running,you can access the API ”http://localhost:42099/detect- vehicles”. Use POSTMAN tool to upload a video as form-data.

### Results

![](https://github.com/tanvigunjal/Vehicles-Detection-and-Speed-Estimation/blob/main/images/image-1.png)
![](https://github.com/tanvigunjal/Vehicles-Detection-and-Speed-Estimation/blob/main/images/image-2.png)

### Future Work
1. Explore libraries such as Detectron2 and supervision, can be considered to further improve the capabilities and efficiency of the vehicle detection system
2. The integration of models such as the Segment Anything Model (SAM) could be explored to potentially improve the accuracy and performance of the system
