from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse 
import uvicorn
from datetime import datetime
import os
from detection import load_model, detect_cars

app = FastAPI()

# save model in model folder 
MODEL_PATH = "model"
DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"

# Load the model
MODEL = load_model(MODEL_PATH)

@app.post("/detect-vehicles")
async def get_info(file: UploadFile = File(...)):
    # Save the video file
    file_path = f"{DATA_FOLDER}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # output file name
    timestamp = datetime.now().strftime("%Y_%m_%d%H_%M_%S")
    output_path = f"{OUTPUT_FOLDER}/output_{timestamp}.avi"

    # Detect cars in the video
    result = detect_cars(file_path, output_path, MODEL)

    return JSONResponse(content={"message": "Car detection complete!",
                                 "result": result})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=42099)