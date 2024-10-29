import sys
import os
import signal
import asyncio
import uvicorn

# Get the current project directory

# Add the project directory to the import path
sys.path.append("autodistill_grounded_sam_2")
sys.path.append("/app/rf_groundingdino")
sys.path.append("/app/segment_anything_2")

from autodistill_grounded_sam_2 import GroundedSAM2

from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2
import os
import tqdm
import supervision as sv

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse


# define an ontology to map class names to our Grounded SAM 2 prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model

# Directory to store uploaded and processed images
INPUT_DIR = "static/uploads/input"
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)

OUTPUT_DIR = "static/uploads/output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_all_images(image_dir):
    # iterate over all the images in the INPUT_DIR
    for image_path in tqdm.tqdm(os.listdir(INPUT_DIR)):
        if not image_path.endswith(".jpg"):
            continue

        image_path = os.path.join(INPUT_DIR, image_path)
        image = cv2.imread(image_path)

        results = base_model.predict(f'{image_path}').with_nms()
        results = results[results.confidence > 0.3]

        mask_annotator = sv.BoxAnnotator()

        annotated_image = mask_annotator.annotate(
            image.copy(), detections=results
        )

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"annotated_{os.path.basename(image_path)}"), annotated_image)

def process_image(image_path, base_model):

    image = cv2.imread(image_path)

    results = base_model.predict(f'{image_path}').with_nms()
    results = results[results.confidence > 0.3]

    mask_annotator = sv.BoxAnnotator()

    annotated_image = mask_annotator.annotate(
        image.copy(), detections=results
    )    

    processed_image_path = os.path.join(OUTPUT_DIR, f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(processed_image_path, annotated_image)
    
    return processed_image_path

app = FastAPI()

# config = uvicorn.Config(app, host='0.0.0.0', port=8080, log_level="critical")
# app = uvicorn.Server(config)
# app.run()

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):

    # output v1
    base_model = GroundedSAM2(
        ontology=CaptionOntology(
            {
                "hole in the fabric": 'fault',
                "torn fabric edge": 'fault',
            }
        ),
        model = "Grounding DINO",
        grounding_dino_box_threshold=0.3,
        grounding_dino_text_threshold=0.8
    )

    # Save the uploaded image
    original_image_path = os.path.join(INPUT_DIR, image.filename)
    with open(original_image_path, "wb") as buffer:
        buffer.write(await image.read())

    # Process the image (this is where your image processing happens)
    processed_image_path = process_image(original_image_path, base_model)

    base_model = None

    # Respond with URLs of original and processed images
    return JSONResponse({
        "original_image_url": original_image_path,
        "processed_image_url": processed_image_path
    })

# Serve static files, e.g., the HTML file
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html") as f:
        return f.read()