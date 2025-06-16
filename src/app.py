import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
from transformers import AutoImageProcessor
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import uvicorn
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def test():
    MODEL_PATH = "onnx_models/fp32.onnx"
    ort_sess = ort.InferenceSession(MODEL_PATH)
    input_name = ort_sess.get_inputs()[0].name
    input_shape = ort_sess.get_inputs()[0].shape

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")

    # Prepare image processor
    BACKBONE_MODEL = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(BACKBONE_MODEL, use_fast=False)
    processor.do_resize = False
    processor.do_center_crop = False

    # Create a dummy image for testing
    dummy_image = Image.new("RGB", (1242, 375), color=(255, 255, 255))
    print(f"Dummy image shape: {dummy_image.size}")
    image_np = preprocess_image(dummy_image, processor=processor)
    onnx_input = {input_name: image_np}
    print(f"ONNX input prepared: {onnx_input.keys()}")
    print(f"Processed image shape: {onnx_input[input_name].shape}")
    output = ort_sess.run(None, onnx_input)[0]
    print(f"Dummy inference successful: {output.shape}")

def create_segmentation_image(pred_array, num_classes=33):
    """Create a properly colored segmentation image like in your notebook"""
    # Use the same colormap and normalization as your notebook
    cmap = plt.get_cmap("viridis", num_classes)
    norm = BoundaryNorm(boundaries=np.arange(num_classes + 1) - 0.5, ncolors=num_classes)
    
    # Apply colormap to the prediction
    colored_pred = cmap(norm(pred_array))
    
    # Convert to 0-255 uint8 format for PIL
    colored_pred_uint8 = (colored_pred[:, :, :3] * 255).astype(np.uint8)
    
    return colored_pred_uint8

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global ort_sess, input_name, input_shape, input_type, PROCESSOR

    # Load ONNX model
    MODEL_PATH = "onnx_models/fp32.onnx"
    ort_sess = ort.InferenceSession(MODEL_PATH)
    input_name = ort_sess.get_inputs()[0].name
    input_shape = ort_sess.get_inputs()[0].shape
    input_type = ort_sess.get_inputs()[0].type

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")

    # Prepare image processor
    BACKBONE_MODEL = "facebook/dinov2-base"
    PROCESSOR = AutoImageProcessor.from_pretrained(BACKBONE_MODEL, use_fast=False)
    PROCESSOR.do_resize = False
    PROCESSOR.crop_size["height"] = 364
    PROCESSOR.crop_size["width"] = 1232

    yield
    
    # Shutdown (cleanup if needed)
    print("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="ONNX Single Image Inference API", 
    version="1.0.0",
    lifespan=lifespan
)

def preprocess_image(image: Image.Image, processor=None):
    """Preprocess image for model inference"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to RGB if needed
    processed_image = processor(image, return_tensors="np").pixel_values

    return processed_image

@app.get("/")
async def root():
    return {
        "message": "ONNX Model Inference API", 
        "model_input_shape": input_shape,
        "model_input_name": input_name,
        "model_input_type": input_type
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Single image inference"""
    try:
        print(f"Received file: {file.filename}, content type: {file.content_type}")
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        image_data = await file.read()
        print(f"Image data length: {len(image_data)}")

        image = Image.open(io.BytesIO(image_data))
        print(f"Image opened: {image.size}, mode: {image.mode}, type: {type(image)}")

        image_np = preprocess_image(image, processor=PROCESSOR)
        print(f"Preprocessed shape: {image_np.shape}")

        onnx_input = {input_name: image_np}

        # Run inference
        t1 = time.perf_counter()
        output = ort_sess.run(None, onnx_input)[0].squeeze(0)
        t2 = time.perf_counter()
        ort_inference_time = t2 - t1
        print(f"ONNX inference time: {ort_inference_time:.4f} seconds")
        print(f"Output shape: {output.shape}")  # [C, H, W]

        # Postprocess results
        pred = np.argmax(output, axis=0)        # [C, H, W] -> [H, W]
        pred = pred.astype(np.uint8)            # Convert to uint8 for image representation
        print(f"Prediction shape: {pred.shape}, type: {type(pred)}, dtype: {pred.dtype}")

        # Create colored segmentation image
        # colored_pred = create_segmentation_image(pred, num_classes=33)

        # Convert to PIL image
        pil_image = Image.fromarray(pred)
        print(f"PIL image created: {pil_image.size}, mode: {pil_image.mode}")

        # Convert to binary
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        return Response(
            content=img_buffer.getvalue(),
            media_type="image/png",
        )
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(f"ERROR TYPE: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/info")
async def model_info():
    """Get model information"""
    return {
        "input_name": input_name,
        "input_shape": input_shape,
        "input_type": input_type
    }

def main():
    # Run a fake inference to ensure everything is set up correctly
    # try:
    #     test()
    # except Exception as e:
    #     print(f"Error during dummy inference: {str(e)}")

    # Start the FastAPI app
    print("\n\n Starting FastAPI app...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
