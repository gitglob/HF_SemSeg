# Get model information from the API
## Use GET method to retrieve model info from the /info endpoint
curl -X GET "http://localhost:8000/info"

# Send a POST request to the /predict endpoint for single image prediction
## Use POST method to send data to the /predict endpoint
## Set the Accept header to expect a JSON response
## Set the Content-Type header for file upload
## Attach the image file to the request as form data
curl -X POST "http://localhost:8000/predict" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@assets/example.png" \
    --output assets/example_pred.png