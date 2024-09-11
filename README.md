# Ml-project-1

Potato Disease Detection using FastAPI and TensorFlow

Project Overview
This project is a Potato Disease Detection application that uses a machine learning model to classify images of potato leaves into three categories:

Early Blight
Late Blight
Healthy
The backend is built using FastAPI, and the machine learning model is served to make predictions based on uploaded images. The frontend is developed with HTML, CSS, and JavaScript for an interactive user interface.

Features
Model Fetching: The machine learning model is automatically downloaded from Google Drive during the startup of the FastAPI server.
Prediction Endpoint: Users can upload an image of a potato leaf, and the model will classify it into one of the three classes (Early Blight, Late Blight, or Healthy).
Frontend Interface: Users can upload an image directly from their computer or capture one using their webcam, and get instant predictions.
Cross-Origin Resource Sharing (CORS): CORS is enabled to allow interaction between the frontend and backend.
Table of Contents
Technologies Used
Setup Instructions
Running the Application
Endpoints
Frontend Instructions
Technologies Used
Backend:

FastAPI
TensorFlow
Google Drive API (via gdown)
Python (v3.7+)
Uvicorn
Frontend:

HTML/CSS
TailwindCSS (for styling)
JavaScript (for interaction)
Setup Instructions
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/potato-disease-detection.git
cd potato-disease-detection
Install required Python packages: Ensure you have Python 3.7 or higher installed. Then, create a virtual environment and install dependencies:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
Google Drive Model Download: The model is hosted on Google Drive. The gdown package will download it automatically during the app startup.

Run the application: Start the FastAPI server by running the following command:

bash
Copy code
uvicorn main:app --reload --host 0.0.0.0 --port 5500
This will run the backend server on localhost:5500.

Running the Application
After starting the FastAPI server:

Access the backend API on http://localhost:5500/docs to explore the API documentation.
Use the frontend by opening the index.html file in a web browser.
Requirements:
Python 3.7+
pip for package management
Backend
The application automatically downloads the ML model from Google Drive when the server starts.

Endpoints
/ping [GET]
Description: A simple health check endpoint.
Response:
json
Copy code
{
    "message": "Hello, Badi"
}
/predict [POST]
Description: Accepts an uploaded image of a potato leaf and returns the predicted class along with confidence.
Request:
File (image)
Response:
json
Copy code
{
    "class": "Early Blight",
    "confidence": 0.95
}
Frontend Instructions
Uploading Image:

The frontend provides an option to upload an image from your device or capture an image using your webcam.
Drag and drop the image into the designated area or use the file input buttons.
Prediction:

Once an image is uploaded or captured, click the Predict button.
The application sends the image to the backend FastAPI server for prediction.
The predicted class (Early Blight, Late Blight, or Healthy) and the confidence score are displayed on the webpage.
Frontend Technologies:
TailwindCSS: For responsive design.
FontAwesome: For icons.
Future Improvements
Expand the Dataset: Incorporate more diseases and classes for broader detection.
Enhance UI: Improve the frontend design and user experience.
Deploy on Cloud: Deploy both backend and frontend on a cloud platform like Railway, Vercel, or Netlify.
License
This project is licensed under the MIT License.

Feel free to customize and expand the README as per your specific needs or project updates.
