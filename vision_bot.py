# vision_bot.py

import cv2
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch

# Initialize the BLIP processor and model
print("Loading the BLIP model and processor...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Function to capture an image from the webcam
def capture_image():
    print("Accessing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        raise IOError("Failed to capture image")

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Convert the captured frame to a PIL Image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

# Function to process the image and generate an answer
def answer_question(image, question):
    print("Processing image and question...")
    inputs = processor(image, question, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

def main():
    print("Welcome to Vision Bot!")
    print("Capturing an image from the webcam...")
    image = capture_image()

    # Ask the user for a question
    question = input("Please type your question about the captured image:\n")

    # Get the answer to the question
    print("Generating an answer...")
    answer = answer_question(image, question)

    # Display the answer
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
