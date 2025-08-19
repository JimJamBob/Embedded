import os
import base64
import io
import cv2
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not found in .env file")

# Initialize the OpenAI client with xAI API endpoint
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

def capture_webcam_image():
    """Capture an image from the webcam."""
    try:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Ensure it's connected and accessible.")

        print("Webcam opened. Press 's' to capture an image, 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame from webcam.")

            # Display the live preview
            cv2.imshow('Webcam Preview - Press s to capture, q to quit', frame)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            # Capture on 's'
            if key == ord('s'):
                cap.release()
                cv2.destroyAllWindows()
                return frame

            # Quit on 'q'
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("User quit webcam capture.")

    except Exception as e:
        raise RuntimeError(f"Webcam capture error: {str(e)}")

def encode_image(frame):
    """Encode a captured frame (numpy array) to base64."""
    # Encode frame to JPEG in memory
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = 'image/jpeg'
    return base64_image, mime_type

def describe_webcam_image():
    """Capture image from webcam and get a description from Grok 4 API."""
    try:
        # Capture image from webcam
        frame = capture_webcam_image()

        # Encode to base64
        base64_image, mime_type = encode_image(frame)

        # Prepare the API request
        response = client.chat.completions.create(
            model="grok-4-0709",  # Use exact model name for vision support
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant created by xAI."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please provide a concise description of this image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "auto"  # 'auto' to balance token usage
                            }
                        }
                    ]
                }
            ],
            temperature=0.5,
            max_tokens=500  # Allow room for reasoning and output
        )

        # Debug the full response
        print("Full API Response:", response)

        # Check if response contains content
        if not response.choices or not response.choices[0].message.content:
            return "Error: Empty response from API (check token usage or model access)"

        # Extract and return the response
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Optionally list models first to verify
    # list_available_models()  # Uncomment if needed from previous version

    description = describe_webcam_image()
    print("Image Description:", description)