import os
import base64
import subprocess
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

def capture_pi_camera_image(output_path="temp_image.jpg"):
    """Capture an image using Raspberry Pi camera with rpicam-still."""
    try:
        # Run rpicam-still to capture image
        cmd = ["rpicam-still", "-o", output_path, "--timeout", "5000", "--immediate"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to capture image with rpicam-still: {result.stderr}")

        # Read the captured image
        frame = cv2.imread(output_path)
        if frame is None:
            raise RuntimeError(f"Failed to read captured image from {output_path}")

        # Clean up temporary file
        os.remove(output_path)
        return frame

    except Exception as e:
        raise RuntimeError(f"Pi camera capture error: {str(e)}")

def encode_image(frame):
    """Encode a captured frame (numpy array) to base64."""
    # Encode frame to JPEG in memory
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = 'image/jpeg'
    return base64_image, mime_type

def describe_pi_camera_image():
    """Capture image from Raspberry Pi camera and get a description from Grok 4 API."""
    try:
        # Capture image from Pi camera
        frame = capture_pi_camera_image()

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
    description = describe_pi_camera_image()
    print("Image Description:", description)