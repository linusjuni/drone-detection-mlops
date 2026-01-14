from locust import HttpUser, task, between
from io import BytesIO
from PIL import Image


class DroneDetectorUser(HttpUser):
    """Simulates users making requests to the drone detector API."""

    # Wait 1-3 seconds between requests
    wait_time = between(1, 3)

    def on_start(self):
        """Generate a test image once when user starts (not per request)."""
        # Create a simple 224x224 RGB test image
        img = Image.new("RGB", (224, 224), color=(73, 109, 137))

        # Convert to JPEG bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        self.test_image = img_bytes.getvalue()

    @task(1)
    def health_check(self):
        """Check API health - lightweight baseline."""
        self.client.get("/health")

    @task(2)
    def get_info(self):
        """Get API info."""
        self.client.get("/v1/info")

    @task(10)
    def predict_drone(self):
        """Make prediction - primary workload (10x weight)."""
        files = {"file": ("test.jpg", self.test_image, "image/jpeg")}
        with self.client.post("/v1/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                # Optionally validate response structure
                try:
                    data = response.json()
                    if "prediction" in data and "metadata" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except Exception as e:
                    response.failure(f"Response parsing failed: {e}")
            else:
                response.failure(f"Got status code {response.status_code}")
