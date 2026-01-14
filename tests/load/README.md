# Load Testing Guide

## Run the Test

```bash
uv run locust -f tests/load/locustfile.py --host https://drone-detector-api-66108710596.europe-north2.run.app
```

Open <http://localhost:8089>, configure:

- **Users:** 10
- **Spawn rate:** 2
- **Run time:** 10m
- Click **START**

## Read Results

Go to **Statistics** tab, look at **POST /v1/predict**:

- **Median:** Typical response time
- **95th percentile:** What most users see
- **99th percentile:** Cold starts
- **Failures:** Should be 0%

Download CSV from **Download Data** tab for comparison.
