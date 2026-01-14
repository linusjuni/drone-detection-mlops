# Drone Detector Frontend

SvelteKit frontend for bird vs drone classification.

## Quick Start

```bash
npm install
npm run dev
```

Open `http://localhost:5173` and upload an image to get predictions!

## Configuration

The frontend connects to the production API by default. Configure in `.env`:

```bash
# Production API
PUBLIC_API_URL=https://drone-detector-api-66108710596.europe-north2.run.app
```

## Project Structure

```plaintext
src/
├── routes/
│   ├── +page.svelte      # Main UI with image upload
│   ├── +layout.svelte    # App layout
│   ├── api.ts            # API client (predictImage, checkHealth)
│   └── layout.css        # Tailwind + Red Bull theme
└── app.html              # HTML template
```

## API Integration

The `api.ts` module provides:

- `predictImage(file: File)` - Upload image and get prediction
- `checkHealth()` - Check API status
