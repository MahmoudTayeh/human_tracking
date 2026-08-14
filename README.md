# Human Tracking Vision

[![Validate project](https://github.com/MahmoudTayeh/human_tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/MahmoudTayeh/human_tracking/actions/workflows/ci.yml)
[![Deploy showcase](https://github.com/MahmoudTayeh/human_tracking/actions/workflows/pages.yml/badge.svg)](https://github.com/MahmoudTayeh/human_tracking/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-172234.svg)](LICENSE)

**Human Tracking Vision** is a modular computer-vision pipeline for person detection, multi-object tracking, optional face recognition, and movement analytics. The project is designed to keep the processing path inspectable: configuration lives in YAML, the main loop lives in Python, and the output includes artifacts that a human can review.

## Live showcase

The repository includes a static engineering showcase built with React, Vite, and TypeScript. GitHub Actions builds and deploys it automatically to GitHub Pages after every push to `main`.

**Live demo:** <https://mahmoudtayeh.github.io/human_tracking/>

**Source:** <https://github.com/MahmoudTayeh/human_tracking>

The showcase is intentionally a product and engineering presentation, not a fake live camera feed. The annotated frame is a visual explanation of the pipeline; actual processing runs locally with the Python application.

## What it does

| Layer | Responsibility | Implementation |
| --- | --- | --- |
| Detection | Locate people in each video frame | YOLOv8 via Ultralytics |
| Tracking | Preserve track continuity across frames | DeepSort RealTime |
| Recognition | Optionally compare face embeddings | InsightFace |
| Analytics | Produce trajectories, heatmaps, and statistics | OpenCV, pandas, matplotlib |
| Acceleration | Use GPU execution when available | CUDA / ONNX Runtime |

## Repository layout

```text
.
├── configs/config.yaml          # Runtime configuration
├── src/
│   ├── detector.py              # Person detection
│   ├── tracker.py               # Multi-object tracking
│   ├── face_recognizer.py       # Optional face recognition
│   ├── visualizer.py            # Charts and visual outputs
│   └── utils.py                 # Shared helpers
├── main.py                      # Video processing entry point
├── register_faces.py            # Face enrollment utility
├── requirements.txt             # Python dependencies
├── site/                        # React/Vite showcase for GitHub Pages
└── .github/workflows/           # CI and deployment automation
```

## Quick start

The processing application requires Python 3.11 or newer. A CUDA-capable GPU is optional but recommended for real-time workloads; the default configuration can also run on CPU with lower throughput.

```bash
git clone https://github.com/MahmoudTayeh/human_tracking.git
cd human_tracking

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create the runtime directories, register known faces only when the use case and applicable law permit it, then run the configured input:

```bash
mkdir -p data/known_faces data/videos outputs/videos outputs/statistics outputs/visualizations
python register_faces.py --mode webcam --name "Your Name" --num-photos 5
python main.py --config configs/config.yaml
```

The input video, model path, device, confidence threshold, recognition interval, and output settings are controlled in [`configs/config.yaml`](configs/config.yaml).

## Local showcase development

The showcase is a small, self-contained Vite app under `site/` so it can be built independently from the Python runtime.

```bash
cd site
pnpm install
pnpm dev
```

For a production build:

```bash
pnpm build
```

The GitHub Pages workflow uses the same build command and publishes `site/dist` through the official Pages deployment action.

## Responsible use

Face recognition and biometric data can affect real people. Use this project only for a documented, lawful purpose with appropriate consent or legal basis. Apply access controls, encryption, retention limits, audit trails, bias testing, and human review. An automated track or recognition result is an input to a decision—not a decision by itself.

Do not use the system to identify or monitor people in a way that violates privacy, safety, or applicable regulations. Teams deploying it should review local requirements, including data-protection and biometric rules, before collecting or processing any face data.

## Performance notes

The original project notes describe an approximate reference of **25 FPS with an NVIDIA GPU** and **7 FPS on CPU**, depending on model size, resolution, scene density, and recognition settings. Treat these numbers as indicative rather than a benchmark; measure on the target hardware and workload before making an operational commitment.

## Contributing

Contributions are welcome. Keep changes focused, update the relevant configuration or documentation, and make sure the validation workflow passes before opening a pull request.

## License

Released under the [MIT License](LICENSE).
