# Video Game Recommendation System

Course project for **CS 550: Massive Data Mining and Learning**.

This project builds an end-to-end recommender system using the Amazon Review Data 2018 Video Games dataset. It includes data preprocessing, per-user train/test splitting, baseline rating prediction, item-item collaborative filtering, Top-10 recommendation, evaluation, an interactive Next.js demo, a report, and presentation slides.

## Project Structure

```text
.
├── src/                         # Python recommendation pipeline
├── demo/video-game-recommender/  # Next.js interactive demo
├── report/                      # LaTeX report and figures
├── slides/                      # Beamer presentation and figures
├── docs/screenshots/            # Demo screenshots
├── Class-Project/               # Course-provided PDFs
├── framework.txt                # Project framework and checklist
├── requirements.txt             # Python dependencies
└── README.md
```

Large raw/processed data files are not included in the repository.

## Dataset

Use the Amazon Review Data 2018 Video Games files:

```text
Video_Games_5.json.gz
meta_Video_Games.json.gz
```

Create the expected raw data directory:

```bash
mkdir -p data/raw
```

Download the files:

```bash
cd data/raw
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz
```

By default, the pipeline reads and writes data under `./data`. To keep large data outside the repository, set `MDM_DATA_ROOT`:

```bash
export MDM_DATA_ROOT=/path/to/external/data/MDM
mkdir -p "$MDM_DATA_ROOT/raw"
```

Then place the two `.json.gz` files in:

```text
$MDM_DATA_ROOT/raw/
```

## Python Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Pipeline

Quick smoke test:

```bash
python src/run_pipeline.py \
  --max-review-rows 50000 \
  --max-metadata-rows 50000 \
  --max-recommendation-users 100 \
  --max-candidate-items 500
```

Final-style run used for the report:

```bash
python src/run_pipeline.py \
  --max-review-rows 500000 \
  --max-metadata-rows 500000 \
  --max-recommendation-users 1000 \
  --max-candidate-items 3000
```

Main outputs:

```text
data/results/metrics.json
data/results/rating_predictions.csv
data/results/top10_recommendations.csv
data/results/popularity_top10_recommendations.csv
demo/video-game-recommender/public/data/
```

If `MDM_DATA_ROOT` is set, `data/results/` is replaced by `$MDM_DATA_ROOT/results/`.

## Demo

The demo loads static JSON exported by the Python pipeline.

```bash
cd demo/video-game-recommender
npm install
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Open:

```text
http://localhost:3000
```

If port `3000` is busy, choose another port:

```bash
npm run dev -- --hostname 0.0.0.0 --port 3001
```

## Report and Slides

Compile the report:

```bash
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Compile the slides:

```bash
cd slides
pdflatex -interaction=nonstopmode slides.tex
pdflatex -interaction=nonstopmode slides.tex
```

Final PDFs are generated as:

```text
report/main.pdf
slides/slides.pdf
```

## Model Summary

The main model is item-item collaborative filtering:

- build an item-user rating matrix from training ratings;
- compute cosine similarity between items;
- predict held-out ratings from similar items the user already rated;
- rank unseen candidate items by predicted rating;
- return the Top-10 games per user.

Evaluation metrics:

- rating prediction: MAE, RMSE;
- Top-10 recommendation: Precision@10, Recall@10, F1@10, NDCG@10.

## Reported Results

The final experiment used:

```text
Train ratings: 380,249
Test ratings: 93,178
Users: 55,223
Items: 17,400
```

Main metrics:

```text
Mean baseline MAE: 0.7726
Item-item CF MAE: 0.7459
Popularity Precision@10: 0.0025
Item-item CF Precision@10: 0.0127
Item-item CF NDCG@10: 0.0402
```

## Public Release Notes

Do not commit raw data, processed data, `.next/`, `node_modules/`, Python caches, or LaTeX build artifacts. The `.gitignore` file is configured for these files.
