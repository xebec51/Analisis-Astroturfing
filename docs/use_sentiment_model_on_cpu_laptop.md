# RM2 Sentiment IndoBERT V4 CPU Inference

This guide runs the published RM2 Sentiment IndoBERT V4 final model on a CPU laptop. No training is required.

## Model

- Final model: `indobenchmark/indobert-base-p2`
- Local inference artifact: `artifacts/rm2_sentiment/indobert_v4_final/base_reference`
- Final freeze pointer: `artifacts/rm2_sentiment/indobert_v4_final/final_selected_model`
- Compatibility artifact: `output/rm2_sentiment/model/indobert_v4_final_candidate`
- Checkpoint SHA256: `2b13c75df848ca0abe02cabf8a1ace0b26814730762a4a7d506095bb518c3c52`
- Frozen before locked test in commit: `05247a3050e14cdaabe5b7af8930daa37c45ad29`

Labels are emitted in this order:

```text
Negative
Neutral
Positive
```

## Setup

Install Git LFS before cloning, then download the model checkpoint:

```powershell
git clone https://github.com/xebec51/Analisis-Astroturfing.git
cd Analisis-Astroturfing
git lfs install
git lfs pull
```

Create a Python environment and install runtime packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch transformers pandas safetensors
```

## Verify Artifact

The checkpoint must be a real safetensors file, not a Git LFS pointer. The expected file size is about 498 MB.

```powershell
Get-Item artifacts\rm2_sentiment\indobert_v4_final\base_reference\model.safetensors
Get-Content artifacts\rm2_sentiment\indobert_v4_final\base_reference\SHA256SUMS.txt
```

The prediction script verifies `SHA256SUMS.txt` before loading the model.

## Predict One Text

```powershell
python scripts\predict_rm2_sentiment_cpu.py `
  --model-dir artifacts\rm2_sentiment\indobert_v4_final\base_reference `
  --text "produk ini cukup baik dan sesuai deskripsi"
```

The output contains:

- `predicted_label`: one of `Negative`, `Neutral`, `Positive`
- `confidence`: probability of the predicted class
- `prob_negative`, `prob_neutral`, `prob_positive`: class probabilities that sum to 1

## Predict CSV

Prepare a CSV with a text column, for example:

```csv
id,text
1,produk ini cukup baik dan sesuai deskripsi
2,layanan buruk dan pengiriman sangat lambat
```

Run inference:

```powershell
python scripts\predict_rm2_sentiment_cpu.py `
  --model-dir artifacts\rm2_sentiment\indobert_v4_final\base_reference `
  --input-csv input.csv `
  --text-column text `
  --output-csv sentiment_predictions.csv
```

## Troubleshooting

If loading fails with a Git LFS pointer error, run:

```powershell
git lfs install
git lfs pull
```

If CPU memory is limited, close other applications before loading the model. The published final model is the base IndoBERT checkpoint selected over the staged large candidates using development OOF metrics only.
