# Surgical Phase Detection and Video Indexing with MongoDB

An end-to-end system for **automatic surgical phase recognition and temporal video indexing**.

The project converts long surgical videos into searchable phase intervals:

```text
Surgical video
      ↓
1 frame/sec
      ↓
ResNet-50
      ↓
2048-D feature / second
      ↓
MS-TCN
      ↓
phase / second
      ↓
temporal cleanup
      ↓
phase intervals
      ↓
MongoDB
```

![Surgical Phase Detection Pipeline](docs/surgical_phase_detection_pipeline.png)

## 1. Project goal

Surgical videos are long and difficult to inspect or search manually. The project implements a pipeline that:

1. reads annotated surgical workflow data;
2. prepares per-second phase labels;
3. samples surgical videos at approximately 1 frame per second;
4. extracts a 2048-dimensional visual representation with pretrained ResNet-50;
5. predicts the surgical phase over time using a Multi-Stage Temporal Convolutional Network (MS-TCN);
6. converts second-level predictions into clean temporal intervals;
7. stores the resulting metadata and phase intervals in MongoDB;
8. makes the surgical video collection searchable by phase and time.

The **video itself is not stored in MongoDB**. MongoDB stores metadata and model outputs, while the original video can remain in controlled storage.

---

## 2. Model architecture

### ResNet-50 visual encoder

Each video is sampled at **1 frame per second**. A pretrained ResNet-50 is used as a feature extractor. Its final ImageNet classification layer is replaced with an identity layer.

For every sampled frame:

```text
RGB frame
   ↓
ResNet-50
   ↓
2048-D feature vector
```

For a video with `T` sampled seconds, the resulting feature matrix has shape:

```text
[T, 2048]
```

### MS-TCN temporal model

The feature sequence is then passed to an MS-TCN:

- 4 stages
- 10 dilated temporal layers per stage
- 64 feature maps
- 2048 input features
- 7 surgical phase classes in the current dataset

The model predicts one phase label for each sampled second.

Internally the network receives:

```text
[B, 2048, T]
```

and produces:

```text
[num_stages, B, num_classes, T]
```

The prediction from the final stage is used as the model output.

---

## 3. Surgical annotations

The original annotation spreadsheet contains two related concepts:

### Surgical phases

These indicate **where the procedure is in the surgical workflow**, for example:

- Adhesiolysis / colon / splenic / liver release
- Hilar dissection
- Kidney release / fatty detachment
- Tumor boundaries identification / tumorectomy preparation
- Tumor resection
- Tumor bed reconstruction / hemostasis
- Outside patient

### Surgical tags

Tags describe an event or condition occurring during the procedure, for example:

- Echography
- Outside patient
- Unknown break

In the current V1 model:

- the model is trained on **surgical phases**;
- `Outside patient` is treated as a phase-like visual class;
- `Unknown break` and unlabeled positions are ignored using label `-100`;
- tag prediction is reserved for future work.

---

## 4. Dataset preparation

`preparedataset.py` reads the Excel annotation file and creates a processed dataset.

Important design choices:

- annotations are converted to a 1 Hz timeline;
- label dictionaries are created automatically;
- `IGNORE` is mapped to `-100`, compatible with PyTorch `CrossEntropyLoss(ignore_index=-100)`;
- train / validation / test splitting is performed by **complete surgery**, not by video fragment, to avoid data leakage;
- the original phase labels are preserved.

A processed fragment has a structure similar to:

```text
processed/
├── label_maps.json
├── splits.csv
├── U57L_1_raw/
│   ├── features.npy
│   ├── phase_labels.npy
│   ├── phase_labels_aligned.npy
│   ├── tag_labels.npy
│   └── timeline.csv
└── ...
```

### Current usable split

The processed dataset used in the reported experiments contained:

| Split | Video fragments | Surgeries |
|---|---:|---:|
| Train | 87 | 16 |
| Validation | 14 | 3 |
| Test | 22 | 5 |

Fragments containing no usable phase labels are skipped.

---

## 5. Feature extraction

`feature.py` extracts visual features from the annotated video collection.

For each video:

1. open the video with OpenCV;
2. sample approximately one frame per second;
3. convert BGR to RGB;
4. apply the standard preprocessing associated with pretrained ResNet-50;
5. extract a 2048-D feature vector;
6. align the feature sequence with the annotation sequence;
7. save:
   - `features.npy`
   - `phase_labels_aligned.npy`

Small feature/annotation duration differences are cropped to the common temporal length while the original phase annotation is preserved separately.

---

## 6. Dataset loader

`dataset.py` implements `SurgicalPhaseDataset`.

It:

- loads the surgery-level split from `splits.csv`;
- extracts a surgery ID from the video folder name;
- verifies `features.npy` and `phase_labels_aligned.npy`;
- validates feature dimensions;
- validates temporal alignment;
- rejects NaN / infinite feature values;
- skips fragments with no valid labels;
- returns features as `[2048, T]` and labels as `[T]`.

---

## 7. Training

`train.py` trains the unweighted MS-TCN baseline.

### Training configuration

| Parameter | Value |
|---|---:|
| Epochs | 50 |
| Batch size | 1 |
| Learning rate | `5e-4` |
| MS-TCN stages | 4 |
| Layers per stage | 10 |
| Feature maps | 64 |
| Feature dimension | 2048 |
| Ignore label | -100 |
| Temporal smoothing weight | 0.15 |

The loss is the sum, over all MS-TCN stages, of:

```text
Cross-Entropy Loss
        +
0.15 × Temporal Smoothing Loss
```

The temporal smoothing term penalizes abrupt changes in adjacent predictions at valid annotated positions.

The best checkpoint is selected using **validation Macro-F1** and saved as:

```text
training_output/best_model.pt
```

---

## 8. Evaluation

### Frame / second-level evaluation

`evaluate.py` evaluates the best checkpoint on the test surgeries and produces:

- global accuracy;
- Macro-F1;
- Weighted-F1;
- per-phase precision / recall / F1;
- classification report;
- confusion matrix;
- per-video metrics;
- JSON summary.

### Reported baseline results

| Metric | Test result |
|---|---:|
| Accuracy | **0.7904** |
| Macro-F1 | **0.8010** |
| Weighted-F1 | **0.7939** |
| Best validation Macro-F1 | **≈ 0.8803** |

The main observed errors occur between visually similar neighboring phases. `Outside patient` is particularly distinctive, while tumor preparation is one of the more difficult phases.

Class-weighting experiments improved some minority-class behavior but reduced overall performance, so the **unweighted model was retained as the baseline**.

---

## 9. Temporal segmentation evaluation

Second-level accuracy alone does not tell whether predicted phase boundaries are clean. Therefore the project also evaluates temporal segments.

`evaluate_segments.py` computes:

- normalized Edit score;
- F1@10;
- F1@25;
- F1@50.

`postprocess_segments.py` evaluates different minimum segment durations on the validation set.

### Raw MS-TCN temporal results

| Metric | Raw prediction |
|---|---:|
| Edit | 55.61 |
| F1@10 | 52.51 |
| F1@25 | 50.15 |
| F1@50 | 43.07 |

### 15-second post-processing experiment

The best validation setting among the tested thresholds was **15 seconds**, with validation F1@50 = **74.13**.

When this fixed threshold was evaluated on the test set:

| Metric | Test result after 15 s cleanup |
|---|---:|
| Edit | **67.49** |
| F1@10 | **70.12** |
| F1@25 | **66.93** |
| F1@50 | **57.37** |

> **Important:** the current end-to-end batch inference script is configured with a **5-second** minimum segment duration because a less aggressive cleanup may better preserve short clinically meaningful predictions. The temporal metrics above correspond to the separately evaluated **15-second** setting and should not be reported as metrics for the 5-second configuration.

The cleanup rule merges short segments:

- into the common neighboring phase when both neighbors have the same label;
- otherwise into the longer adjacent segment;
- at a sequence boundary, into the only neighboring segment.

---

## 10. Prediction workflows

The repository contains several prediction scripts for different stages of development.

### `predict_intervals.py`

Use this when `features.npy` already exists.

It:

```text
features.npy
    ↓
MS-TCN
    ↓
post-processing
    ↓
phase intervals
    ↓
JSON
```

### `predict_new_video.py`

End-to-end inference for **one new raw video**:

```text
raw video
    ↓
1 fps sampling
    ↓
ResNet-50
    ↓
MS-TCN
    ↓
temporal cleanup
    ↓
phase intervals
    ↓
JSON / MongoDB
```

### `predict_new_videos_mongo.py`

Main entry point for **all supported videos in a folder**.

Supported extensions currently include:

```text
.mp4
.avi
.mov
.mkv
.m4v
```

For every video it:

1. extracts ResNet-50 features;
2. predicts phases with MS-TCN;
3. applies temporal cleanup;
4. converts predictions to intervals;
5. writes a local JSON result;
6. upserts the video document into MongoDB;
7. continues to the next video if one video fails.

---

## 11. MongoDB integration

The current MongoDB configuration uses:

```text
Database:   surgical_phase_detection
Collection: video_phase_predictions
```

The connection URI is read from an environment variable instead of being committed to source control:

```bash
export MONGO_URI='mongodb://USERNAME:PASSWORD@127.0.0.1:27017/?authSource=admin'
```

Do **not** commit real credentials.

### Example document

A stored video document follows this general structure:

```json
{
  "video_id": "U141-Channel0-part-0001",
  "source_video": "U141-Channel0-part-0001.mp4",
  "model": {
    "visual_encoder": "ResNet50",
    "temporal_model": "MS-TCN",
    "checkpoint": "best_model.pt",
    "checkpoint_epoch": 0,
    "validation_macro_f1": 0.0
  },
  "sampling": {
    "fps": 1,
    "number_timesteps": 1200
  },
  "video_duration": {
    "seconds": 1200.0,
    "time": "00:20:00"
  },
  "postprocessing": {
    "minimum_segment_duration_sec": 5,
    "selected_on": "validation set"
  },
  "number_segments": 5,
  "segments": [
    {
      "phase_id": 1,
      "phase": "Hilar dissection",
      "start_sec": 320.0,
      "end_sec": 610.0,
      "duration_sec": 290.0,
      "start_time": "00:05:20",
      "end_time": "00:10:10",
      "duration": "00:04:50"
    }
  ]
}
```

The numerical seconds are kept for computation and database filtering, while `HH:MM:SS` fields are included for human readability.

### Duplicate handling

The batch pipeline uses an upsert based on `video_id`:

```python
collection.replace_one(
    {"video_id": video_id},
    result,
    upsert=True,
)
```

Running the pipeline again for the same video therefore updates the document instead of intentionally creating another copy.

---

## 12. Example MongoDB queries

Connect with `mongosh`, then:

```javascript
use surgical_phase_detection
```

### Count indexed videos

```javascript
db.video_phase_predictions.countDocuments()
```

### List indexed videos

```javascript
db.video_phase_predictions.find(
  {},
  {
    video_id: 1,
    number_segments: 1,
    "video_duration.time": 1,
    _id: 0
  }
)
```

### Find videos containing a phase

```javascript
db.video_phase_predictions.find(
  {
    "segments.phase": "Tumor resection"
  },
  {
    video_id: 1,
    "segments.$": 1,
    _id: 0
  }
)
```

### Find videos containing a long phase

```javascript
db.video_phase_predictions.find(
  {
    segments: {
      $elemMatch: {
        phase: "Tumor resection",
        duration_sec: { $gt: 300 }
      }
    }
  },
  {
    video_id: 1,
    segments: 1,
    _id: 0
  }
)
```

These queries illustrate the main purpose of the project: turning a video archive into a **semantic surgical-video index**.

---

## 13. Repository structure

Repository layout:

```text
surgical-phase-detection/
├── README.md
├── NoSQL_Project_Overview.md
├── dataset.py
├── mstcn.py
├── preparedataset.py
├── feature.py
├── train.py
├── evaluate.py
├── evaluate_segments.py
├── postprocess_segments.py
├── predict_intervals.py
├── predict_new_video.py
├── predict_new_videos_mongo.py
├── test.py
├── test_dataset.py
├── test_model.py
├── docs/
│   └── surgical_phase_detection_pipeline.png
├── processed/                  # usually excluded from Git
├── training_output/            # model checkpoints; manage separately
├── evaluation_output/
└── new_video_predictions/
```

Use the following filenames in the repository:

```text
mstcn.py
feature.py
```

because the project code imports `from mstcn import MultiStageModel`.

---

## 14. Installation

A Python environment needs the packages used by the project:

```bash
pip install \
  numpy \
  pandas \
  torch \
  torchvision \
  opencv-python \
  pillow \
  scikit-learn \
  matplotlib \
  pymongo \
  openpyxl
```

A CUDA-enabled PyTorch installation is recommended for GPU inference/training where supported.

MongoDB is required only for the database integration scripts.

---

## 15. Typical workflow

### Step 1 — Prepare annotations

Update the paths in `preparedataset.py`, then:

```bash
python preparedataset.py
```

### Step 2 — Extract ResNet-50 features

Update `VIDEO_DIR` / `PROCESSED_DIR` in `feature.py`:

```bash
python feature.py
```

### Step 3 — Validate dataset

```bash
python test_dataset.py
python test_model.py
```

### Step 4 — Train MS-TCN

```bash
python train.py
```

### Step 5 — Evaluate

```bash
python evaluate.py
python evaluate_segments.py
```

### Step 6 — Tune temporal cleanup on validation data

```bash
python postprocess_segments.py
```

### Step 7 — Run folder-level inference and MongoDB indexing

Set the MongoDB URI:

```bash
export MONGO_URI='mongodb://USERNAME:PASSWORD@127.0.0.1:27017/?authSource=admin'
```

Edit `VIDEO_FOLDER`, `PROCESSED_DIR`, and `CHECKPOINT_PATH` in `predict_new_videos_mongo.py`, then:

```bash
python predict_new_videos_mongo.py
```

---

## 16. Configuration notes

Several scripts currently use hard-coded local filesystem paths. Before publishing the repository for wider reuse, consider moving these values to:

- command-line arguments;
- a YAML/JSON config file;
- environment variables.

At minimum, review:

```text
PROCESSED_DIR
VIDEO_DIR
VIDEO_FOLDER
CHECKPOINT_PATH
OUTPUT_DIR
---

## 17. Security and clinical-data note

This repository is a technical system. When applied to real clinical/surgical data:

- keep source videos in approved secure storage;
- store only the metadata needed by the application;
- use authentication and least-privilege database users;
- restrict MongoDB network access;
- keep credentials outside source code;
- follow the applicable organizational, ethical, regulatory, and data-governance requirements.

The current setup uses a local MongoDB deployment so indexing metadata can remain on the controlled machine rather than being sent to a public cloud service.

---

## 18. Main result

The system connects **computer vision**, **temporal action segmentation**, and **NoSQL data management**:

> A new surgical video can be automatically transformed into structured phase intervals and indexed as a searchable MongoDB document.

The architecture supports future functionality such as:

- phase-based video search;
- direct navigation to relevant video intervals;
- phase-duration statistics;
- quality-control dashboards;
- model-version tracking;
- processing-status monitoring;
- integration with wider clinical data-management workflows.

---

