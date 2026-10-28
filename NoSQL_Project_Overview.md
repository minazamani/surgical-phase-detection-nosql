# NoSQL and MongoDB in the Surgical Phase Detection Project

## 1. Why this document exists

This document explains two things:

1. the **NoSQL concepts** relevant to this project;
2. how those concepts were applied to build a searchable surgical-video index using MongoDB.

The project combines:

```text
Machine Learning / Computer Vision
              │
              ▼
    Surgical Phase Recognition
              │
              ▼
         Phase intervals
              │
              ▼
        Data Engineering
              │
              ▼
            MongoDB
              │
              ▼
   Semantic surgical-video index
```

The main objective is not to put large video files inside the database. Instead, the database stores **structured metadata and model-derived temporal information**, allowing the video archive to be queried without manually opening every video.

---

# Part I — NoSQL concepts

## 2. What is NoSQL?

**NoSQL** is commonly interpreted as **“Not Only SQL.”**

It refers to a family of non-relational database technologies rather than one single database model.

Traditional relational databases usually organize information in:

```text
tables
  ├── rows
  └── columns
```

and relationships between tables are typically represented using keys.

NoSQL systems can represent information differently and are often selected when applications benefit from:

- flexible data structures;
- nested or hierarchical records;
- horizontal scalability;
- high-throughput workloads;
- distributed architectures;
- application-oriented data models.

NoSQL does **not** automatically replace relational databases. The choice should depend on the data, required queries, relationships, consistency requirements, deployment model, and operational constraints.

---

## 3. Main NoSQL models

Four commonly used NoSQL categories are:

### 3.1 Document databases

Store records as JSON-like or BSON documents.

Examples:

- MongoDB
- CouchDB

A document can contain:

- fields;
- nested objects;
- arrays;
- related metadata.

This is the model used in this project.

### 3.2 Key-value databases

Each unique key points to a value.

Typical uses include:

- caching;
- session storage;
- very fast key-based lookup.

Redis is a common example.

### 3.3 Wide-column databases

Designed around rows, keys, and column families, often for large distributed datasets and predefined access patterns.

Examples include:

- Cassandra
- HBase

### 3.4 Graph databases

Represent data as nodes and relationships as edges.

They are particularly useful when the main task is exploring connections, paths, or networks of entities.

Neo4j is a common example.

---

## 4. Relational versus document-oriented modeling

A relational system typically normalizes information into multiple related tables.

A document database can instead keep information that is commonly accessed together inside a single nested record.

For this project, the natural application object is:

```text
one surgical video
```

The information that is normally requested together includes:

```text
video metadata
model metadata
sampling metadata
post-processing parameters
predicted phase intervals
```

These fields form a coherent document, which makes MongoDB's document model a reasonable fit for the demonstrator.

---

## 5. Flexible schema does not mean “no design”

A MongoDB collection can contain documents with differing fields, but a production-quality application still needs clear rules.

Useful design questions include:

- What does one document represent?
- Which fields are required?
- Which identifiers must remain stable?
- Which values are numeric versus human-readable strings?
- Which arrays can grow?
- Which queries will be executed frequently?
- Which fields should be indexed?
- How will schema changes be versioned?

The prototype uses a consistent representation where **one document represents one predicted video**.

---

## 6. Embedding versus references

### Embedding

Embedding stores related information inside the same document.

The project embeds phase intervals inside the video document:

```json
{
  "video_id": "example",
  "segments": [
    {
      "phase": "Tumor resection",
      "start_sec": 100,
      "end_sec": 400
    }
  ]
}
```

This is useful because the intervals belong directly to that prediction and are normally retrieved together with the video metadata.

### References

References would instead store an identifier pointing to another document.

References can be useful when information:

- is shared by many records;
- changes independently;
- is large;
- should be managed centrally.

For example, a future system might maintain model definitions in a separate `models` collection and reference a `model_version_id`.

---

## 7. CRUD

CRUD stands for:

- **Create**
- **Read**
- **Update**
- **Delete**

Examples for this project:

### Create / update

The Python application uses `replace_one(..., upsert=True)`.

An **upsert** means:

- update the matching video if it exists;
- insert a new document if it does not.

This supports repeatable processing without intentionally creating duplicate records.

### Read

Example:

```javascript
db.video_phase_predictions.findOne(
  { video_id: "U141-Channel0-part-0001" }
)
```

### Delete

A demonstration record could be removed with:

```javascript
db.video_phase_predictions.deleteOne(
  { video_id: "example-video" }
)
```

Deletion should be handled carefully in real data-management workflows.

---

## 8. MongoDB aggregation

MongoDB's aggregation pipeline can transform and summarize documents.

Important stages include:

- `$match` — filter records;
- `$group` — group records and calculate summaries;
- `$project` — reshape returned fields;
- `$sort` — order results;
- `$unwind` — expand array elements such as phase segments.

For example, a future analysis can calculate how often each predicted phase appears:

```javascript
db.video_phase_predictions.aggregate([
  { $unwind: "$segments" },
  {
    $group: {
      _id: "$segments.phase",
      occurrences: { $sum: 1 }
    }
  },
  { $sort: { occurrences: -1 } }
])
```

Or calculate average predicted phase duration:

```javascript
db.video_phase_predictions.aggregate([
  { $unwind: "$segments" },
  {
    $group: {
      _id: "$segments.phase",
      average_duration_sec: {
        $avg: "$segments.duration_sec"
      }
    }
  },
  { $sort: { average_duration_sec: -1 } }
])
```

---

## 9. Indexes

An index helps MongoDB find records for supported queries without scanning the entire collection.

Indexes improve reads but:

- consume storage;
- require maintenance on writes;
- should be chosen according to real access patterns.

For this project, useful candidate indexes include:

```javascript
db.video_phase_predictions.createIndex(
  { video_id: 1 },
  { unique: true }
)

db.video_phase_predictions.createIndex(
  { "segments.phase": 1 }
)

db.video_phase_predictions.createIndex(
  { "segments.start_sec": 1 }
)
```

The single-video inference version explicitly creates these indexes. The batch demonstrator relies primarily on the stable `video_id` upsert and can be extended to create the same indexes during initialization.

---

# Part II — Project implementation

## 10. Problem addressed

Long surgical videos contain multiple workflow phases.

Without temporal indexing, answering questions such as:

> Which videos contain tumor resection?

or:

> At what time does kidney release occur?

requires manual video inspection or separate annotation files.

The project creates a machine-readable semantic index:

```text
Video
  ↓
phase recognition
  ↓
temporal intervals
  ↓
MongoDB document
  ↓
searchable video collection
```

---

## 11. End-to-end pipeline

The implemented pipeline is:

```text
Surgical video
      ↓
sample at 1 fps
      ↓
pretrained ResNet-50
      ↓
2048-D visual feature / second
      ↓
MS-TCN
      ↓
phase prediction / second
      ↓
temporal post-processing
      ↓
phase intervals
      ↓
local JSON + MongoDB
```

This connects the AI output directly with the data-management layer.

---

## 12. Data preparation

The annotation spreadsheet contains:

- surgery identifier;
- video / fragment identifier;
- start time;
- end time;
- surgical phase;
- surgical tag.

`preparedataset.py` transforms these annotations into machine-learning timelines.

At 1 Hz:

```text
second 0 → phase
second 1 → phase
second 2 → phase
...
```

Unlabeled / ignored regions use:

```text
-100
```

which is ignored by the PyTorch classification loss.

Splitting is performed by complete surgery rather than fragment so that fragments of the same surgery do not appear in different train/test partitions.

---

## 13. Visual representation

The project uses a pretrained ResNet-50.

For every sampled second:

```text
frame
  ↓
ResNet-50
  ↓
2048-dimensional feature
```

Thus a video becomes:

```text
T × 2048
```

where `T` is approximately the video duration in seconds.

---

## 14. Temporal phase recognition

The visual sequence is processed with MS-TCN.

The current configuration is:

- 4 stages;
- 10 temporal layers per stage;
- 64 feature maps;
- 2048-D input;
- 7 current phase classes.

MS-TCN uses dilated temporal convolutions and repeated refinement stages to model the sequence and improve temporal predictions.

---

## 15. Training strategy

The baseline is trained using:

1. cross-entropy classification loss;
2. temporal smoothing loss.

The smoothing term discourages unnecessarily abrupt changes between neighboring predictions.

Ignored annotation positions are excluded.

The model checkpoint with the highest validation Macro-F1 is retained.

---

## 16. Model performance

The retained unweighted baseline achieved:

| Metric | Test |
|---|---:|
| Accuracy | **79.04%** |
| Macro-F1 | **80.10%** |
| Weighted-F1 | **79.39%** |

Best validation Macro-F1 was approximately:

```text
88.03%
```

The experiments showed that class weighting could improve some weaker phases, particularly tumor preparation, but reduced overall discrimination between neighboring phases. Therefore the unweighted MS-TCN was retained as the baseline.

---

## 17. Why temporal post-processing was needed

The model can produce short unstable segments such as:

```text
Hilar dissection
Hilar dissection
Kidney release        ← very short prediction
Hilar dissection
Hilar dissection
```

This can preserve reasonable per-second accuracy while producing poor phase intervals.

The post-processing algorithm:

1. identifies segments shorter than a selected duration;
2. if both neighbors have the same phase, merges the short segment into that phase;
3. otherwise merges it into the longer neighboring phase.

This substantially improved temporal segmentation performance in the 15-second evaluation experiment.

### Raw temporal results

```text
Edit       55.61
F1@10      52.51
F1@25      50.15
F1@50      43.07
```

### Test results after the validation-selected 15-second cleanup

```text
Edit       67.49
F1@10      70.12
F1@25      66.93
F1@50      57.37
```

The production-style batch inference prototype currently uses a **5-second** threshold to make cleanup less aggressive. That setting has not been assigned the 15-second evaluation metrics above.

---

## 18. Conversion from predictions to database records

After prediction, consecutive seconds with the same phase are combined.

Instead of storing:

```text
second 100 → Hilar
second 101 → Hilar
second 102 → Hilar
...
```

the system stores:

```json
{
  "phase": "Hilar dissection",
  "start_sec": 100,
  "end_sec": 320,
  "duration_sec": 220,
  "start_time": "00:01:40",
  "end_time": "00:05:20",
  "duration": "00:03:40"
}
```

This representation is compact, readable, and practical for video navigation.

---

## 19. MongoDB data model

The current database is:

```text
surgical_phase_detection
```

and the collection is:

```text
video_phase_predictions
```

One document represents one video prediction.

General structure:

```json
{
  "video_id": "...",
  "source_video": "...",
  "model": {
    "visual_encoder": "ResNet50",
    "temporal_model": "MS-TCN",
    "checkpoint": "best_model.pt",
    "checkpoint_epoch": 0,
    "validation_macro_f1": 0.0
  },
  "sampling": {
    "fps": 1,
    "number_timesteps": 0
  },
  "video_duration": {
    "seconds": 0,
    "time": "00:00:00"
  },
  "postprocessing": {
    "minimum_segment_duration_sec": 5
  },
  "number_segments": 0,
  "segments": []
}
```

This is a document-oriented design because the video and all phase intervals that belong to its prediction are represented together.

---

## 20. Why the video is not stored in MongoDB

The source surgical video can be large and sensitive.

The design therefore separates:

### File / video storage

Stores:

```text
MP4 / video data
```

### MongoDB

Stores:

```text
video identifier
source filename
model information
duration
sampling information
phase intervals
```

This lets users search and summarize the archive without loading every video.

It is also aligned with the broader NoSQL/data-management concept in which large binary source objects remain in controlled file/object storage while the database holds their searchable metadata.

---

## 21. Python-to-MongoDB connection

The project uses PyMongo:

```python
from pymongo import MongoClient
```

The MongoDB connection URI is kept outside the code:

```python
MONGO_URI = os.getenv("MONGO_URI")
```

Example environment configuration:

```bash
export MONGO_URI='mongodb://USERNAME:PASSWORD@127.0.0.1:27017/?authSource=admin'
```

This avoids committing a real password to GitHub.

The batch pipeline verifies the connection using:

```python
client.admin.command("ping")
```

and stores each result using an upsert.

---

## 22. Useful queries implemented / demonstrated

### Number of indexed videos

```javascript
db.video_phase_predictions.countDocuments()
```

### Video summary

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

### Videos containing kidney release

```javascript
db.video_phase_predictions.find(
  {
    "segments.phase":
      "Kidney release / fatty detachment"
  },
  {
    video_id: 1,
    "segments.$": 1,
    _id: 0
  }
)
```

### Videos with a phase longer than five minutes

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

These examples demonstrate that the database is not only a storage destination: it enables clinically meaningful temporal search of the processed video collection.

---

## 23. Security choices in the demonstrator

The MongoDB setup was designed to remain local rather than using an external public database.

Important principles are:

- bind the database to controlled interfaces;
- enable authentication;
- keep credentials outside code;
- use least-privilege users where possible;
- do not commit passwords to Git;
- do not store unnecessary patient information;
- keep source videos in approved storage;
- use approved encryption / backup procedures for production data.

For a real clinical deployment, organizational security and regulatory requirements take precedence over this prototype setup.

---

## 24. NoSQL value demonstrated by the project

The project demonstrates several practical NoSQL concepts.

### Flexible nested documents

Model output naturally contains nested objects and an array of phase intervals.

### Application-oriented modeling

The schema is designed around the main unit users want to retrieve:

```text
one video + its timeline
```

### Upserts

The same video can be reprocessed and updated without intentionally creating duplicate documents.

### Queryable arrays

MongoDB can directly query phase names and durations inside the `segments` array.

### Metadata indexing

The system converts otherwise difficult-to-search video content into a semantic index.

### Separation of content and metadata

Large media files remain outside MongoDB while searchable information is stored in the database.

---

## 25. What has been achieved

The prototype currently connects all of the following:

```text
Excel phase annotations
        ↓
dataset preparation
        ↓
video sampling
        ↓
ResNet-50 features
        ↓
MS-TCN training
        ↓
test evaluation
        ↓
temporal evaluation
        ↓
post-processing
        ↓
new-video inference
        ↓
folder-level batch processing
        ↓
JSON outputs
        ↓
MongoDB upsert
        ↓
semantic queries
```

This moves the project beyond a standalone AI model.

It is now an **end-to-end surgical phase detection and data-management demonstrator**.

---

## 26. Possible next steps

### Data engineering

- MongoDB JSON schema validation;
- dedicated application user rather than administrator credentials;
- explicit model-version identifiers;
- processing timestamps;
- prediction status and error fields;
- source-video storage URI;
- automatic index initialization;
- backup / restore procedure.

### Analytics

- average duration per phase;
- phase occurrence counts;
- phase distribution by surgery;
- processing success rate;
- model version comparison.

### Visualization

A dashboard could show:

```text
video
timeline
phase intervals
duration per phase
model version
processing status
```

### Machine learning

- video / motion-aware encoders;
- multi-task phase + tag prediction;
- confidence scores;
- improved boundary modeling;
- automatic optimization of clinically appropriate cleanup thresholds.

---

## 27. Summary

The central idea of the project is:

> **Use AI to convert unstructured surgical video into structured temporal metadata, then use a document database to make that metadata searchable and operationally useful.**

The AI side answers:

```text
What phase is occurring, and when?
```

The NoSQL side answers:

```text
How can those predictions be stored, updated, searched, summarized, and used by downstream applications?
```

Together they form the first version of a semantic surgical-video indexing platform.
