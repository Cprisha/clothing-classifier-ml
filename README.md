# Clothing Classifier ML

A multi-modal PyTorch pipeline for automated clothing attribute classification and text detail matching, combining ResNet-18 visual feature extraction with TF-IDF metric embeddings.

---

## Libraries - How? Where? Which?

- **PyTorch (`torch`, `torch.nn`, `torch.utils.data`)**: Serves as the core deep learning framework. It defines the multi-output architecture (`FashionModel`), constructs custom dataset loaders (`ClothingDataset`), and handles the combined loss computation and backpropagation during training.
- **torchvision**: Supplies the pre-trained `resnet18` backbone used for visual feature extraction, as well as the image transformation pipelines (`transforms`) for dataset augmentation and normalization.
- **pandas**: Used across all scripts for loading, cleaning, filtering, and indexing catalog metadata from CSV files.
- **numpy**: Handles array manipulation, manages random train/validation split shuffling, and saves numerical text embedding matrices as `.npy` binaries.
- **scikit-learn (`TfidfVectorizer`)**: Extracts numerical TF-IDF feature representations from unstructured free-text detail columns (`extra1`, `extra2`, `extra3`).
- **PIL (`Image`)**: Loads catalog images from disk and ensures standard RGB conversion before passing them to visual transform pipelines.
- **joblib**: Serializes dataset mapping dictionaries (`attribute_maps.pkl`) and TF-IDF vectorizers so they can be reloaded seamlessly during inference.
- **OS & RE (`os`, `re`)**: Handles file system path construction for images and dataset artifacts, along with regular expressions for rule-based text cleaning.

---

## Why this code

What a cloth feels like is not necessarily what the cloth actually is. Eyes can be deceiving but alas it's what one sees of your clothes that makes a difference. This was my attempt of using computer vision to describe a single clothing article.
---

## Features

- **Multi-Task Vision Model**: Uses a shared ResNet-18 feature extractor to simultaneously predict multiple structured clothing attributes.
- **Text Embedding Regression**: Maps image features directly into TF-IDF vector spaces using MSE loss regression, enabling similarity matching for unstructured text descriptions.
- **Rule-Based Data Cleaning**: Standardizes messy raw catalog values into normalized categories before feeding them to the network.
- **Cosine Similarity Retrieval**: Matches predicted visual features against dataset text embeddings to surface detailed garment notes (e.g., trims, drapes, or closures).

---

## MVP Structure

- [clean_dataset.py](clean_dataset.py) - Preprocesses raw catalog metadata, normalizes string fields, and standardizes attribute values.
- [train.py](train.py) - Builds TF-IDF vectorizers, processes images, trains the multi-head PyTorch model, and saves model checkpoints.
- [test.py](test.py) - Runs inference on a single test image, predicts categorical tags, and computes vector similarity for unstructured details.
- [clothing_catalog_enhanced_final.csv](clothing_catalog_enhanced_final.csv) - Raw input dataset containing catalog attributes and metadata.

---

## Dataset Setup & CSV Schema

Place your image folder inside the `dataset/` directory and ensure your raw metadata matches the required column structure.

```text
.
├── dataset/
│   └── images/
│       ├── dress0001.jpg
│       └── dress0002.jpg
├── clothing_catalog_enhanced_final.csv
├── clean_dataset.py
├── train.py
└── test.py
```

---

### CSV Schema

| Column Name | Category Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `image_name` | Identifier | Filename inside `dataset/images/` | `dress0001.jpg` |
| `clothing_type` | Profile Tag | Primary garment type | `Gown (Evening/Maxi)` |
| `neckline` | Profile Tag | Garment neckline structure | `Cowl / Sweetheart` |
| `pattern` | Profile Tag | Visual surface pattern | `Solid` |
| `waistline` | Profile Tag | Waist construction style | `Natural` |
| `sleeves` | Profile Tag | Arm coverage / sleeve style | `Cap sleeve` |
| `fabric` | Profile Tag | Material composition | `Chiffon` |
| `silhouette` | Profile Tag | Garment cut and outline | `Mermaid` |
| `extra1`–`extra3` | Free Text | Unstructured textual metadata | `crepe fabric with tulle train` |

---

## How to Run the Project

1. Clone the repository or download the files.
2. Install dependencies.
3. Run [clean_dataset.py](clean_dataset.py) to sort, filter, structure, and clean the dataset.
4. Train the model via [train.py](train.py) to generate feature mappings and saved weights.
5. Run inference using [test.py](test.py) on a test image (e.g., `testsubject.jpg`).
   
---

## Future Upgrades

- Add confidence score thresholds to mask low-certainty predictions as "N/A".
- Support multi-label classification for garments with hybrid patterns or fabric blends.
- Outfit comparison system for events

---

## Notes

- Raw images are not included in this repository due to copyright constraints; populate `dataset/images/` with your own catalog images.
- Do not delete generated `.pkl` or `.npy` files after training—they are required by [test.py](test.py) to map predictions back to text values.
- Loss weight scaling (e.g., text regression multiplier) can be tuned inside [train.py](train.py) depending on dataset size.
- Feel free to explore various epochs, batch sizes, and learning rates in [train.py](train.py).
