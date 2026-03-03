# Funko Pop CNN (MansooNet) — Image Attribute Classification with Imbalance Handling

![Funko Pop example](https://github.com/cmansoo/funko-pop-cnn/assets/70994968/560b883f-ff72-4424-9705-9d99768fee59)

## Definition of a “Funko Pop!”
Funko Pop! figures are small collectible figurines known for their exaggerated, stylized features. They depict popular characters from a wide range of franchises and genres.  
Source: https://toysforapound.com

---

## Goal
Design an AI/computer vision model to identify visual attributes from Funko Pop images, including:
- Hair / facial hair  
- Item in hand  
- Gender (multi-class)  

---

## Real-World Application
This project mirrors real-world manufacturing and operations use cases—e.g., **AI-enabled visual inspection** to detect **defects** or **attribute mismatches** on production lines to improve quality control and efficiency.

---

## Dataset
- **256 total images**: 32 Funko Pops × 8 angles  
- Most imbalanced/complex label: **Gender** (3 classes)

### Class distribution (project dataset)
| Attribute | Class Counts |
|---|---|
| Human / Non-Human | Human: 19, Non-Human: 13 |
| Gender | Male: 18, Female: 1, Other: 13 |
| Facial Hair | Yes: 6, No: 26 |
| Glasses | Yes: 2, No: 30 |
| Hat | Yes: 7, No: 25 |
| Item in Hand | Yes: 17, No: 15 |

> Labels and image paths are managed via `funko_file_map.csv`.  
> (Images are stored in a shared Google Drive and are not included in this GitHub repo.)

---

## Approach

### 1) MansooNet — CNN from scratch
A custom CNN architecture (“**MansooNet**”) was implemented and trained using image augmentation and a train/validation/test workflow.

**Model structure (high-level):**  
**4 Convolutional layers → Flatten → Dense stack → Output layer**

**Key specs (as implemented):**
- Input size: **(224, 224, 3)**
- Filters: **72 → 144 → 216 → 360**
- Kernel sizes: **(11×11) → (7×7) → (5×5) → (3×3)**
- Activations: ReLU
- Pooling: MaxPooling (3×3) and (2×2)
- Dense stack: multiple layers with dropout
- Total parameters: **~47M trainable parameters** (dense-heavy design)

---

### 2) Key contribution — SMOTE on flattened hidden features (embedding space)
A major challenge was severe class imbalance (especially for **Gender**). Instead of attempting to oversample in pixel space, we used SMOTE in **CNN embedding space**:

1. Train MansooNet up to the **flatten** layer (hidden feature representation)
2. Extract **flattened hidden features** for each image
3. Apply **SMOTE** to oversample minority classes **in embedding space**
4. Train a classifier head on the balanced embedding dataset

**Why this works better than pixel-level SMOTE:**  
SMOTE behaves more sensibly in a learned feature space than on raw pixels, and it directly targets class imbalance without generating unrealistic images.

---

## Key Insights
- For small image datasets, **class imbalance can dominate performance** and cause models to overfit to the majority class.
- A practical solution is to combine **representation learning (CNN)** with classical imbalance handling (**SMOTE**) in embedding space.
- This approach is a useful pattern for real-world scenarios like **quality inspection**, where rare classes (defects) are underrepresented.

---

## Repository Contents
- `MansooNet_gender_and_initial_training.ipynb` — baseline training and model development (gender + initial experiments)
- `MansooNet_SMOTE.ipynb` — embedding extraction + SMOTE + post-SMOTE model training/evaluation
- `grid_search_example_using_a_loop.ipynb` — experimental hyperparameter loop example
- `funko_file_map.csv` — label + image path mapping (paths point to shared Drive)
​

<!--![image](https://github.com/cmansoo/funko-pop-cnn/assets/70994968/560b883f-ff72-4424-9705-9d99768fee59)-->
