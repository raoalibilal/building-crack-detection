# Building Crack Detection and Localization using Deep Learning

**7th Semester Deep Learning Project, Superior University (Gold Campus)**

**Student:** Ali Bilal (SU92-BSCSM-F23-213)  
**Supervisor:** Dr. Munir

A deep learning system that checks photos of walls, floors and concrete surfaces. It tells **whether there is a crack**, shows **exactly where the crack is** (highlighted in red), and gives the **crack area and a rough severity level**.

## Problem
Cracks are early warning signs of structural damage in buildings, bridges and roads. Manual inspection is slow, costly and depends on the inspector. This project automates it with deep learning.

## What the system does
| Step | Task | Model |
|---|---|---|
| 1 | Crack / No crack | EfficientNetB0 (transfer learning), compared with a Custom CNN baseline |
| 2 | Crack location (pixel level) | U-Net with a pretrained MobileNetV2 encoder |
| 3 | Report | Crack area % and severity: Minor (under 1%), Moderate (1 to 3%), Severe (over 3%) |

Large photos are split into 256x256 tiles so thin cracks are not lost.

## Datasets
1. **Concrete Crack Images for Classification** (Özgenel, 2018): 40,000 images (227x227), 20,000 crack and 20,000 no crack. Split 24,000 train / 8,000 validation / 8,000 test.
2. **Concrete crack segmentation dataset**: 800 images (256x256) with hand-labelled crack masks. Split 640 / 80 / 80.

Both download automatically inside the notebook.

## Methods
- **Data augmentation:** flips, rotation, brightness and contrast changes
- **Classification:** EfficientNetB0 pretrained on ImageNet, trained in two stages (head first, then fine-tuning the top layers with a small learning rate). Baseline: a 3-block CNN trained from scratch.
- **Segmentation:** U-Net (encoder-decoder with skip connections). Loss = Binary cross-entropy + Dice loss, which works well for thin objects like cracks.
- **Metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC (classification). Dice and IoU (segmentation).

## Results
Saved automatically in `results/` after training:
| File | Shows |
|---|---|
| `metrics.json` | All final numbers (accuracy, AUC, Dice, IoU, model size, training time) |
| `classification_report.txt` | Precision, recall, F1 per class |
| `confusion_matrix_roc.png` | Confusion matrix and ROC curve |
| `model_comparison.png` | Custom CNN vs EfficientNetB0 |
| `cnn_curves.png`, `effnet_curves.png`, `unet_curves.png` | Training curves |
| `segmentation_results.png` | Image, true mask, predicted mask, overlay |
| `inspector_demo.png` | Full system output |

## Live website
Project page: https://building-crack-detection.vercel.app

## Project structure
```
Building_Crack_Detection.ipynb   full pipeline: data, training, evaluation, demo
src/predict.py                   run the Crack Inspector on any photo
index.html                       project website (Vercel)
models/                          trained models (.keras)
results/                         charts, metrics, reports
requirements.txt
```

## How to run
**Train on Google Colab**
1. Open the notebook in Colab: [Open in Colab](https://colab.research.google.com/github/raoalibilal/building-crack-detection/blob/main/Building_Crack_Detection.ipynb)
2. `Runtime > Change runtime type > T4 GPU`
3. `Runtime > Run all` (about 25 to 35 minutes)
4. The notebook downloads `trained_outputs.zip` with the models and results
5. The last cell starts an optional live web demo (public link) for the viva

**Use the trained model on your computer**
```bash
pip install -r requirements.txt
python src/predict.py wall_photo.jpg
```
Output: result, confidence, crack area, severity, and `wall_photo_result.jpg` with cracks in red.

## Limitations
- Trained on concrete surfaces; painted, tiled or very dirty walls may give false alarms.
- Severity is based on crack area only, not depth or width in real units. It is a screening tool, not a replacement for a structural engineer.

## Tech stack
Python, TensorFlow / Keras, EfficientNetB0, U-Net, MobileNetV2, scikit-learn, Matplotlib, Gradio, Google Colab
