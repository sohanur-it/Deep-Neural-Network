# Deep Neural Network — ML/DL Learning Path

A hands-on collection of scripts and notebooks for learning machine learning,
deep learning, and computer vision from the ground up. Everything is organized
into numbered folders meant to be worked through roughly in order — each one
builds on ideas from the previous one.

## How to use this repo

1. Install dependencies: `pip install -r requirements.txt`
2. Start with `01-python-numpy-basics/` and work forward through the folders.
3. Open notebooks with `jupyter notebook` (or JupyterLab); run scripts with
   `python <file>.py`.
4. Every notebook starts with a markdown cell, and every script starts with a
   docstring, explaining what that file demonstrates.

## Learning path

### 01 — Python & NumPy Basics
Array creation, indexing/slicing, and a first look at `sklearn.datasets`.
| File | What it covers |
|---|---|
| `01-numpy-basics.ipynb` | NumPy array creation, indexing, slicing |
| `02-python-and-sklearn-intro.ipynb` | Plain Python basics + first peek at scikit-learn datasets |

### 02 — Data Visualisation
Plotting fundamentals with Matplotlib.
| File | What it covers |
|---|---|
| `01-matplotlib-basics.ipynb` | Line plots, titles, labels, legends |
| `02-histogram-plot-example.py` | Stacked histogram of two overlapping distributions |

### 03 — scikit-learn Fundamentals
The classic ML workflow: load data → split → train → evaluate. Uses the Iris
and handwritten-digits datasets throughout.
| File | What it covers |
|---|---|
| `01-euclidean-distance.py` | `scipy.spatial.distance.euclidean` basics |
| `02-custom-knn-classifier.py` | KNN implemented from scratch, to see how it works internally |
| `03-knn-classifier-iris.py` | `KNeighborsClassifier` with train/test split + accuracy |
| `04-decision-tree-classifier-iris.py` | `DecisionTreeClassifier` on the same split, for comparison |
| `05-knn-classifier-iris-notebook.ipynb` | KNN fit/predict on the full Iris dataset |
| `06-explore-iris-dataset.ipynb` | Inspecting the Iris dataset's structure |
| `07-cross-validation.ipynb` | `cross_val_score` across multiple folds |
| `08-train-test-split-evaluation.ipynb` | `train_test_split` + accuracy scoring |
| `09-choosing-best-k-for-knn.ipynb` | Sweeping `n_neighbors` to pick the best k |
| `10-linear-classifier-iris.ipynb` | Feature scaling with `StandardScaler` + a linear classifier |
| `11-digit-classification-intro.ipynb` | Loading and visualizing the digits dataset |
| `12-knn-classifier-digits.ipynb` | KNN applied to handwritten digit images |

### 04 — TensorFlow Basics
| File | What it covers |
|---|---|
| `01-mnist-handwritten-digits.py` | Legacy TF1 MNIST setup (kept as historical reference — see file for compatibility notes) |

### 05 — PyTorch Basics
| File | What it covers |
|---|---|
| `01_tensors.py` | Tensor creation, ops, reshaping, NumPy interop, device check |
| `02_autograd.py` | Gradient tracking and `.backward()` |
| `03_simple_nn.py` | A minimal `nn.Module` feedforward network |
| `04_train_iris.py` | Full training loop on the Iris dataset |
| `06_card_image_classifier.ipynb` | Playing-card image classifier: `ImageFolder` dataset, pretrained `timm` EfficientNet-B0, train/val loop, loss plot, predictions |

### 06 — Computer Vision with OpenCV
Webcam capture, face detection, and motion detection, building in complexity.
| File | What it covers |
|---|---|
| `01-image-show.ipynb` | Reading and displaying a static image |
| `02-face-detection-opencv.ipynb` | Haar cascade face detection on a static image |
| `03-image-capture-webcam.ipynb` | Grabbing a single frame from the webcam |
| `04-video-capture-webcam.ipynb` | Webcam video capture (scratch) |
| `05-video-capture-online-example.ipynb` | Live grayscale webcam feed loop |
| `06-motion-detector.ipynb` | Frame-diffing motion detector with logged timestamps |
| `07-face-recognition-webcam.ipynb` | Real-time face detection on a live webcam feed |
| `08-face-recognition-training.ipynb` | Building a labeled image list for face-recognition training |
| `09-scratchpad.ipynb` | Misc. experiment notebook |

## Stack

Python 3, NumPy, pandas, Matplotlib/Seaborn, scikit-learn, TensorFlow, PyTorch,
OpenCV, Jupyter.

## Reference links

- [Iris ML walkthrough (Colab)](https://colab.research.google.com/drive/1aCb1hSTPjJhVP09_QTaPW_GJmeltjTHU)
- [Iris flower dataset with scikit-learn (Colab)](https://colab.research.google.com/drive/1Obln0GBeOcbsv0HHcaPAwYsiIOIL-ykL)
- [Iris/fashion dataset with TensorFlow (Colab)](https://colab.research.google.com/drive/1MtmmG0SxXUHDKwXSvYC2NGzGozpglUaM)
