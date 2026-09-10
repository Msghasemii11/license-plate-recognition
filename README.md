# License Plate Recognition

A computer vision and machine learning project for recognizing digits on a license plate.

## Technologies

- Python
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Logistic Regression

## How it works

1. Loads digit images from the dataset.
2. Resizes each digit to **8 × 32 pixels**.
3. Converts each image to a 256-feature vector.
4. Splits the dataset into training and test sets.
5. Trains a Logistic Regression classifier.
6. Preprocesses the license-plate image with OpenCV.
7. Uses vertical pixel projection to find possible digit regions.
8. Resizes each detected region to 8 × 32 pixels.
9. Predicts the digits.
10. Displays the detected plate and model accuracy.

## Dataset structure

```text
pelak/
├── 1/
├── 2/
├── 3/
├── 4/
├── 5/
├── 6/
├── 7/
├── 8/
└── 9/
```

Each folder should contain images of that digit.

**Note:** The current code trains on digits 1–9. For digit 0, add a `0` folder and change `range(1, 10)` to include it.

## Installation

```bash
pip install -r requirements.txt
```

## Run

Place the test image next to the Python file and name it:

```text
pelak.jpg
```

Then run:

```bash
python license_plate_recognition.py
```

The program prints the model accuracy and predicted plate number.

## Project structure

```text
license-plate-recognition/
├── license_plate_recognition.py
├── requirements.txt
├── .gitignore
├── sample_plate.jpg
└── pelak/
    ├── 1/
    ├── 2/
    ├── 3/
    ├── 4/
    ├── 5/
    ├── 6/
    ├── 7/
    ├── 8/
    └── 9/
```

## Model

**Logistic Regression**

Input size: **8 × 32 = 256 features**

The test accuracy is calculated automatically when the program runs.

## Future improvements

- Add digit 0
- Improve digit segmentation
- Support Persian license-plate characters
- Use a larger and more diverse dataset
- Compare Logistic Regression with CNN models
- Improve robustness to lighting and different plate layouts

## Author

Machine learning and computer vision portfolio project.

watch the demo.mp4
