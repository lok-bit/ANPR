# License Plate Recognition — Part 2

**Character Segmentation, CNN Recognition & Batch Inference**

This repository contains the **second stage** of a two-part License Plate Recognition (LPR) system:

* **Part 1 (not in this repo):** License plate detection using a Haar cascade to localize the plate region and export `haar_carplate.xml`.
* **Part 2 (this repo):** From a cropped plate image to **character segmentation**, **CNN-based character recognition**, and **batch processing** for multiple images.

> This repo assumes you already have `haar_carplate.xml` from Part 1.

---

## Pipeline (This Repo)

1. **Plate ROI Detection (Haar cascade)** – use `haar_carplate.xml` to crop the plate region.
2. **Preprocessing** – grayscale → denoise → adaptive threshold → morphology.
3. **Character Segmentation** – contour filtering, geometric checks, left-to-right sorting, ROI normalization (e.g., 28×28).
4. **CNN Training** – train a small CNN for alphanumeric (0–9, A–Z).
5. **Inference** – predict characters and assemble the final plate string.
6. **Batch Inference** – run over a folder and print results to the console.

---

## Notes & Implementation Details

* **Preprocessing:** grayscale → Gaussian blur → adaptive threshold → morphology (open/close).
* **Segmentation:** contour filtering by aspect ratio/area/height; sort left→right.
* **CNN (reference):** `Conv-BN-ReLU`×2 → MaxPool → Dropout → Dense → Softmax (36 classes).
* **Evaluation:** character accuracy, confusion matrix, and per-plate exact match rate.
* **Locale:** alphanumeric plates by default—extend label set as needed.

---

## Acknowledgements

* OpenCV for image processing & contour analysis
* TensorFlow/Keras for CNN training
* Haar cascade (`haar_carplate.xml`) produced in Part 1
