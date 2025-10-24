# License Plate Recognition — Part 2

**Character Segmentation, CNN Recognition & Batch Inference**

This repository contains the **second stage** of a two-part License Plate Recognition (LPR) system:

* **Part 1 (not in this repo):** License plate detection using a Haar cascade to localize the plate region and export `haar_carplate.xml`.[→ Go to Part 1: Build a Haar feature cascade](https://github.com/lok-bit/haar-training-tool)
* **Part 2 (this repo):** From a cropped plate image to **character segmentation**, **CNN-based character recognition**, and **batch processing** for multiple images.

> This repo assumes you already have `haar_carplate.xml` from Part 1.

---

## Pipeline (This Repo)

1. **Plate ROI Detection (Haar cascade)** – use `haar_carplate.xml` to crop the plate region.
2. **Preprocessing** – grayscale → denoise → adaptive threshold → morphology.
3. **Character Segmentation** – contour filtering, geometric checks, left-to-right sorting, ROI normalization (e.g., 28×28).
4. **CNN Training** – train a small CNN for alphanumeric (0–9, A–Z, excluding I/O if desired).
5. **Inference** – predict characters and assemble the final plate string.
6. **Batch Inference** – run over a folder and print results to the console.

---

## Notes & Implementation Details

* **Preprocessing:** grayscale → Gaussian blur → adaptive threshold → morphology (open/close).
* **Segmentation:** contour filtering by aspect ratio/area/height; sort left→right.
* **CNN :** `Conv2D(20,5×5)+ReLU` → `MaxPool(2×2)` → `Conv2D(50,5×5)+ReLU` → `MaxPool(2×2)` → `Flatten` → `Dense(500)+ReLU` → `Dense(34)+Softmax`.
* **Evaluation:** character accuracy, confusion matrix, and per-plate exact match rate.
* **Locale:** alphanumeric plates by default—extend label set as needed.

## Dataset & Labels

* **Classes:** 34 (digits 0–9, letters A–Z **excluding I and O**). :contentReference[oaicite:9]{index=9}
* **Character crops:** normalized to **18×38**; for very narrow glyphs (e.g., **'1'**) we pad left/right with white margins to avoid shape distortion after resizing. :contentReference[oaicite:10]{index=10}
* **Augmentation:** per-class images expanded to ~**500** by adding **random noise dots** to simulate dirt/imperfections on plates. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

## Implementation Details

* **Preprocessing:** grayscale → denoise → adaptive threshold → morphology (open/close).  
* **Segmentation:** contour detection → bounding rect → geometric filtering (aspect/area/height) → left-to-right sorting → resize to 18×38. :contentReference[oaicite:13]{index=13}
* **CNN Training:** train/val split **80%/20%**, **batch=32**, **epochs=10** (small CNN for 34-way classification). :contentReference[oaicite:14]{index=14}

## Evaluation & Known Limitations

* In a small test set, **five sample images were all recognized**; however, with two **real-world** photos, letters **'N'** and **'R'** were misclassified. Root cause: the **plate was tilted**, and the upstream Haar-based crop did **not fully capture** the plate region before segmentation. :contentReference[oaicite:15]{index=15}

## Troubleshooting Notes

* If segmentation misses characters, tune contour **aspect/area thresholds** and visualize with `cv2.rectangle()` to iterate quickly. :contentReference[oaicite:16]{index=16}
* For the upstream detector, increasing **`minNeighbors` to 5** improved stability (fewer false boxes / missing plates in our tests). :contentReference[oaicite:17]{index=17}

---

## Acknowledgements

* OpenCV for image processing & contour analysis
* TensorFlow/Keras for CNN training
* Haar cascade (`haar_carplate.xml`) produced in Part 1
