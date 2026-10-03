# Exercise 23 — Case Studies: Image Classification and Object Detection

---

## Section A — Multiple Choice

**1.** In the image classification pipeline, what is the role of the softmax layer?

- A) It extracts spatial features from the input image
- B) It converts raw output scores into a probability distribution over classes
- C) It flattens the feature volume into a 1D vector
- D) It applies non-maximum suppression to remove duplicate predictions

&nbsp;

**2.** A detection model outputs the bounding box (cx=0.5, cy=0.4, w=0.3, h=0.2) for a detected object. What do these coordinates represent?

- A) The pixel coordinates of the top-left and bottom-right corners
- B) The normalised centre coordinates, width, and height relative to image dimensions
- C) The filter size and stride used in the final convolutional layer
- D) The class probabilities for the four most likely categories

&nbsp;

**3.** Why is sliding window detection computationally expensive?

- A) It requires training a separate CNN for each window size
- B) It applies a full CNN forward pass to thousands of overlapping windows per image
- C) It cannot handle images larger than 224×224 pixels
- D) It requires ground truth bounding boxes at test time

&nbsp;

**4.** In YOLO, which grid cell is responsible for detecting an object?

- A) The cell with the largest anchor box
- B) The cell closest to the top-left corner of the bounding box
- C) The cell whose centre is closest to the image centre
- D) The cell that contains the centre of the object's bounding box

&nbsp;

**5.** What does non-maximum suppression (NMS) do in object detection?

- A) Removes anchor boxes that are too small to contain an object
- B) Discards predicted boxes with low classification confidence before training
- C) Keeps the highest-confidence box and removes overlapping boxes above an IoU threshold
- D) Normalises bounding box coordinates to the range [0, 1]

&nbsp;

---

## Section B — Short Answer

**1.** Explain why object detection is harder than image classification. Name two specific properties of detection that make it more difficult.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** What is the purpose of anchor boxes in region-based detection? How do they differ from the windows used in sliding window detection?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** YOLO produces a fixed-size output tensor regardless of how many objects are in the image. Explain how this is possible and what the output tensor dimensions represent.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** A classification model outputs softmax probabilities [0.72, 0.21, 0.07] for classes [cat, dog, bird]. The model is deployed in a system that must also handle images with no recognisable object. What is the fundamental limitation of this setup, and how might it be addressed?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation

A detection model predicts a bounding box for a cat. The ground truth box and the predicted box are given in corner format (x_min, y_min, x_max, y_max), with coordinates in pixels on a 10×10 grid.

```
Ground truth box A:   (0, 0, 6, 4)
Predicted box B:      (2, 1, 8, 5)
```

**(i)** Compute the area of box A and the area of box B.

&nbsp;

&nbsp;

&nbsp;

**(ii)** Identify the coordinates of the intersection rectangle. Compute its area.

&nbsp;

&nbsp;

&nbsp;

**(iii)** Compute the area of the union of A and B.

&nbsp;

&nbsp;

&nbsp;

**(iv)** Compute IoU = intersection / union. Express as a fraction and as a decimal.

&nbsp;

&nbsp;

&nbsp;

**(v)** A detection is considered correct if IoU ≥ 0.5. Is this prediction correct? What does this IoU value tell you about the quality of the predicted box?

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** YOLO is significantly faster than Faster R-CNN but was originally less accurate, particularly on small objects. Explain why the grid cell design causes YOLO to struggle with small objects, and describe what architectural change would help.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A student argues: "Sliding window detection is obsolete — region proposal networks are strictly better because they are faster and more accurate." Evaluate this claim. Is there any scenario where sliding window remains the right approach?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
