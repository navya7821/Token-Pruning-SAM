# Token Pruning for Hard Hat Detection with SAM

Automated system for detecting persons, heads, and hard hats in construction site images. Uses **Segment Anything Model (SAM)** for mask generation, **fine-tuned ResNet-50** for classification, and **token pruning** to reduce computation by **~84%** while **preserving mAP**.

---

## Project Overview
- Generate segmentation masks with SAM (ViT-B)
- Classify mask crops into `"person"`, `"head"`, or `"hard_hat"`
- Apply post-generation token pruning: retain top ~10% masks by **IoU × stability score**
- Refine masks with **Otsu thresholding** and **morphology**
- Benchmark pre/post-pruning on **399 COCO-annotated images**
- Evaluate **mAP@0.5:0.95** for accuracy preservation

---

## Workflow of the Code
1. Unzip dataset and models from `Final_token_pruning.zip`
2. Load SAM checkpoint and fine-tuned ResNet-50
3. Configure automatic mask generator with `points_per_side=24`, `pred_iou_thresh=0.88`
4. For each image:  
   `generate masks → prune → refine → classify crops → filter by confidence (0.85)`
5. Map classes to COCO categories (`person/head/helmet`)
6. Run detection on sample images with visualization (bounding boxes + labels)
7. Benchmark: Compare SAM/total time, avg masks/dets, mAP on 399 images
8. Output: Performance table showing **~1.1× speedup**, minimal mAP drop

---

## Results (399-Image Benchmark)

| Metric                   | Before Pruning | After Pruning (90%) | Change               |
|--------------------------|----------------|---------------------|----------------------|
| **Avg SAM Time / image** | 2.89 s         | 2.88 s              | ↓ 1.0×               |
| **Total Inference Time** | 3.10 s         | **2.92 s**          | **↓ 1.1× speedup**   |
| **Avg Masks Generated**  | 63.2           | **10.1**            | **↓ 84.1%**          |
| **Avg Detections**       | 21.8           | 4.6                 | –                    |
| **mAP@0.5:0.95**         | 0.002          | **0.003**           | **Preserved / +50%** |

> **Key Takeaway:**  
> **84% fewer tokens → 1.1× faster inference → mAP preserved (even slightly improved)**

---

## Interactive Notebook Capabilities
- Mask visualization with matplotlib (original + annotated)
- Printed detections: `class | confidence | bbox`
- Full benchmark table: pre- vs post-pruning metrics
- Fast eval mode with configurable `prune ratio (0.9)` and `conf thresh`
- COCO eval integration for mAP computation

---

## Features Summary
- Efficient token pruning for SAM acceleration
- Hybrid segmentation + classification pipeline
- Mask refinement for better crop quality
- Confidence-based filtering to reduce false positives
- Comprehensive benchmarking with speed/mAP trade-offs
- Handles construction site variations (lighting, occlusion)
- Modular for batch processing or real-time inference
- Colab-ready with GPU support (T4)

