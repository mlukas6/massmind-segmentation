# Project Proposal and Deliveries

## Proposal 1 — Obstacle Semantic Segmentation

**Complexity:** ★★★ (High)

### Motivation

Semantic segmentation for Autonomous Surface Vehicles (ASVs) is critical in enabling real-time obstacle detection and navigation in complex maritime environments. By accurately segmenting objects like buoys, boats, debris, and natural obstacles such as rocks and marine vegetation at the pixel level, ASVs can safely chart paths, avoid hazards, and make real-time course adjustments, even in dynamic waters. Given the challenges of varied lighting, water reflections, and wave interference, robust segmentation models enhance ASV autonomy, allowing them to reliably interpret and respond to their surroundings. This capability not only improves the safety and efficiency of ASV operations in industries like shipping, environmental monitoring, and offshore maintenance but also broadens their application to more challenging and congested waters where precise obstacle segmentation is essential.

### Dataset

- **Dataset Size:** 2916 LWIR images.
- **Annotations:** Each image is labeled with pixel-level segmentation across seven classes (sky, water, bridge, obstacle, living obstacle, background, self).
- **Location:** Captured in and around Boston Harbor. Collected over two years, the images reflect a wide range of marine environments and conditions, including busy harbors and open waters, across different seasons and times of day.
- **Format:** Each data entry includes an image filename and a corresponding semantic segmentation mask, structured as: `data/filename.png mask/filename.png`

**Download Links:**

- Data: <https://drive.google.com/file/d/1T572f0oqy5JmuTvVEwkSUeXLWOSHl4hL/view>
- Label: <https://drive.google.com/file/d/1pHp48O_Q-s72RoDf1nD7ERzsv9yZTDE1/view>

### Challenge

1. Propose a custom AI-model that performs semantic segmentation from the LWIR image.
2. Train and test the AI-model without and with data augmentation.
3. Use the following metrics to assess the quality of your implementation:
   - IoU of training and testing;
   - Precision and recall;
   - Model complexity (# parameters).
4. Compare the results of your AI-model with at least 1 existing model (e.g., U-Net with VGG backbone).
5. Discuss the obtained results taking into consideration the following paper: <https://journals.sagepub.com/doi/10.1177/02783649231153020>

### References

- <https://github.com/uml-marine-robotics/MassMIND>
- <https://journals.sagepub.com/doi/10.1177/02783649231153020>
- <https://ieeexplore.ieee.org/document/9659477>

---

## Deliveries

*Group of two (up to three) persons.*

For the second assignment take into account the following considerations:

1. Computer vision applications should be developed in **Pytorch**;
2. The **code file** and **report** will be the **colaboratory** to be submitted in Moodle;
3. The colaboratory needs to be ready to work (the import of dataset must be automatically performed. Save the colab after running all cells of the program);
4. The report in the colaboratory should include:
   - **Introduction** summarizing the objectives of the work;
   - **Methodology** describing the algorithm or the NN model as well as, the description of the dataset used for training/evaluation;
   - **Results** section that demonstrate the improvement/work that was made. Try to focus on benchmarking your algorithm/model or showing quantitative metrics. Special attention must be made in presenting the results.
5. Each group must do a 5 minute presentation of the work (focused on the method and the results).
