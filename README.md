# 🎯 Acceleration-Based User Authentication System

![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-orange) ![Status](https://img.shields.io/badge/Status-Completed-success)

This repository contains the implementation and results for an acceleration-based user authentication system. The project leverages motion data (Accelerometer & Gyroscope) to distinguish between legitimate users and imposters using Neural Networks (NN), Support Vector Machines (SVM), and Dynamic Time Warping (DTW).

## 📄 Final Report
The complete detailed analysis is available in the project report:
👉 **[Extended Version of AI & ML Report.pdf](Extended_Version_of_AI_&_ML_Report.pdf)**

---

## 🛠️ Pipeline Overview

The system follows a standard machine learning pipeline for biometric authentication:
1.  **Preprocessing:** Sliding window segmentation.
2.  **Feature Extraction:** Time Domain (TF), Frequency Domain (FD), and Hybrid (TFDF).
3.  **Optimization:** Feature selection (ANOVA), Dimensionality Reduction (PCA).
4.  **Classification:** Neural Networks vs SVM.

### 1. Preprocessing
We utilized a sliding window approach with a 3-second window and 50% overlap to ensure continuity in gait patterns.

![Sliding Window](G1%20Intra%20&%20Inter%20Variances/Sliding%20Window.png)

### 2. Feature Analysis
We extracted 131 features and analyzed their variance. A good feature should have low intra-class variance (user is consistent) and high inter-class variance (users are different from each other).

* **Feature Correlation:**
    ![Heatmap](G1%20Intra%20&%20Inter%20Variances/Correlation%20Heatmap.png)

* **Feature Variability:**
    ![Feature Variability](G1%20Intra%20&%20Inter%20Variances/Feature%20Variability.png)

* **Mean Feature Values:**
    ![Mean Features](G1%20Intra%20&%20Inter%20Variances/Mean%20Features.png)

* **Distance Distribution:**
    ![Distance Histogram](G1%20Intra%20&%20Inter%20Variances/Distance%20Histogram.png)

* **ANOVA Discriminability:** We used Analysis of Variance (ANOVA) to rank features based on their p-values (discriminative capability).
    ![ANOVA](G1%20Intra%20&%20Inter%20Variances/ANOVA.png)

* **3D Variance Analysis:**
    ![Variance Analysis](G1%20Intra%20&%20Inter%20Variances/Variance%20Analysis2.png)

---

## 🧠 Dimensionality Reduction (PCA)

We applied PCA to reduce the feature space while retaining 95% of the variance.
* **User Clusters (PC1 vs PC2):** The scatter plot shows how distinct different users are in the reduced space.
    ![PCA Scatter](G3%20All%20Other%20Figures/PCA%20Dimensionality%20Reduction/Figure_6.png)
* **Optimization:** We found that 15-20 Principal Components provided the optimal balance for EER.
    ![PCA Trend](G3%20All%20Other%20Figures/PCA%20Dimensionality%20Reduction/Figure_54.png)

---

## 📈 Results & Optimization

### Neural Network Tuning
We performed hyperparameter tuning on the number of hidden neurons.
* **Trend:** Increasing neurons improved accuracy up to a point (Hidden=30), after which overfitting risks increased.
    ![NN Tuning](G2%20Initial%20NN%20&%20Model%20Graphs/Hidden%20neuron%20tuning%20&%20trends6.png)

### Feature Selection (ANOVA)
We ranked features based on their p-values. Using the **Top-40 features** yielded comparable results to the full set with significantly less computation.
![Feature Selection](G3%20All%20Other%20Figures/Feature%20Selection%20(ANOVA)/Figure_46.png)

### Window Optimization
We tested different window lengths and overlap percentages.
* **Length:** 3 seconds provided the best balance between latency and accuracy.
* **Overlap:** 50% overlap maximized data usage without excessive redundancy.
    ![Window Opt](G3%20All%20Other%20Figures/Window_Optimization/Figure_40.png)

### Dynamic Time Warping (DTW)
DTW was used as a baseline comparison to measure template similarity.
    ![DTW Matrix](G3%20All%20Other%20Figures/DTW_Analysis/Figure_28.png)

### Final Comparison: NN vs SVM
We benchmarked our optimized Neural Network against an SVM (RBF Kernel).
* **Result:** The SVM achieved a slightly better Equal Error Rate (EER) in specific split scenarios, demonstrating high robustness.
    ![NN vs SVM](G3%20All%20Other%20Figures/Final_Comparisons/Figure_61.png)

---

## 🚀 How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/sasindu26/AI-ML-Coursework.git](https://github.com/sasindu26/AI-ML-Coursework.git)
    ```
2.  **Setup MATLAB:** Ensure you have the Statistics and Machine Learning Toolbox and Deep Learning Toolbox installed.
3.  **Run the Main Script:**
    Execute `main.m` to run the full pipeline (Preprocessing -> Feature Extraction -> Training -> Evaluation).

## 👥 Contributors

<table>
    <td align="center">
      <a href="https://github.com/sasindu26">
        <img src="https://avatars.githubusercontent.com/sasindu26" width="100px;" alt=""/><br />
        <sub><b>Sasindu Nilupul</b></sub>
      </a>
    </td>
  <br>
    <td align="center">
      <a href="https://github.com/DidulakaHirusha">
        <img src="https://avatars.githubusercontent.com/DidulakaHirusha" width="100px;" alt=""/><br />
        <sub><b>Didulaka Hirusha</b></sub>
      </a>
    </td>
  <br>
    <td align="center">
      <a href="https://github.com/wathsara02">
        <img src="https://avatars.githubusercontent.com/wathsara02" width="100px;" alt=""/><br />
        <sub><b>Wathsara Kalhara</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/ShanushkiBodhinayaka">
        <img src="https://avatars.githubusercontent.com/ShanushkiBodhinayaka" width="100px;" alt=""/><br />
        <sub><b>Shanushki Bodhinayaka</b></sub>
      </a>
    </td>
  </tr>
</table>
