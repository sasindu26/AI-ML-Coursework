% Main Execution Script for Gait Authentication
clc; clear; close all;

% Parameters
dataFolder = fullfile(pwd, 'dataset');
fs = 30; 
winLenSec = 3; 
overlap = 0.5;
hiddenSizes = 10;
numThresholds = 200;
mainTargetImpostorRatio = 3;
rng(1); % reproducibility

% 1. Load Data
fprintf('Loading data...\n');
allSessions = load_all_data(dataFolder, fs);

% 2. Preprocessing
fprintf('Preprocessing, filtering and windowing...\n');
[allWindows, labelsUser, labelsDay] = build_windows(allSessions, fs, winLenSec, overlap);

% Visualizations (Optional - kept from original)
plot_example_raw_filtered(allSessions(1), fs);
plot_sliding_window_diagram(allSessions(1), fs, winLenSec, overlap);

% 3. Feature Extraction
fprintf('Extracting TF, FD and TFDF features...\n');
[X_TF, X_FD, X_TFDF] = extract_features(allWindows, fs);

% 4. Run Analysis Modules
fprintf('Running Variance Visualization...\n');
visualize_variance(X_TF, X_FD, X_TFDF, labelsUser);

fprintf('Running PCA Analysis...\n');
run_pca_analysis(X_TFDF, labelsUser);

fprintf('Running ANOVA Feature Selection...\n');
% Note: ANOVA logic was inline in original, wrapped here for consistency
plot_feature_discriminability_anova(X_TFDF, labelsUser, 'TFDF'); 

fprintf('Running Neural Network Experiments...\n');
run_neural_network(X_TF, X_FD, X_TFDF, labelsUser, labelsDay, hiddenSizes, numThresholds, mainTargetImpostorRatio);

fprintf('Running SVM Comparison...\n');
run_svm_comparison(X_TF, X_FD, X_TFDF, labelsUser, labelsDay, numThresholds, mainTargetImpostorRatio);

fprintf('Running DTW Analysis...\n');
run_dtw(allWindows, labelsUser, fs);

fprintf('All tasks completed.\n');
