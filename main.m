% Main Execution Script for Gait Authentication
clc; clear; close all;

% ===================== USER PARAMETERS ==========================
dataFolder   = fullfile(pwd, 'dataset');
fs           = 30;    % sampling rate (Hz)
winLenSec    = 3;     % window length (s)
overlap      = 0.5;   % 50% overlap

hiddenSizes   = 10;
numThresholds = 200;
mainTargetImpostorRatio = 3;

rng(1); % reproducibility

% ===================== 1. LOAD DATA =============================
fprintf('Loading data...\n');
allSessions = load_all_data(dataFolder, fs);

% ===================== 2. PREPROCESS & SEGMENT ==================
fprintf('Preprocessing, filtering and windowing...\n');
[allWindows, labelsUser, labelsDay] = ...
    build_windows(allSessions, fs, winLenSec, overlap);

% (Optional) Plot Example raw vs filtered (first session) defined in main script originally
% plot_example_raw_filtered(allSessions(1), fs); 
% plot_sliding_window_diagram(allSessions(1), fs, winLenSec, overlap);

% ===================== 3. FEATURE EXTRACTION ====================
fprintf('Extracting TF, FD and TFDF features...\n');
[ X_TF, X_FD, X_TFDF ] = extract_features(allWindows, fs);

% ===================== 4. VISUALIZE VARIANCE ====================
fprintf('Generating Variance and Feature Analysis Plots...\n');
visualize_variance(X_TF, X_FD, X_TFDF, labelsUser);

% ===================== 5. NEURAL NETWORK TRAINING ===============
fprintf('Running Neural Network Training and Evaluation...\n');
% Define main split (Split A)
idxTrain = (labelsDay == 1);
idxTest  = (labelsDay == 2);

run_neural_network(X_TF, X_FD, X_TFDF, labelsUser, idxTrain, idxTest, ...
    hiddenSizes, numThresholds, mainTargetImpostorRatio, labelsDay);

% ===================== 6. PCA ANALYSIS ==========================
fprintf('Running PCA Analysis...\n');
run_pca_analysis(X_TFDF, labelsUser, idxTrain, idxTest, ...
    hiddenSizes, numThresholds, mainTargetImpostorRatio);

% ===================== 7. SVM COMPARISON ========================
fprintf('Running SVM Comparison...\n');
run_svm_comparison(X_TF, X_FD, X_TFDF, labelsUser, idxTrain, idxTest, ...
    numThresholds, mainTargetImpostorRatio);

% ===================== 8. DTW ANALYSIS ==========================
fprintf('Running DTW Analysis...\n');
run_dtw(allWindows, labelsUser, fs);

fprintf('\n=== All tasks completed. ===\n');
