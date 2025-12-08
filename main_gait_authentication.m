clc; clear; close all;

%% ===================== USER PARAMETERS ==========================
dataFolder   = fullfile(pwd, 'dataset');
fs           = 30;    % sampling rate (Hz)
winLenSec    = 3;     % window length (s)
overlap      = 0.5;   % 50% overlap

hiddenSizes   = 10;
numThresholds = 200;

mainTargetImpostorRatio = 3;

userToInspect           = 3;
featureSetForUserPlots  = 'TFDF';

rng(1); % reproducibility

%% ===================== 1. LOAD DATA =============================
fprintf('Loading data...\n');
allSessions = load_all_data(dataFolder, fs);

%% ===================== 2. PREPROCESS & SEGMENT ==================
fprintf('Preprocessing, filtering and windowing...\n');

[allWindows, labelsUser, labelsDay] = ...
    build_windows(allSessions, fs, winLenSec, overlap);

% Example raw vs filtered (first session)
plot_example_raw_filtered(allSessions(1), fs);

% Visualise one sample window (first window, Ax)
figure;
t_win = (0:size(allWindows,2)/6-1)/fs;
plot(t_win, allWindows(1,1:numel(t_win)));
xlabel('Time (s)'); ylabel('Acceleration X');
title('Example 3-second window (Ax)');
grid on;

% Sliding window diagram
plot_sliding_window_diagram(allSessions(1), fs, winLenSec, overlap);

%% ===================== 3. FEATURE EXTRACTION ====================
fprintf('Extracting TF, FD and TFDF features...\n');

[ X_TF, X_FD, X_TFDF ] = extract_all_feature_sets(allWindows, fs);

plot_feature_statistics(X_TFDF, labelsUser, 'TFDF');
plot_feature_correlation(X_TFDF, 'TFDF');
plot_pca_visualisation(X_TFDF, labelsUser, 'TFDF');
plot_feature_variability(X_TFDF, labelsUser, 'TFDF');
plot_user_feature_means_heatmap(X_TFDF, labelsUser, 'TFDF');
plot_intra_inter_distances(X_TFDF, labelsUser, 'TFDF');
plot_feature_discriminability_anova(X_TFDF, labelsUser, 'TFDF');

plot_intra_inter_variance_detailed(X_TF, labelsUser, 'TF');
plot_feature_variance_surface(X_TF, labelsUser, 'TF');

plot_intra_inter_variance_detailed(X_FD, labelsUser, 'FD');
plot_feature_variance_surface(X_FD, labelsUser, 'FD');

plot_intra_inter_variance_detailed(X_TFDF, labelsUser, 'TFDF');
plot_feature_variance_surface(X_TFDF, labelsUser, 'TFDF');

numUsers = numel(unique(labelsUser));
N        = numel(labelsUser);

fprintf('\n================ Neural Network Configuration ================\n');
fprintf('Network type         : Feed-forward patternnet\n');
fprintf('Hidden layer neurons : %d\n', hiddenSizes);
fprintf('Hidden activation    : Sigmoid\n');
fprintf('Output activation    : Sigmoid\n');
fprintf('Training algorithm   : Scaled Conjugate Gradient (trainscg)\n');
fprintf('Loss function        : Cross-entropy\n');
fprintf('Regularization (L2)  : %.2f\n', 0.1);
fprintf('Training/Validation split : 70/15/15\n');
fprintf('Early stopping       : Enabled\n');
fprintf('===============================================================\n');

%% ===================== 4. DEFINE TRAIN/TEST SPLITS ==============
% Split A: Day1 train, Day2 test
splitConfigs(1).fieldName   = 'Day1Train_Day2Test';
splitConfigs(1).title       = 'Split A: Day1 train, Day2 test';
splitConfigs(1).idxTrain    = (labelsDay == 1);
splitConfigs(1).idxTest     = (labelsDay == 2);

% Split B: Both days combined, random 80/20 split
idxAll      = (1:N).';
permAll     = randperm(N);
Ntrain_all  = round(0.8 * N);
idxTrainB   = false(N,1); idxTrainB(permAll(1:Ntrain_all)) = true;
idxTestB    = ~idxTrainB;

splitConfigs(2).fieldName   = 'BothDays_80_20';
splitConfigs(2).title       = 'Split B: Both days, 80/20 split';
splitConfigs(2).idxTrain    = idxTrainB;
splitConfigs(2).idxTest     = idxTestB;

% Split C: Day1 only, 80/20 split
idxDay1         = (labelsDay == 1);
idxDay1List     = find(idxDay1);
permDay1        = idxDay1List(randperm(numel(idxDay1List)));
Ntrain_day1     = round(0.8 * numel(idxDay1List));
idxTrainC       = false(N,1);
idxTestC        = false(N,1);
idxTrainC(permDay1(1:Ntrain_day1))           = true;
idxTestC(permDay1(Ntrain_day1+1:end))       = true;

splitConfigs(3).fieldName   = 'Day1Only_80_20';
splitConfigs(3).title       = 'Split C: Day1 only, 80/20 split';
splitConfigs(3).idxTrain    = idxTrainC;
splitConfigs(3).idxTest     = idxTestC;

%% ===================== 5. RUN BINARY EXPERIMENTS ================
featureSets = {'TF','FD','TFDF'};
resultsSplits = struct;

for k = 1:numel(splitConfigs)
    sc = splitConfigs(k);
    fprintf('\n===============================================\n');
    fprintf('Running binary experiments for %s\n', sc.title);
    fprintf('===============================================\n');
    
    resultsSplits.(sc.fieldName) = run_binary_experiment( ...
        sc.title, ...
        X_TF, X_FD, X_TFDF, ...
        labelsUser, ...
        sc.idxTrain, sc.idxTest, ...
        featureSets, ...
        hiddenSizes, numThresholds, ...
        mainTargetImpostorRatio);
end

%% ===================== 6. CHOOSE BEST SPLIT (TFDF) ==============
splitFieldNames = {'Day1Train_Day2Test','BothDays_80_20','Day1Only_80_20'};
splitLabels     = {'A: D1->D2','B: Both 80/20','C: D1 80/20'};

accTDFD = zeros(1,numel(splitFieldNames));
eerTDFD = zeros(1,numel(splitFieldNames));
aucTDFD = zeros(1,numel(splitFieldNames));

for i = 1:numel(splitFieldNames)
    sf = splitFieldNames{i};
    accTDFD(i) = resultsSplits.(sf).TFDF.meanAccuracyPct;
    eerTDFD(i) = resultsSplits.(sf).TFDF.meanEERPct;
    aucTDFD(i) = resultsSplits.(sf).TFDF.meanAUC;
end

figure;
tiledlayout(1,3, 'Padding','compact','TileSpacing','compact');
sgtitle('Comparison of train/test splits (TFDF, averaged over users)');

nexttile;
bar(accTDFD);
set(gca,'XTickLabel',splitLabels);
ylabel('Accuracy (%)');
title('Mean Accuracy');
grid on;

nexttile;
bar(eerTDFD);
set(gca,'XTickLabel',splitLabels);
ylabel('EER (%)');
title('Mean Equal Error Rate');
grid on;

nexttile;
bar(aucTDFD);
set(gca,'XTickLabel',splitLabels);
ylabel('AUC');
title('Mean ROC AUC');
grid on;

fprintf('\n=== Split comparison done. Using Split A + TFDF as reference. ===\n');

fprintf('\n=== Target:Impostor Ratio Analysis (all feature sets, Split A) ===\n');

ratioResultsGlobal = analyze_ratio_performance_all_features( ...
    X_TF, X_FD, X_TFDF, labelsUser, ...
    splitConfigs(1).idxTrain, splitConfigs(1).idxTest, ...
    hiddenSizes, numThresholds);

%% ===================== 8. TABLE FOR SPLIT A, TFDF ===============
perUserTable_TDFD_SplitA = resultsSplits.Day1Train_Day2Test.TFDF.perUserTable;
fprintf('\nPer-user evaluation metrics (Split A, TFDF):\n');
disp(perUserTable_TDFD_SplitA);

fprintf('\n=== Building user similarity matrix (Split A, TFDF) ===\n');

idxTestA   = (labelsDay == 2);
testUsers  = labelsUser(idxTestA);
resA_TFDF  = resultsSplits.Day1Train_Day2Test.TFDF;
numUsers   = numel(unique(labelsUser));

simMean = zeros(numUsers);
simStd  = zeros(numUsers);

for modelU = 1:numUsers
    scoresU = resA_TFDF.users(modelU).scores;
    
    for dataU = 1:numUsers
        idx = (testUsers == dataU);
        if any(idx)
            vals = scoresU(idx);
            simMean(modelU, dataU) = mean(vals);
            simStd(modelU,  dataU) = std(vals);
        else
            simMean(modelU, dataU) = NaN;
            simStd(modelU,  dataU) = NaN;
        end
    end
end

figure;
imagesc(simMean, [0 1]);
colorbar;
axis square;
xlabel('User ID (test data)');
ylabel('User ID (model)');
title('User similarity scores (Split A, TFDF)');
set(gca,'XTick',1:numUsers,'YTick',1:numUsers);

for r = 1:numUsers
    for c = 1:numUsers
        if isnan(simMean(r,c)), continue; end
        txt = sprintf('%.2f\n(±%.3f)', simMean(r,c), simStd(r,c));
        if simMean(r,c) > 0.5
            txtColor = 'w';
        else
            txtColor = 'k';
        end
        text(c, r, txt, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize',7, ...
            'Color',txtColor);
    end
end

anomalyThresh = 0.5;
hold on;
for r = 1:numUsers
    for c = 1:numUsers
        if r ~= c && simMean(r,c) >= anomalyThresh
            rectangle('Position',[c-0.5, r-0.5, 1, 1], ...
                      'EdgeColor','r', 'LineWidth',1.5);
        end
    end
end
hold off;

fprintf('Similarity matrix computed.\n');

%% ===================== 9. CONFUSION MATRICES + ROC (TFDF) =======
resA_TDFD       = resultsSplits.Day1Train_Day2Test.TFDF;
usersToInspect  = [3 8];
for k = 1:numel(usersToInspect)
    u = usersToInspect(k);

    if u < 1 || u > numUsers
        warning('usersToInspect(%d) = %d is out of range.', k, u);
        continue;
    end

    cm = resA_TDFD.users(u).confMat;
    figure;
    confusionchart(cm, {'Impostor','Genuine'}, 'RowSummary','row-normalized');
    title(sprintf('Binary NN Confusion (User %d, TFDF, Split A, Ratio 1:%d)', ...
        u, mainTargetImpostorRatio));

    rocFpr = resA_TDFD.users(u).rocFpr;
    rocTpr = resA_TDFD.users(u).rocTpr;

    figure;
    plot(rocFpr, rocTpr, 'LineWidth', 1.5); hold on;
    plot([0 1], [0 1], 'k--');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    grid on;
    title(sprintf('ROC curve (User %d, TFDF, Split A, Ratio 1:%d)', ...
        u, mainTargetImpostorRatio));
    legend({'NN verifier','Random guess'}, 'Location','SouthEast');
end

fprintf('\n=== Binary NN evaluation finished. ===\n');

%% ===================== 10. DTW ANALYSIS =========================
fprintf('\n=== DTW-based analysis (distance matrix + LOOCV) ===\n');

numSamplesPerUserForCV = 40;

[dtwDistMat, dtwConfMat, dtwPerUserAcc, dtwOverallAcc] = ...
    run_dtw_analysis(allWindows, labelsUser, fs, numSamplesPerUserForCV);

fprintf('DTW overall LOOCV accuracy (multiclass user ID): %.2f %%\n', ...
    dtwOverallAcc);
disp('DTW per-user accuracy (%):');
disp(dtwPerUserAcc);

%% 11. WINDOW OPTIMISATION (Split A, TFDF)

fprintf('\n=== 11. WINDOW OPTIMISATION (Split A, TFDF) ===\n');

idxTrainA_base = (labelsDay == 1);
idxTestA_base  = (labelsDay == 2);
baselineAcc = resultsSplits.Day1Train_Day2Test.TFDF.meanAccuracyPct;
baselineEER = resultsSplits.Day1Train_Day2Test.TFDF.meanEERPct;
baselineAUC = resultsSplits.Day1Train_Day2Test.TFDF.meanAUC;

%% 11.1 Vary window length (fixed overlap = 50 %)

winLens   = [2 3 4 5];
fixedOv   = 0.5;
nWL       = numel(winLens);

accWL = zeros(nWL,1);
eerWL = zeros(nWL,1);
aucWL = zeros(nWL,1);

for i = 1:nWL
    wl = winLens(i);
    fprintf('\n[Window length %.1f s, overlap %.0f %%]\n', wl, fixedOv*100);

    [allWinTmp, labUserTmp, labDayTmp] = build_windows(allSessions, fs, wl, fixedOv);
    [~, ~, X_TFDF_tmp] = extract_all_feature_sets(allWinTmp, fs);

    idxTrainA = (labDayTmp == 1);
    idxTestA  = (labDayTmp == 2);

    resWL = run_binary_experiment( ...
        sprintf('Split A (D1->D2) | win=%.1fs, ov=%.0f%%', wl, fixedOv*100), ...
        [], [], X_TFDF_tmp, ...
        labUserTmp, ...
        idxTrainA, idxTestA, ...
        {'TFDF'}, ...
        hiddenSizes, numThresholds, ...
        mainTargetImpostorRatio);

    accWL(i) = resWL.TFDF.meanAccuracyPct;
    eerWL(i) = resWL.TFDF.meanEERPct;
    aucWL(i) = resWL.TFDF.meanAUC;
end

T_winLen = table(winLens', accWL, eerWL, aucWL, ...
    'VariableNames', {'WindowLen_s','Accuracy','EER','AUC'});
fprintf('\n=== Window-length optimisation (overlap=50%%) ===\n');
disp(T_winLen);

figure;
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
sgtitle('Window-length optimisation (Split A, TFDF, overlap=50%)');

nexttile;
plot(winLens, accWL,'-o','LineWidth',1.5);
hold on; yline(baselineAcc,'k--','Baseline 3s','LabelHorizontalAlignment','left');
xlabel('Window length (s)'); ylabel('Accuracy (%)'); grid on;

nexttile;
plot(winLens, eerWL,'-o','LineWidth',1.5);
hold on; yline(baselineEER,'k--','Baseline 3s','LabelHorizontalAlignment','left');
xlabel('Window length (s)'); ylabel('EER (%)'); grid on;

nexttile;
plot(winLens, aucWL,'-o','LineWidth',1.5);
hold on; yline(baselineAUC,'k--','Baseline 3s','LabelHorizontalAlignment','left');
xlabel('Window length (s)'); ylabel('AUC'); grid on;

%% 11.2 Vary overlap (fixed window length = 3 s)

overlaps = [0.25 0.5 0.75];
nOV      = numel(overlaps);

accOV = zeros(nOV,1);
eerOV = zeros(nOV,1);
aucOV = zeros(nOV,1);

for i = 1:nOV
    ov = overlaps(i);
    fprintf('\n[Window length 3.0 s, overlap %.0f %%]\n', ov*100);

    [allWinTmp, labUserTmp, labDayTmp] = build_windows(allSessions, fs, 3.0, ov);
    [~, ~, X_TFDF_tmp] = extract_all_feature_sets(allWinTmp, fs);

    idxTrainA = (labDayTmp == 1);
    idxTestA  = (labDayTmp == 2);

    resOV = run_binary_experiment( ...
        sprintf('Split A (D1->D2) | win=3.0s, ov=%.0f%%', ov*100), ...
        [], [], X_TFDF_tmp, ...
        labUserTmp, ...
        idxTrainA, idxTestA, ...
        {'TFDF'}, ...
        hiddenSizes, numThresholds, ...
        mainTargetImpostorRatio);

    accOV(i) = resOV.TFDF.meanAccuracyPct;
    eerOV(i) = resOV.TFDF.meanEERPct;
    aucOV(i) = resOV.TFDF.meanAUC;
end

T_overlap = table(overlaps'*100, accOV, eerOV, aucOV, ...
    'VariableNames', {'Overlap_pct','Accuracy','EER','AUC'});
fprintf('\n=== Overlap optimisation (winLen=3s) ===\n');
disp(T_overlap);

figure;
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
sgtitle('Overlap optimisation (Split A, TFDF, winLen=3s)');

nexttile;
plot(overlaps*100, accOV,'-o','LineWidth',1.5);
hold on; yline(baselineAcc,'k--','Baseline 50%','LabelHorizontalAlignment','left');
xlabel('Overlap (%)'); ylabel('Accuracy (%)'); grid on;

nexttile;
plot(overlaps*100, eerOV,'-o','LineWidth',1.5);
hold on; yline(baselineEER,'k--','Baseline 50%','LabelHorizontalAlignment','left');
xlabel('Overlap (%)'); ylabel('EER (%)'); grid on;

nexttile;
plot(overlaps*100, aucOV,'-o','LineWidth',1.5);
hold on; yline(baselineAUC,'k--','Baseline 50%','LabelHorizontalAlignment','left');
xlabel('Overlap (%)'); ylabel('AUC'); grid on;

%% 12. FEATURE SELECTION USING ANOVA (Split A, TFDF)

fprintf('\n=== 12. FEATURE SELECTION USING ANOVA (TFDF, Split A) ===\n');

idxTrainA = (labelsDay == 1);
idxTestA  = (labelsDay == 2);

numFeat = size(X_TFDF,2);
pVals   = zeros(numFeat,1);

for f = 1:numFeat
    pVals(f) = anova1(X_TFDF(:,f), labelsUser, 'off');
end

[~, sortIdx] = sort(pVals, 'ascend');

topKList = [10 20 40 60 80];
topKList = topKList(topKList <= numFeat);

nK = numel(topKList);
accAnova = zeros(nK,1);
eerAnova = zeros(nK,1);
aucAnova = zeros(nK,1);

for i = 1:nK
    K = topKList(i);
    featIdx = sortIdx(1:K);
    X_TFDF_red = X_TFDF(:, featIdx);

    resK = run_binary_experiment( ...
        sprintf('Split A (D1->D2) | TFDF ANOVA top-%d', K), ...
        [], [], X_TFDF_red, ...
        labelsUser, ...
        idxTrainA, idxTestA, ...
        {'TFDF'}, ...
        hiddenSizes, numThresholds, ...
        mainTargetImpostorRatio);

    accAnova(i) = resK.TFDF.meanAccuracyPct;
    eerAnova(i) = resK.TFDF.meanEERPct;
    aucAnova(i) = resK.TFDF.meanAUC;
end

T_anova = table(topKList', accAnova, eerAnova, aucAnova, ...
    'VariableNames', {'NumFeatures','Accuracy','EER','AUC'});
fprintf('\n=== ANOVA-based feature selection results ===\n');
disp(T_anova);

figure;
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
sgtitle('ANOVA feature selection (Split A, TFDF)');

nexttile;
plot(topKList, accAnova,'-o','LineWidth',1.5);
hold on; yline(baselineAcc,'k--','Baseline all','LabelHorizontalAlignment','left');
xlabel('Number of TFDF features (top ANOVA)'); ylabel('Accuracy (%)'); grid on;

nexttile;
plot(topKList, eerAnova,'-o','LineWidth',1.5);
hold on; yline(baselineEER,'k--','Baseline all','LabelHorizontalAlignment','left');
xlabel('Number of TFDF features'); ylabel('EER (%)'); grid on;

nexttile;
plot(topKList, aucAnova,'-o','LineWidth',1.5);
hold on; yline(baselineAUC,'k--','Baseline all','LabelHorizontalAlignment','left');
xlabel('Number of TFDF features'); ylabel('AUC'); grid on;

%% 13. PCA DIMENSIONALITY REDUCTION (TFDF, Split A)

fprintf('\n=== 13. PCA DIMENSIONALITY REDUCTION (TFDF, Split A) ===\n');

X_train_TFDF = X_TFDF(idxTrainA,:);
X_test_TFDF  = X_TFDF(idxTestA,:);

[coeffPCA, ~, ~, ~, explainedPCA, muPCA] = pca(X_train_TFDF);

cumExpl = cumsum(explainedPCA);
figure;
plot(cumExpl,'LineWidth',1.5);
xlabel('Number of principal components');
ylabel('Cumulative explained variance (%)');
grid on;
title('PCA cumulative explained variance (TFDF, training set)');

pcaDims = [5 10 15 20 30 40];
pcaDims = pcaDims(pcaDims <= size(coeffPCA,2));

nP = numel(pcaDims);
accPCA = zeros(nP,1);
eerPCA = zeros(nP,1);
aucPCA = zeros(nP,1);

for i = 1:nP
    K = pcaDims(i);
    fprintf('  -> PCA with %d components\n', K);

    Z_train = (X_train_TFDF - muPCA) * coeffPCA(:,1:K);
    Z_test  = (X_test_TFDF  - muPCA) * coeffPCA(:,1:K);

    Z_full = zeros(size(X_TFDF,1), K);
    Z_full(idxTrainA,:) = Z_train;
    Z_full(idxTestA,:)  = Z_test;

    resP = run_binary_experiment( ...
        sprintf('Split A (D1->D2) | TFDF PCA-%d', K), ...
        [], [], Z_full, ...
        labelsUser, ...
        idxTrainA, idxTestA, ...
        {'TFDF'}, ...
        hiddenSizes, numThresholds, ...
        mainTargetImpostorRatio);

    accPCA(i) = resP.TFDF.meanAccuracyPct;
    eerPCA(i) = resP.TFDF.meanEERPct;
    aucPCA(i) = resP.TFDF.meanAUC;
end

T_pca = table(pcaDims', accPCA, eerPCA, aucPCA, ...
    'VariableNames', {'NumPCs','Accuracy','EER','AUC'});
fprintf('\n=== PCA-based dimensionality reduction results ===\n');
disp(T_pca);

figure;
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
sgtitle('PCA dimensionality reduction (Split A, TFDF)');

nexttile;
plot(pcaDims, accPCA,'-o','LineWidth',1.5);
hold on; yline(baselineAcc,'k--','Baseline full','LabelHorizontalAlignment','left');
xlabel('Number of principal components'); ylabel('Accuracy (%)'); grid on;

nexttile;
plot(pcaDims, eerPCA,'-o','LineWidth',1.5);
hold on; yline(baselineEER,'k--','Baseline full','LabelHorizontalAlignment','left');
xlabel('Number of principal components'); ylabel('EER (%)'); grid on;

nexttile;
plot(pcaDims, aucPCA,'-o','LineWidth',1.5);
hold on; yline(baselineAUC,'k--','Baseline full','LabelHorizontalAlignment','left');
xlabel('Number of principal components'); ylabel('AUC'); grid on;

%% 14. HIDDEN NEURON TUNING (TFDF, Split A)

fprintf('\n=== 14. HYPERPARAMETER TUNING: HIDDEN NEURONS (TFDF, Split A) ===\n');

hiddenList = [5 10 15 20 30];
nH = numel(hiddenList);

accH = zeros(nH,1);
eerH = zeros(nH,1);
aucH = zeros(nH,1);

for i = 1:nH
    h = hiddenList(i);
    fprintf('  -> Hidden neurons: %d\n', h);

    resH = run_binary_experiment( ...
        sprintf('Split A (D1->D2) | TFDF, hidden=%d', h), ...
        X_TF, X_FD, X_TFDF, ...
        labelsUser, ...
        idxTrainA, idxTestA, ...
        {'TFDF'}, ...
        h, numThresholds, ...
        mainTargetImpostorRatio);

    accH(i) = resH.TFDF.meanAccuracyPct;
    eerH(i) = resH.TFDF.meanEERPct;
    aucH(i) = resH.TFDF.meanAUC;
end

T_hid = table(hiddenList', accH, eerH, aucH, ...
    'VariableNames', {'HiddenNeurons','Accuracy','EER','AUC'});
fprintf('\n=== Hidden-neuron tuning results ===\n');
disp(T_hid);

figure;
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
sgtitle('Hidden-neuron tuning (Split A, TFDF)');

nexttile;
plot(hiddenList, accH,'-o','LineWidth',1.5);
hold on; yline(baselineAcc,'k--','Baseline h=10','LabelHorizontalAlignment','left');
xlabel('Hidden neurons'); ylabel('Accuracy (%)'); grid on;

nexttile;
plot(hiddenList, eerH,'-o','LineWidth',1.5);
hold on; yline(baselineEER,'k--','Baseline h=10','LabelHorizontalAlignment','left');
xlabel('Hidden neurons'); ylabel('EER (%)'); grid on;

nexttile;
plot(hiddenList, aucH,'-o','LineWidth',1.5);
hold on; yline(baselineAUC,'k--','Baseline h=10','LabelHorizontalAlignment','left');
xlabel('Hidden neurons'); ylabel('AUC'); grid on;

%% 15. ALTERNATIVE CLASSIFIER: SVM VS NN (Split A, TFDF)

fprintf('\n=== 15. ALTERNATIVE CLASSIFIER: SVM VS NN (Split A, TFDF) ===\n');

idxTrainA = (labelsDay == 1);
idxTestA  = (labelsDay == 2);

resSVM_SplitA = run_svm_experiment( ...
    'Split A: Day1 train, Day2 test (SVM)', ...
    X_TF, X_FD, X_TFDF, ...
    labelsUser, ...
    idxTrainA, idxTestA, ...
    {'TFDF'}, ...
    numThresholds, ...
    mainTargetImpostorRatio);

resSVM_SplitA_TFDF = resSVM_SplitA.TFDF;
resNN = resultsSplits.Day1Train_Day2Test.TFDF;

nn_acc = resNN.meanAccuracyPct;
nn_eer = resNN.meanEERPct;
nn_auc = resNN.meanAUC;
nn_far = resNN.meanFARPct;
nn_frr = resNN.meanFRRPct;
nn_f1  = resNN.meanF1Pct;

svm_acc = resSVM_SplitA_TFDF.meanAccuracyPct;
svm_eer = resSVM_SplitA_TFDF.meanEERPct;
svm_auc = resSVM_SplitA_TFDF.meanAUC;
svm_far = resSVM_SplitA_TFDF.meanFARPct;
svm_frr = resSVM_SplitA_TFDF.meanFRRPct;
svm_f1  = resSVM_SplitA_TFDF.meanF1Pct;

models      = categorical({'NN','SVM'});
models      = reordercats(models,{'NN','SVM'});

figure;
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
sgtitle(sprintf('NN vs SVM (Split A, TFDF, ratio 1:%d)', mainTargetImpostorRatio));

nexttile;
bar(models, [nn_acc svm_acc]);
ylabel('Accuracy (%)');
title('Mean Accuracy');
grid on;

nexttile;
bar(models, [nn_eer svm_eer]);
ylabel('EER (%)');
title('Mean Equal Error Rate');
grid on;

nexttile;
bar(models, [nn_auc svm_auc]);
ylabel('AUC');
title('Mean ROC AUC');
grid on;

nexttile;
bar(models, [nn_far svm_far]);
ylabel('FAR (%)');
title('Mean False Acceptance Rate');
grid on;

nexttile;
bar(models, [nn_frr svm_frr]);
ylabel('FRR (%)');
title('Mean False Rejection Rate');
grid on;

nexttile;
bar(models, [nn_f1 svm_f1]);
ylabel('F1-score (%)');
title('Mean F1-score');
grid on;

eerNN  = resNN.perUserTable.EER;
eerSVM = resSVM_SplitA_TFDF.perUserTable.EER;
usersE = resNN.perUserTable.User;

figure;
bar(usersE, [eerNN eerSVM]);
xlabel('User ID');
ylabel('EER (%)');
title(sprintf('Per-user EER (NN vs SVM, Split A, TFDF, ratio 1:%d)', ...
    mainTargetImpostorRatio));
legend({'NN','SVM'}, 'Location','best');
grid on;

fprintf('\n=== All experiments completed. ===\n');

%% =================================================================
%% =============== HELPER FUNCTIONS (LOCAL) ========================
%% =================================================================

function allSessions = load_all_data(dataFolder, defaultFs)

allSessions = [];
k = 0;

for userID = 1:10
    for day = 1:2
        
        if day == 1
            dayTag = 'FD';
        else
            dayTag = 'MD';
        end
        
        fileName = fullfile(dataFolder, sprintf('U%dNW_%s.csv', userID, dayTag));
        
        if ~isfile(fileName)
            warning('File not found: %s (skipping)', fileName);
            continue;
        end
        
        raw = readmatrix(fileName);
        
        if size(raw,2) < 7
            error('File %s has fewer than 7 columns.', fileName);
        end
        
        Ax = raw(:,2); Ay = raw(:,3); Az = raw(:,4);
        Gx = raw(:,5); Gy = raw(:,6); Gz = raw(:,7);
        
        sig = [Ax, Ay, Az, Gx, Gy, Gz];
        
        k = k + 1;
        allSessions(k).signal = sig;
        allSessions(k).fs     = defaultFs;
        allSessions(k).userID = userID;
        allSessions(k).day    = day;
    end
end

end

function [allWindows, labelsUser, labelsDay] = ...
    build_windows(allSessions, fs, winLenSec, overlap)

winLenSamples = round(winLenSec * fs);
stepSamples   = round(winLenSamples * (1 - overlap));

allWindows = [];
labelsUser = [];
labelsDay  = [];

for k = 1:numel(allSessions)
    sig   = allSessions(k).signal;
    user  = allSessions(k).userID;
    day   = allSessions(k).day;
    
    N = size(sig,1);
    startIdx = 1;
    
    while (startIdx + winLenSamples - 1) <= N
        idxRange = startIdx : (startIdx + winLenSamples - 1);
        winSeg   = sig(idxRange, :);
        
        allWindows = [allWindows; winSeg(:)'];  %#ok<AGROW>
        labelsUser = [labelsUser; user];       %#ok<AGROW>
        labelsDay  = [labelsDay; day];         %#ok<AGROW>
        
        startIdx = startIdx + stepSamples;
    end
end

end

function plot_example_raw_filtered(session, fs)

sig = session.signal;
t = (0:size(sig,1)-1)/fs;

fc = 7;
[b,a] = butter(2, fc/(fs/2), 'low');
sigF = filtfilt(b,a,sig);

figure;
subplot(2,1,1);
plot(t, sig(:,1)); hold on;
plot(t, sigF(:,1), 'LineWidth',1.2);
xlabel('Time (s)'); ylabel('Ax');
legend('Raw','Filtered');
title('Accel X - Raw vs Filtered');
grid on;

subplot(2,1,2);
plot(t, sig(:,4)); hold on;
plot(t, sigF(:,4), 'LineWidth',1.2);
xlabel('Time (s)'); ylabel('Gx');
legend('Raw','Filtered');
title('Gyro X - Raw vs Filtered');
grid on;

end

function [X_TF, X_FD, X_TFDF] = extract_all_feature_sets(allWindows, fs)

numWindows = size(allWindows,1);
numSamples = size(allWindows,2)/6;

X_TF   = [];
X_FD   = [];
X_TFDF = [];

for i = 1:numWindows
    winVec = allWindows(i,:);
    winMat = reshape(winVec, numSamples, 6);
    
    accel = winMat(:,1:3);
    gyro  = winMat(:,4:6);
    
    tf_feat = extract_TF_features(accel, gyro);
    fd_feat = extract_FD_features(accel, gyro, fs);
    
    X_TF   = [X_TF; tf_feat];              %#ok<AGROW>
    X_FD   = [X_FD; fd_feat];              %#ok<AGROW>
    X_TFDF = [X_TFDF; [tf_feat, fd_feat]]; %#ok<AGROW>
end

end

function tf_feat = extract_TF_features(accel, gyro)

sig = [accel, gyro];
N   = size(sig,1);

means   = mean(sig,1);
medians = median(sig,1);
stds    = std(sig,0,1);
vars    = var(sig,0,1);
mins    = min(sig,[],1);
maxs    = max(sig,[],1);
ranges  = maxs - mins;
rmsVal  = sqrt(mean(sig.^2,1));
skews   = skewness(sig,0,1);
kurts   = kurtosis(sig,0,1);
iqrs    = iqr(sig,1);

zcr = zeros(1,6);
for c = 1:6
    s = sig(:,c);
    zcr(c) = sum(abs(diff(sign(s)))>0) / (N-1);
end

sma     = sum(abs(sig),1) / N;
energy  = sum(sig.^2,1) / N;

tf_feat = [means, medians, stds, vars, mins, maxs, ranges, ...
           rmsVal, skews, kurts, iqrs, zcr, sma, energy];

end

function fd_feat = extract_FD_features(accel, gyro, fs)

sig = [accel, gyro];
N   = size(sig,1);

fd_feat = [];

for c = 1:6
    x = sig(:,c);
    X = fft(x);
    mag = abs(X(1:floor(N/2)));
    freqs = (0:numel(mag)-1)' * fs/N;
    
    magN = mag / sum(mag + eps);
    
    [~, idxMax] = max(mag);
    domFreq = freqs(idxMax);
    
    specCent   = sum(freqs .* magN);
    specSpread = sqrt(sum(((freqs - specCent).^2) .* magN));
    bandPow    = sum(mag.^2)/numel(mag);
    
    p = magN + eps;
    specEnt = -sum(p .* log2(p));
    
    geoMean   = exp(mean(log(mag + eps)));
    arithMean = mean(mag + eps);
    specFlat  = geoMean / arithMean;
    
    fd_feat = [fd_feat, domFreq, specCent, specSpread, bandPow, specEnt, specFlat]; %#ok<AGROW>
end

end

function plot_feature_statistics(X, labelsUser, featName)

numUsers = numel(unique(labelsUser));
featIdx = [1, 2, 3, 4];
numF    = numel(featIdx);

figure;
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
sgtitle(['Per-user distribution (selected features, ' featName ')']);

for i = 1:numF
    f = featIdx(i);
    nexttile;
    boxplot(X(:,f), labelsUser);
    xlabel('User ID');
    ylabel(sprintf('Feature %d', f));
    title(sprintf('Feature %d', f));
    grid on;
end

end

function plot_feature_correlation(X, featName)
R = corrcoef(X);
figure;
imagesc(R);
colorbar;
title(['Feature Correlation - ' featName]);
xlabel('Feature index'); ylabel('Feature index');
axis square;
end

function plot_pca_visualisation(X, labelsUser, featName)
[~, score, ~] = pca(X);
figure;
gscatter(score(:,1), score(:,2), labelsUser);
xlabel('PC1'); ylabel('PC2');
title(['PCA scatter (' featName ')']);
grid on;
end

function plot_feature_variability(X, labelsUser, featName)
numUsers = numel(unique(labelsUser));
featIdx = [1, 2, 10, 20];

figure;
tiledlayout(1,numel(featIdx),'TileSpacing','compact','Padding','compact');
sgtitle(['Per-user feature variability (' featName ')']);

for i = 1:numel(featIdx)
    f = featIdx(i);
    vals = X(:,f);
    muPerUser = zeros(numUsers,1);
    sdPerUser = zeros(numUsers,1);
    for u = 1:numUsers
        muPerUser(u) = mean(vals(labelsUser==u));
        sdPerUser(u) = std(vals(labelsUser==u));
    end
    nexttile;
    errorbar(1:numUsers, muPerUser, sdPerUser, 'o-','LineWidth',1.2);
    xlabel('User ID'); ylabel(sprintf('Feature %d', f));
    title(sprintf('Feature %d (mean \x00b1 std)', f));
    grid on;
end

end

function plot_user_feature_means_heatmap(X, labelsUser, featName)
numUsers = numel(unique(labelsUser));
K = min(20, size(X,2));

userMeans = zeros(numUsers, K);

for u = 1:numUsers
    idx = (labelsUser == u);
    userMeans(u,:) = mean(X(idx,1:K), 1);
end

figure;
imagesc(userMeans);
colorbar;
xlabel('Feature index (1..K)');
ylabel('User ID');
title(sprintf('Per-user mean feature values (first %d, %s)', K, featName));
set(gca,'XTick',1:K,'YTick',1:numUsers);
end

function plot_intra_inter_distances(X, labelsUser, featName)
numUsers = numel(unique(labelsUser));
numPairsPerType = 2000;

N = size(X,1);
intraD = zeros(numPairsPerType,1);
interD = zeros(numPairsPerType,1);

cnt = 0;
while cnt < numPairsPerType
    u = randi(numUsers);
    idx = find(labelsUser == u);
    if numel(idx) < 2, continue; end
    ab = randsample(idx,2);
    d = norm(X(ab(1),:) - X(ab(2),:));
    cnt = cnt + 1;
    intraD(cnt) = d;
end

cnt = 0;
while cnt < numPairsPerType
    ab = randsample(N,2);
    if labelsUser(ab(1)) == labelsUser(ab(2))
        continue;
    end
    d = norm(X(ab(1),:) - X(ab(2),:));
    cnt = cnt + 1;
    interD(cnt) = d;
end

figure;
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
sgtitle(['Intra- vs Inter-user distances (' featName ')']);

nexttile;
hold on;
histogram(intraD, 'Normalization','pdf');
histogram(interD, 'Normalization','pdf');
xlabel('Distance');
ylabel('PDF');
legend('Intra-user','Inter-user');
title('Histogram');
grid on;

nexttile;
boxplot([intraD; interD], ...
    [ones(size(intraD)); 2*ones(size(interD))], ...
    'Labels',{'Intra-user','Inter-user'});
ylabel('Distance');
title('Boxplot');
grid on;

end

function plot_feature_discriminability_anova(X, labelsUser, featName)
K = min(30, size(X,2));
pVals = zeros(K,1);

for f = 1:K
    pVals(f) = anova1(X(:,f), labelsUser, 'off');
end

figure;
stem(1:K, -log10(pVals), 'filled');
xlabel('Feature index');
ylabel('-log_{10}(p-value)');
title(['Feature discriminability (ANOVA, ' featName ')']);
grid on;

end

function ratioResultsGlobal = analyze_ratio_performance_all_features( ...
    X_TF, X_FD, X_TFDF, labelsUser, idxTrain, idxTest, ...
    hiddenSizes, numThresholds)

featureSets = {'TF','FD','TFDF'};
ratios      = 1:7;
numRatios   = numel(ratios);
numFeats    = numel(featureSets);
numUsers    = numel(unique(labelsUser));

XTrain_TF   = X_TF(idxTrain,:);   XTest_TF   = X_TF(idxTest,:);
XTrain_FD   = X_FD(idxTrain,:);   XTest_FD   = X_FD(idxTest,:);
XTrain_TFDF = X_TFDF(idxTrain,:); XTest_TFDF = X_TFDF(idxTest,:);

yTrainUsers = labelsUser(idxTrain);
yTestUsers  = labelsUser(idxTest);

metrics = zeros(numRatios, numFeats, 6);

for r = 1:numRatios
    R = ratios(r);
    fprintf('  -> Ratio 1:%d\n', R);

    for f = 1:numFeats
        featName = featureSets{f};
        switch featName
            case 'TF'
                XTrain = XTrain_TF;   XTest = XTest_TF;
            case 'FD'
                XTrain = XTrain_FD;   XTest = XTest_FD;
            case 'TFDF'
                XTrain = XTrain_TFDF; XTest = XTest_TFDF;
        end

        acc = zeros(numUsers,1);
        far = zeros(numUsers,1);
        frr = zeros(numUsers,1);
        eer = zeros(numUsers,1);
        auc = zeros(numUsers,1);
        f1  = zeros(numUsers,1);

        for u = 1:numUsers
            genIdx = find(yTrainUsers == u);
            impIdx = find(yTrainUsers ~= u);

            numGen = numel(genIdx);
            numImpNeeded = min(numGen * R, numel(impIdx));
            impSel = randsample(impIdx, numImpNeeded);

            XTrain_u = [XTrain(genIdx,:); XTrain(impSel,:)];
            yTrain_u = [ones(numGen,1); zeros(numImpNeeded,1)];
            yTestBin = (yTestUsers == u);

            [~, yPred, scores] = train_binary_nn_verifier( ...
                XTrain_u, yTrain_u, XTest, hiddenSizes);

            [m, ~] = evaluate_binary_classification(yTestBin, yPred);
            [farCurve, frrCurve, eerVal, ~, rocFpr, rocTpr] = ...
                compute_far_frr_eer_binary(yTestBin, scores, numThresholds);

            [~, idxE] = min(abs(farCurve - frrCurve));
            farAtE = farCurve(idxE);
            frrAtE = frrCurve(idxE);

            [sortedFpr, idxSort] = sort(rocFpr);
            sortedTpr = rocTpr(idxSort);
            aucVal = trapz(sortedFpr, sortedTpr);

            acc(u) = m.accuracy  * 100;
            far(u) = farAtE      * 100;
            frr(u) = frrAtE      * 100;
            eer(u) = eerVal      * 100;
            auc(u) = aucVal;
            f1(u)  = m.f1        * 100;
        end

        metrics(r,f,1) = mean(acc);
        metrics(r,f,2) = mean(far);
        metrics(r,f,3) = mean(frr);
        metrics(r,f,4) = mean(eer);
        metrics(r,f,5) = mean(auc);
        metrics(r,f,6) = mean(f1);
    end
end

ratioResultsGlobal.ratios    = ratios;
ratioResultsGlobal.metrics   = metrics;
ratioResultsGlobal.features  = featureSets;

titles = {'Accuracy (%)','FAR (%)','FRR (%)','EER (%)','AUC','F1-score (%)'};
figure;
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
sgtitle('Performance vs Target:Impostor ratio (Split A)');

for mIdx = 1:6
    nexttile; hold on;
    for f = 1:numFeats
        plot(ratios, squeeze(metrics(:,f,mIdx)), '-o', 'LineWidth',1.2);
    end
    xlabel('Impostor ratio (1:N)');
    ylabel(titles{mIdx});
    title(titles{mIdx});
    grid on;
    if mIdx == 1
        legend(featureSets, 'Location','best');
    end
end
hold off;

end

function res = run_binary_experiment( ...
        experimentTitle, ...
        X_TF, X_FD, X_TFDF, ...
        labelsUser, ...
        idxTrain, idxTest, ...
        featureSets, ...
        hiddenSizes, numThresholds, ...
        mainTargetImpostorRatio)

yTrainUsers = labelsUser(idxTrain);
yTestUsers  = labelsUser(idxTest);

numUsers = numel(unique(labelsUser));

for s = 1:numel(featureSets)
    featName = featureSets{s};
    fprintf('\n=== %s | Features: %s ===\n', experimentTitle, featName);
    
    switch featName
        case 'TF'
            X = X_TF;
        case 'FD'
            X = X_FD;
        case 'TFDF'
            X = X_TFDF;
    end
    
    XTrain_full = X(idxTrain,:);
    XTest       = X(idxTest,:);
    
    userID   = (1:numUsers).';
    accPct   = zeros(numUsers,1);
    aucVal   = zeros(numUsers,1);
    precPct  = zeros(numUsers,1);
    recPct   = zeros(numUsers,1);
    f1Pct    = zeros(numUsers,1);
    farPct   = zeros(numUsers,1);
    frrPct   = zeros(numUsers,1);
    eerPct   = zeros(numUsers,1);
    
    res.(featName).users = struct([]);
    
    for u = 1:numUsers
        fprintf('  -> User %d (ratio 1:%d)\n', u, mainTargetImpostorRatio);
        
        genIdx = find(yTrainUsers == u);
        impIdx = find(yTrainUsers ~= u);
        numGen = numel(genIdx);
        numImpNeeded = min(numGen * mainTargetImpostorRatio, numel(impIdx));
        
        impSel = randsample(impIdx, numImpNeeded);
        
        XTrain_u = [XTrain_full(genIdx,:); XTrain_full(impSel,:)];
        yTrain_u = [ones(numGen,1); zeros(numImpNeeded,1)];
        
        yTest_bin = (yTestUsers == u);
        
        [netBin, yPred_bin, score_bin] = train_binary_nn_verifier( ...
            XTrain_u, yTrain_u, XTest, hiddenSizes);

        if u == 1 && strcmp(featName,'TFDF') && contains(experimentTitle,'Day1 train')
            figure;
            view(netBin);
            title('Neural Network Architecture (TFDF, User 1, Split A)');
        end
        
        [binMetrics, binConfMat] = evaluate_binary_classification(yTest_bin, yPred_bin);
        
        [farBin, frrBin, eerBin, thrBin, rocFprBin, rocTprBin] = ...
            compute_far_frr_eer_binary(yTest_bin, score_bin, numThresholds);
        
        [~, idxEbin] = min(abs(farBin - frrBin));
        farAtEER = farBin(idxEbin);
        frrAtEER = frrBin(idxEbin);
        
        [sortedFpr, idxSort] = sort(rocFprBin);
        sortedTpr = rocTprBin(idxSort);
        auc = trapz(sortedFpr, sortedTpr);
        
        accPct(u)  = binMetrics.accuracy  * 100;
        precPct(u) = binMetrics.precision * 100;
        recPct(u)  = binMetrics.recall    * 100;
        f1Pct(u)   = binMetrics.f1        * 100;
        farPct(u)  = farAtEER             * 100;
        frrPct(u)  = frrAtEER             * 100;
        eerPct(u)  = eerBin               * 100;
        aucVal(u)  = auc;
        
        res.(featName).users(u).userID     = u;
        res.(featName).users(u).net        = netBin;
        res.(featName).users(u).confMat    = binConfMat;
        res.(featName).users(u).metrics    = binMetrics;
        res.(featName).users(u).scores     = score_bin;
        res.(featName).users(u).yTrueBin   = yTest_bin;
        res.(featName).users(u).far        = farBin;
        res.(featName).users(u).frr        = frrBin;
        res.(featName).users(u).eer        = eerBin;
        res.(featName).users(u).thresholds = thrBin;
        res.(featName).users(u).rocFpr     = rocFprBin;
        res.(featName).users(u).rocTpr     = rocTprBin;
        res.(featName).users(u).far_eer    = farAtEER;
        res.(featName).users(u).frr_eer    = frrAtEER;
        res.(featName).users(u).auc        = auc;
    end
    
    perUserTable = table(userID, ...
        accPct, aucVal, precPct, recPct, f1Pct, farPct, frrPct, eerPct, ...
        'VariableNames', {'User','Accuracy','AUC','Precision','Recall', ...
                          'F1','FAR','FRR','EER'});
    
    res.(featName).perUserTable    = perUserTable;
    res.(featName).meanAccuracyPct = mean(accPct);
    res.(featName).meanEERPct      = mean(eerPct);
    res.(featName).meanAUC         = mean(aucVal);
    res.(featName).meanF1Pct       = mean(f1Pct);
    res.(featName).meanFARPct      = mean(farPct);
    res.(featName).meanFRRPct      = mean(frrPct);
    
    fprintf('\nPer-user metrics (%s, %s):\n', experimentTitle, featName);
    disp(perUserTable);
end

featOrder = featureSets;
farVals = zeros(1,numel(featOrder));
frrVals = zeros(1,numel(featOrder));
eerVals = zeros(1,numel(featOrder));
accVals = zeros(1,numel(featOrder));
f1Vals  = zeros(1,numel(featOrder));

for i = 1:numel(featOrder)
    fName   = featOrder{i};
    farVals(i) = res.(fName).meanFARPct;
    frrVals(i) = res.(fName).meanFRRPct;
    eerVals(i) = res.(fName).meanEERPct;
    accVals(i) = res.(fName).meanAccuracyPct;
    f1Vals(i)  = res.(fName).meanF1Pct;
end

figure;
tiledlayout(2,3, 'Padding','compact','TileSpacing','compact');
sgtitle(sprintf('Binary NN metrics across feature sets (%s)', experimentTitle));

nexttile;
bar(farVals);
set(gca, 'XTickLabel', featOrder);
ylabel('FAR (%)');
title('Mean FAR');
grid on;

nexttile;
bar(frrVals);
set(gca, 'XTickLabel', featOrder);
ylabel('FRR (%)');
title('Mean FRR');
grid on;

nexttile;
bar(eerVals);
set(gca, 'XTickLabel', featOrder);
ylabel('EER (%)');
title('Mean EER');
grid on;

nexttile;
bar(accVals);
set(gca, 'XTickLabel', featOrder);
ylabel('Accuracy (%)');
title('Mean Accuracy');
grid on;

nexttile;
bar(f1Vals);
set(gca, 'XTickLabel', featOrder);
ylabel('F1-score (%)');
title('Mean F1');
grid on;

end

function res = run_svm_experiment( ...
        experimentTitle, ...
        X_TF, X_FD, X_TFDF, ...
        labelsUser, ...
        idxTrain, idxTest, ...
        featureSets, ...
        numThresholds, ...
        mainTargetImpostorRatio)

yTrainUsers = labelsUser(idxTrain);
yTestUsers  = labelsUser(idxTest);

numUsers = numel(unique(labelsUser));

for s = 1:numel(featureSets)
    featName = featureSets{s};
    fprintf('\n=== %s | SVM | Features: %s ===\n', experimentTitle, featName);
    
    switch featName
        case 'TF'
            X = X_TF;
        case 'FD'
            X = X_FD;
        case 'TFDF'
            X = X_TFDF;
    end
    
    XTrain_full = X(idxTrain,:);
    XTest       = X(idxTest,:);
    
    userID   = (1:numUsers).';
    accPct   = zeros(numUsers,1);
    aucVal   = zeros(numUsers,1);
    precPct  = zeros(numUsers,1);
    recPct   = zeros(numUsers,1);
    f1Pct    = zeros(numUsers,1);
    farPct   = zeros(numUsers,1);
    frrPct   = zeros(numUsers,1);
    eerPct   = zeros(numUsers,1);
    
    res.(featName).users = struct([]);
    
    for u = 1:numUsers
        fprintf('  -> SVM for User %d (ratio 1:%d)\n', u, mainTargetImpostorRatio);
        
        genIdx = find(yTrainUsers == u);
        impIdx = find(yTrainUsers ~= u);
        numGen = numel(genIdx);
        numImpNeeded = min(numGen * mainTargetImpostorRatio, numel(impIdx));
        impSel = randsample(impIdx, numImpNeeded);
        
        XTrain_u = [XTrain_full(genIdx,:); XTrain_full(impSel,:)];
        yTrain_u = [ones(numGen,1); zeros(numImpNeeded,1)];
        
        yTest_bin = (yTestUsers == u);
        
        [svmModel, yPred_bin, score_bin] = train_binary_svm_verifier( ...
            XTrain_u, yTrain_u, XTest); %#ok<NASGU>
        
        [binMetrics, binConfMat] = evaluate_binary_classification(yTest_bin, yPred_bin);
        
        [farBin, frrBin, eerBin, thrBin, rocFprBin, rocTprBin] = ...
            compute_far_frr_eer_binary(yTest_bin, score_bin, numThresholds);
        
        [~, idxEbin] = min(abs(farBin - frrBin));
        farAtEER = farBin(idxEbin);
        frrAtEER = frrBin(idxEbin);
        
        [sortedFpr, idxSort] = sort(rocFprBin);
        sortedTpr = rocTprBin(idxSort);
        auc = trapz(sortedFpr, sortedTpr);
        
        accPct(u)  = binMetrics.accuracy  * 100;
        precPct(u) = binMetrics.precision * 100;
        recPct(u)  = binMetrics.recall    * 100;
        f1Pct(u)   = binMetrics.f1        * 100;
        farPct(u)  = farAtEER             * 100;
        frrPct(u)  = frrAtEER             * 100;
        eerPct(u)  = eerBin               * 100;
        aucVal(u)  = auc;
        
        res.(featName).users(u).userID     = u;
        res.(featName).users(u).confMat    = binConfMat;
        res.(featName).users(u).metrics    = binMetrics;
        res.(featName).users(u).scores     = score_bin;
        res.(featName).users(u).yTrueBin   = yTest_bin;
        res.(featName).users(u).far        = farBin;
        res.(featName).users(u).frr        = frrBin;
        res.(featName).users(u).eer        = eerBin;
        res.(featName).users(u).thresholds = thrBin;
        res.(featName).users(u).rocFpr     = rocFprBin;
        res.(featName).users(u).rocTpr     = rocTprBin;
        res.(featName).users(u).far_eer    = farAtEER;
        res.(featName).users(u).frr_eer    = frrAtEER;
        res.(featName).users(u).auc        = auc;
    end
    
    perUserTable = table(userID, ...
        accPct, aucVal, precPct, recPct, f1Pct, farPct, frrPct, eerPct, ...
        'VariableNames', {'User','Accuracy','AUC','Precision','Recall', ...
                          'F1','FAR','FRR','EER'});
    
    res.(featName).perUserTable    = perUserTable;
    res.(featName).meanAccuracyPct = mean(accPct);
    res.(featName).meanEERPct      = mean(eerPct);
    res.(featName).meanAUC         = mean(aucVal);
    res.(featName).meanF1Pct       = mean(f1Pct);
    res.(featName).meanFARPct      = mean(farPct);
    res.(featName).meanFRRPct      = mean(frrPct);
    
    fprintf('\nPer-user SVM metrics (%s, %s):\n', experimentTitle, featName);
    disp(perUserTable);
end

end

function [netBin, yPredTest, scoreTest] = train_binary_nn_verifier( ...
    XTrain, yTrainBin, XTest, hiddenSizes)

T = double(yTrainBin)';

XTrainT = XTrain';
XTestT  = XTest';

netBin = patternnet(hiddenSizes);
netBin.performParam.regularization = 0.1;
netBin.trainParam.showWindow = false;

netBin = train(netBin, XTrainT, T);

scoreTest = netBin(XTestT)';
yPredTest = scoreTest >= 0.5;

end

function [svmModel, yPredTest, scoreTest] = train_binary_svm_verifier( ...
    XTrain, yTrainBin, XTest)

yTrainBin = double(yTrainBin(:));

svmModel = fitcsvm( ...
    XTrain, yTrainBin, ...
    'KernelFunction','rbf', ...
    'KernelScale','auto', ...
    'Standardize',true, ...
    'ClassNames',[0 1]);

[yPredTest, scoreMat] = predict(svmModel, XTest);
scoreTest = scoreMat(:,2);

end

function [metrics, confMat] = evaluate_binary_classification(yTrueBin, yPredBin)

yTrueBin = double(yTrueBin(:));
yPredBin = double(yPredBin(:));

confMat = confusionmat(yTrueBin, yPredBin);

if numel(confMat) == 1
    if yTrueBin(1) == 0
        confMat = [confMat 0; 0 0];
    else
        confMat = [0 0; 0 confMat];
    end
end

TN = confMat(1,1);
FP = confMat(1,2);
FN = confMat(2,1);
TP = confMat(2,2);

denAll = max(1, (TP+TN+FP+FN));
accuracy  = (TP + TN) / denAll;
precision = TP / max(1, (TP + FP));
recall    = TP / max(1, (TP + FN));
f1        = 2 * precision * recall / max(1e-12, (precision + recall));

metrics.accuracy  = accuracy;
metrics.precision = precision;
metrics.recall    = recall;
metrics.f1        = f1;

end

function [far, frr, eer, thresholds, rocFpr, rocTpr] = ...
    compute_far_frr_eer_binary(yTrueBin, scores, numThresholds)

genuineIdx  = (yTrueBin == 1);
impostorIdx = (yTrueBin == 0);

thresholds = linspace(0,1,numThresholds);
far = zeros(size(thresholds));
frr = zeros(size(thresholds));

for k = 1:numel(thresholds)
    t = thresholds(k);
    if any(impostorIdx)
        far(k) = mean(scores(impostorIdx) >= t);
    else
        far(k) = 0;
    end
    if any(genuineIdx)
        frr(k) = mean(scores(genuineIdx)  <  t);
    else
        frr(k) = 0;
    end
end

[~, idxMin] = min(abs(far - frr));
eer = (far(idxMin) + frr(idxMin)) / 2;

scoreVals = scores(:);
labelsBin = yTrueBin(:);

rocThresh = sort(unique(scoreVals));
rocFpr = zeros(size(rocThresh));
rocTpr = zeros(size(rocThresh));

for k = 1:numel(rocThresh)
    t = rocThresh(k);
    preds = scoreVals >= t;

    TP = sum(preds==1 & labelsBin==1);
    FP = sum(preds==1 & labelsBin==0);
    FN = sum(preds==0 & labelsBin==1);
    TN = sum(preds==0 & labelsBin==0);
    
    rocTpr(k) = TP / max(1, (TP + FN));
    rocFpr(k) = FP / max(1, (FP + TN));
end

end

function plot_sliding_window_diagram(session, fs, winLenSec, overlap)

sig = session.signal(:,1);
N   = numel(sig);

maxTime    = 12;
maxSamples = min(N, round(maxTime*fs));
sig = sig(1:maxSamples);
t   = (0:maxSamples-1)/fs;

winLenSamples = round(winLenSec * fs);
stepSamples   = round(winLenSamples * (1 - overlap));

w1_start = 1;
w1_end   = w1_start + winLenSamples - 1;

w2_start = w1_start + stepSamples;
w2_end   = w2_start + winLenSamples - 1;

w1_end = min(w1_end, maxSamples);
w2_end = min(w2_end, maxSamples);

t_w1_start = (w1_start-1)/fs;
t_w1_end   = (w1_end-1)/fs;
t_w2_start = (w2_start-1)/fs;
t_w2_end   = (w2_end-1)/fs;

t_ov_start = t_w2_start;
t_ov_end   = t_w1_end;

figure;
plot(t, sig, 'LineWidth',1.0);
hold on;
xlabel('Time (s)');
ylabel('Magnitude / Ax');
title('Overlap Sliding Window');
grid on;

yl = ylim;

drawRect = @(x1,x2,color) patch( ...
    [x1 x2 x2 x1], [yl(1) yl(1) yl(2) yl(2)], color, ...
    'FaceAlpha',0.08, 'EdgeColor',color, 'LineWidth',1.5);

drawRect(t_w1_start, t_w1_end, [0 0.6 0]);       % Window 1
drawRect(t_w2_start, t_w2_end, [0.9 0.4 0]);     % Window 2
drawRect(t_ov_start, t_ov_end, [0.8 0 0]);       % Overlap

plot([t_w1_start t_w1_start], yl, 'k--','LineWidth',0.8);
plot([t_w1_end   t_w1_end],   yl, 'k--','LineWidth',0.8);
plot([t_w2_start t_w2_start], yl, 'k--','LineWidth',0.8);
plot([t_w2_end   t_w2_end],   yl, 'k--','LineWidth',0.8);

text((t_w1_start+t_w1_end)/2, yl(1)+0.05*(yl(2)-yl(1)), ...
    'Window 1', 'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
    'FontWeight','bold', 'Color',[0 0.4 0]);

text((t_w2_start+t_w2_end)/2, yl(1)+0.05*(yl(2)-yl(1)), ...
    'Window 2', 'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
    'FontWeight','bold', 'Color',[0.7 0.3 0]);

text((t_ov_start+t_ov_end)/2, yl(1)+0.55*(yl(2)-yl(1)), ...
    'Overlap', 'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
    'FontWeight','bold', 'Color',[0.7 0 0]);

hold off;

end

function [dtwDistMat, confMat, perUserAccPct, overallAccPct] = ...
    run_dtw_analysis(allWindows, labelsUser, fs, numPerUser)

users    = unique(labelsUser);
numUsers = numel(users);
L        = size(allWindows,2)/6;
N        = size(allWindows,1);

magSeries = zeros(N, L);
for i = 1:N
    winMat = reshape(allWindows(i,:), L, 6);
    magSeries(i,:) = sqrt(sum(winMat.^2, 2));
end

templates = zeros(numUsers, L);
for uIdx = 1:numUsers
    u = users(uIdx);
    idx = (labelsUser == u);
    templates(uIdx,:) = mean(magSeries(idx,:), 1);
end

dtwDistMat = zeros(numUsers);
for i = 1:numUsers
    for j = 1:numUsers
        dtwDistMat(i,j) = dtw_distance(templates(i,:), templates(j,:));
    end
end

figure;
imagesc(dtwDistMat);
colorbar;
axis square;
xlabel('User ID'); ylabel('User ID');
title('DTW distance matrix between user templates');
set(gca,'XTick',1:numUsers,'YTick',1:numUsers);

selIdx = [];
for uIdx = 1:numUsers
    u = users(uIdx);
    idxU = find(labelsUser == u);
    if numel(idxU) <= numPerUser
        selIdx = [selIdx; idxU(:)];
    else
        selIdx = [selIdx; randsample(idxU, numPerUser)];
    end
end

numSel = numel(selIdx);
predLabels = zeros(numSel,1);
trueLabels = labelsUser(selIdx);

for k = 1:numSel
    testIdx = selIdx(k);
    trainIdx = setdiff(selIdx, testIdx);

    tmpl = zeros(numUsers, L);
    for uIdx = 1:numUsers
        u = users(uIdx);
        idxU = trainIdx(labelsUser(trainIdx) == u);
        tmpl(uIdx,:) = mean(magSeries(idxU,:), 1);
    end

    testSeq = magSeries(testIdx,:);
    dists = zeros(numUsers,1);
    for uIdx = 1:numUsers
        dists(uIdx) = dtw_distance(testSeq, tmpl(uIdx,:));
    end

    [~, bestIdx] = min(dists);
    predLabels(k) = users(bestIdx);
end

confMat = confusionmat(trueLabels, predLabels);

figure;
imagesc(confMat);
colorbar;
axis square;
xlabel('Predicted user'); ylabel('True user');
title('DTW leave-one-out confusion matrix');
set(gca,'XTick',users,'YTick',users);

perUserAccPct = zeros(numUsers,1);
for uIdx = 1:numUsers
    row = confMat(uIdx,:);
    perUserAccPct(uIdx) = 100 * row(uIdx)/max(1,sum(row));
end
overallAccPct = 100 * trace(confMat) / max(1,sum(confMat(:)));

figure;
bar(1:numUsers, perUserAccPct);
xlabel('User ID');
ylabel('Accuracy (%)');
title('DTW LOOCV per-user accuracy');
grid on;

louoMat = zeros(numUsers);

for u0Idx = 1:numUsers
    u0 = users(u0Idx);
    
    trainUsers = users(users ~= u0);
    numTrainUsers = numel(trainUsers);
    
    trainTmpl = zeros(numTrainUsers, L);
    for tIdx = 1:numTrainUsers
        u = trainUsers(tIdx);
        idxU = (labelsUser == u);
        trainTmpl(tIdx,:) = mean(magSeries(idxU,:), 1);
    end
    
    idxTest = (labelsUser == u0);
    testSeqs = magSeries(idxTest,:);
    numTest  = sum(idxTest);
    
    for k = 1:numTest
        dists = zeros(numTrainUsers,1);
        for tIdx = 1:numTrainUsers
            dists(tIdx) = dtw_distance(testSeqs(k,:), trainTmpl(tIdx,:));
        end
        [~, bestIdx] = min(dists);
        predictedUser = trainUsers(bestIdx);
        
        louoMat(u0Idx, predictedUser) = louoMat(u0Idx, predictedUser) + 1;
    end
end

figure;
imagesc(louoMat);
colorbar;
axis square;
xlabel('Closest enrolled user ID');
ylabel('Left-out user ID');
title('DTW Leave-One-User-Out similarity matrix');
set(gca,'XTick',users,'YTick',users);

end

function d = dtw_distance(x, y)

x = x(:); y = y(:);
n = numel(x); m = numel(y);

D = inf(n+1, m+1);
D(1,1) = 0;

for i = 1:n
    for j = 1:m
        cost = (x(i) - y(j))^2;
        D(i+1,j+1) = cost + min( [ D(i,j+1), D(i+1,j), D(i,j) ] );
    end
end

d = sqrt(D(n+1,m+1));
end

function plot_intra_inter_variance_detailed(X, labelsUser, featName)

users    = unique(labelsUser);
numUsers = numel(users);
numFeat  = size(X,2);

muUser   = zeros(numUsers, numFeat);
varUser  = zeros(numUsers, numFeat);

for ui = 1:numUsers
    u   = users(ui);
    idx = (labelsUser == u);
    Xi  = X(idx,:);
    muUser(ui,:)  = mean(Xi, 1);
    varUser(ui,:) = var(Xi, 0, 1);
end

intraMeanVar = mean(varUser, 1);
intraStdVar  = std(varUser, 0, 1);

interVar = var(muUser, 0, 1);

featIdx = 1:numFeat;

figure;
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
sgtitle(sprintf('Intra / Inter-variance (%s)', featName));

nexttile; hold on;
for ui = 1:numUsers
    plot(featIdx, varUser(ui,:), 'LineWidth', 0.8);
end
plot(featIdx, intraMeanVar, 'k', 'LineWidth', 2);
xlabel('Feature index');
ylabel('Variance');
title('Intra-variance per user');
grid on;
legendStrings = arrayfun(@(u) sprintf('User %d',u), users, 'UniformOutput', false);
legendStrings{end+1} = 'Mean across users';
legend(legendStrings,'Location','bestoutside');
hold off;

nexttile;
plot(featIdx, intraStdVar, 'LineWidth', 1.5);
xlabel('Feature index');
ylabel('Std of intra-variance');
title('Std of intra-variance');
grid on;

nexttile;
plot(featIdx, interVar, 'r', 'LineWidth', 1.5);
xlabel('Feature index');
ylabel('Variance');
title('Inter-variance (user means)');
grid on;

end

function plot_feature_variance_surface(X, labelsUser, featName)

users    = unique(labelsUser);
numUsers = numel(users);
numFeat  = size(X,2);

varUser = zeros(numUsers, numFeat);

for ui = 1:numUsers
    u   = users(ui);
    idx = (labelsUser == u);
    Xi  = X(idx,:);
    varUser(ui,:) = var(Xi, 0, 1);
end

[featGrid, userGrid] = meshgrid(1:numFeat, 1:numUsers);

figure;
surf(featGrid, userGrid, varUser, 'EdgeColor','none');
xlabel('Feature Index');
ylabel('User Index');
zlabel('Variance');
title(sprintf('Feature Variances Across Users (%s)', featName));
colorbar;
view(45, 30);
grid on;

end
