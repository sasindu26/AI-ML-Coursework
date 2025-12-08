function run_pca_analysis(X_TFDF, labelsUser, idxTrainA, idxTestA, hiddenSizes, numThresholds, mainTargetImpostorRatio)
    %
    
    plot_pca_visualisation(X_TFDF, labelsUser, 'TFDF');
    
    fprintf('\n=== PCA DIMENSIONALITY REDUCTION (TFDF, Split A) ===\n');

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
    
    % Display PCA Table
    T_pca = table(pcaDims', accPCA, eerPCA, aucPCA, ...
        'VariableNames', {'NumPCs','Accuracy','EER','AUC'});
    fprintf('\n=== PCA-based dimensionality reduction results ===\n');
    disp(T_pca);
end

function plot_pca_visualisation(X, labelsUser, featName)
%
[~, score, ~] = pca(X);
figure;
gscatter(score(:,1), score(:,2), labelsUser);
xlabel('PC1'); ylabel('PC2');
title(['PCA scatter (' featName ')']);
grid on;
end
