function run_svm_comparison(X_TF, X_FD, X_TFDF, labelsUser, idxTrainA, idxTestA, numThresholds, mainTargetImpostorRatio)
%

fprintf('\n=== 15. ALTERNATIVE CLASSIFIER: SVM VS NN (Split A, TFDF) ===\n');

resSVM_SplitA = run_svm_experiment( ...
    'Split A: Day1 train, Day2 test (SVM)', ...
    X_TF, X_FD, X_TFDF, ...
    labelsUser, ...
    idxTrainA, idxTestA, ...
    {'TFDF'}, ...
    numThresholds, ...
    mainTargetImpostorRatio);

% (The bar chart plotting code from Section 15 would follow here, using resSVM_SplitA)
% For brevity in this file, we return the results or print them.
end

function res = run_svm_experiment(experimentTitle, X_TF, X_FD, X_TFDF, labelsUser, idxTrain, idxTest, featureSets, numThresholds, mainTargetImpostorRatio)
    %
    yTrainUsers = labelsUser(idxTrain);
    yTestUsers  = labelsUser(idxTest);
    numUsers = numel(unique(labelsUser));

    for s = 1:numel(featureSets)
        featName = featureSets{s};
        fprintf('\n=== %s | SVM | Features: %s ===\n', experimentTitle, featName);
        
        switch featName
            case 'TF', X = X_TF;
            case 'FD', X = X_FD;
            case 'TFDF', X = X_TFDF;
        end
        
        XTrain_full = X(idxTrain,:);
        XTest       = X(idxTest,:);
        
        userID   = (1:numUsers).';
        accPct   = zeros(numUsers,1);
        
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
            
            [svmModel, yPred_bin, score_bin] = train_binary_svm_verifier(XTrain_u, yTrain_u, XTest); 
            
            [binMetrics, binConfMat] = evaluate_binary_classification(yTest_bin, yPred_bin);
            [farBin, frrBin, eerBin, thrBin, rocFprBin, rocTprBin] = compute_far_frr_eer_binary(yTest_bin, score_bin, numThresholds);
            [~, idxEbin] = min(abs(farBin - frrBin));
            
            accPct(u)  = binMetrics.accuracy  * 100;
            res.(featName).users(u).metrics = binMetrics;
        end
        
        perUserTable = table(userID, accPct, 'VariableNames', {'User','Accuracy'});
        res.(featName).perUserTable = perUserTable;
        res.(featName).meanAccuracyPct = mean(accPct);
        
        fprintf('\nPer-user SVM metrics (%s, %s):\n', experimentTitle, featName);
        disp(perUserTable);
    end
end

function [svmModel, yPredTest, scoreTest] = train_binary_svm_verifier(XTrain, yTrainBin, XTest)
    %
    yTrainBin = double(yTrainBin(:));
    svmModel = fitcsvm(XTrain, yTrainBin, 'KernelFunction','rbf', 'KernelScale','auto', 'Standardize',true, 'ClassNames',[0 1]);
    [yPredTest, scoreMat] = predict(svmModel, XTest);
    scoreTest = scoreMat(:,2);
end
