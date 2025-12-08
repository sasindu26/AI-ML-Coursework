function res = run_binary_experiment(experimentTitle, X_TF, X_FD, X_TFDF, labelsUser, idxTrain, idxTest, featureSets, hiddenSizes, numThresholds, mainTargetImpostorRatio)
%

    yTrainUsers = labelsUser(idxTrain);
    yTestUsers  = labelsUser(idxTest);
    numUsers = numel(unique(labelsUser));

    for s = 1:numel(featureSets)
        featName = featureSets{s};
        fprintf('\n=== %s | Features: %s ===\n', experimentTitle, featName);
        
        switch featName
            case 'TF', X = X_TF;
            case 'FD', X = X_FD;
            case 'TFDF', X = X_TFDF;
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
            
            [netBin, yPred_bin, score_bin] = train_binary_nn_verifier(XTrain_u, yTrain_u, XTest, hiddenSizes);
            [binMetrics, binConfMat] = evaluate_binary_classification(yTest_bin, yPred_bin);
            [farBin, frrBin, eerBin, thrBin, rocFprBin, rocTprBin] = compute_far_frr_eer_binary(yTest_bin, score_bin, numThresholds);
            
            [~, idxEbin] = min(abs(farBin - frrBin));
            farAtEER = farBin(idxEbin);
            frrAtEER = frrBin(idxEbin);
            [sortedFpr, idxSort] = sort(rocFprBin);
            sortedTpr = rocTprBin(idxSort);
            auc = trapz(sortedFpr, sortedTpr);
            
            accPct(u)  = binMetrics.accuracy * 100;
            precPct(u) = binMetrics.precision * 100;
            recPct(u)  = binMetrics.recall * 100;
            f1Pct(u)   = binMetrics.f1 * 100;
            farPct(u)  = farAtEER * 100;
            frrPct(u)  = frrAtEER * 100;
            eerPct(u)  = eerBin * 100;
            aucVal(u)  = auc;
            
            res.(featName).users(u).scores = score_bin;
            res.(featName).users(u).confMat = binConfMat;
        end
        
        perUserTable = table(userID, accPct, aucVal, precPct, recPct, f1Pct, farPct, frrPct, eerPct, ...
            'VariableNames', {'User','Accuracy','AUC','Precision','Recall','F1','FAR','FRR','EER'});
        res.(featName).perUserTable = perUserTable;
        fprintf('\nPer-user metrics (%s, %s):\n', experimentTitle, featName);
        disp(perUserTable);
    end
end

function [netBin, yPredTest, scoreTest] = train_binary_nn_verifier(XTrain, yTrainBin, XTest, hiddenSizes)
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

function [metrics, confMat] = evaluate_binary_classification(yTrueBin, yPredBin)
    yTrueBin = double(yTrueBin(:));
    yPredBin = double(yPredBin(:));
    confMat = confusionmat(yTrueBin, yPredBin);
    if numel(confMat) == 1
        if yTrueBin(1) == 0, confMat = [confMat 0; 0 0]; else, confMat = [0 0; 0 confMat]; end
    end
    TN = confMat(1,1); FP = confMat(1,2); FN = confMat(2,1); TP = confMat(2,2);
    denAll = max(1, (TP+TN+FP+FN));
    metrics.accuracy  = (TP + TN) / denAll;
    metrics.precision = TP / max(1, (TP + FP));
    metrics.recall    = TP / max(1, (TP + FN));
    metrics.f1        = 2 * metrics.precision * metrics.recall / max(1e-12, (metrics.precision + metrics.recall));
end

function [far, frr, eer, thresholds, rocFpr, rocTpr] = compute_far_frr_eer_binary(yTrueBin, scores, numThresholds)
    genuineIdx  = (yTrueBin == 1);
    impostorIdx = (yTrueBin == 0);
    thresholds = linspace(0,1,numThresholds);
    far = zeros(size(thresholds));
    frr = zeros(size(thresholds));
    for k = 1:numel(thresholds)
        t = thresholds(k);
        if any(impostorIdx), far(k) = mean(scores(impostorIdx) >= t); else, far(k) = 0; end
        if any(genuineIdx), frr(k) = mean(scores(genuineIdx)  <  t); else, frr(k) = 0; end
    end
    [~, idxMin] = min(abs(far - frr));
    eer = (far(idxMin) + frr(idxMin)) / 2;
    scoreVals = scores(:); labelsBin = yTrueBin(:);
    rocThresh = sort(unique(scoreVals));
    rocFpr = zeros(size(rocThresh)); rocTpr = zeros(size(rocThresh));
    for k = 1:numel(rocThresh)
        t = rocThresh(k);
        preds = scoreVals >= t;
        TP = sum(preds==1 & labelsBin==1); FP = sum(preds==1 & labelsBin==0);
        FN = sum(preds==0 & labelsBin==1); TN = sum(preds==0 & labelsBin==0);
        rocTpr(k) = TP / max(1, (TP + FN)); rocFpr(k) = FP / max(1, (FP + TN));
    end
end
