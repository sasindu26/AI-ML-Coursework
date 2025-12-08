function run_neural_network(X_TF, X_FD, X_TFDF, labelsUser, idxTrain, idxTest, hiddenSizes, numThresholds, mainTargetImpostorRatio, labelsDay)

    %
    featureSets = {'TF','FD','TFDF'};
    resultsSplits = struct;

    fprintf('\n===============================================\n');
    fprintf('Running binary experiments for Split A\n');
    fprintf('===============================================\n');
    
    resultsSplits.Day1Train_Day2Test = run_binary_experiment( ...
        'Split A: Day1 train, Day2 test', ...
        X_TF, X_FD, X_TFDF, ...
        labelsUser, ...
        idxTrain, idxTest, ...
        featureSets, ...
        hiddenSizes, numThresholds, ...
        mainTargetImpostorRatio);

    %
    perUserTable_TDFD_SplitA = resultsSplits.Day1Train_Day2Test.TFDF.perUserTable;
    fprintf('\nPer-user evaluation metrics (Split A, TFDF):\n');
    disp(perUserTable_TDFD_SplitA);
    
    % User Similarity Matrix
    idxTestA   = (labelsDay == 2);
    testUsers  = labelsUser(idxTestA);
    resA_TFDF  = resultsSplits.Day1Train_Day2Test.TFDF;
    numUsers   = numel(unique(labelsUser));
    
    simMean = zeros(numUsers);
    
    for modelU = 1:numUsers
        scoresU = resA_TFDF.users(modelU).scores;
        for dataU = 1:numUsers
            idx = (testUsers == dataU);
            if any(idx)
                vals = scoresU(idx);
                simMean(modelU, dataU) = mean(vals);
            else
                simMean(modelU, dataU) = NaN;
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

end
