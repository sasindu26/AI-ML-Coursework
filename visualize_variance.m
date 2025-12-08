function visualize_variance(X_TF, X_FD, X_TFDF, labelsUser)
    % Wrapper to run all visualization plots
    
    plot_feature_statistics(X_TFDF, labelsUser, 'TFDF');
    plot_feature_correlation(X_TFDF, 'TFDF');
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
end

function plot_feature_statistics(X, labelsUser, featName)
%
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
%
R = corrcoef(X);
figure;
imagesc(R);
colorbar;
title(['Feature Correlation - ' featName]);
xlabel('Feature index'); ylabel('Feature index');
axis square;
end

function plot_feature_variability(X, labelsUser, featName)
%
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
%
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
%
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
%
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

function plot_intra_inter_variance_detailed(X, labelsUser, featName)
%
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
%
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
