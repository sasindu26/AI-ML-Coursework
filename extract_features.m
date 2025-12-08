function [X_TF, X_FD, X_TFDF] = extract_features(allWindows, fs)
%
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
%
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
%
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
