clc;
clear;
close all;

%% ================== COUGH SOUND CLASSIFICATION PROJECT ==================
% Enhanced DSP + Ensemble ML version
% Classes: COVID, Tuberculosis, Healthy
% Author: Team (4 Members)
% -------------------------------------------------------------------------

%% Dataset Paths
baseFolder = 'C:\Users\apoti\Downloads\cough dataset';

covidFolder        = fullfile(baseFolder, 'Covid');
tuberculosisFolder = fullfile(baseFolder, 'Tuberculosis');
healthyFolder      = fullfile(baseFolder, 'Healthy');

% Create Audio Datastore
ads = audioDatastore({covidFolder, tuberculosisFolder, healthyFolder}, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.wav', ...
    'LabelSource', 'foldernames');

disp('Dataset Summary:');
disp(countEachLabel(ads));

%% ================== FEATURE EXTRACTION ==================
features = [];
labels   = [];

files = ads.Files;
for k = 1:numel(files)
    [y, Fs] = audioread(files{k});
    y = preprocessAudio(y, Fs);
    feat = extractFeatures(y, Fs);
    features = [features; feat];
    labels   = [labels; string(ads.Labels(k))];
end

labels = categorical(labels);

%% ================== CLASS BALANCING ==================
disp('Balancing dataset using oversampling...');
tbl = table(features, labels);
[grpCounts, grpNames] = groupcounts(tbl.labels);

maxCount = max(grpCounts);
newFeatures = [];
newLabels = [];

for i = 1:numel(grpNames)
    idx = tbl.labels == grpNames(i);
    f = tbl.features(idx,:);
    n = size(f,1);
    reps = ceil(maxCount / n);
    fAug = repmat(f, reps, 1);
    fAug = fAug(1:maxCount,:);
    newFeatures = [newFeatures; fAug];
    newLabels = [newLabels; repmat(grpNames(i), maxCount, 1)];
end

features = newFeatures;
labels = categorical(newLabels);

%% ================== TRAIN / TEST SPLIT ==================
cv = cvpartition(labels, 'HoldOut', 0.3);
XTrain = features(training(cv), :);
YTrain = labels(training(cv));
XTest  = features(test(cv), :);
YTest  = labels(test(cv));

%% ================== TRAIN CLASSIFIER ==================
disp('Training RUSBoost Ensemble Classifier...');
t = templateTree('MaxNumSplits', 50);
model = fitcensemble(XTrain, YTrain, 'Method', 'RUSBoost', ...
                     'NumLearningCycles', 150, 'Learners', t);

%% ================== TEST & EVALUATE ==================
YPred = predict(model, XTest);
YPred = categorical(YPred);
accuracy = mean(YPred == YTest) * 100;
disp(['Classification Accuracy: ', num2str(accuracy,'%.2f'), '%']);

% Confusion chart
figure;
confusionchart(YTest, YPred);
title('Cough Classification: COVID vs TB vs Healthy');

% Class-level precision and recall
C = confusionmat(YTest, YPred);
precision = diag(C)./sum(C,2);
recall = diag(C)./sum(C,1)';
disp(table(categories(YTest), precision, recall));

%% ================== INTERACTIVE DEMO ==================
disp('Select a cough file to classify...');
classifyCoughInteractive(model);

%% ================== FUNCTIONS ==================

% -------------------- Preprocessing --------------------
function y = preprocessAudio(y, Fs)
    if size(y,2) > 1, y = mean(y,2); end
    y = y / max(abs(y));
    frameLen = round(0.02*Fs);
    energy = buffer(y, frameLen, 0, 'nodelay');
    energy = sum(energy.^2);
    idx = find(energy > 0.001, 1, 'last');
    if ~isempty(idx)
        y = y(1:idx*frameLen);
    end
end

% -------------------- Feature Extraction --------------------
function feat = extractFeatures(y, Fs)
    if size(y,2) > 1, y = mean(y,2); end

    % Time-domain
    zcr = mean(abs(diff(sign(y))));
    rmsEnergy = rms(y);
    entropy = -sum((y.^2) .* log(y.^2 + eps));

    % Frequency-domain
    N = length(y);
    Y = abs(fft(y));
    f = (0:N-1)*(Fs/N);
    half = 1:floor(N/2);
    f = f(half); Y = Y(half);
    specCentroid = sum(f'.*Y)/sum(Y);
    specBandwidth = sqrt(sum(((f'-specCentroid).^2).*Y)/sum(Y));
    specRollOff = f(find(cumsum(Y) >= 0.85*sum(Y),1));

    % Cepstral-domain
    coeffs = mfcc(y, Fs, "NumCoeffs", 13);
    mfccMean = mean(coeffs,1);
    mfccStd = std(coeffs,0,1);

    feat = [zcr, rmsEnergy, entropy, specCentroid, specBandwidth, ...
            specRollOff, mfccMean, mfccStd];
end

% -------------------- Interactive Classification --------------------
function classifyCoughInteractive(model)
    [file, path] = uigetfile('*.wav', 'Select a cough audio file');
    if isequal(file,0)
        disp('User cancelled.');
        return;
    end
    filePath = fullfile(path, file);
    [y, Fs] = audioread(filePath);
    y = preprocessAudio(y, Fs);
    t = (0:length(y)-1)/Fs;

    feat = extractFeatures(y, Fs);
    YPred = predict(model, feat);
    if iscell(YPred), YPred = categorical(YPred); end
    disp(['Predicted Class: ', char(YPred)]);

    % DSP Visualization
    zcr = mean(abs(diff(sign(y))));
    N = length(y);
    Y = abs(fft(y));
    f = (0:N-1)*(Fs/N);
    half = 1:floor(N/2);
    f = f(half); Y = Y(half);
    specCentroid = sum(f'.*Y)/sum(Y);
    specBandwidth = sqrt(sum(((f'-specCentroid).^2).*Y)/sum(Y));

    window = round(0.03*Fs);
    noverlap = round(0.02*Fs);
    nfft = 1024;

    figure;
    subplot(3,1,1);
    plot(t, y); xlabel('Time (s)'); ylabel('Amplitude');
    title(['Waveform | ZCR = ', num2str(zcr,'%.3f')]);

    subplot(3,1,2);
    plot(f, Y); hold on;
    xline(specCentroid,'r','LineWidth',1.5);
    xline(specCentroid+specBandwidth,'g--');
    xline(specCentroid-specBandwidth,'g--');
    xlabel('Frequency (Hz)'); ylabel('Magnitude');
    title(['FFT | Centroid=', num2str(specCentroid,'%.1f'),' Hz, BW=', num2str(specBandwidth,'%.1f')]);

    subplot(3,1,3);
    spectrogram(y, window, noverlap, nfft, Fs, 'yaxis');
    title(['Spectrogram | Predicted: ', char(YPred)]);
    colorbar;
end
