clc; clear; close all;

%% Dataset Paths
baseFolder ='C:\Users\apoti\Downloads\cough dataset';
covidFolder = fullfile(baseFolder, 'Covid');
tuberculosisFolder = fullfile(baseFolder, 'Tuberculosis');
healthyFolder = fullfile(baseFolder, 'Healthy');

%% Create Audio Datastore
ads = audioDatastore({covidFolder, tuberculosisFolder, healthyFolder}, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.wav', ...
    'LabelSource', 'foldernames');

disp(countEachLabel(ads)); % show number of files per class

%% Extract Features
features = [];
labels = [];
files = ads.Files; % all file paths
for k = 1:numel(files)
    [y, Fs] = audioread(files{k});
    y = preprocessAudio(y, Fs); % clean audio
    feat = extractFeatures(y, Fs); % DSP features
    features = [features; feat]; % Bug: fails if feat lengths differ
    labels = [labels; string(ads.Labels(k))];
end
labels = categorical(labels);

%% Train/Test Split
cv = cvpartition(labels,'HoldOut',0.3);
XTrain = features(training(cv),:);
YTrain = labels(training(cv));
XTest  = features(test(cv),:);
YTest  = labels(test(cv));

%% Train Classifier
disp('Training Random Forest Classifier...');
model = TreeBagger(200, XTrain, YTrain, ...
                   'OOBPrediction','On', 'Method','classification');

%% Test Accuracy
YPred = predict(model, XTest);
YPred = categorical(YPred); % Bug: may fail if YPred already categorical
accuracy = mean(YPred == YTest) * 100;
disp(['Classification Accuracy: ', num2str(accuracy,'%.2f'), '%']);

figure; confusionchart(YTest, YPred);
title('Cough Classification: COVID vs TB vs Healthy');

%% Interactive Demo
disp('Now select a cough file for classification...');
classifyCoughInteractive(model);

%% ================= FUNCTIONS =================

% Audio Preprocessing (remove trailing silence)
function y = preprocessAudio(y, Fs)
    if size(y,2) > 1, y = mean(y,2); end
    y = y / max(abs(y)); % normalize
    frameLen = round(0.02*Fs);
    energy = buffer(y, frameLen, 0, 'nodelay');
    energy = sum(energy.^2);
    idx = find(energy > 0.001, 1, 'last'); % last non-silent frame
    y = y(1:idx*frameLen); % Bug: fails if idx is empty (silent audio)
end

% Feature Extraction
function feat = extractFeatures(y, Fs)
    % Time-domain: Zero Crossing Rate
    zcr = mean(abs(diff(sign(y))));
    
    % Frequency-domain
    N = length(y);
    Y = abs(fft(y));
    f = (0:N-1)*(Fs/N);
    half = 1:floor(N/2);
    f = f(half); Y = Y(half);
    specCentroid = sum(f'.*Y)/sum(Y);
    specBandwidth = sqrt(sum(((f'-specCentroid).^2).*Y)/sum(Y));
    
    % Cepstral-domain: MFCC
    coeffs = mfcc(y, Fs, "NumCoeffs", 13); % Bug: may error if y too short
    mfccMean = mean(coeffs,1); 
    
    % Final vector
    feat = [zcr, specCentroid, specBandwidth, mfccMean]; % Bug: length mismatch if coeffs empty
end

% Interactive Classification + Visualization
function classifyCoughInteractive(model)
    [file, path] = uigetfile('*.wav', 'Select a cough audio file');
    if isequal(file,0)
        disp('User cancelled file selection.');
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
    
    % DSP Analysis
    zcr = mean(abs(diff(sign(y))));
    N = length(y);
    Y = abs(fft(y));
    f = (0:N-1)*(Fs/N);
    half = 1:floor(N/2);
    f = f(half); Y = Y(half);
    specCentroid = sum(f'.*Y)/sum(Y);
    specBandwidth = sqrt(sum(((f'-specCentroid).^2).*Y)/sum(Y));
    
    % Spectrogram
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
    title(['FFT | Centroid=',  num2str(specCentroid,'%.1f'),' Hz, BW=', num2str(specBandwidth,'%.1f')]);
    
    subplot(3,1,3);
    mspectrogram(y, window, noverlap, nfft, Fs, 'yaxis'); % Bug: mspectrogram may not exist; should be 'spectrogram'
    title(['Spectrogram | Predicted Class: ', char(YPred)]);
    colorbar;
end
