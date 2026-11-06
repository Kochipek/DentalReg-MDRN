%% =========================================================================
%% DentalReg-MDRN: Dental Radiograph Analysis Pipeline
%% Step 6: Automated Registration Benchmark (Intensity vs Phase Correlation)
%% Author: İpek Koçhisarlı
%% =========================================================================
clear; clc; close all;

% Resolve and add data paths
script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, '..', 'data')));
addpath(genpath(fullfile(script_dir, '..')));

% 1. Load data
ref = imread('ref_roi.bmp');
test_matched = imread('test_matched.bmp');
outputRef = imref2d(size(ref));

% 2. Intensity-Based Automatic Registration (imregtform)
% Gradient descent optimization minimizing Mean Squared Error / Mutual Information
fprintf('=== Intensity-Based Automatic Registration (imregtform) ===\n');
[optimizer, metric] = imregconfig('monomodal');
optimizer.MaximumIterations = 300;
autoTypes = {'similarity', 'affine'};

for k = 1:length(autoTypes)
    tType = autoTypes{k};
    tform_auto = imregtform(test_matched, ref, tType, optimizer, metric);
    reg_auto = imwarp(test_matched, tform_auto, 'OutputView', outputRef);
    reg_auto_m = imhistmatch(reg_auto, ref);
    
    p_b = psnr(reg_auto, ref);
    p_a = psnr(reg_auto_m, ref);
    fprintf('%-11s | Before Matching: %7.3f dB | After Matching: %7.3f dB | Net Gain: %+6.3f dB\n', ...
        tType, p_b, p_a, p_a - p_b);
end

% 3. Frequency-Domain Phase Correlation Registration (imregcorr)
% Analytical cross-power spectrum peak localization in 2D FFT domain
fprintf('\n=== Frequency-Domain Phase Correlation Registration (imregcorr) ===\n');
corrTypes = {'translation', 'rigid', 'similarity'};

for k = 1:length(corrTypes)
    cType = corrTypes{k};
    tform_corr = imregcorr(test_matched, ref, cType);
    reg_corr = imwarp(test_matched, tform_corr, 'OutputView', outputRef, 'Interpolation', 'linear');
    reg_corr_m = imhistmatch(reg_corr, ref);
    
    p_b = psnr(reg_corr, ref);
    p_a = psnr(reg_corr_m, ref);
    fprintf('%-11s | Before Matching: %7.3f dB | After Matching: %7.3f dB | Net Gain: %+6.3f dB\n', ...
        cType, p_b, p_a, p_a - p_b);
end
