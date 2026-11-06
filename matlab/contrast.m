%% =========================================================================
%% DentalReg-MDRN: Dental Radiograph Analysis Pipeline
%% Step 2: Photometric Alignment via Histogram Matching
%% Author: İpek Koçhisarlı
%% =========================================================================
clear; clc; close all;

% Resolve and add data paths
script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, '..', 'data')));
addpath(genpath(fullfile(script_dir, '..')));

% 1. Load cropped ROI images
ref_roi  = imread('ref_roi.bmp');
test_roi = imread('test_roi.bmp');

% 2. Histogram matching for global contrast correction
% Maps the intensity distribution of the moving test image to the reference image
test_matched = imhistmatch(test_roi, ref_roi);

% 3. Save contrast-corrected moving image
imwrite(test_matched, 'test_matched.bmp');

% 4. Compute preliminary metrics
p_before = psnr(test_roi, ref_roi);
p_after  = psnr(test_matched, ref_roi);

fprintf('Step 2 complete: Photometric Alignment via Histogram Matching\n');
fprintf('  PSNR before contrast matching: %.2f dB\n', p_before);
fprintf('  PSNR after contrast matching:  %.2f dB (+%.2f dB gain)\n', p_after, p_after - p_before);

% 5. Display comparative results
figure('Name', 'Contrast Correction', 'NumberTitle', 'off');
subplot(1, 3, 1); imshow(ref_roi);      title('Reference ROI', 'FontWeight', 'bold');
subplot(1, 3, 2); imshow(test_roi);     title(sprintf('Original Test (%.2f dB)', p_before), 'FontWeight', 'bold');
subplot(1, 3, 3); imshow(test_matched); title(sprintf('Matched Test (%.2f dB)', p_after), 'FontWeight', 'bold');
