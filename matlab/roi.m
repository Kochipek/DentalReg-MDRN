%% =========================================================================
%% DentalReg-MDRN: Dental Radiograph Analysis Pipeline
%% Step 1: Region of Interest (ROI) Extraction
%% Author: İpek Koçhisarlı
%% =========================================================================
clear; clc; close all;

% Resolve and add data paths
script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, '..', 'data')));
addpath(genpath(fullfile(script_dir, '..')));

% 1. Load raw periapical dental radiograph images
ref_img  = imread('res1.bmp');
test_img = imread('res2.bmp');

% 2. Define uniform ROI bounding box: [xmin, ymin, width, height]
% Eliminates peripheral sensor borders, film holder artifacts, and zero-padding
rect = [85, 2, 351, 363];

% 3. Crop anatomical regions containing crown, pulp cavities, and root apices
ref_roi  = imcrop(ref_img, rect);
test_roi = imcrop(test_img, rect);

% 4. Save cropped ROI images
imwrite(ref_roi, 'ref_roi.bmp');
imwrite(test_roi, 'test_roi.bmp');

% 5. Display comparative visualization
figure('Name', 'ROI Extraction', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(ref_roi);
title(sprintf('Reference ROI (%dx%d)', size(ref_roi, 1), size(ref_roi, 2)), 'FontWeight', 'bold');

subplot(1, 2, 2);
imshow(test_roi);
title(sprintf('Test ROI (%dx%d)', size(test_roi, 1), size(test_roi, 2)), 'FontWeight', 'bold');

fprintf('Step 1 complete: Extracted ref_roi.bmp and test_roi.bmp (%dx%d pixels).\n', ...
    size(ref_roi, 1), size(ref_roi, 2));
