%% =========================================================================
%% DentalReg-MDRN: Dental Radiograph Analysis Pipeline
%% Step 3: Intensity Histogram & Cumulative Distribution Analysis
%% Author: İpek Koçhisarlı
%% =========================================================================
clear; clc; close all;

% Resolve and add data paths
script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, '..', 'data')));
addpath(genpath(fullfile(script_dir, '..')));

% 1. Load images
ref_roi     = imread('ref_roi.bmp');
test_before = imread('test_roi.bmp');
test_after  = imread('test_matched.bmp');

% 2. Compute normalized histograms (Probability Density Function - PDF)
ref_norm    = imhist(ref_roi) / numel(ref_roi);
before_norm = imhist(test_before) / numel(test_before);
after_norm  = imhist(test_after) / numel(test_after);

% 3. Compute cumulative frequency histograms (Cumulative Distribution Function - CDF)
ref_cdf     = cumsum(ref_norm);
before_cdf  = cumsum(before_norm);
after_cdf   = cumsum(after_norm);

% 4. 2x3 Grid Visualization: Individual PDF and CDF curves
figure('Name', 'Histogram Analysis (PDF & CDF)', 'NumberTitle', 'off');
subplot(2, 3, 1); bar(0:255, ref_norm, 'k'); title('Reference - Normalized PDF'); xlim([0 255]); grid on;
subplot(2, 3, 2); bar(0:255, before_norm, 'r'); title('Test (Before) - Normalized PDF'); xlim([0 255]); grid on;
subplot(2, 3, 3); bar(0:255, after_norm, 'g'); title('Test (After) - Normalized PDF'); xlim([0 255]); grid on;

subplot(2, 3, 4); plot(0:255, ref_cdf, 'k', 'LineWidth', 1.5); title('Reference - Cumulative CDF'); grid on; xlim([0 255]); ylim([0 1]);
subplot(2, 3, 5); plot(0:255, before_cdf, 'r', 'LineWidth', 1.5); title('Test (Before) - Cumulative CDF'); grid on; xlim([0 255]); ylim([0 1]);
subplot(2, 3, 6); plot(0:255, after_cdf, 'g', 'LineWidth', 1.5); title('Test (After) - Cumulative CDF'); grid on; xlim([0 255]); ylim([0 1]);

% 5. Overlaid Comparative Visualization
figure('Name', 'Overlaid Histogram Comparison', 'NumberTitle', 'off');
subplot(1, 2, 1);
plot(0:255, ref_norm, 'k', 0:255, before_norm, 'r--', 0:255, after_norm, 'g:', 'LineWidth', 1.2);
title('Normalized Histogram (PDF) Overlay', 'FontWeight', 'bold');
xlabel('Gray Level Intensity'); ylabel('Normalized Frequency');
legend('Reference', 'Test (Before)', 'Test (After)');
grid on; xlim([0 255]);

subplot(1, 2, 2);
plot(0:255, ref_cdf, 'k', 'LineWidth', 1.5); hold on;
plot(0:255, before_cdf, 'r--', 'LineWidth', 1.5);
plot(0:255, after_cdf, 'g:', 'LineWidth', 2.0);
title('Cumulative Histogram (CDF) Overlay', 'FontWeight', 'bold');
xlabel('Gray Level Intensity'); ylabel('Cumulative Probability');
legend('Reference', 'Test (Before)', 'Test (After)', 'Location', 'SouthEast');
grid on; xlim([0 255]); ylim([0 1]);

fprintf('Step 3 complete: Computed and plotted normalized PDFs and CDFs.\n');
