%% =========================================================================
%% DentalReg-MDRN: Dental Radiograph Analysis Pipeline
%% Step 5: Secondary Histogram Matching Benchmark across 18 Transformations
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
load('control_points.mat', 'movingPoints', 'fixedPoints');

% 2. Define configurations
transformNames = {'similarity', 'affine', 'projective', 'polynomial', 'pwl', 'lwm'};
interpNames = {'nearest', 'linear', 'cubic'};
outputRef = imref2d(size(ref));

fprintf('=== Benchmark: Secondary Histogram Matching across 18 Transformations ===\n');
fprintf('%-11s %-9s %9s %9s %8s %9s %9s\n', ...
    'Transform', 'Interp.', 'PSNR old', 'PSNR new', 'Gain', 'MSE old', 'MSE new');
fprintf('%s\n', repmat('-', 1, 72));

% 3. Iterate through all combinations
for t = 1:length(transformNames)
    tName = transformNames{t};
    if strcmp(tName, 'polynomial')
        tform = fitgeotrans(movingPoints, fixedPoints, 'polynomial', 2);
    elseif strcmp(tName, 'lwm')
        tform = fitgeotrans(movingPoints, fixedPoints, 'lwm', 12);
    else
        tform = fitgeotrans(movingPoints, fixedPoints, tName);
    end
    
    for i = 1:length(interpNames)
        iName = interpNames{i};
        reg = imwarp(test_matched, tform, iName, 'OutputView', outputRef);
        reg_matched = imhistmatch(reg, ref);
        
        p_old = psnr(reg, ref);
        p_new = psnr(reg_matched, ref);
        mse_old = mean((double(ref(:)) - double(reg(:))).^2);
        mse_new = mean((double(ref(:)) - double(reg_matched(:))).^2);
        gain = p_new - p_old;
        
        fprintf('%-11s %-9s %9.3f %9.3f %8.3f %9.2f %9.2f\n', ...
            tName, iName, p_old, p_new, gain, mse_old, mse_new);
    end
end
