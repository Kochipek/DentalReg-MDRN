%% =========================================================================
%% DentalReg-MDRN: Dental Radiograph Analysis Pipeline
%% Step 4: Multi-Model Geometric Registration (6 Transforms x 3 Interpolations)
%% Author: İpek Koçhisarlı
%% =========================================================================
clear; clc; close all;

% Resolve and add data paths
script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, '..', 'data')));
addpath(genpath(fullfile(script_dir, '..')));

% 1. Load ROI images
ref_img  = imread('ref_roi.bmp');
test_img = imread('test_matched.bmp');

% 2. Load 64 curated anatomical control points (Landmarks)
% If manual_select is true, cpselect GUI is invoked
manual_select = false;
if manual_select
    [movingPoints, fixedPoints] = cpselect(test_img, ref_img, 'Wait', true);
    save('control_points.mat', 'movingPoints', 'fixedPoints');
else
    load('control_points.mat', 'movingPoints', 'fixedPoints');
end

fprintf('Loaded %d corresponding anatomical landmark pairs.\n', size(movingPoints, 1));

% 3. Define 6 geometric transformation models and 3 interpolation methods
tforms  = {'similarity', 'affine', 'projective', 'polynomial', 'pwl', 'lwm'};
interps = {'nearest', 'linear', 'cubic'};

outputRef = imref2d(size(ref_img));
results_psnr = zeros(length(tforms), length(interps));
results_mse  = zeros(length(tforms), length(interps));

best_psnr = -Inf;
best_registered = test_img;
best_mask = true(size(ref_img));
best_t = '';
best_i = '';

fprintf('\n%-12s %-10s %10s %10s\n', 'Transform', 'Interp.', 'MSE', 'PSNR (dB)');
fprintf('%s\n', repmat('-', 1, 46));

% 4. Comprehensive 18-combination evaluation loop
for t = 1:length(tforms)
    t_name = tforms{t};
    
    % Fit geometric transformation
    if strcmp(t_name, 'polynomial')
        tform = fitgeotrans(movingPoints, fixedPoints, 'polynomial', 2);
    elseif strcmp(t_name, 'lwm')
        tform = fitgeotrans(movingPoints, fixedPoints, 'lwm', 12);
    else
        tform = fitgeotrans(movingPoints, fixedPoints, t_name);
    end
    
    for i = 1:length(interps)
        i_name = interps{i};
        
        % Warp image using specific interpolation
        warped = imwarp(test_img, tform, i_name, 'OutputView', outputRef, 'FillValues', 0);
        
        % Exclude rotation fill-value boundary artifacts from error computation
        mask = (warped > 0);
        mse_val = mean((double(ref_img(mask)) - double(warped(mask))).^2);
        psnr_val = 10 * log10((255^2) / mse_val);
        
        results_psnr(t, i) = psnr_val;
        results_mse(t, i)  = mse_val;
        
        fprintf('%-12s %-10s %10.2f %10.2f\n', t_name, i_name, mse_val, psnr_val);
        
        if psnr_val > best_psnr
            best_psnr = psnr_val;
            best_t = t_name;
            best_i = i_name;
            best_registered = warped;
            best_mask = mask;
        end
    end
end

% 5. Save optimal registered image
imwrite(best_registered, 'test_registered.bmp');
save('registration_results.mat', 'results_psnr', 'results_mse', 'tforms', 'interps');

fprintf('\nRegistration Summary:\n');
fprintf('  Optimal Model:        %s + %s interpolation\n', upper(best_t), upper(best_i));
fprintf('  Optimal Tissue PSNR:  %.2f dB\n', best_psnr);

% 6. Visual comparison of optimal result
diff_before = imabsdiff(ref_img, test_img);
diff_after  = imabsdiff(ref_img, best_registered);

figure('Name', 'Registration Results', 'NumberTitle', 'off');
subplot(2, 2, 1); imshow(test_img);        title('Test (Before Registration)', 'FontWeight', 'bold');
subplot(2, 2, 2); imshow(best_registered);  title(sprintf('Optimal: %s + %s (%.2f dB)', best_t, best_i, best_psnr), 'FontWeight', 'bold');
subplot(2, 2, 3); imshow(diff_before, []); title('Absolute Difference (Before)', 'FontWeight', 'bold'); colorbar;
subplot(2, 2, 4); imshow(diff_after, []);  title(sprintf('Absolute Difference (After - %.2f dB)', best_psnr), 'FontWeight', 'bold'); colorbar;
