
close all;
clear;
clc;

%% ================================================================
%% READ  IMAGE
%% ================================================================

u = im2double(imread( ...
    "D:\Image_Dataset_collection\Copyright Dataset_SIPI\3.2.25.tiff"));

if size(u,3) == 3
    u = rgb2gray(u);
end

u = double(u);

fprintf('Image size = %d x %d\n',size(u,1),size(u,2));


%% ================================================================
%% GAUSSIAN NOISE
%% ================================================================

sigma_input = 10;

mean_val = 0;
var_val = (sigma_input/255)^2;

% ------------------------------------------------
% FIXED NOISE REALIZATION
% ------------------------------------------------

rng(42,'twister');

noisy_img = imnoise( ...
    u, ...
    'gaussian', ...
    mean_val, ...
    var_val);

noisy_img = min(max(noisy_img,0),1);

% Noisy observation
u0 = noisy_img;


%% ================================================================
%% MODEL PARAMETERS
%% ================================================================

% Caputo fractional order
alpha = 0.01;

% Hausdorff fractal order
beta = 0.01;

% Time step
dt = 0.05;

% Number of iterations
Niter = 10;

% Gaussian smoothing parameter
sigma = 0.1;

% Diffusion threshold
K = 0.10;

% Fidelity parameter
lambda = ;


%% ================================================================
%% INITIAL CONDITION
%% ================================================================

u_current = u0;

% Solution history:
%
% u_hist{1} = u^0
% u_hist{2} = u^1
% ...
%
u_hist = cell(1,Niter+1);

u_hist{1} = u_current;


%% ================================================================
%% RESTORATION - CORRECT L1 CAPUTO UPDATE
%% ================================================================

u0 = noisy_img;
u_current = u0;

u_hist = cell(1,Niter+1);
u_hist{1} = u_current;

tic;

for n = 0:Niter-1

    % ------------------------------------------------------------
    % Diffusion term at current time level
    % ------------------------------------------------------------

    [div_term,c,grad_H] = diffusion_HFD( ...
        u_current, ...
        beta, ...
        sigma, ...
        K);


    % ------------------------------------------------------------
    % Fidelity term
    % ------------------------------------------------------------

     fidelity = lambda * (u_current-u0);


    % ------------------------------------------------------------
    % Right-hand side
    % ------------------------------------------------------------

    RHS = div_term - fidelity;


    % ------------------------------------------------------------
    % Caputo L1 memory term
    %
    % For n = 0, there is no previous history.
    % ------------------------------------------------------------

    history = zeros(size(u_current));

    if n >= 1

        for k = 1:n

            a_k = (k+1)^(1-alpha) - k^(1-alpha);

            history = history + ...
                a_k .* ...
                (u_hist{n-k+2} - u_hist{n-k+1});

        end

        history = history ./ ...
            (gamma(2-alpha) * dt^alpha);

    end


    % ------------------------------------------------------------
    % Intensity constraint
    % ------------------------------------------------------------

    u_new = min(max(u_new,0),1);


    % ------------------------------------------------------------
    % Store
    % ------------------------------------------------------------

    u_hist{n+2} = u_new;

    u_current = u_new;

end

execution_time = toc;

u_final = u_current;

%% ================================================================
%% METRICS
%% ================================================================

psnr_val = psnr(u_final,u);

ssim_val = ssim(u_final,u);

entropy_val = entropy(u_final);


%% ================================================================
%% FINAL RESULTS
%% ================================================================

fprintf('\n');
fprintf('====================================================\n');
fprintf('          FINAL RESTORATION RESULTS\n');
fprintf('====================================================\n');

fprintf('Gaussian noise sigma = %g\n',sigma_input);
fprintf('Caputo order alpha   = %.2f\n',alpha);
fprintf('HFD order beta       = %.2f\n',beta);
fprintf('Time step dt         = %.3f\n',dt);
fprintf('Iterations           = %d\n',Niter);
fprintf('Gaussian sigma       = %.2f\n',sigma);
fprintf('Diffusion K          = %.3f\n',K);
fprintf('Fidelity lambda      = %.4f\n',lambda);

fprintf('----------------------------------------------------\n');

fprintf('PSNR    = %.4f dB\n',psnr_val);
fprintf('SSIM    = %.4f\n',ssim_val);
fprintf('Entropy = %.4f\n',entropy_val);
fprintf('Time    = %.4f seconds\n',execution_time);

fprintf('====================================================\n');


%% ================================================================
%% DIFFUSION COEFFICIENT STATISTICS
%% ================================================================

fprintf('\n');
fprintf('Diffusion coefficient statistics:\n');

fprintf('c min  = %.6f\n',min(c(:)));
fprintf('c max  = %.6f\n',max(c(:)));
fprintf('c mean = %.6f\n',mean(c(:)));

fprintf('\n');
fprintf('HFD gradient statistics:\n');

fprintf('HFD min  = %.6f\n',min(grad_H(:)));
fprintf('HFD max  = %.6f\n',max(grad_H(:)));
fprintf('HFD mean = %.6f\n',mean(grad_H(:)));


%% ================================================================
%% DISPLAY RESULTS
%% ================================================================

figure( ...
    'Color','w', ...
    'Position',[100 100 1250 430]);

subplot(1,3,1);

imshow(u,[]);

title('Original', ...
    'FontWeight','bold');


subplot(1,3,2);

imshow(noisy_img,[]);

title( ...
    sprintf('Gaussian Noise \\sigma = %d',sigma_input), ...
    'FontWeight','bold');


subplot(1,3,3);

imshow(u_final,[]);

title('Restored', ...
    'FontWeight','bold');


%% ================================================================
%% HFD-BASED DIFFUSION FUNCTION
%% ================================================================

function [div_term,c,grad_H] = diffusion_HFD( ...
                    u_current, ...
                    beta, ...
                    sigma, ...
                    K)

    u_current = double(u_current);

    [M,N] = size(u_current);


    %% ------------------------------------------------------------
    % Gaussian smoothing
    %% ------------------------------------------------------------

    G = fspecial( ...
        'gaussian', ...
        [5 5], ...
        sigma);

 
    %% ------------------------------------------------------------

    [gx,gy] = gradient(u_smooth);


    % Pixel coordinates
    x = 1:N;
    y = 1:M;


    % Hausdorff coordinate factors
    hx = (x.^(1-beta))/beta;
    hy = (y.^(1-beta))/beta;


    % Convert to 2-D arrays
    Hx = repmat(hx,M,1);
    Hy = repmat(hy',1,N);


    %% ------------------------------------------------------------
    % Hausdorff gradient
    %% ------------------------------------------------------------

    D_Hx = Hx .* gx;
    D_Hy = Hy .* gy;


    %% ------------------------------------------------------------
    % Magnitude of Hausdorff gradient
    %% ------------------------------------------------------------

    grad_H = sqrt( ...
        D_Hx.^2 + ...
        D_Hy.^2);


    %% ------------------------------------------------------------
    % Ordinary gradient of current image
    %% ------------------------------------------------------------

    [ux,uy] = gradient(u_current);

    grad_u = sqrt( ...ux.^2 + ...
        uy.^2);


    %% ------------------------------------------------------------
    % Diffusion coefficient
       %
    %% ------------------------------------------------------------

    c = 1 ./ ...
        (1 + ...
        (grad_H .* grad_u) ./ (K^2));


    %% ------------------------------------------------------------
    % Diffusion flux
    %% ------------------------------------------------------------

    flux_x = c .* ux;

    flux_y = c .* uy;


    %% ------------------------------------------------------------
    % Divergence
    %% ------------------------------------------------------------

    [flux_xx,~] = gradient(flux_x);

    [~,flux_yy] = gradient(flux_y);


    div_term = flux_xx + flux_yy;

end