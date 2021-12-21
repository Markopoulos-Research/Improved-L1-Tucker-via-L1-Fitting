%% 0-Clear Workspace
clear; clc; close all;

%% 1-Experiment parameters
D = [5,8,5,8,5];        % Tensor dimensionality
d = [3,3,2,2,2];        % low-rank dimensionality
G_std = 3;              % Standard-deviation of the core tensor (Generated by zero-centered Normal distribution)


%% 2-Setup the experiment
% Generates G_true (tensor core), Un_True (true bases)
I = length(D);
G_true = tensor(normrnd(0, G_std, d));
Un_true = generate_orth_basis(I, D, d);
% Generate clean tensor
X_clean = ttm(G_true, Un_true, 1:I);        % Noise free tensor	


%% 2-Experiment parameters
sigma_n = 1;            % Noise std
N_o = 10;                % Number of Outlier entries
sigma_o = 8;				% outlier std
init_method = 'HOSVD';
maxit = 100;

outdir = fullfile(experiment_folder,['sN' num2str(sigma_n) '-pO' num2str(N_o) '-sO' num2str(sigma_o)]);

%%
%% Pre-test
onr = getONR_sparse(Ds, 'sigma_o', sigma_o, 'sigma_n', sigma_n, 'P', P_o, 'P_type', P_type);

%% Corrupt data
Z_n = normrnd(0, sigma_n, size(X_clean));           % Additive Noise tensor with mean = 0, and standard deviation = $sigma_n
X_n = X_clean + Z_n;
outlier_mask = gen_rand_sparse_indices(P_o, Ds, 'P_type', P_type);
Z_o = outlier_mask.*normrnd(0, sigma_o, Ds);          % Additive Outlier tensor with mean = 0, and standard deviation = $sigma_o
X_corr = X_n + Z_o;             % X_corr = ttm(G,Un_true,'t') + N + O

% Initialize Uns
[U0_L1, U0_L2] = initialize_bases(I, Ds, ds, init_method, 'X', X_corr, 'tol', tol);

%% L1 HOOI / L2 proj algorithms
[U_L1L2, G_L1L2, ~,~,stats_L1L2, ~, stats_L1L2_T1] = L1HOOI(X_corr, ds, U0_L1, 'maxit', maxit, 'tol', tol, 'X_clean', X_clean, 'Un_true', Un_true, 'proj', 'L2');
%% L1 HOOI / L1 proj
[U, G, ~,~,stats_L1L1, ~,stats_L1L2_T1] = L1HOOI(X_corr, ds, U0_L1, 'maxit', maxit, 'tol', tol, 'X_clean', X_clean, 'Un_true', Un_true, 'proj','L1');
%% L2 HOOI / L2 proj algorithms
T = tucker_als(X_corr, ds, 'init', U0_L2, 'maxiters',maxit, 'tol', tol);