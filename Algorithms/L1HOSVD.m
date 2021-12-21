function [U, G, Xhat, L1met, stats]=L1HOSVD(X, Ks, Uin, varargin)
%% Basic Usage: [U, G, Xhat, L1met_end, stats] = L1HOSVD(X, Ks, U0)
%  X: I-way input Tensor of size D = [D1, D2, ..., DI];
%  Ks: [d1, ..., dI]
%  Uin: is a cell array. U{i} is the i th basis, a Di by di matrix.
%  G: tensor core.
%  U: cell array of same size as Uin, contains the computed bases.
%  L1met: The value of the objective function (L1-metric of tensor core) at
%  the end of computation
%  stats: struct containing some information about algorithm execution
%  stats.update_types : list of the indexes of basis (0 for updating
%  B), in the order they are updated in algorithm execution
%  stats.RERR: list of reconstruction errors after each basis update
%  stats.SERR: list of suspace errors after each basis update
%% Input params definition
params = inputParser;
params.addParameter('tol',1e-4,@isscalar);
params.addParameter('maxit',50,@(x) isscalar(x) & x > 0);
params.addParameter('X_clean',nan, @(x) isequal(size(x),size(X)))
params.addParameter('Un_true',nan,@(x) iscell(x) & isequal(length(x),length(Ks)))
params.parse(varargin{:});
%%
tol = params.Results.tol;
maxit = params.Results.maxit;
%% 
if ~isnan(params.Results.Un_true)
    se_flag = true;
    Un_true = params.Results.Un_true;
else
    se_flag = false;
end
if ~isnan(params.Results.X_clean)
    re_flag = true;
    X_clean = params.Results.X_clean;
else
    re_flag = false;
end

Subspace_errors = [];
Reconstruction_errors = [];

%% Initialze algorithm
L1_metric = [];
update_type = [];

D = size(X);
n = ndims(X);

U = Uin;

%% Calculate initial L1 metric
G = ttm(X,U,1:n,'t');
L1_metric = [L1_metric, sum(abs(G(:)))];
%% Algorithm
for i=1:n
    K  = Ks(i);
    if K == D(i) 
        Ui=eye(K); 
    else 
        [Ui,~,~, l1met]=L1PCA(X, K, U, i, maxit, tol);
        L1_metric = [L1_metric l1met];
        update_type = [update_type,  repmat([0, i],1, floor(length(l1met)/2))];
    end
    U{i}=Ui;
    if se_flag
        Subspace_errors = [Subspace_errors, ERR_subspace(Un_true, U, Ks)];
    end
    if re_flag
        G = ttm(X,U,'t');
        Xhat = ttm(G, U);
        Reconstruction_errors = [Reconstruction_errors, ERR_reconstruction(X_clean, Xhat)];
    end
end
G = ttm(X,U,1:n,'t');
Xhat = ttm(G,U,1:n).data;
L1met = sum(abs(G(:)));           % L1 norm of the computed tensor core
G = double(G);
stats = struct();
stats.exec_time = 0;
stats.L_metric = L1_metric;
stats.B_metric = [];
stats.SERR = Subspace_errors;
stats.RERR = Reconstruction_errors;

end

function [Q,B, met, L1met]=L1PCA(X, K, U, i, maxit, tol)        % this function is Modified
    Q = U{i};
    Z  = tenmat(X,i);
    Xi = Z.data;
    metmax=0;
    met=zeros(1,maxit);
    L1met = zeros(1,maxit);
    t = 0;
    while true
       t = t+1;
       A = Xi'*Q;
       met(t) = sum(abs(A(:)));
       B = sign(A);               % B update
      [UK,~,VK] = mSVD(Xi*B,K);
       Q = UK*VK';                % U update
       %% Calculate L1 metric
       U{i} = Q;
       G = ttm(tensor(X),U,'t');
       L1met(t) = sum(abs(G(:)));
       %% Check for convergence
       if (abs(met(t)-metmax)/abs(metmax))<=tol || t==maxit
           break
       else
           metmax = met(t);
       end
    end
    met=met(1:t);
    L1met = L1met(1:t);
end

function [U,S,V]=mSVD(X,K)
       [UK,SK,VK]=svd(X,'econ');
       S=SK(1:K,1:K);
       U=UK(:,1:K);
       V=VK(:,1:K);
end