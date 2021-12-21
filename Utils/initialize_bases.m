function [UL1, UL2] = initialize_bases(N, D, d, init_method, varargin)
% * if init_method is 'HOSVD', returns UL1 initialized by L1HOSVD (for L1
% methods), and UL2, initialized by L2HOSVD (for L2 methods)
% * if init_method is  'default', UL1=UL2 are both randomly initialized.

valid_init_strs = ["default", "HOSVD"];%, "L1HOSVD"];
validatestring(init_method,valid_init_strs);

params = inputParser;
params.addParameter('X', nan, @(x) isequal(size(x), D))
params.addParameter('tol', 1e-6, @(x) isscalar(x) & x>0)
params.parse(varargin{:});


U = generate_orth_basis(N,D,d);
if isequal(init_method, 'default')
    UL1 = U;
    UL2 = U;
elseif isequal(init_method, 'HOSVD')        % Returns one U initialized by L1 HOSVD, and another U initialized by L2-HOSVD
    X = params.Results.X;
    T = hosvd(X, params.Results.tol, 'rank', d, 'verbosity', 0);
    UL2 = T.U;
    UL1 = L1HOSVD(X, d, UL2, 'maxit', 1000, 'tol', params.Results.tol);
end

end

