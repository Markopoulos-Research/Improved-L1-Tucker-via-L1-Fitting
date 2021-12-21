function [U] = generate_orth_basis(N, D, d)
% generates N, orthonormal basis matrices of size (Di, di)
    U = cell(N,1);
    for i = 1:N
        U{i} = orth(randn(D(i),d(i)));  % True Orthonormal bases for low rank tensor core
    end
end

