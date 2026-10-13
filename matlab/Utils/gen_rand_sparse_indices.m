function [mask] = gen_rand_sparse_indices(P, sz, varargin)
% input arguments:
% P: Probability of entry getting corrupted (from a tensor of size sz)
% sz: size of the tensor
    DEFAULT_ = -1;
    params = inputParser;
    params.addParameter('P_type',DEFAULT_, @(x) ismember(x,{'probability','count'}))
    params.parse(varargin{:})
    
    P_type = params.Results.P_type;
        
    if isequal(P_type,'probability')
        mask = rand(sz);
        mask = mask <= P;
    elseif isequal(P_type,'count')
        mask = zeros(sz);
        rand_idx = randsample(prod(sz),P,false);
        mask(rand_idx) = 1;
    else
        error('P_type argument is not given, or has invalid value.')
    end

end

