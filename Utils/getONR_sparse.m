function [onr] = getONR_sparse(sz, varargin)
% 
    DEFAULT_ = -1;
    params = inputParser;
    params.addParameter('sigma_o',DEFAULT_, @(x) isscalar(x) && (x>=0))
    params.addParameter('sigma_n',DEFAULT_, @(x) isscalar(x) & x>0)
    params.addParameter('P_type',DEFAULT_, @(x) ismember(x,{'probability','count'}))
    params.addParameter('P',DEFAULT_,@(x) isnumeric(x))
    params.parse(varargin{:})
    
    sigma_n = params.Results.sigma_n;
    sigma_o = params.Results.sigma_o;
    P = params.Results.P;
    P_type = params.Results.P_type;
    
    
    %% Validation
    if (sigma_n == DEFAULT_) || (sigma_o == DEFAULT_) || (P == DEFAULT_) || isequal(P_type,DEFAULT_)
        error('[ERROR] Not enough input parameters.')
    end
    
    if isequal(P_type,'probability')
        assert((P>=0) && (P<1)) 
    elseif isequal(P_type,'count')
        assert (P>=0 && P<prod(sz))
    else
        error('Invalid input for P_type argument')
    end
    %% compute ONR
    
    ExpN = prod(sz)*sigma_n^2;
    
    if isequal(P_type,'probability')
        ExpO =prod(sz)*P*sigma_o^2;
    elseif isequal(P_type,'count')
        ExpO = P*sigma_o^2;
    end
    
    onr = ExpO/ExpN;
    
end

