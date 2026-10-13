function [ERR] = ERR_subspace(U, Uhat, d)
%   U : true bases
%   Uhat: approximated bases
    
    I = length(d);
    %% Input validity check
    
    if ~ (iscell(U) && iscell(Uhat))
        error('[ERROR] Inputs must be cell arrays.');
    end
    if isempty(Uhat)
        ERR = [];
        return
    elseif ~isequal(size(U),size(Uhat))
        error('[ERROR] The cell arrays must be of the same sizes.')
    end
    for i = 1:I
        if ~ (isequal(size(U{i},2),d(i)) && isequal(size(Uhat{i},2),d(i)))
            error('[Error] The sizes of basis matrices do not agree with input variable d.')
        end
    end
    
    %% Compute Error
    err = zeros(1,I);
    for i = 1:I
        err(i) = (1/(2*d(i)))*norm(U{i}*U{i}' - Uhat{i}*Uhat{i}','fro')^2;
    end
    ERR = sum(err)/I;
    
end

