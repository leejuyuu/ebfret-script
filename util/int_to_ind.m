function zind = int_to_ind(z, K)
	% Converts a set of categorical variables from integer
    % to indicator format 
    %
    % Inputs
    % ------
    %
    % z : (N,) 
    %	Set of categorical variables, represented as integers 1:K
    %
    % K : int (optional)
    %   Number of categorical
    %
    % Outputs
    % -------
    %
	% zind : (N, K)
    %   Variables in indicator format
    if (nargin < 2)
        K = max(z(:));
    end
    if length(z(:)) == length(z)
        z = z(:);
        d = 1;
    else
        % number of dimensions of data
        d = ndims(z);
    end
    % convert to indicator format
    zind = bsxfun(@eq, z, reshape(1:K, [ones([1 d]) K]));
	
