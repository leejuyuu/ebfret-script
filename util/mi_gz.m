function [I, H] = mi_gz(g, z)
	% Calculates I(g ; z), the mutual information between
    % inferred state probabilities and the true states.
    %
    % Inputs
    % ------
    %
    % g : (T,K)
    %	Estimates of responsibilities g(t,k) = p(z(t)=k | x)
    %
    % z : (T,) or (T,L) 
    %	True states (either integer or indicator format)
    %
    % Outputs
    % -------
    %
	% I : The normalized mutual information between 
	%  	gamma(t,k) and z(t,k)
	%
	% H : Normalization factor: joint entropy between 
	%  	gamma(t,k) and z(t,k)
    [T K] = size(g);
    if length(z(:)) == T
        z = int_to_ind(z(:));
    end 
    [T L] = size(z);    
    % calculate joint probabilities
    pgz = squeeze(mean(bsxfun(@times, reshape(g, [T K 1]), ...
                                      reshape(z, [T 1 L])), 1));
    % calculate marginals
    pg = sum(pgz,2);
    pz = sum(pgz,1);
    % calculate mi and H
    H = sum(-pgz(:) .* log(pgz(:)));
    I = nan_to_zero(pgz .* (log(pgz) - log(bsxfun(@times, pg, pz))));
    I = sum(I(:));    
	
	
