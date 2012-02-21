function [I, H] = mi_xiz(xi, z)
	% Calculates I(xi ; z), the mutual information between inference
	% estimates of the joint probabilities p(z(t), z(t+1) | x) and
	% the true states z(t).
    %
    % Inputs
    % ------
    %
    % xi : (T,K,K)
    %	Estimates of joint xi(t,k,l) = p(z(t)=k,z(t+1)=l | x)
    %
    % z : (T+1,) or (T+1,K) or (T+1,K,K)
    %	True states, in integer, indicator or joint indicator format
    %
    % Outputs
    % -------
    %
	% I : The normalized mutual information between 
	%  	xi(t,k,l) and (z(t,k),z(t+1,l))
	%
	% H :  Normalization factor: joint entropy between
	%  	xi(t,k,l) and (z(t,k),z(t+1,l))
	[T K K] = size(xi);
	if length(z(:)) == T+1
		z = int_to_ind(z(:));
	end
    L = size(z, 2);    
	if length(z(:)) == (T+1) * L 	
		% zz(t,k,l) = z(t,k)*z(t+1,l)
		zz = bsxfun(@times, reshape(z(2:end,:), [T 1 L]), z(1:end-1,:));
	else
		% assume joint has been previously calculated
		% (this is necessary when lumping data from an ensemble together)
		zz = z;
	end
	% calculate joint probabilities
	pxiz = squeeze(mean(bsxfun(@times, reshape(xi, [T K^2 1]), ...
	                                   reshape(zz, [T 1 L^2])), 1));
	% calculate marginals
	pxi = sum(pxiz,2);
	pz = sum(pxiz,1);
	% calculate mi and H
	H = sum(-pxiz(:) .* log(pxiz(:)));
	I = nan_to_zero(pxiz .* (log(pxiz) - log(bsxfun(@times, pxi, pz))));
	I = sum(I(:));	
