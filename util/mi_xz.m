function [mi, hz] = mi_xz(theta, T, varargin)
	% Calculates I(x ; z) the mutual information between 
    % observations x and latent states z, using the parameters
    % theta of the emmissions distribution
    %
    %   p(x | z, theta) = Norm(x | theta.mu(z), theta.sigma(z))
    %
    % And an averaged estimate of the marginal p(z) across 
    % a specified number of timepoints T
    %
    %   p(z=k | theta) = 1/T Sum_t p(z(t)=k | theta)
    %
    % The mutual information is calculated from the joint
    % p(x, z) = p(x | z) p(z), where a numerical integration
    % step is performed for the x-dependence.
    %
    % TODO: Write a better explanation here
    %
    %
    % Inputs
    % ------
    %
    % theta : struct
    %   Model parameters
    %
    % T : int
    %   Time points in trace
    %
    % Variable Arguments
    % ------------------
    %
    % 'Lim' : int, default : 20
    %   Integration limits across x (in units of theta.sigma)   
    %
    % 'Precise' : boolean, default: false
    %   If set, calculate mutual information using a separate 
    %   integration step over x for each t. The default is to 
    %   calculate the average p(z_av) = 1/T Sum_t p(z(t)) and 
    %   return mi = T * mi(x ; z_av)
    %
    % Outputs
    % -------
    %
	% mi : float
    %   The mutual information between x and z
	%
	% hz : float 
    %   Entropy of p(z)

    % parse variable arguments
    lim = 20;
    precise = false;
    for i = 1:length(varargin)
        if isstr(varargin{i})
            switch lower(varargin{i})
            case {'lim'}
                lim = varargin{i+1};
            case {'precise'}
                precise = varargin{i+1};
            end
        end
    end 

    % conditional distribution p(x | z, theta)
    function p = px_z(x, k, theta)
        dx = bsxfun(@minus, x, theta.mu(k));
        D = bsxfun(@rdivide, dx, theta.sigma(k));
        log_p = bsxfun(@plus, -0.5 * log(2*pi) - log(theta.sigma(k)), ...
                              -0.5 * D.^2);
        p = exp(log_p);
    end

    % joint distribution p(x, z | theta)
    pxz = @(x, k, theta, pz) px_z(x, k, theta) .* pz(k);

    % marginal distribution p(x | theta)
    px = @(x, K, theta, pz) sum(bsxfun(@times, px_z(x, 1:K, theta), pz(:)), 1);

    % calculate pz(t,k) p(z(t)=k)
    K = length(theta.mu);
    pz = zeros(T, K);
    pz(1,:) = theta.pi(:);
    for t = 2:T
        pz(t,:) = pz(t-1,:) * theta.A;
    end 

    % if we're doing the precise version of algorithm,
    % perform numerical integration for each time point
    if precise
        mi = zeros(T, K);
        h = zeros(T, K);
        for t = 1:T
            for k = 1:K
                mi(t,k) = quad(@(x) nan_to_zero(pxz(x,k,theta,pz(t,:)) ...
                                               .* log(px_z(x,k,theta) ...
                                                      ./ px(x,K,theta,pz(t,:)))), ...
                             -lim*theta.sigma(k), lim*theta.sigma(k));
            end
            hz(t,:) = -pz(t,:) .* log(pz(t,:));
        end
    % for approximate varsion of algorithm, calculate 
    % mutual information between x, z, using p(z) averaged 
    % over trace
    else
        pz = mean(pz, 1);
        mi = zeros(1,K);
        for k = 1:K
            mi(k) = quad(@(x) nan_to_zero(pxz(x,k,theta,pz) ...
                                       .* log(px_z(x,k,theta) ...
                                              ./ px(x,K,theta,pz))), ...
                     -lim*theta.sigma(k), lim*theta.sigma(k));
        end
        hz = -pz .* log(pz);
        % multiply by T
        mi = T * mi;
        hz = T * hz;
    end
    
    mi = sum(mi(:));
    hz = sum(hz(:));
end
