function w = init_w_shmm(x, u, varargin)
	% w = init_w_shmm(x, u, varargin)
	%
    % Initialization of the posterior parameters w for a trace
    % with datapoints x. Parameters are drawn from the prior and
    % optionally refined using hard and/or soft kmeans estimation.
    %
    % Inputs
    % ------
    % 
    % x : (T x D)
    %   Signal to learn posterior from.
    %
    % u : struct
    %   Hyperparameters for VBEM/EB algorithm
    %
    % Outputs
    % -------
    %
    % u : struct
    %   Hyperparameters with sampled state means
    %
    % Jan-Willem van de Meent
    % $Revision: 1.0$  $Date: 2011/02/14$

    % parse inputs
    ip = inputParser();
    ip.StructExpand = true;
    ip.addRequired('x', @isnumeric);
    ip.addRequired('u', @isstruct);
    ip.addParamValue('threshold', 1e-5, @isscalar);
    ip.addParamValue('quiet', true, @isscalar);
    ip.parse(x, u, varargin{:});
    args = ip.Results;
    x = args.x;
    u = args.u;
    K = size(u.A,1);
    T = size(x, 1);
    D = size(x, 2);

    % draw precision matrix from wishart
    theta.Lambda = wishrnd(u.W, u.nu);
    % draw mu from multivariate normal
    theta.dmu = mvnrnd(u.dmu, inv(u.beta * theta.Lambda));
    % draw transition matrix from dirichlet
    theta.A = dirrnd(u.A);

    % add draw A ~ Dir(u.A) to prior with count (T-1)/K for each row  
    if size(u.A, 1) == 1
        w.A = u.A + theta.A .* (T-1);
    else
        w.A = zeros(size(u.A));
        % one count for first state
        w.A(1, :) = u.A(1, :) + theta.A(1, :);
        % one count for last state
        w.A(end, :) = u.A(end, :) + theta.A(end, :);
        % remaining counts in middle
        w.A(2:end-1, :) = u.A(2:end-1, :) + theta.A(2:end-1, :) * (T-3) / (K-2);
    end

    % add T/K counts to beta and nu
    w.beta = u.beta + T;
    w.nu = u.nu + T;

    % take weighted average for state offest
    w.dmu = (u.beta * u.dmu + T * theta.dmu) / w.beta;

    % set W such that W nu = L 
    % TODO: this should be a proper update, but ok for now
    w.W = bsxfun(@times, theta.Lambda, 1 ./ w.nu);

    % ensure fields in w are in same order as u
    w = orderfields(w, fieldnames(u));