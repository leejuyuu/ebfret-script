function w = init_w_gmm(x, u)
	% w = init_w_gmm(signal, u, varargin)
	%
    % Runs a Gaussian Mixture Model EM on the data to initialize the
    % posterior parameters of the emissions model. The posterior
    % parameters for the transition matrix are randomly sampled 
    % from the prior.
    %
    % Inputs
    % ------
    % 
    % x : (Tx1)
    %   Signal to learn posterior from.
    %
    % u : struct
    %   Hyperparameters for VBEM/HMI algorithm
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
    ip = InputParser();
    ip.StructExpand = true;
    ip.addRequired('x', @isnumeric);
    ip.addRequired('u', @isstruct);
    ip.parse(x, u);
    args = ip.Results;
    x = args.x;
    u = args.u;
    K = length(u.mu);
    T = length(x);

    % this is necessary just so matlab does not complain about 
    % structs being dissimilar because of the order of the fields
    w = u;

    hard_kmeans = false;
    gmm_opts.TolFun = 1e-3;

    if hard_kmeans
        % run hard kmeans to get cluster centres
        [idx mu] = kmeans(x, K);
        % run soft kmeans to get mean and variance of each state;
        gmm = gmdistribution.fit(x, K, 'Start', idx, ...
                                 'CovType', 'diagonal', 'Regularize', 1e-6, 'Options', gmm_opts);
    else
        gmm = gmdistribution.fit(x, K, ...
                                 'CovType', 'diagonal', 'Regularize', 1e-6, 'Options', gmm_opts);
    end

    % get emission modelparameters
    [mu, kdxs] = sort(gmm.mu);
    lambda = 1 ./ gmm.Sigma(kdxs);
    p = gmm.PComponents(kdxs);

    % randomly sample transition matrix
    A = dirrnd(u.A);
    
    % assign posterior parameters         
    w.pi = u.pi + p(:); 
    w.A = u.A + (T-1)/K * A;
    w.mu = mu(:);
    w.beta = u.beta + T/K;
    w.nu = u.nu + T/K;
    w.W = lambda(:) ./ w.nu;
