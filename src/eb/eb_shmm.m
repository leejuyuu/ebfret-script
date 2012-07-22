function [u, L, vb, omega] = eb_shmm(data, u0, mu0, varargin)
% [u, L, vb, omega] = eb_shmm(data, u0)
%
% Runs a empirical Bayes inferenceon a collection of single 
% molecule time series using a Stepping Hidden Markov Model.
% See vbem_shmm for further information.
%
% The inference process iteratively performs two steps:
%
% 1. Run Variational Bayes Expectation Maximization (VBEM)
%    on each Time Series. 
%
%    This yields two distributions q(theta | w) and q(z) that 
%    approximate the posterior over the model parameters and latent 
%    states for each trace.
%
% 2. Update the hyperparameters for the distribution p(theta | u).
%
%    This distribution, which plays the role of a prior on the 
%    parameters in the VBEM iterations, represents an aggregate
%    over the ensemble of time series, reporting on both the average
%    parameter value and the variability from molecule to molecule.
%    
%
% Inputs
% ------
%
%   data : (N x 1) cell 
%     Time series data on which inference is to be performed.
%     Each data{n} should be a 1d vector of time points.
%   
%   u0 : (M x 1) struct 
%     Initial guesses for hyperparameters of the ensemble distribution
%     p(theta | u) over the model parameters.
%
%       .A : K x 2
%           Dirichlet prior on self and forward transitions. If K = 1, 
%           the prior is assumed to be identical for all states
%       .dmu : scalar
%           Normal-Wishart prior - state offset 
%       .beta : scalar
%           Normal-Wishart prior - state occupation count
%       .W : scalar
%           Normal-Wishart prior - state precision
%       .nu : scalar 
%           Normal-Wishart prior - degrees of freedom
%
%     Note: The current implementation only supports D=1 signals
%
%   w0 : (N x M) or (N x 1) struct (optional)
%     Intial guesses for variational parameters. 
%    
%
% Variable Inputs
% ---------------
%
%   threshold : float
%     Convergence threshold. Execution halts when fractional increase 
%     in total summed evidence drops below this value.
%
%   max_iter : int (default 100)
%     Maximum number of iterations 
%
%   restarts : int
%     Number of VBEM restarts to perform for each trace.
%
%   do_restarts : {'init', 'always'} (default: 'init')
%     Specifies whether restarts should be performed only when 
%     determining the initial value of w (default, faster), or at 
%     every hierarchical iteration (slower, but in some cases more 
%     accurate).
%
%   display : {'hstep', 'trace', 'off'} (default: 'off')
%     Print status information.
%
%   vbem : struct
%     Struct storing variable inputs for VBEM algorithm
%     (see function documentation)
%
%
% Outputs
% -------
%
%   u : (M x 1) struct
%     Optimized hyperparameters (same fields as u0)
%
%   L : (I x 1) 
%     Total lower bound summed over all traces for each 
%     Empirical Bayes iteration 
%
%   vb : (N x M) struct
%     Output of VBEM algorithm for each trace
%
%     .w : struct
%         Variational parameters for appromate posterior q(theta | w) 
%         (same fields as u)
%     .L : float
%         Lower bound for evidence
%
%   vit : (N x 1) struct
%     Viterbi paths for each trace
%
%     .x : (T x 1)
%         FRET level of viterbi path for every time point
%     .y : int
%         Mixture component for trace
%     .z : (T x 1)
%         State index of viterbi path for every time point
%
%
%   omega : struct
%     Mixture parameters for priors
%
%     .gamma : (N x M)
%         Responsibilities for each trace and mixture component
%     .pi : (M x 1)
%         Mixture component prior weights 
%
% Jan-Willem van de Meent
% $Revision: 1.0$  $Date: 2011/08/04$

% parse inputs
ip = inputParser();
ip.StructExpand = true;
ip.addRequired('data', @iscell);
ip.addRequired('u0', @isstruct);
ip.addRequired('mu0', @isnumeric);
ip.addOptional('w0', struct(), @(w) isstruct(w) & isfield(w, 'mu'));
ip.addParamValue('threshold', 1e-5, @isscalar);
ip.addParamValue('max_iter', 100, @isscalar);
ip.addParamValue('restarts', 10, @isscalar);
ip.addParamValue('do_restarts', 'init', ...
                  @(s) any(strcmpi(s, {'always', 'init'})));
ip.addParamValue('display', 'off', ...
                  @(s) any(strcmpi(s, {'hstep', 'trace', 'off'})));
ip.addParamValue('vbem', struct(), @isstruct);
ip.parse(data, u0, mu0, varargin{:});

try
    % collect inputs
    args = ip.Results;
    data = args.data;
    u0 = args.u0;

    % get dimensions
    N = length(data);
    M = length(u0);
    K = size(mu0, 1);

    converged = false;
    it = 1;
    u(it, :) = u0;

    % main loop for hierarchical inference process
    omega.pi = ones(M,1) ./ M;
    while ~converged
        % initialize guesses for w    
        if (it == 1) | strcmpi(args.do_restarts, 'always')
            R = args.restarts;
            % on first iteration, initial values for w are either 
            % supplied as arguments, or inferred by running a gaussian
            % mixture model on the data
            %
            % on subsequent iterations, the result from the previous
            % iteration is used in the fist restart

            if (strcmpi(args.display, 'hstep') | strcmpi(args.display, 'trace'))
                fprintf('[%s] eb: %d states, it %d, initializing w\n', ...
                         datestr(now, 'yymmdd HH:MM:SS'), K, it)
            end

            if (it == 1)
                switch length(args.w0(:))
                    case N*M
                        [w0(:, :, 1)] =  args.w0;
                    case N
                        for m = 1:M
                            [w0(:, m, 1)] =  args.w0(:);
                        end
                    otherwise
                        for n = 1:N
                            if strcmpi(args.display, 'trace')
                                fprintf('[%s] eb: %d states, it %d, trace %d of %d\n', ...
                                         datestr(now, 'yymmdd HH:MM:SS'), K, 0, n, N);
                            end    
                            for m = 1:M
                                w0(n, m, 1) =  init_w_shmm(data{n}, u0(m));
                            end
                        end
                end
            else
                % use value from previous iteration
                w0(:, :, 1) = w(it-1, :, :);
            end

            % additional restarts use a randomized guess for the
            % prior parameters
            for r = 2:R
                for n = 1:N
                    for m = 1:M
                        % draw w0 from prior u for other restarts
                        w0(n, m, r) = init_w_shmm(data{n}, u(it, m));
                    end
                end
            end
        else
            % only do one restart and use w from last iteration as guess
            R = 1;
            w0(:, :, 1) = w(it-1, :, :);
        end

        if (strcmpi(args.display, 'hstep') | strcmpi(args.display, 'trace'))
            fprintf('[%s] eb: %d states, it %d, running VBEM\n', ...
                     datestr(now, 'yymmdd HH:MM:SS'), K, it)
        end

        % run vbem on each trace 
        L(it,:,:) = -Inf * ones(N, M);
        w_it = cell(N, M);
        stat_it = cell(N, M);
        parfor n = 1:N
            if strcmpi(args.display, 'trace')
                fprintf('[%s] eb: %d states, it %d, trace %d of %d\n', ...
                         datestr(now, 'yymmdd HH:MM:SS'), K, it, n, N);
            end 
            % loop over prior mixture components
            for m = 1:M
                % loop over restarts
                for r = 1:R
                    [w_, L_, s_] = vbem_shmm(data{n}, w0(n, m, r), u(it, m), (1:K)', args.vbem);
                    % keep result if L better than previous restarts
                    if L_(end) > L(it, n, m)
                        w_it{n, m} = w_;
                        s_it{n, m} = s_;
                        L(it, n, m) = L_(end);
                        restart(n, m) = r;
                    end
                end
            end
        end
        w(it,:,:) = reshape([w_it{:}], size(w_it));
        s(it,:,:) = reshape([s_it{:}], size(s_it));

        % calculate prior mixture responsiblities for each trace
        Lit = reshape(L(it,:,:), [N M]);
        L0 = bsxfun(@minus, Lit , mean(Lit, 2));
        omega(it).gamma = normalize(bsxfun(@times, exp(L0), omega(it).pi'), 2);
        omega(it+1).pi = normalize(sum(omega(it).gamma, 1))';

        % calculate summed evidence
        sL(it) = sum(sum(omega(it).gamma .* Lit, 2), 1);

        if strcmpi(args.display, 'hstep') | strcmpi(args.display, 'trace')
            fprintf('[%s] eb: %d states, it %d, L: %e, rel increase: %.2e, randomized: %.3f\n', ...
                    datestr(now, 'yymmdd HH:MM:SS'), K, it, sL(it), (sL(it)-sL(max(it-1,1)))/sL(it), sum(restart(:)~=1) / length(restart(:)));
        end    

        % check for convergence
        if (it > 1) & (abs((sL(it) - sL(it-1))) < args.threshold * abs(sL(it-1)) | it > args.max_iter)
            if sL(it) < sL(it-1)
              it = it-1;
            end
            break;
        end

        % run hierarchical updates
        for m = 1:M
            u_new = hstep_shmm(w(it, :, m), omega(it).gamma(:, m));
            u(it+1, m) = u_new;
        end

        % proceed with next iteration
        it = it + 1;
    end

    % place vbem output in struct 
    for n = 1:N
        for m = 1:M
            vb(n,m) = struct('w', w(it,n,m), ...
                             's', s(it,n,m), ...
                             'L', L(it,n,m), ...
                             'restart', restart(n,m));
        end
    end

    % % calculate viterbi paths
    % for n = 1:N
    %     m = find(bsxfun(@eq, L(it,n,:), max(L(it,n,:))));
    %     [vit(n).z, vit(n).x] = viterbi_vb(w(it,n,m), data{n});
    %     vit(n).y = m; 
    % end 

    % assign outputs
    L = sL(1:it);
    u = u(it,:);
    omega = omega(it);
catch ME
    % ok something went wrong here, so dump workspace to disk for inspection
    day_time =  datestr(now, 'yymmdd-HH.MM');
    save_name = sprintf('crashdump-eb_shmm-%s.mat', day_time);
    save(save_name);
    rethrow(ME);
end
