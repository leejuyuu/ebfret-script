function [u, L, vb, vit] = hmi(data, u0, varargin)
% [u, L, vb, vit] = hmi(data, u0)
%
% Runs a hierarchical inference process on a collection of single 
% molecule FRET time series (or traces) using a Hidden Markov Model.
%
% The inference process iteratively performs two steps:
%
% 1. Run Variational Bayes Expectation Maximization (VBEM)
%    on each Time Series. 
%
%    This yields two distributions q(theta | w) and q(z) that approximate
%    the posterior over the model parameters and latent states 
%    for each trace.
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
%   u0 : struct 
%     Initial guess for hyperparameters of the ensemble distribution
%     p(theta | u) over the model parameters.
%
%     .A : (K x K)
%         Dirichlet prior for each row of transition matrix
%     .pi : (K x 1)
%         Dirichlet prior for initial state probabilities
%     .mu : (K x D)
%         Normal-Wishart prior - state means 
%     .beta : (K x 1)
%         Normal-Wishart prior - state occupation count
%     .W : (K x D x D)
%         Normal-Wishart prior - state precisions
%     .nu : (K x 1)
%         Normal-Wishart prior - degrees of freedom
%         (must be equal to beta+1)
%
%     Note: The current implementation only supports D=1 signals
%
%
% Variable Inputs
% ---------------
%
%   hstep : 'ml', 'mm'   
%     Method to use for hierarchical updates of hyperparameters
%     'ml' for maximum likelihood, 'mm' for method of moments.
%
%   restarts : int
%     Number of VBEM restarts to perform for each trace.
%
%   do_restarts : 'init', 'always'
%     Specifies whether restarts should be performed only when 
%     determining the initial value of w (default, faster), or at 
%     every hierarchical iteration (slower, but in some cases more 
%     accurate).
%
%   threshold : float
%     Convergence threshold. Execution halts when fractional increase 
%     in total summed evidence drops below this value.
%
%   maxiter : int (default 100)
%     Maximum number of iterations 
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
%   u : struct
%     Optimized hyperparameters (same fields as u0)
%
%   L : (I x 1) 
%     Total lower bound summed over all traces for each 
%     HMI iteration 
%
%   vb : (N x 1) struct
%     Output of VBEM algorithm for each trace
%
%     .w : struct
%         Variational parameters for appromate posterior q(theta | w) 
%         (same fields as u)
%     .L : float
%         Lower bound for evidence
%     .stat : struct
%         Expectation values under q(z) (see vbem documentation)
%
%   vit : (N x 1) struct
%     Viterbi paths for each trace
%
%     .x : (T x 1)
%         FRET level of viterbi path for every time point
%     .z : (T x 1)
%         State index of viterbi path for every time point
%
% Jan-Willem van de Meent
% $Revision: 1.0$  $Date: 2011/08/04$

% parse inputs
ip = InputParser();
ip.StructExpand = true;
ip.addRequired('data', @iscell);
ip.addRequired('u0', @isstruct);
ip.addParamValue('hstep', 'ml', ...
                  @(s) any(strcmpi(s, {'ml', 'mm'})));
ip.addParamValue('restarts', 10, @isscalar);
ip.addParamValue('do_restarts', 'init', ...
                  @(s) any(strcmpi(s, {'always', 'init'})));
ip.addParamValue('threshold', 1e-5, @isscalar);
ip.addParamValue('maxiter', 100, @isscalar);
ip.addParamValue('boolean', true, @isscalar);
ip.addParamValue('display', 'off', ...
                  @(s) any(strcmpi(s, {'hstep', 'trace', 'off'})));
ip.addParamValue('vbem', struct(), @isstruct);
ip.parse(data, u0, varargin{:});

% collect inputs
args = ip.Results;
data = args.data;
u0 = args.u0;

% get dimensions
N = length(data);
K = length(u0.pi);

converged = false;
it = 1;
u(it) = u0;
% main loop for hierarchical inference process
while ~converged
    % initialize guesses for w    
    if (it == 1) | strcmpi(args.do_restarts, 'always')
        R = args.restarts;
        for n = 1:N
            % do not randomize first restart
			if (it == 1)
            	w0(n, 1) =  init_w(u0, length(data{n}), 'randomize', false);
			else
        		w0(n, 1) = w(it-1, n);
			end
       		% randomize guess w0 for other restarts
            for r = 2:R
                w0(n, r) = init_w(u0, length(data{n}));
            end
        end
    else
        % only do one restart and use w from last iteration as guess
        R = 1;
        w0(:, 1) = w(it-1, :);
    end

    % run vbem on each trace 
    for n = 1:N
        if strcmpi(args.display, 'trace')
            fprintf('hmi: %d states, it %d, trace %d of %d\n', K, it, n, N);
        end 
        L{it,n} = [-Inf];
        % loop over restarts
        for r = 1:R
            [w_, L_, stat_] = vbem(data{n}, w0(n,r), u(it), args.vbem);
            % keep result if L better than previous restarts
            if L_(end) > L{it, n}(end)
                w(it, n) = w_;
                L{it, n} = L_;
                stat(it, n) = stat_;
            end
        end
    end

    % calculate summed evidence
    sL(it) = sum(cellfun(@(l) l(end), {L{it,:}}));

    if strcmpi(args.display, 'hstep')
        fprintf('hmi: %d, L: %e, rel increase: %.2e\n', it, sL(it), (sL(it)-sL(max(it-1,1)))/sL(it));
    end    

    % check for convergence
    if it > 1
        if (sL(it) - sL(it-1)) < args.threshold * abs(sL(it-1)) | it > args.maxiter
            if sL(it) < sL(it-1)
              it = it-1;
            end
            break;
        end
    end

    % run hierarchical updates
    if strcmp(args.hstep, 'ml')
      u(it+1) = hstep_ml(w(it,:), u(it));
    else
      u(it+1) = hstep_mm(w(it,:), u(it), stat(it,:));
    end

    % proceed with next iteration
    it = it + 1;
end

% place vbem output in struct 
for n = 1:N
    vb(n) = struct('w', w(it,n), ...
                   'L', L{it,n}(end), ...
                   'stat', stat(it,n));
end

% calculate viterbi paths
for n = 1:N
    [vit(n).z, vit(n).x] = viterbi_vb(w(it,n), data{n});
end 

L = sL(1:it);
u = u(it);
