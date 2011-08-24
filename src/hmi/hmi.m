function [u, L, vb, vit] = hmi(data, u0, varargin)
% [u, L, vb. vit] = hmi(data, u0)
%
% Runs a hierarchical inference proces on a collection of single 
% molecule FRET time series to determine the levels of the FRET
% states, as well as the transition rates between the states.
%
% In addition to detecting the best parameters for each trace,
% this method also calculates a set of hyperparameters that
% define the distribution of parameters over all traces. 
%
% Inputs
% ------
%
%   data : (N x 1) cell 
%     Time series data on which inference is to be performed.
%     Each data{n} should be a 1-dim vector of time points.
%   
%   u0 : struct 
%     Initial guess for hyperparameters of the prior p(theta | u) 
%     over the model parameters theta.
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
%
% Variable Inputs
% ---------------
%
%   hstep : 'ml', 'mm'   
%     Method to use for hierarchical updates of hyperparameters
%     'ml' for maximum likelihood, 'mm' for method of moments.
%
%   restarts : int
%     Number of VBEM restarts to perform when to determine the 
%     optimal values for the variational parameters w.
%
%   do_restarts : 'init', 'always'
%     Specifies whether restarts should be performed only when 
%     determining the initial value of w (default, faster), or at 
%     every hierarchical iteration (more accurate).
%
%   threshold : float
%     Convergence threshold. Execution halts when fractional increase 
%     in total summed evidence drops below this value.
%
%   verbose : boolean
%     Print status information.
%
%   maxIter : int (default 100)
%     Maximum number of iterations 
%
% Outputs
% -------
%
%   theta : struct
%     Maximum likelihood parameters calculated from the 
%     hyperparameters of best run. (same fields as u)
%
%     .A : (K x K)
%         Transition matrix. A(i, j) gives probability of switching 
%         from state i to state j at each time point
%     .pi : (1 x K)
%         Initial probability of each state
%     .mu : (1 x K)
%         FRET emission levels for each state
%     .sigma : (1 x K)
%         FRET emission noise for each FRET state
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
%         Variational parameters (same fields as u)
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


% parse variable arguments
hstep = 'ml';
restarts = 20;
do_restarts = 'init';
threshold = 1e-5;
verbose = false;
maxIter = 100;
for i = 1:length(varargin)
    if isstr(varargin{i})
        switch lower(varargin{i})
        case {'hstep'}
            val = lower(varargin{i+1});
            switch val
            case {'ml','mm'}
                hstep = val;
            otherwise
                err = MException('HMI:HstepUnknown', ...
                                 'hstep must be one of ''ml'' or ''mm''');
                raise(err);
            end
        case {'restarts'}
            restarts = varargin{i+1};
        case {'do_restarts'}
            val = lower(varargin{i+1});
            switch val
            case {'init', 'always'}
                do_restarts = val;
            end
        case {'threshold'}
            threshold = varargin{i+1};
        case {'verbose'}
            verbose = varargin{i+1};
        case {'maxiter'}
            maxIter = varargin{i+1};
        end
    end
end 

% get dimensions
N = length(data);
K = length(u0.pi);

converged = false;
it = 1;
u(it) = u0;
while ~converged
    % initialize guesses for w    
    if (it == 1) | strcmp(do_restarts, 'always')
        R = restarts;
        for n = 1:N
            % do not randomize first restart
            w0(n, 1) =  init_w(u0, length(data{n}), 'randomize', false);
            for r = 2:R
                % randomize guess w0 for other restarts
                w0(n, r) = init_w(u0, length(data{n}));
            end
        end
    else
        % only do one restart and use w from last iteration as guess
        R = 1;
        w0(:, 1) = w(it-1, :);
    end

    % run vbem on each trace 
    options.threshold = threshold;
    options.maxIter = maxIter; 
    for n = 1:N
        % if verbose
        %     fprintf('hmi init: n = %d\n', n);
        % end
        L{it,n} = [-Inf];
        % loop over restarts
        for r = 1:R
            [w_, L_, stat_] = vbem(data{n}, w0(n,r), u(it), options);
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

    if verbose
        fprintf('hmi iteration: %d, L: %e, rel increase: %.2e\n', it, sL(it), (sL(it)-sL(max(it-1,1)))/sL(it));
    end    

    % check for convergence
    if it > 1
        if (sL(it) - sL(it-1)) < threshold * abs(sL(it-1)) | it > maxIter
            if sL(it) < sL(it-1)
              it = it-1;
            end
            break;
        end
    end

    % run hierarchical updates
    if strcmp(hstep, 'ml')
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
    [vit(n).z, vit(n).x] = viterbi(w(it,n), data{n});
end 

L = sL(1:it);
u = u(it);
