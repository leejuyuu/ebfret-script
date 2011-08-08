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
%     .nu : (K x 1)
%         Normal-Wishart prior - degrees of freedom
%         (must be equal to beta+1)
%     .W : (K x D x D)
%         Normal-Wishart prior - state precisions
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
%     Number of restarts to perform when determining initial values
%     for the variational parameters w (before starting hierarchical
%     updates)
%
%   threshold : float
%     Convergence threshold. Execution halts when fractional increase 
%     in total summed evidence drops below this value.
%
%   verbose : boolean
%     Print status information.
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
threshold = 1e-5;
verbose = false;
for i = 1:length(varargin)
    if isstr(varargin{i})
        switch lower(varargin{i})
        case {'hstep'}
            method = varargin{i+1};
            switch lower(method)
            case {'ml','mm'}
                hstep = varargin{i+1};
            otherwise
                err = MException('HMI:HstepUnknown', ...
                                 'hstep must be one of ''ml'' or ''mm''');
                raise(err);
            end
        case {'restarts'}
            restarts = varargin{i+1};
        case {'threshold'}
            threshold = varargin{i+1};
        case {'verbose'}
            verbose = varargin{i+1};
        end
    end
end 

% TODO: make options for these
verbose = true;
R = 20;
threshold = 1e-5;

% get dimensions
N = length(data);
K = length(u0.pi);

converged = false;
it = 1;
u(it) = u0;
while ~converged
    % initialize guesses for w    
    if it == 1
        for n = 1:N
            % if verbose
            %     fprintf('it: 00, n: %03d\n', n);
            % end
            % run vbem restarts and keep best result
            Lm = -inf;
            for r = 1:R
                w0_ = init_w(u0, length(data{n}));
                [w_, L_, stat_] = vbem(data{n}, w0_, u0);
                if L_(end) > Lm
                    w0(n) = w_;
                    Lm = L_(end);
                end
            end 
        end
    else
        % use output from last iteration as initial guess
        w0 = w(it-1, :);
    end

    % run vbem on each trace 
    for n = 1:N
        % if verbose
        %     fprintf('it: %02d, n: %03d\n', it, n);
        % end
        [w(it,n), L{it,n}, stat(it,n)] = vbem(data{n}, w0(n), u(it));
    end

    % calculate summed evidence
    sL(it) = sum(cellfun(@(l) l(end), {L{it,:}}));

    if verbose
        fprintf('hmi iteration: %d, L: %e\n', it, sL(it));
    end    

    % check for convergence
    if it > 1
        if (sL(it) - sL(it-1)) < (threshold * sL(it-1))
            if sL(it) < sL(it-1)
              it = it-1;
            end
            break;
        end
    end

    % run hierarchical updates
    %u(it+1) = hstep_mm(w(it,:), u(it), stat(it,:));
    %dbstop in hstep_ml at 216
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
