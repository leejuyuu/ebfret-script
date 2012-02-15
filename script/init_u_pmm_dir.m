function u = init_u_pmm_dir(M, vb, u, varargin)
	% u = init_u_pmm_dir(M, vb, u)
	%
	% Initializes a set of hyperparameters with M subpopulations.
    % Hyperparameters for A are determined by running a mixture 
    % model on the posterior distributions. Distributions for
    % for the other parameters are identical.
    %
    % Transitions between subpopulations are assumed to be forbidden
    % so the resulting hyperparameters on A have M KxK blocks on the
    % diagonal, and zeros elsewhere.
    %
    % Inputs
    % ------
    % 
    % vb : (1xN) struct
    %   VBEM output for each trace (see function docs)
    %
    % u : struct
    %   Hyperparameters obtained from HMI process (see function docs)
    %
    % M : int
    %   Number of subpopulations. If input hyperparameters have K 
    %   states, output parameters will have K x M states.
    %
    % Variable Inputs
    % ---------------
    %
    % restarts : int
    %   Number of VBEM restarts to perform for each trace.
    %
    % display : {'off', 'final', 'all'} (default: 'off')
    %   Display progress after final or for every restart
    %
    % Outputs
    % -------
    %
    % u : struct
    %   Hyperparameters for K x M state system. 
    %
    %
    % TODO: we could initialize u.pi based on the pmm_dir results
    % but for now, the u.pi paramaters are duplicated across 
    % subpopulations
    %
    % Jan-Willem van de Meent
    % $Revision: 1.0$  $Date: 2011/02/14$

    % parse inputs
    ip = InputParser();
    ip.StructExpand = true;
    ip.addRequired('M', @isscalar);
    ip.addRequired('vb', @isstruct);
    ip.addRequired('u', @isstruct);
    ip.addParamValue('restarts', 10, @isscalar);
    ip.addParamValue('display', 'off', ...
                      @(s) any(strcmpi(s, {'off', 'final', 'all'})));
    ip.parse(M, vb, u, varargin{:});
    args = ip.Results;
    M = args.M;
    K = length(args.u.mu);

    % duplicate hyperparameters for each subpopulation
    append_u = @(u, v) struct('pi', [u.pi; v.pi], ...
                              'A', cat(1, cat(2, u.A, eps + zeros(size(u.A,1), size(v.A,2))), ...  
                                        cat(2, eps + zeros(size(v.A,1), size(u.A,2)), v.A)), ...
                              'mu', [u.mu; v.mu], ...
                              'beta', [length(u.beta) * u.beta; ...
                                     length(v.beta) * v.beta] ...
                                     / (length(u.beta) + length(v.beta)), ...
                              'W', [u.W; v.W], ...
                              'nu', [length(u.nu) * u.nu; ...
                                     length(v.nu) * v.nu] ...
                                     / (length(u.nu) + length(v.nu)));
    u = args.u;
    for m = 2:M
        u = append_u(u, args.u);
    end

    % run parameter mixture model to refine guess for trans matrix
    w = [args.vb(:).w];
    Xi = arrayfun(@(w) w.A - args.u.A, w, 'UniformOutput', false);
    Xi = permute(cat(3, Xi{:}), [3 1 2]);
    L_max = -inf;
    for r = 1:args.restarts
        if r == 1
            % do not randomize for first restart
            u_A0 = permute(repmat(args.u.A, [1 1 M]), [3 1 2]);
        else 
            % randomize by draw from dirichlet               
            u_A0 = zeros(M, K, K);
            for m = 1:M
                u_A0(m, :, :) = sum(args.u.A(:)) * normalize(dirrnd(args.u.A));
            end
        end

        % run prior mixture model
        warning('off', 'pmm_dir:DecreasedL')
        % choosing the prior strength can be a little finicky,
        % so try orders of magnitude until something converges
        for u_strength = [1, 0.1, 10, 0.01, 100]
            [u_A g L exitflag] = pmm_dir(Xi, u_strength * u_A0, ones(m, 1));
            if exitflag == 1
                break
            end
        end

        if strcmpi(args.display, 'all')
            fprintf('pmm_dir: restart = %d, L = %.4e, iterations: %d, exitflag: %d\n', r, L(end), length(L), exitflag);
        end

        % keep results if pmm converged and L improved
        if (exitflag ~= -1) & (L(end) > L_max)
            L_max = L(end);
            rng = @(m) (1:K) + (m-1) * K;
            for m = 1:M
                % set transition matrix prior
                u.A(rng(m), rng(m)) = squeeze(u_A(m, :, :));
            end
        end
    end

    if strcmpi(args.display, 'final')
        fprintf('pmm_dir: restarts = %d, L = %.4e\n', args.restart, L_max);
    end
end
