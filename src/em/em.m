function [theta, L, stat] = em(x, theta0, varargin)
% [theta, L, stat] = em(x, theta0, varargin)
%
% Variational Bayes Expectation Maximization for a Hidden Markov Model
% with Gaussian emissions.
%
% Given a set of observations x(1:T), this algorithm returns an 
% approximation for the posterior distribution q(theta | w) over
% the model paramaters theta.
%
% Inputs
% ------
%
%   x : (T x D) float
%       Observation time series. May be one dimensional 
%       (e.g. a FRET signal), or higher dimensional 
%       (e.g. D=2 for a donor/acceptor signal)
%
%   theta0 : struct 
%       Initial guess for the model parameters.
%
%       .A (K x K)
%           Transition matrix
%       .pi (K x 1)
%           Initial state probabilities
%       .mu (K x D)
%           State emissions means 
%       .Lambda (K x D x D)
%           State emissions precision
%
%
% Variable Inputs
% ---------------
%
%   threshold : float (default: 1e-5)
%      Convergence threshold. Execution halts when the relative
%      increase in the lower bound evidence drop below threshold 
%
%   max_iter : int (default: 100)
%      Maximum number of iteration before execution is truncated
%
%   min_var : float (default: 1e-6)
%      Minimum variance for points in a state. Used to prevent
%      a state converging on a single datapoint (resulting in
%      an infinite precision and a divergence of the likelihood)  
%
%   ignore : {'none', 'spike', 'intermediate', 'all'}
%      Ignore states with length 1 on viterbi path that either
%      collapse back to the previous state ('spike') or move
%      to a third state ('intermediate').
%
%
% Outputs
% -------
%
%   theta : struct
%       Maximum likelihood value for parameters (same fields as theta0)
%
%   L : (I x 1) float
%       Lower bound estimate of evidence for each iteration
%
%   stat : struct
%       Sufficient statistics calculated in forward backward algorithm
%
%       gamma : (T x K)
%           Posterior probability for z(t)
%               gamma(t, k) = E_q(z)[z(t, k)] 
%                           = p(z(t)=k | x(1:T))
%
%       xi : (T x K x K)
%           Joint posterior probilities for z(t) and z(t+1)
%               xi(t, k, l) = E_q(z)[z(t, k) z(t+1, l)]  
%                           = p(z(t)=k, z(t+1)=l | x(1:T))
%
%       ln_Z : float
%           Log normalization constant of q(z).
%
% TODO: this does not work yet for D>1
%
% Jan-Willem van de Meent
% $Revision: 1.00 $  $Date: 2011/08/03$

Debug = false;

% parse inputs
ip = inputParser();
ip.StructExpand = true;
ip.addRequired('x', @(x) isnumeric(x) & (ndims(x)==2));
ip.addRequired('theta0', @isstruct);
ip.addParamValue('threshold', 1e-5, @isscalar);
ip.addParamValue('max_iter', 100, @isscalar);
ip.addParamValue('min_var', 1e-6, @isscalar);
ip.addParamValue('ignore', 'none', ...
                 @(s) any(strcmpi(s, {'none', 'spike', 'intermediate', 'all'})));
ip.parse(x, theta0, varargin{:});

% collect inputs
args = ip.Results;
x = args.x;
theta0 = args.theta0;

% set theta to initial guess
theta = theta0;

% get dimensions
[T D] = size(x);
K = length(theta.pi);

if Debug
    iter = struct();
end

% Main loop of algorithm
for it = 1:args.max_iter
    % E-STEP: UPDATE Q(Z)
    if D>1
        % Calculate Mahalanobis distance
        Lambda = permute(theta.Lambda, [2 3 1]);
        % dx(d, t, k) = x(t, d) - mu(k, d)
        dx = bsxfun(@minus, x', reshape(theta.mu', [D 1 K]));
        % dxLdx(t, k) = Sum_de dx(d,t,k) * l(d, e, k) * dx(e, t, k)
        Delta2 = squeeze(mtimesx(reshape(dx, [1 D T K]), ...
                                 mtimesx(reshape(Lambda, [D D 1 K]), ...
                                         reshape(dx, [D 1 T K]))));
        % note, the mtimesx function applies matrix multiplication to
        % the first two dimensions of an N-dim array, while using singleton
        % expansion to the remaining dimensions. 
        % 
        % TODO: make mtimesx usage optional? (needs compile on Linux/MacOS)

        % calculate precision matrix determinant
        ln_det_Lambda = zeros([K 1]);
        for k = 1:K
            ln_det_Lambda(k) = log(det(Lambda(:,:,k)));
        end
    else
        % dx(t, k) = x(t) - mu(k)
        dx = bsxfun(@minus, x, theta.mu');
        % dxLdx(t, k) = Sum_de dx(t,k) * L(k) * dx(t, k)
        Delta2 = bsxfun(@times, dx, bsxfun(@times, theta.Lambda', dx));
        % get precision 'determinant'
        ln_det_Lambda = log(theta.Lambda);
    end

    % Log emissions probability p(x | z, theta) 
    %
    % ln_px_z(t, k)
    %   = log(1 / 2 pi) * (D / 2)
    %     + 0.5 * E[ln(|Lambda(k,:,:)|]]
    %     - 0.5 * E[Delta(t,k)^2]
    ln_px_z = log(2 * pi) * (-D / 2) ...
                + bsxfun(@plus, 0.5 * ln_det_Lambda', ...
                               -0.5 * Delta2);

    % Forward-back algorithm - computes expecation under q(z) of
    %
    %   g(t, k) = p(z(t)=k | x, theta) 
    %   xi(t, k, l) = p(z(t)=k, z(t+1)=l | x, theta)
    %   Z_(q(z)) = p(x | theta) 
    [g, xi, ln_Z] = forwback(exp(ln_px_z), theta.A, theta.pi);  

    % Get log likelihood
    L(it) = ln_Z;

    if Debug
        iter(it).ln_Z = ln_Z;
    end

    % print warning if lower bound decreases
    if it>2 && ((L(it) - L(it-1)) < -10 * args.threshold * abs(L(it))) 
        fprintf('Warning!!: Lower bound decreased by %e \n', ...
                L(it) - L(it-1));
    end

    % check if the lower bound increase/decrease is less than threshold
    if (it>2)    
        if abs((L(it) - L(it-1)) / L(it-1)) < args.threshold || ~isfinite(L(it)) 
            L(it+1:end) = [];  
            break;
        end
    end

    % M STEP: UPDATE THETA

    % check whether points need to be masked out
    if strcmp(args.ignore, 'none')
        g_ = g;
        xi_ = xi;
    else
        z_hat = viterbi(ln_px_z, log(theta.A), log(theta.pi));
        [g_, xi_] = jitter_filter(z_hat, g, xi, args.ignore); 
    end

    % Calculate sufficient statistics under q(z)
    %
    %   G(k) = Sum_t gamma(t, k)
    %   xmean(k,d) = Sum_t gamma(t, k)/G(k) x(t,d) 
    %   xvar(k,d,e) = Sum_t gamma(t, k)/G(k) 
    %                           (x(t,d1) - xmean(k,d))
    %                           (x(t,d2) - xmean(k,e))
    % G(k) = Sum_t gamma(t, k)
    G = sum(g, 1)';
    G = G + 1e-10;
    % g0(t,k) = g(t,k) / G(k)
    g0 = bsxfun(@rdivide, g, reshape(G, [1 K]));
    % xmean(k,d) = Sum_t g0(t, k) x(t,d) 
    xmean = sum(bsxfun(@times, ...
                       reshape(g0', [K 1 T]), ...
                       reshape(x', [1 D T])), 3);
    % dx(k,d,t) = x(t,d) - xmean(k,d) 
    dx = bsxfun(@minus, reshape(x', [1 D T]), xmean);
    % xvar(k, d, e) = Sum_t g0(t, k) dx(k, d, t) dx(k, e, t)
    xvar = squeeze(sum(bsxfun(@times, ...
                              reshape(g0', [K 1 1 T]), ... 
                              bsxfun(@times, ...
                                     reshape(dx, [K D 1 T]), ...
                                     reshape(dx, [K 1 D T]))), 4));
    
    % hack: check for divergences in likelihood
    if any(xvar < args.min_var)
        for k = 1:K
            msk = xvar(k,:,:) < args.min_var;
            if any(msk)
                % set variance to entire signal range 
                x_range = max(x(:), [], 1) - min(x(:), [], 1);
                xvar(k, :, :) = bsxfun(@times, x_range, x_range');
            end
        end
        % skip m-step updates (gamma's messed up)
        continue        
    end

    % update for pi
    theta.pi = g_(1, :);

    % update for A
    theta.A = normalize(squeeze(sum(xi_, 1)), 2);

    % update for mu
    theta.mu = xmean;

    % update for Lambda
    if D > 1
        for k = 1:K
            theta.Lambda(k, :, :) = inv(squeeze(xvar(k, :, :)));
        end
    else
        theta.Lambda = 1 ./ xvar;
    end

    %fprintf('it: %d, pi: %s, mu: %s\n', it, sprintf('%.2f  ', theta.pi), sprintf('%.2f  ', theta.mu))
end

stat = struct();
stat.gamma = g;
stat.xi = xi;
stat.ln_Z = ln_Z;
% stat.xmean = xmean;
% stat.xvar = xvar;
% stat.ln_px_z = ln_px_z;

% print debugging output
if Debug
    fprintf(['\nRUN SUMMARY:\n', ...
             '  iterations   ', sprintf('%d', it), '\n', ...
             '  F:           ', sprintf('% 7.1e', L(end)), '\n', ...
             '    ln(Z):     ', sprintf('% 7.1e', ln_Z), '\n','\n\n']);
end