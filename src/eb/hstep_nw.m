function u = hstep_nw(w, weights)
% u = hstep_nw(w, weights) 
%
% Empericial Bayes step for a Normal-Wishart prior.
%
% TODO: not implemented for observation data with D>1
% 
% Inputs
% ------
%
%   w : struct (N x 1)
%       Variational parameters of approximate posterior distribution 
%       for parameters q(theta | w), for each of N traces 
%
%       .mu (K x D)
%           Normal-Wishart prior - state means 
%       .beta (K x 1)
%           Normal-Wishart prior - state occupation count
%       .W (K x D x D)
%           Normal-Wishart prior - state precisions
%       .nu (K x 1)
%           Normal-Wishart prior - degrees of freedom
%           (must be equal to beta+1)
%
%   u : struct 
%       Hyperparameters for the prior distribution p(theta | u)
%       (same fields as w)
%
%   weights : (N x 1) optional
%       Weighting for each trace in updates.
%
%
% Outputs
% -------
%
%   u_new : struct 
%       Updated hyperparameters
%
% Jan-Willem van de Meent
% $Revision: 1.10$  $Date: 2012/07/12$


% Get dimensions
N = length(w);
K = length(w(1).mu);

% intialize empty weights if unspecified
if nargin < 2
    weights = ones(N,1);
end

% assign weights to w
w0 = num2cell(weights ./ sum(weights(:)));
[w(:).wt] = deal(w0{:});

% initialize u
u = struct();
fields = {'mu', 'beta', 'W', 'nu'};
for f = 1:length(fields)
    D = ndims(w(1).(fields{f}));
    % initialize each variable from naive weighted average
    u.(fields{f}) = mean(bsxfun(@times, ...
                                cat(D+1, w.(fields{f})), ...
                                cat(D+1, w.wt)), D+1); 
end

% Expectation Value E[lambda] = w.nu * w.W
E_l = arrayfun(@(w) w.nu .* w.W .* w.wt, w, 'UniformOutput', false);
E_l = sum([E_l{:}], 2);

% Expectation Value E[mu * lambda] = w.nu * w.W * w.mu
E_ml = arrayfun(@(w) w.nu .* w.W .* w.mu .* w.wt, w, ...
                'UniformOutput', false);
E_ml = sum([E_ml{:}], 2);

% Expectation Value E[mu^2 * lambda] = 1 / w.beta + w.mu^2 w.W w.nu
E_m2l = arrayfun(@(w) (1./w.beta + w.mu.^2 .* w.W .* w.nu) .* w.wt, w, ...
                'UniformOutput', false);
E_m2l = sum([E_m2l{:}], 2);

% Expectation Value E[log lambda] = psi(w.nu / 2) + log(2 w.W)
E_log_l = arrayfun(@(w) (psi(0.5 * w.nu) + log(2 * w.W)) .* w.wt, w, ...
                   'UniformOutput', false);
E_log_l = sum([E_log_l{:}], 2);

% set small number > eps
EPS = 10 * eps;

% set optimization settings
threshold = 1e-6;
opts = optimset('display', 'off', 'tolX', threshold, 'tolFun', eps);

% (mu, lambda): Solve for u.nu
%
%   psi(u.nu/2) - log(u.nu/2) 
%       = E[log(lambda)] - log(E[lambda])
root_fun = @(nu) psi(0.5 * nu) - log(0.5 * nu) - E_log_l + log(E_l);
u.nu = lsqnonlin(root_fun, ...
                 u.nu, ...
                 ones(size(u.nu)), ...
                 Inf + zeros(size(u.nu)), ...
                 opts);

% (mu, lambda): Solve u.mu, u.beta and u.W
u.mu = E_ml ./ E_l;
u.beta = 1 ./ (E_m2l - E_ml.^2 ./ E_l);
u.W  = E_l ./ u.nu;
