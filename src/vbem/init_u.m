function u = init_u(K, varargin)
% u = init_u(K, counts, varargin)
%
% Initializes a set of hyperparameters u used to define the prior 
% distribution p(theta | u).
%
%
% Inputs
% ------
%
%   K : int
%       Number of states.
%
%
% Variable Inputs
% ---------------
%
% 'mu_counts' : int 
%   Strength of prior, specified in number of pseudocounts,
%   e.g. the number of virtual obeservations that have informed
%   the prior. If specified, beta = mu_counts / K * ones(K, 1), 
%   otherwise beta = 1.
%
% 'mu_sep' : float (default 0.02)
%   Minimum separation of states.
%
% 'mu_min' : float (default 0)
%   Minimum level for states.
%
% 'mu_max' : float (default 1)
%   Maximum level for states.
%
% 'A_counts' : int (default 0)
%   Number of pseudocounts to assign to transition matrix prior,
%   e.g. the number of virtual transitions that have been previously
%   observed. Note that u.A = 1 is equivalent to an ininformative
%   prior, so these counts are added to a matrix ones(K, K)
%
% 'A_tau' : float
%   If unset, additional counts are assigned uniformly to prior. 
%   If specified, diagonal elements are set to ensure that the 
%   expected life-time of each state is equal to A_tau. 
%
%
% Outputs
% -------
%
%   u : struct
%       Hyperparameters for prior distribution p(theta | u)
%
%       .A (KxK)
%           Dirichlet prior for each row of transition matrix
%       .pi (Kx1)
%           Dirichlet prior for initial state probabilities
%       .mu (KxD)
%           Gaussian-Wishart posterior for state means 
%       .beta (Kx1)
%           Gaussian-Wishart posterior for state occupation count
%       .W (KxDxD)
%           Gaussian-Wishart posterior for state precisions
%       .nu (Kx1)
%           Gaussian-Wishart posterior for degrees of freedom
%           (must be equal to beta+1)
%
%
% Jan-Willem van de Meent (modified from Jonathan Bronson)
% $Revision: 1.00 $  $Date: 2011/08/03$

% TODO
% ----
%
% * This function currently does not support D=2 donor/acceptor inference
%   (have no idea how to set priors for mu for this case)

% Parse variable arguments
mu_counts = K;
mu_sep = 0.02;
mu_min = 0;
mu_max = 1;
A_counts = 0;
A_tau = 0;
for i = 1:length(varargin)
    if isstr(varargin{i})
        switch lower(varargin{i})
        case {'mu_counts'}
            mu_counts = varargin{i+1};
        case {'mu_sep'}
            mu_sep = varargin{i+1};
        case {'mu_min'}
            mu_min = varargin{i+1};
        case {'mu_max'}
            mu_max = varargin{i+1};
        case {'a_counts'}
            A_counts = varargin{i+1};
        case {'a_tau'}
            A_tau = varargin{i+1};
        end
    end
end 

% Prior for initial probabilities 
%
%   p(z(1) = k | u.pi)  =  pi(k)
%
% make this uniform with one count in each state
u.pi = ones(K, 1);

% Prior for transition probabilities 
%
%   p(z(t)=l | z(t-1)=k)  =  A(k, l)
if A_tau
    % exp(-1 ./ A_tau) on diagonal, constant elsewhere
    A = exp(-1 ./ A_tau) * eye(K) ...
        + (1 - exp(-1 ./ A_tau)) * (~eye(K)) / (K - 1);
else
    % constant
    A = ones(K, K);
end
% ensure u.A contains A_counts total counts
u.A = A_counts * A  + ones(K, K);


% Prior for emission model parameters
%
% p(x(t) | z(t)=k) = Norm(x(t) | mu(k), Lambda(k))

% range of mu values
rng_mu = (mu_max - mu_min);

% check whether minimum separation of states fits into range
if mu_sep > rng_mu / (K-1)
    warning(['Minimum separation of states ''mu_sep = %.2f'' is too', ...
             'large for specified range. Setting ''mu_sep = %.2f''.'], ... 
             mu_sep, rng_mu / (K-1));
    mu_sep = rng_mu / (K-1);
end

if mu_sep > 0
    % draw K+1 intervals from dirichlet, such that intervals
    % add up to specified range, discounted for minimum sep 
    int_mu = (rng_mu - (K - 1) * mu_sep) * dirrnd(ones(1, K + 1));
    mu = cumsum(int_mu(1:K))' + mu_sep * (0:(K-1))';
else
    % take evenly spaced fret levels over range
    mu = mu_min + mu_sep * (0:(K-1))';
end 

% set mu  
u.mu = mu;

% beta: uniform distribution of counts
u.beta = mu_counts * ones(K, 1) / K;

% W: set uniformly to 400 / nu. This implies implies an expectation value for 
% the emission noise of 0.05
%
%    <lambda> = nu * W = 400
%    <sigma> = 1./sqrt(<lambda>) = 0.05
u.W = 400 ./ (u.beta + 1); 

% nu: fully determined by beta
u.nu = u.beta + 1;
