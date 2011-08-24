function u = init_u(K, counts, varargin)
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
%	counts : int
%		Strength of prior, specified in number of pseudocounts,
%		e.g. the number of virtual obeservations that have informed
%		the prior. 
%
% Variable Inputs
% ---------------
%
% 'sep' : float (default 0.02)
%   Minimum separation of states.
%
% 'min' : float (default 0)
%   Minimum level for states.
%
% 'max' : float (default 1)
%   Maximum level for states.
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
sep_mu = 0.02;
min_mu = 0;
max_mu = 1;
for i = 1:length(varargin)
    if isstr(varargin{i})
        switch lower(varargin{i})
        case {'sep'}
            sep_mu = varargin{i+1};
        case {'min'}
            min_mu = varargin{i+1};
        case {'max'}
            max_mu = varargin{i+1};
        end
    end
end 

% Prior for initial probabilities 
%
%   p(z(1) = k | u.pi)  =  pi(k)
%
% make this uniform with one count
u.pi = ones(K, 1) ./ K;

% Prior for transition probabilities 
%
%   p(z(t)=l | z(t-1)=k)  =  A(k, l)
%
% uniform distribtuion of counts
u.A = counts * ones(K, K) / K^2;

% Prior for emission model parameters
%
% p(x(t) | z(t)=k) = Norm(x(t) | mu(k), Lambda(k))

% range of mu values
rng_mu = (max_mu - min_mu);

% check whether minimum separation of states fits into range
if sep_mu > rng_mu / (K-1)
    warning(['Minimum separation of states ''sep_mu = %.2f'' is too',
             'large for specified range. Setting ''sep_mu = %.2f''.'], ... 
             sep_mu, rng_mu / (K-1));
    sep_mu = rng_mu / (K-1);
end

if sep_mu > 0
    % draw K+1 intervals from dirichlet, such that intervals
    % add up to specified range, discounted for minimum sep 
    int_mu = (rng_mu - (K - 1) * sep_mu) * dirrnd(ones(1, K + 1));
    mu = cumsum(int_mu(1:K))' + sep_mu * (0:(K-1))';
else
    % take evenly spaced fret levels over range
    mu = min_mu + sep_mu * (0:(K-1))';
end 

% set mu  
u.mu = mu;

% beta: uniform distribution of counts
u.beta = counts * ones(K, 1) / K;

% W: set uniformly to 400 / nu. This implies implies an expectation value for 
% the emission noise of 0.05
%
%    <lambda> = nu * W = 400
%    <sigma> = 1./sqrt(<lambda>) = 0.05
u.W = 400 ./ (u.beta + 1); 

% nu: fully determined by beta
u.nu = u.beta + 1;
