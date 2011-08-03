function w0 = init_w(u, counts)
% w0 = init_w(u, counts)
%
% Initializes a first guess for the variational parameters that 
% specify the approximating distribution for the q(theta | w0).
%
% This first guess is constructed by drawing a set of parameters 
% theta = [pi, A, mu, lambda] from the priors:
%
%   pi ~ Dir(u.pi)
%   A ~ Dir(u.A)
%   mu, lambda ~ Gaussian-Wishart(u.mu, u.beta, u.W, u.nu)
%
% These parameters are then used to update the u, weighted by
% the specified number of pseudocounts. The initial guess is 
% therefore analogous to the posterior after observation of a
% time series with length T=counts and parameters theta. 
%
% Counts are distributed evenly between states.
%
%
% Inputs
% ------
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
%   counts : integer
%       Number of pseudocounts to use for randomization. 
%		This should be equal to the number of time points
%       in the FRET series on which inference is performed.
%
%
% Outputs
% -------
%
%   w0 : struct
%       Initial guess for variational parameters of posterior
%       distribution q(theta | w0). Contains same fields as u.
%
% Jan-Willem van de Meent (modified from Jonathan Bronson)
% $Revision: 1.00 $  $Date: 2011/08/03$

% References
% ----------
%
%   [Bishop] equations 10.60-1.63
%   [Beal] equations 3.54 and 3.56
%
% 
% TODO
% ---- 
%  * Work around use of mvnrnd (needs statistics toolbox)
%  * When u.pi and u.A are not uniform, counts should not be evenly
%    distributed among states.
%  * W update is not weighted correctly in terms of counts
%    (or indeed weighted at all)

% number of states
K = length(u.pi);
% number of time points
T = counts;
% signal dimension (1 for FRET or 2 for Donor/Acceptor inference)
D = size(u.W, 2);

% add draw pi ~ Dir(u.pi) to prior with count 1
w0.pi = u.pi + dirrnd(u.pi', 1)';

% add draw A ~ Dir(u.A) to prior with count (T-1)/K for each row  
w0.A = u.A + dirrnd(u.A) .* (T-1) ./ K;

% TODO: why did JonBron add 0.1 to every element here?
% w0.ua(k, :) = u.ua(k, :) + dirrnd(u.ua(k, :) + 0.1, 1) .* (T-1) ./ K;

% add T/K counts to beta and nu
w0.beta = u.beta + T/K;
w0.nu = w0.beta + 1;

% draw state means and emission precision matrices
%
%   mu ~ N(u.mu, Inv(u.beta * Lambda))
%   Lambda ~ Wish(u.W, u.nu) for each k
mu = zeros(K, D);
Lambda = zeros(K, D, D);
for k = 1:K
    % draw precision matrix from wishart
    Lambda(k, :, :) = wishrnd(u.W(k, :, :), u.nu(k));
    % draw mu from multivariate normal
    w_mu_k = mvnrnd(u.mu(k, :), inv(u.beta(k) * Lambda(k, :, :)));
    % w0.mu = (u.beta * u.mu + T/K * mu) / (u.beta + T/K)
    w0.mu(k, :) = (u.beta(k) * u.mu(k, :) + T/K * w_mu_k) / w0.beta(k);;
end

% set W such that W nu = precision 
w0.W = bsxfun(@times, Lambda, 1 ./ w0.nu);