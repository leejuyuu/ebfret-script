function w = init_w(u, counts, varargin)
% w = init_w(u, counts, varargin)
%
% Initializes a first guess for the variational parameters that 
% specify the approximating distribution for the q(theta | w).
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
% Variable Inputs
% ---------------
%
% 	randomize : boolean (default true)
%		If set to true, a set of random parameters is drawn from the 
%       prior and use to generate the pseudocounts. If set to false,
%		the pseudocounts are obtained using the expectation values of
% 		theta under the prior.
%
% Outputs
% -------
%
%   w : struct
%       Initial guess for variational parameters of posterior
%       distribution q(theta | w). Contains same fields as u.
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

% Parse variable arguments
randomize = true;
for i = 1:length(varargin)
    if isstr(varargin{i})
        switch lower(varargin{i})
        case {'randomize'}
            randomize = varargin{i+1};
        end
    end
end 

% number of states
K = length(u.pi);
% number of time points
T = counts;
% signal dimension (1 for FRET or 2 for Donor/Acceptor inference)
D = size(u.W, 2);

% this is necessary just so matlab does not complain about 
% structs being dissimilar because of the order of the fields
w = u;

if randomize
	% draw pi ~ Dir(u.pi) 
	theta.pi = dirrnd(u.pi', 1)';
	% draw A ~ Dir(u.A) 
	theta.A = dirrnd(u.A);
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
	    mu(k, :) = mvnrnd(u.mu(k, :), inv(u.beta(k) * Lambda(k, :, :)));
	end
	theta.mu = mu;
	theta.L = Lambda;
else
	% set parameters to expectation under prior
	theta.pi = normalize(u.pi);
	theta.A = normalize(u.A, 2);
	theta.mu = u.mu;
	theta.L = u.W .* u.nu; 
end

% add pi to prior with count 1
w.pi = u.pi + theta.pi;

% add draw A ~ Dir(u.A) to prior with count (T-1)/K for each row  
w.A = u.A + theta.A .* (T-1) ./ K;

% TODO: why did JonBron add 0.1 to every element here?
% w.ua(k, :) = u.ua(k, :) + dirrnd(u.ua(k, :) + 0.1, 1) .* (T-1) ./ K;

% add T/K counts to beta and nu
w.beta = u.beta + T/K;
w.nu = w.beta + 1;

for k = 1:K
    % w.mu = (u.beta * u.mu + T/K * mu) / (u.beta + T/K)
    w.mu(k, :) = (u.beta(k) * u.mu(k, :) + T/K * theta.mu(k,:)) / w.beta(k);;
end

% set W such that W nu = L 
% TODO: this should be a proper update, but ok for now
w.W = bsxfun(@times, theta.L, 1 ./ w.nu);

% this is grossly retarded, but apparently the only way to get the fields in order
w.pi = w.pi;
w.A = w.A;
w.mu = w.mu;
w.beta = w.beta;
w.W = w.W;
w.nu = w.nu;
