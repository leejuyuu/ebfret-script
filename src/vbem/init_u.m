function u = init_u(K, varargin)
% u = init_u(K, varargin)
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
% 'mu' : string
%   Specifies type of prior on levels of states.
%       
%       'linear'
%           Evenly spaced values mu = (1:K) / (K+1)
%
%       'random'
%           Take K random values between 0 and 1, with a minimum
%           separation of 0.02.
%       
%       'rand_ends'
%           Evenly space values between a random minimum and maximum.
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
mu_type = 'random'
for i = 1:length(varargin)
    if isstr(varargin{i})
        switch lower(varargin{i})
        case {'mu'}
            mu_type = lower(varargin{i+1});
            valid_types = {'linear', 'random', 'rand_ends'};
            % check whether mu_type has a valid value
            valid = arrayfun(@(t) strcmp(mu_type, valid_types(i)), ... 
                             1:length(mu_types));
            if ~any(valid)
                err = MException('init_u:invalid_mu_type', ...
                                 '[init_u] Error: ''mu'' must be one of {''linear'', ''random'', ''randend''}');
                throw(err)
            end 
        end
    end
end 

% Prior for initial probabilities 
%
%   p(z(1) = k | u.pi)  =  pi(k)  ~  Dir(u.pi)
%
% make this uniform
u.pi = ones(1,K);

% Prior for transition probabilities 
%
%   p(z(t)=l | z(t-1)=k)  =  A(k, l)  ~  Dir(u.A)
%
% make this uniform too.
u.A = ones(K);

% Prior for emission model state levels and precision
%
%   lambda(k) ~ Wishart(u.nu(k), u.W(k))
%   mu(k) ~ Normal(u.mu(k), 1/(u.beta(k) lambda(k)))
%
%   p(x(t) | z(t)=k)  =  Normal(mu(k), 1/lambda(k))

% mu: choice between 'linear', 'random', and 'rand_ends'
switch(mu_type)
    case {'linear'}
        % linearly space states between 1/(K+1) and K/(K+1)
        mu = (1:K)'/(K+1);

    case {'random'}
        % ensure spacing of states is at least  0.02
        dmu = zeros(K, K);
        while min(dmu(:) < 0.02)
            % generate a set of random mu values
            mu = rand(K, 1);
            % calculate spacing between states
            dmu = abs(bsxfun(@minus, mu, mu'));
        end
        % output values in ascending order
        mu = sort(mu);

    case {'rand_ends'}
        % ensure spacing of states is at least  0.02
        dmu = zeros(K, K);
        while min(dmu(:) < 0.02)
            % generate a set of random mu values
            mu = rand(K, 1);
            % now space evenly between min and max
            mu = linspace(min(mu), max(mu), K);
            % calculate spacing between states
            dmu = abs(bsxfun(@minus, mu, mu'));
        end
end        

% beta: uniform, one count for each state 
u.beta = ones(K, 1);
% nu: fully determined by beta
u.nu = u.beta + 1;

% W: set uniformly to 400 / nu. This implies implies an expectation value for 
% the emission noise of 0.05
%
%    <lambda> = nu * W = 400
%    <sigma> = 1./sqrt(<lambda>) = 0.05
u.W = 400 ./ u.nu; 
