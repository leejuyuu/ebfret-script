function PriorPar = get_priors(K,h)

% Hyperparameters for 1D
PriorPar.upi = ones(1,K);

% mus are a bit complicated
if h == 1
    % first try spreding states evenly between 0 and 1
    mu = (1:K)/(K+1);
else
    % then try at random, subject to some constraints below
    mu = rand(1,K);
end

% this just makes later analysis and calculations easier
mu = sort(mu);

% smoth things out a bit
if mod(h,2)
    % make things uniform between mu(min) and mu(max)
    mu = linspace(mu(1),mu(end),K);
else 
    % no point in having 2 states directly on top of each other
    while min(mu) < 0.02
        mu = rand(1,K);
        mu = sort(mu);
    end
end

PriorPar.mu = mu;
PriorPar.beta = 1*ones(K,1);
% might want to change this later
PriorPar.W = 50*ones(1,K);
PriorPar.v = 5*ones(K,1);
PriorPar.ua = ones(K);