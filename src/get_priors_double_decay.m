function PriorPar = get_priors_double_decay(K,h)

% Hyperparameters for 1D
PriorPar.upi = ones(1,K);

% mus are a bit complicated
if h < 3
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

if K==2
    PriorPar.mu = [0.3 0.7]
    PriorPar.ua = ones(2);
end

SMALL = 0.0001;
STAY = 1;
TRANS = 0.1;

if K==3
    PriorPar.mu = [0.3 0.3 0.7];
    
    PriorPar.ua = [1       SMALL  1;...
                   SMALL   STAY   TRANS;...
                   1       1      1];
end

if K==4 
    PriorPar.mu = [0.3 0.3 0.7 0.7];
    PriorPar.ua = [1       SMALL  SMALL  1;...
                   SMALL   STAY   TRANS  SMALL;...
                   SMALL   1      1      SMALL;...
                   1       SMALL  SMALL  1];
end
