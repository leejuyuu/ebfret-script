function M0 = M0_from_prior(PriorPar,T)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function uses the hyperparameters in PriorPar to generate 
% guesses for the starting parameters (i.e. the M0 step).
% See equations CB 10.60-10.63 and MJB 3.54, 3.56. 
%
% Initial count matrices must sum up to the length of the trace, T.
%
% Counts are assigned evenly to all hidden states.
%
% beta and v are Kx1, m is DxK, W is DxDxK.
% Wpi is 1xK and Wa is KxK with all rows identical. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% number of states
K = length(PriorPar.upi);

% pi gets 1 count
M0.upi = PriorPar.upi + dirrnd(PriorPar.upi,1);
% beta and v just get T/K counts
M0.beta = PriorPar.beta + T/K;
M0.v = PriorPar.v + T/K;
% generate precision of each state using *prior* W, v
lambda = zeros(1,K);
M0.ua = zeros(K);
for k = 1:K
    lambda(k) = wishrnd(PriorPar.W(k),PriorPar.v(k));
    % each row of A should have (T-1)/K counts
    M0.ua(k,:) = PriorPar.ua(k,:) + dirrnd(PriorPar.ua(k,:)+0.1,1)*(T-1)/K;
end
% set W such that Wv = covariance 
M0.W = lambda ./ M0.v';

% use covariances and *prior* beta to calculate mus
sigma_mu = sqrt( (1 ./ (lambda .* PriorPar.beta') ) );

M0.mu = sigma_mu.*randn(1,K) + PriorPar.mu;



% if nargin == 1
%     M0 = PriorPar;
% else
%     M0.upi = 2*PriorPar.upi;
%     M0.mu = PriorPar.mu;
%     M0.beta = PriorPar.beta + [20.5 28.333 84.167]';
%     M0.v = PriorPar.v + [20.5 28.333 84.167]';
%     M0.W = PriorPar.v' .* PriorPar.W ./M0.v';
%     M0.ua = normalise(PriorPar.ua,2) .* [20.5 28.333 84.167; 20.5 28.333 84.167;20.5 28.333 84.167]';
%     keyboard
% end
