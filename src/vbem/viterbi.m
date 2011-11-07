function [z_hat x_hat] = viterbi(w, x)
% function [z_hat x_hat] = viterbi(w, x)
%
% Determines Viterbi path on time series x using posterior parameter
% estimates w.
%
% Inputs
% ------
%
%   w : struct
%       Variational parameters of approximate posterior distribution 
%       for parameters q(theta | w), for each of N traces 
%
%       .A (K x K)
%           Dirichlet prior for each row of transition matrix
%       .pi (K x 1)
%           Dirichlet prior for initial state probabilities
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
%   x : (TxD)
%       Observation sequence (i.e. FRET signal)
%
% Outputs
% -------
%
%   z_hat : (Tx1)
%       Index of most likely state at every time point
%
%   x_hat : (TxD)
%       Mean emissions level of most likely state at every time point
%
%
% TODO: untested for use with D>1 time series
%
% Jan-Willem van de Meent
% $Revision: 1.00$  $Date: 2011/11/07$

% A lot of paths have 0 probablity. Not a problem for the calculation, but
% creates a lot of warning messages.
warning('off','MATLAB:log:logOfZero')

% get dimensions
[K D] = size(w.mu);
T = length(x);

% get MAP estimate for transition matrix
A = normalize(w.A, 2);

% calculate emission probabilities

% define gaussian distribution
if D == 1
	gauss = @(x, mu, l) (l/(2*pi))^0.5 * exp(-0.5 * l * (x - mu)^2);
else
	gauss = @(x, mu, L) ...
	   (2*pi).^(-0.5*D) * det(L).^(0.5) ...
	   .* exp(-0.5 .* (x - mu)' * L * (x - mu));
end

% Compute values for timestep 1
% omega(z1) = ln(p(z1)) + ln(p(x1|z1))
% CB 13.69
pZ0 = normalize(w.pi);
omega = zeros(T, K);
for k=1:K
   omega(1, k) = log(pZ0(k)) + log(gauss(x(1, :), w.mu(k, :), w.W(k, :, :) * w.nu(k)));
end


% stores most likely previous state at each timepoint (dependent on the state)
z_max = zeros(T, K);

% arbitrary value, since there is no predecessor to t=1
z_max(1, :) = 0;

% forward pass
% omega(zt) = ln(p(xt|zt)) + max{ ln(p(zt|zt-1)) + omega(zt-1) }
% CB 13.68
for t=2:T
    for k=1:K
        [omega(t, k) z_max(t, k)] = max(log(A(:, k)') + omega(t-1, :));
        omega(t, k) = omega(t, k) + log(gauss(x(t,:), w.mu(k,:), w.W(k,:,:) * w.nu(k)));
    end
end
    
% backward pass
z_hat = zeros(T, 1);
[L z_hat(T)] = max(omega(T,:));
for t=(T-1):-1:1
    z_hat(t) = z_max(t+1, z_hat(t+1));
end
x_hat = w.mu(z_hat, :);

warning('on','MATLAB:log:logOfZero')
