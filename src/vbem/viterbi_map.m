function [z_hat x_hat] = viterbi_map(w, x)
% function [z_hat x_hat] = viterbi_map(w, x)
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

% get dimensions
[K D] = size(w.mu);

% get MAP estimates

% initial state probabilities
ln_pi = log(normalize(w.pi));
% transition matrix
ln_A = log(normalize(w.A, 2));
% emission levels
mu = w.mu;

% calculate emission probabilities
if D == 1
	% calculate Mahalanobis distance
	l = w.W .* w.nu;
	Delta2 = bsxfun(@times, l(:)', bsxfun(@minus, x(:), mu(:)') .^ 2);
	ln_px_z = 0.5 * bsxfun(@minus, log(l(:)' / (2*pi)),  Delta2);
else
	% precision matrix
	L = bsxfun(@times, w.W, w.nu);
	gauss = @(x, mu, L) ...
	   (2*pi).^(-0.5*D) * det(L).^(0.5) ...
	   .* exp(-0.5 .* (x - mu)' * L * (x - mu));
    for k = 1:K
        px_z(:,k) = gauss(x, mu(k), squeeze(L(k,:,:)));
    end
    ln_px_z = log(px_z);
end

% calculate viterbi paths
z_hat = viterbi(ln_px_z, ln_A, ln_pi);

% generate idealized trace
x_hat = mu(z_hat, :);
