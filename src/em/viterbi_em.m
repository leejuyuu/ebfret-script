function [z_hat x_hat] = viterbi_em(theta, x)
% function [z_hat x_hat] = viterbi_em(theta, x)
%
% Determines Viterbi path on time series x using posterior parameter
% estimates w.
%
% Inputs
% ------
%
%   theta : struct 
%       Model parameters.
%
%       .A (K x K)
%           Transition matrix
%       .pi (K x 1)
%           Initial state probabilities
%       .mu (K x D)
%           State emissions means 
%       .Lambda (K x D x D)
%           State emissions precision
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
[K D] = size(theta.mu);

% calculate emission probabilities
if D == 1
	Delta2 = bsxfun(@times, theta.Lambda(:)', bsxfun(@minus, x(:), theta.mu(:)') .^ 2);
	ln_px_z = 0.5 * bsxfun(@minus, log(theta.Lambda(:)' / (2*pi)),  Delta2);
else
	% precision matrix
	gauss = @(x, mu, L) ...
	   (2*pi).^(-0.5*D) * det(theta.Lambda).^(0.5) ...
	   .* exp(-0.5 .* (x - theta.mu)' * L * (x - theta.mu));
    for k = 1:K
        px_z(:,k) = gauss(x, theta.mu(k), squeeze(theta.Lambda(k,:,:)));
    end
    ln_px_z = log(px_z);
end

% calculate viterbi paths
z_hat = viterbi(ln_px_z, log(theta.A), log(theta.pi));

% generate idealized trace
x_hat = theta.mu(z_hat, :);
