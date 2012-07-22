function  [gamma, xi, ln_Z] = forwback_banded(px_z, a, pi, d)
% [gamma, xi, ln_Z] = forwback_banded(px_z, a, pi)
%          
% Performs forward-backward message passing for HMMs.
% 
% Inputs
% ------
%
%   px_z : T x K 
%       Observation likelihood p(x(t) | z(t)=k, theta) = px_z(t, k) 
%       given the latent state at each time point.
%
%   a : K x L 
%       Transition probabilities for states
%
%         a(k, l) = p(z(t+1)=k+d(l) | z(t)=k, theta)
%
%   pi : K x 1
%       Prior probabilities for states
%
%         p(z(1)=k | theta)  =  pi(k)
%
%   d : L x 1
%       Indices for diagonals.
%
% Outputs
% -------
%
%   gamma : T x K
%       Posterior probabilities for states
%         p(z(t)=k | x(1:T))  =  gamma(t, k)
%
%   xi : K x L
%       Posterior joint counts for states
%         xi(k, l) = sum_t p(z(t+1)=k+d(l), z(t)=k | x(1:T)) 
%
%   ln_Z : float
%       Log normalization constant Z = p(x(1:T) | theta)
%
% Jan-Willem van de Meent 
% $Revision: 1.2$  $Date: 2011/08/08$

% get number of columns in banded transition matrix
L = size(a,2);

% assume columns of A centered around diagonal if d unspecified
if nargin < 4
  if mod(L, 2)
    d = -0.5*(L-1):0.5*(L-1);
  else
    d = [-0.5*L:-1 1:0.5*L];
  end
end

% get dimensions
[T, K] = size(px_z);
L = length(d);

% construct sparse transition matrix
A = spdiags(a, d, zeros(K));

% Forward backward message passing  
%                           
% alpha(t, k) = p(x(1:t), z(t)=k) / p(x(1:t))
% beta(t, k) = p(x(t+1:T) | z(t)=k) / p(x(t+1:T) | x(1:t))
% scale(t, k) = p(x(t) | x(1:t-1))
alpha = zeros(T,K);
beta = zeros(T,K);
c = zeros(T,1);

% Forward pass (with scaling)
alpha(1,:) = pi' .* px_z(1, :);
c(1) = sum(alpha(1, :));
alpha(1,:) = alpha(1, :) ./ c(1); 
for t=2:T
  % alpha(t, k) = sum_l p(x(t) | z(t)=k) A(l, k) alpha(t-1, l) 
  alpha(t,:) = (alpha(t-1, :) * A) .* px_z(t, :); 
  % c(t) = p(x(t) | x(1:t-1)) = sum_l alpha(t, l)
  c(t) = sum(alpha(t, :));
  % normalize alpha by factor p(x(t) | p(x(1:t-1)))
  alpha(t,:) = alpha(t,:) ./ c(t); 
  % note: prod(c(1:t)) = p(x(1:t))
end
  
% Backward pass (with scaling)
beta(T,:) = ones(1, K);
for t=T-1:-1:1
  % beta(t, k) = sum_l p(x(t+1) | z(t)=l) A(k, l) beta(t+1, l) 
  beta(t, :) = (beta(t+1,:) .* px_z(t+1,:)) * A' / c(t+1);  
  % note: prod(c(t+1:T)) = p(x(t+1:T) | x(1:t))
end

% Posterior probabilities for states
%
% gamma(t, k) = p(z(t) | x(1:T))
%             = p(x(1:t), z(t)) p(x(t+1:T) | z(t)) / p(x(1:T))
%             = alpha(t) beta(t) 
gamma = alpha .* beta;

% Posterior transition joint probabilities
%
% xi(k, l) = sum_t p(z(t)=k, z(t+1)=k+d(l) | x(1:T))
%          = alpha(t, k) a(k,l) px_z(t+1, k+d(l)) beta(t+1, d(l)) / c(t+1)
pxz_b_c = bsxfun(@times, 1./c(2:T), beta(2:T, :)) .* px_z(2:T, :);
xi_ = ...
  bsxfun(@times, A, ...
         sum(bsxfun(@times, ...
                    reshape(alpha(1:T-1, :)', [K 1 T-1]), ...
                    reshape(pxz_b_c', [1 K T-1])), 3));
xi = spdiags(xi, d);
for l = 1:length(d)
  % spdiags puts the zeros in the wrong place for our purposes
  % so correct this
  xi(:, l) = circshift(xi(:, l), -d(l));
end

% Evidence
%
% Ln Z = Ln[p(x(1:T))] = Sum_t log(c(t))
ln_Z = sum(log(c));