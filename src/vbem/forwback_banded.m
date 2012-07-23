function  [gamma, xi, ln_Z] = forwback_banded(px_z, A, d, alpha_1, beta_T)
% [gamma, xi, ln_Z] = forwback_banded(px_z, A, d, alpha_1, beta_T)
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
%   A : K x L 
%       Transition probabilities for states
%
%         a(k, l) = p(z(t+1)=k+d(l) | z(t)=k, theta)
%
%       Note that a(k, l) may assign a non-zero probability to 
%       transitions outside the range 1:K. These transitions are
%       assumed to imply an invalid trajectory, and will be
%       assigned zero probability in the posteriors gamma and xi. 
%       However, the probability mass assigned to the transitions 
%       does imply a lower log likelihood p(x(1:T) | theta).
%
%   d : L x 1
%       Indices for diagonals.
%
%   alpha_1 : K x 1, optional
%       Weights for state at t=1
%
%   beta_T : K x 1, optional
%       Weights for state at t=T
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
L = size(A, 2);

% get number of time points and number of states
[T, K] = size(px_z);

% assume columns of A centered around diagonal if d unspecified
if nargin < 3
  if mod(L, 2)
    d = -0.5*(L-1):0.5*(L-1);
  else
    d = [-0.5*L:-1 1:0.5*L];
  end
end

if nargin < 4
  alpha_1 = ones(K, 1);
else
  alpha_1 = alpha_1(:);
end

if nargin < 5
  beta_T = ones(K, 1);
else
  beta_T = beta_T(:);
end


% if specified transition probabilities are such that a
% trajectory can go out of bounds, the we need to pad the
% accessible states with sink states that have zero probability
pad = [0 0];
for l = 1:length(d)
  if d(l) < 0
    pad(1) = abs(min(min((A(:,l) > 0) .* ...
                         (d(l):K+d(l)-1)'), ...
                     -pad(1)));
  end
  if d(l) > 0
    pad(2) = max(max((A(:,l) > 0) .* ...
                     (d(l)-K+1:d(l))'), ...
                 pad(2));
  end
end

% pad px_z, alpha_1 and beta_T with zeros
px_z = cat(2, zeros(T, pad(1)), px_z, zeros(T, pad(2)));
alpha_1 = cat(1, zeros(pad(1),1), alpha_1, zeros(pad(2),1));
beta_T = cat(1, zeros(pad(1),1), beta_T, zeros(pad(2),1));

% realign diagonals in A
A = cat(1, zeros(pad(1), length(d)), A, zeros(pad(2), length(d)));
for l = 1:length(d)
  A(:,l) = circshift(A(:,l), d(l)); 
end

% adjust numer of states
K = K + sum(pad);

% construct sparse transition matrix
A = spdiags(A, d, zeros(K));

% Forward backward message passing  
%                           
% alpha(t, k) = p(x(1:t), z(t)=k) / p(x(1:t))
% beta(t, k) = p(x(t+1:T) | z(t)=k) / p(x(t+1:T) | x(1:t))
% scale(t, k) = p(x(t) | x(1:t-1))
alpha = zeros(T,K);
beta = zeros(T,K);
c = zeros(T,1);

% Forward pass (with scaling)
alpha(1,:) = alpha_1' .* px_z(1, :);
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

% assign backward sweep variable for last time point
beta(T,:) = beta_T';

% correct last point to reflect pinning beta_T
alpha(T,:) = alpha(T,:) .* beta(T, :) * c(T);
c(T) = sum(alpha(T, :));
alpha(T,:) = alpha(T,:) / c(T);

% Backward pass (with scaling)  
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
xi = ...
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

% strip padded states from g and xi
gamma = gamma(:, 1+pad(1):end-pad(2));
xi = xi(1+pad(1):end-pad(2),:);

% Evidence
%
% Ln Z = Ln[p(x(1:T))] = Sum_t log(c(t))
ln_Z = sum(log(c));