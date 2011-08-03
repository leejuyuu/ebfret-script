function D_kl = kl_normwish(mu_p, beta_p, W_p, nu_p, ...
                            mu_q, beta_q, W_q, nu_q)
% D_kl = kl_normwish(mu_p, beta_p, W_p, nu_p, ...
%                    mu_q, beta_q, W_q, nu_q)
%
% Returns Kullback-Leibler divergence 
%
%   D_kl(p || q) = Int d theta p(theta) log [p(theta) / q(theta)]
%
% for two Normal-Wishart Priors
%
%   p(mu, L | u_p) = Norm(mu | mu_p, beta_p L_p) Wish(W_p, nu_p)
%   q(mu, L | u_q) = Norm(mu | mu_q, beta_q L_q) Wish(W_q, nu_q)
%
%   p(mu, L | mu0, beta, W, nu) 
%     = B(W, nu) (beta / 2 pi)^D/2 |L|^(nu - D)/2 
%       exp[-1/2 (mu - mu0)T L (mu - mu0)]
%       exp[-1/2 Tr(Inv(W) * L)]
%
%
% Parameters
% ----------
%
%   mu_p, mu_q : (K x D)
%       Mean for distribution mu
%
%   beta_p, beta_q : (K)
%       Pseudo-counts for distribution on mu
%
%   W_p, W_q : (K x D x D)
%       Pseudo-counts for distribution on lambda
%
%   nu_p, nu_q : (K)
%       Pseudo-counts for distribution on lambda
%
%
% Output
% ------
%
%   D_kl : (K x K)
%       Kullback-Leibler divergence for each set of parameters
%       theta_p(k) and theta_q(l)
%
%
% Jan-Willem van de Meent (modified from Matthew J. Beal)
% $Revision: 1.0$ 
% $Date: 2011/08/03$


% HELPER FUNCTIONS
D = @(var) size(var, 1);

% Log of normalisation B for Wishart prior (CB B.79)
% 
% log(B(W, nu)) = log( |W|^nu/2 [ 2^(nu D / 2) pi^(D (D-1) / 4)  
%                                 Prod_d=1:D Gamma((nu + d - 1)/2) ])
ln_B = @(W, nu) ...
       - 0.5 * nu * log(det(W)) ...
       - 0.5 * nu * D(W) * log(2) ...
       - 0.25 * D(W) * (D(W) - 1) * log(pi) ...
       - sum(gammaln(0.5 * bsxfun(@minus, nu+1, 1:D(W))), 2);

% Expectation of log precision matrix Lambda 
% under Wish(L | W, nu) (CB 10.65, JKC 44)
%
% E_ln_det_L(k)  =  E[ln(|Lambda(k)|)]
%                =  ln(|W|) + D ln(2) 
%                   + Sum_d psi((nu(k) + 1 - d)/2)  
E_ln_det_L = @(W, nu) ...
             log(det(W)) ...
             + D(W) * log(2) + ...
             + sum(psi(0.5 * bsxfun(@minus, nu+1, 1:D(W))), 2);

% Expectation of Squared Mahalanobis distance under Normal-Wishart 
%
% E_D2 = E[x^T L dmu] = D / beta + Sum_de nu dmu(d) W(d,e) dmu(e)
E_D2 = @(mu_p, mu_q, beta_q, W_q, nu_q) ...
       D(mu_p) / beta_q ...
       + nu_q * ((mu_p-mu_q)' * W_q * (mu_p-mu_q));

% KL Divergence of two Normal-Wishart Distributions
%
% D_kl(p || q) = E_p[ln p] - E_p[ln q]
%
% E_p[ln p] =  ln B(W_p, nu_p)
%             + 0.5 D ln(beta_p/ 2 pi)
%             + 0.5 (nu_p - D) E_p[ln |L|]
%             - 0.5 D
%             - 0.5 nu_p D
%
% E_p[ln q] =  ln B(W_q, nu_q)
%             + 0.5 D ln(beta_p/ 2 pi)
%             + 0.5 (nu_q - D) E_p[ln |L|]
%             - 0.5 beta_q E_p[(mu - mu_q)T L (mu - mu_q)] 
%             - 0.5 E_p[Tr(Inv(W_q) L)]
D_kl_pq = @(mu_p, beta_p, W_p, nu_p, ...
            mu_q, beta_q, W_q, nu_q) ...
          ln_B(W_p, nu_p) - ln_B(W_q, nu_q) ...
          + 0.5 * (D(mu_p) * log(beta_p / beta_q) ...
                   + (nu_p - nu_q) * E_ln_det_L(W_p, nu_p) ...
                   - D(mu_p) ...
                   + beta_q * E_D2(mu_q, mu_p, beta_p, W_p, nu_p) ... 
                   - nu_p * D(mu_p) ...
                   + trace(inv(W_q) * (nu_p * W_p)));

% Calculate D_kl for every set of params on first axis of every var
K = size(mu_p, 1);
D_kl = arrayfun(@(k) D_kl_pq(mu_p(k), beta_p(k), W_p(k), nu_p(k), ...
                             mu_q(k), beta_q(k), W_q(k), nu_q(k)), 1:K);

