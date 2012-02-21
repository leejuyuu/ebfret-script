function D_kl = kl_dir(alpha_p, alpha_q)
% D_kl = kl_dir(alpha_p, alpha_q)
%
% Returns Kullback-Leibler divergence 
%
%	D_kl(p || q) = Int d theta p(theta) log [p(theta) / q(theta)] 
%
% for two Dirichlet priors
%
% 	p(pi | alpha_p) = Dir(pi | alpha_p)
% 	q(pi | alpha_q) = Dir(pi | alpha_q)
%
% Parameters
% ----------
%
%	alpha_p, alpha_q : (SZ x K)
%		Parameters of Dirichlet distributions
%
% Output
% ------
%
%	D_kl : (SZ)
%		Kullback-Leibler divergence for each set of K parameters
%		on last dimension of alpha_p and alpha_q
%
% Jan-Willem van de Meent (modified from Matthew J. Beal)
% $Revision: 1.0$ 
% $Date: 2011/08/03$

% figure out summation dim
d = ndims(alpha_p);
d = d - (size(alpha_p, d) == 1);

Alpha_p = sum(alpha_p, d);
Alpha_q = sum(alpha_q, d);

% For each set of K parameters
% 
% D_kl = log[Gamma[sum_k alpha_p(k)] 
%            / Gamma[sum_k alpha_q(k)]]
%		 + sum_k log[Gamma(alpha_p(k)) 
%                    / Gamma(alpha_q(k)) ]
%		 + sum_k (alpha_p(k) - alpha_q(k)) 
%                (psi(alpha_p(k)) - psi(Alpha_p))
D_kl = gammaln(Alpha_p) - gammaln(Alpha_q) ...
       - sum(gammaln(alpha_p) - gammaln(alpha_q), d) ...
       + sum((alpha_p - alpha_q) .* ...
             bsxfun(@minus, psi(alpha_p), psi(Alpha_p)), d);