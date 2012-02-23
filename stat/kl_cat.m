function D_kl = kl_cat(p, q)
% D_kl = kl_dir(p, q)
%
% Returns Kullback-Leibler divergence between two categorical 
% distributions 
%
%	D_kl(p || q) = Sum_i theta p(i) log [p(i) / q(i)] 
%
%
% Parameters
% ----------
%
%	p, q : (SZ x K)
%		Set of categorical distributions with K possible values.
%
% Output
% ------
%
%	D_kl : (SZ)
%		Kullback-Leibler divergence for each set of K parameters
%		on last dimension of p and q
%
% Jan-Willem van de Meent
% $Revision: 1.0$ 
% $Date: 2011/08/03$


% figure out summation dim
d = ndims(p);
d = d - (size(p, d) == 1);
D_kl = sum(p .* nan_to_zero(log(p ./ q)), d);