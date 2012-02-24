function pd =  pd_dir(x, alpha)
	% pd = pd_dir(x, alpha)
	%
	% Probability density function for a Dirichlet distribution

	% figure out summation dim
	d = ndims(alpha);
	d = d - (size(alpha, d) == 1);

	ln_Beta = @(alpha, d) sum(gammaln(alpha), d) - gammaln(sum(alpha, d));
	ln_pd = -ln_Beta(alpha, d) + sum(bsxfun(@times, log(x), (alpha-1)), d);
	pd = exp(ln_pd);