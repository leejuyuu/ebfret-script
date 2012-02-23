function pd =  pd_dir(x, alpha)
	% pd = pd_dir(x, alpha)
	%
	% Probability density function for a Dirichlet distribution

	% figure out summation dim
	d = ndims(alpha_p);
	d = d - (size(alpha_p, d) == 1);

	Beta = @(alpha, d) prod(gamma(alpha), d)) ./ gamma(sum(alpha, d);
	pd = 1./Beta(alpha, d) * prod(x.^(alpha-1), d);