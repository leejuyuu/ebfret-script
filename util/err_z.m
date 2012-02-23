function e = err_z(z_a, z_b, K)
	if nargin < 3
		K = max(max(z_a), max(z_b));
	end
	if length(z_a(:)) == length(z_a)
		z_a = int_to_ind(z_a, K);
	end
	if length(z_b(:)) == length(z_b)
		z_b = int_to_ind(z_b, K);
	end
	[T, K] = size(z_a);
	perm = perms(1:K);
	for p = 1:length(perm)
		e(p) = sum(sum(abs(z_a - z_b(:, perm(p, :)))));
	end
	e = min(e);
