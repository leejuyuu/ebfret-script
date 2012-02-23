function prob = norm_hist(counts, bins, lim)
	if nargin < 3
		lim(1) = -inf;
		lim(2) = inf;
	end
	db = zeros(size(bins));
	db(2:end-1) = (bins(3:end) - bins(1:end-2)) / 2;
	db(1) = bins(2) - bins(1);
	db(end) = bins(end) - bins(end-1);
	mask = (db > min(lim)) & (db < max(lim));
	prob = counts ./ (db * sum(counts(mask)));
end
