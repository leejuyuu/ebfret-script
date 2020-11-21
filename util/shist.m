function [h, b] =  shist(data, err, bins, wt)
	% [h, b] =  shist(data, err, bins, wt)
	%
	% Calculates a smoothed histogram based on a set of datapoints
	% with associated standard errors 
	if length(bins) == 1
		[h bins] = hist(data, bins);
	end
	if nargin < 4
		wt = 1;
	end
	% ensure proper orientation with vector input
	if max(size(data)) == length(data(:))
		data = data(:)';
		err = err(:)';
		wt = wt(:)';
	end
	h = zeros(length(bins), size(data,1));
	for d = 1:size(data,1)
		pd = pd_norm(bins(:), data(d,:), 1./err(d,:).^2);
		% normalize distribution over bins
		pd = normalize_old(pd, 1);
		% histogram is sum over all traces
		h(:,d) = sum(bsxfun(@times, wt, pd), 2);
	end
	b = bins(:);
