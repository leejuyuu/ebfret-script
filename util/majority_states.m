function [w, u, idxs] = majority_states(K, w, u, varargin)
	% [w, u, idxs] = majority_states(K, w, u, varargin)
	%
	% Finds K most populated states, and return a filtered 
	% dataset that contains only the majority state data 
    %
    %
    % Inputs
    % ------
    % 
    % K : int
    %   Number of states to select (must be smaller or
    %   equal to number of states in vb and u)
    %
    % w : (1xN) struct
    %   Posterior parameters for each trace (see vbem.m docs)
    %
    % u : struct
    %   Hyperparameters obtained from HMI process (see vbem.m docs)
    %
    %
    % Variable Inputs
    % ---------------
    %
    % 'max_outliers' : float (default: inf)
    %   Maximum number of allowed time points in outlier states. 
    %   Traces where this number is exceeded are excluded from the 
    %   filtered dataset.
    %
    %
    % Outputs
    % -------
    %
    % w : (1xL) struct
    %   Filtered VBEM output for each trace. 
    %
    % u : struct
    %   Hyperparameters optimized from filtered VBEM output
    %
    % idxs : (1xL)
    %   Indices of selected traces
    %
    % Jan-Willem van de Meent
    % $Revision: 1.0$  $Date: 2011/02/14$

    % parse inputs
    ip = inputParser();
    ip.StructExpand = true;
    ip.addRequired('K', @isscalar);
    ip.addRequired('w', @isstruct);
    ip.addRequired('u', @isstruct);
    ip.addParamValue('max_outliers', inf, @isscalar);
    ip.parse(K, w, u, varargin{:});
    args = ip.Results;

    function wf = filter_w(w, kdxs)
        % projects a set of hyperparameters down to the states 
        % specified by kdxs
    	wf = w;
    	field_names = fields(w);
    	for f = 1:length(field_names)
    		field = field_names{f};
    		if strcmp(field, 'A')
    			wf.(field) = w.(field)(kdxs, kdxs);
    		else
    			wf.(field) = w.(field)(kdxs, :);
    		end
    	end
    end

    function statf = filter_stat(stat, kdxs)
        % projects a set of sufficient statistics down to the states 
        % specified by kdxs
    	statf = stat;
    	field_names = fields(u);
    	for f = 1:length(field_names)
    		field = field_names{f};
    		switch field
	    		case {'G', 'xmean', 'xvar'}
	    			statf.(field) = stat.(field)(kdxs, :);
				case {'gamma'}
	    			statf.(field) = normalize(stat.(field)(:, kdxs), 2);
				case {'xi'}
					xi = stat.(field)(:, kdxs, kdxs);
					T = size(xi, 1);
					xi = normalize(reshape(xi, [T k*k]), 2);
	    			statf.(field) = reshape(xi, [T, k, k]);
    		end
    	end
    end

	[K_orig D] = size(w(1).mu);

    if K_orig > K
        % calculate transition matrix posterior counts
        Xi = bsxfun(@minus, cat(3, w.A), u.A);
        % get top "K" most populated states
        [srtd kdxs] = sort(mean(mean(Xi, 3), 1), 'descend');
        kdxs = sort(kdxs(1:K));
        % count all points in non-majority states
        Xi_top = squeeze(sum(sum(Xi(kdxs, kdxs, :),1),2));
        Xi_all = squeeze(sum(sum(Xi,1),2));
        outlier_points = Xi_all - Xi_top;
        % get indexes of traces that only have majority states
        idxs = find(outlier_points <= args.max_outliers);
    	% filter vbem output
    	w = [w(idxs)];
    	% project w and stat down to k states
	    for n = 1:length(w)
		    w(n) = filter_w(w(n), kdxs);
		    %vb(n).stat = filter_stat(vb(n).stat, kdxs);
	   	end
        % construct new set of hyperparameters by running hstep update
        u = filter_w(args.u, kdxs);
        u = hstep_ml(w, u);
	else
		idxs = 1:length(w);
	end
end
