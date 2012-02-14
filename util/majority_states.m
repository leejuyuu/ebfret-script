function [vb, u, idxs] = majority_states(K, vb, u, varargin)
	% [vb, u, idxs] = majority_states(K, vb, u, varargin)
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
    % vb : (1xN) struct
    %   VBEM output for each trace (see function docs)
    %
    % u : struct
    %   Hyperparameters obtained from HMI process (see function docs)
    %
    %
    % Variable Inputs
    % ---------------
    %
    % 'hstep' : {'ml', 'mm'} (default: 'ml')
    %   Algorithm to use 
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
    % vb : (1xM) struct
    %   Filtered VBEM output for each trace. 
    %
    % u : struct
    %   Hyperparameters optimized from filtered VBEM output
    %
    % idxs : (1xN)
    %   Indices of selected traces
    %
    % Jan-Willem van de Meent
    % $Revision: 1.0$  $Date: 2011/02/14$

    % parse inputs
    ip = InputParser();
    ip.StructExpand = true;
    ip.addRequired('K', @isscalar);
    ip.addRequired('vb', @isstruct);
    ip.addRequired('u', @isstruct);
    ip.addParamValue('hstep', 'ml', ...
                      @(s) any(strcmpi(s, {'ml', 'mm'})));
    ip.addParamValue('max_outliers', inf, @isscalar);
    ip.parse(K, vb, u, varargin{:});
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

	[K_orig D] = size(vb(1).w.mu);

    if K_orig > K
        % get posterior params
        w = [args.vb.w];
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
    	vb = args.vb(idxs);
    	% project w and stat down to k states
	    for n = 1:length(vb)
		    vb(n).w = filter_w(vb(n).w, kdxs);
		    vb(n).stat = filter_stat(vb(n).stat, kdxs);
	   	end
        % construct new set of hyperparameters by running hstep update
        u  = filter_w(args.u, kdxs)
        switch args.hstep
            case {'ml'}
                u = hstep_ml([vb.w], u);
            case {'mm'}
                u = hstep_mm([vb.w], u, [vb.stat]);
        end
	else
		idxs = 1:length(vb);
	end
end
