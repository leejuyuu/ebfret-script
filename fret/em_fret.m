function runs = em_fret(x, K_values, restarts, varargin)
    % em_fret(x, K_values, restarts, varargin)
    %
    % Runs EM inference (maximum likelihood) on a set of FRET time series.
	%
	% Inputs
	% ------
	%
    % x : (1xN) cell
    %   Time series to perform inference on.
	%
	% K_values : (1xR)
	%	Number of states to use for each run
	%
	% restarts : int
	%	Number of VBEM restarts to perform for each trace
	%
	%
	% Variable Inputs
	% ---------------
	%
	% 'num_cpu' : int (default: 1)
	% 	Number of cpu's to use
	%
	% 'display' : {'all', 'traces', 'states', 'off'}
	%	Verbosity of progress messages
	%
	% 'em' : struct
	%	Any options to pass to em algorithm
	%
	% Outputs
	% -------
	% 
    % runs : (1xR) struct
    %   Output of em inference for each K value (see em.m)
    %
    %   .u struct
    %       Hyperparameters for ensemble distribution
    %   .em (1xN) struct    
    %       VBEM output for each trace
    %   .vit (1xN) struct
    %       Viterbi path for each trace
    %   .K int
    %       Number of states

	% parse input
    ip = inputParser();
    ip.StructExpand = true;
    ip.addRequired('x', @iscell);
	ip.addRequired('K_values', @isnumeric);
	ip.addRequired('restarts', @isscalar);
	ip.addParamValue('em', struct(), @isstruct);
    ip.addParamValue('display', 'off', ...
                      @(s) any(strcmpi(s, {'all', 'traces', 'states', 'none'})));
	ip.addParamValue('num_cpu', 1, @isscalar);
    ip.parse(x, K_values, restarts, varargin{:});
    opts = ip.Results;

    % open matlabpool if using mutliple CPU's
    if opts.num_cpu > 1
    	matlabpool('OPEN', 'local', opts.num_cpu);
    end

    % set defaults for any missing options
    opts_em = em_defaults();
    fnames = fieldnames(opts_em);
    for f = 1:length(fnames)
    	if ~isfield(opts.em, fnames{f})
    		opts.em.(fnames{f}) = opts_em.(fnames{f});
    	end
    end

	try
		N = length(x);
		R = opts.restarts;

		% ignore warning messages during gmm kmeans parameter init
		warn = warning('off', 'stats:gmdistribution:FailedToConverge');

		for k = 1:length(opts.K_values)
			K = opts.K_values(k);
			if strcmpi(opts.display, 'states')
				fprintf('em_fret: %d states\n', K);
			end
			u = u_defaults(K);
			
			ml = cell(N, R);
			parfor n = 1:N
				if strcmpi(opts.display, 'traces')
					fprintf('em_fret: %d states, trace %d of %d\n', K, n , N);
				end
				for r = 1:R
					if strcmpi(opts.display, 'all')
						fprintf('em_fret: %d states, trace %d of %d, restart %d of %d\n', ...
						         K, n , N, r, R);
					end
					ml{n,r} = struct();
					w0 = init_w_gmm(x{n}, u);
					ml{n,r}.theta0 = theta_map(w0);
					[ml{n,r}.theta, ml{n,r}.L stat] = ...
						em(x{n}, ml{n,r}.theta0, opts.em);
					% hack: set L to -inf if likelihood diverged
					if any(stat.gamma ~= stat.gamma)
						ml{n,r}.L(end) = -inf
					end
				end
			end
			ml = reshape([ml{:}], [N R]);

			% keep best restart
			L = arrayfun(@(m) m.L(end), ml);
			[mx, idx] = max(L, [], 2);
			idxs = (idx(:)-1)*N + (1:N)';
			ml = ml(idxs);

			% get viterbi paths
			vit = struct();
			for n = 1:N
				[vit(n).z vit(n).x] = viterbi_em(ml(n).theta, x{n});
			end

			% assign outputs
			runs(k).K = K;
			runs(k).em = ml;
			runs(k).vit = vit;
		end

		% restore warning status
		warning(warn);

	    % close matlabpool if necessary
	    if opts.num_cpu > 1
	    	matlabpool('CLOSE');
	    end
	catch ME
		% ok something went wrong here, so dump workspace to disk for inspection
		day_time = 	datestr(now, 'yymmdd-HH.MM');
        save_name = sprintf('crashdump-em_fret-%s.mat', day_time);
		save(save_name);

	    % close matlabpool if necessary
	    if opts.num_cpu > 1
	    	matlabpool('CLOSE');
	    end

		rethrow(ME);
	end
