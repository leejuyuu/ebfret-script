function em_fret(save_name, data_files, K_values, restarts, varargin)
    % em_fret(save_name, data_files, K_values, restarts, varargin)
    %
    % Runs EM inference (maximum likelihood) on a set of FRET time series.
	%
	% Inputs
	% ------
	%
	% save_name : string
	%	File name to save results to (without extension)
	%
	% data_files : (1xD) cell
	%	Set of file names to load FRET data from (see load_fret)
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
	% 'work_dir' : string 
	%	Working directory for algorithm
	%
	% 'load_fret' : struct
	%	Any options to pass to load_fret function
	%
	% 'em' : struct
	%	Any options to pass to em algorithm
	%
	% 'display' : {'all', 'traces', 'states', 'off'}
	%	Verbosity of progress messages
	%
	% 'num_cpu' : int (default: 1)
	% 	Number of cpu's to use
	%
	% Outputs
	% -------
	% 
	% Results are saved to 'save_name.mat'.

	% parse input
    ip = InputParser();
    ip.StructExpand = true;
    ip.addRequired('save_name', @isstr);
	ip.addRequired('data_files', @(d) iscell(d) | isstr(d));
	ip.addRequired('K_values', @isnumeric);
	ip.addRequired('restarts', @isscalar);
	ip.addParamValue('work_dir', pwd(), @isstr);
	ip.addParamValue('load_fret', struct(), @isstruct);
	ip.addParamValue('em', struct(), @isstruct);
    ip.addParamValue('display', 'off', ...
                      @(s) any(strcmpi(s, {'all', 'traces', 'states', 'none'})));
	ip.addParamValue('num_cpu', 1, @isscalar);
    ip.parse(save_name, data_files, K_values, restarts, varargin{:});
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
    opts_load_fret = load_fret_defaults();
    fnames = fieldnames(opts_load_fret);
    for f = 1:length(fnames)
    	if ~isfield(opts.load_fret, fnames{f})
    		opts.load_fret.(fnames{f}) = opts_load_fret.(fnames{f});
    	end
    end

	try
		% load data
		data = load_fret(opts.data_files, opts.load_fret)
		fret = cat(2, data.fret);
		N = length(fret);
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
					w0 = init_w_gmm(fret{n}, u);
					ml{n,r}.theta0 = theta_map(w0);
					[ml{n,r}.theta, ml{n,r}.L, ml{n,r}.stat] = ...
						em(fret{n}, ml{n,r}.theta0, opts.em);
					% hack: set L to -inf if likelihood diverged
					if any(ml{n,r}.stat.gamma ~= ml{n,r}.stat.gamma)
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
				[vit(n).z vit(n).x] = viterbi_em(ml(n).theta, fret{n});
			end

			% assign outputs
			runs(k).K = K;
			runs(k).ml = ml;
			runs(k).vit = vit;
		end

		% restore warning status
		warning(warn);

		% save results to disk
		save_name = sprintf('%s.mat', opts.save_name);
		save(save_name, 'data', 'opts', 'runs');
	
	    % close matlabpool if necessary
	    if opts.num_cpu > 1
	    	matlabpool('CLOSE');
	    end
	catch ME
		% ok something went wrong here, so dump workspace to disk for inspection
		day_time = 	datestr(now, 'yymmdd-hh.MM');
		save_name = sprintf('%s-crashdump-%s.mat', opts.save_name, day_time);
		save(save_name);

	    % close matlabpool if necessary
	    if opts.num_cpu > 1
	    	matlabpool('CLOSE');
	    end

		throw(ME);
	end