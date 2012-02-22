function vbem_fret(save_name, x, K_values, restarts, varargin)
	% vbem_fret(save_name, x, K_values, restarts, varargin)
	%
	% Runs VBEM inference (maximum evidence) on a set of FRET traces
	%
	% Inputs
	% ------
	%
	% save_name : string
	%	File name to save results to (without extension)
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
	% 'vbem' : struct
	%	Any options to pass to vbem algorithm
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
    ip = inputParser();
    ip.StructExpand = true;
    ip.addRequired('save_name', @isstr);
    ip.addRequired('x', @iscell);
	ip.addRequired('K_values', @isnumeric);
	ip.addRequired('restarts', @isscalar);
	ip.addParamValue('vbem', struct(), @isstruct);
    ip.addParamValue('display', 'off', ...
                      @(s) any(strcmpi(s, {'all', 'traces', 'states', 'none'})));
	ip.addParamValue('num_cpu', 1, @isscalar);
    ip.parse(save_name, x, K_values, restarts, varargin{:});
    opts = ip.Results;

    % open matlabpool if using mutliple CPU's
    if opts.num_cpu > 1
    	matlabpool('OPEN', 'local', opts.num_cpu);
    end

    % set defaults for any missing options
    opts_vbem = vbem_defaults();
    fnames = fieldnames(opts_vbem);
    for f = 1:length(fnames)
    	if ~isfield(opts.vbem, fnames{f})
    		opts.vbem.(fnames{f}) = opts_vbem.(fnames{f});
    	end
    end

	try
		% load data
		N = length(x);
		R = opts.restarts;

		% ignore warning messages during gmm kmeans parameter init
		warn = warning('off', 'stats:gmdistribution:FailedToConverge');

		for k = 1:length(opts.K_values)
			K = opts.K_values(k);
			if strcmpi(opts.display, 'states')
				fprintf('vbem_fret: %d states\n', K);
			end
			u = u_defaults(K);
			
			vb = cell(N, R);
			parfor n = 1:N
				if strcmpi(opts.display, 'traces')
					fprintf('vbem_fret: %d states, trace %d of %d\n', K, n , N);
				end
				for r = 1:R
					if strcmpi(opts.display, 'all')
						fprintf('vbem_fret: %d states, trace %d of %d, restart %d of %d\n', ...
						         K, n , N, r, R);
					end
					vb{n,r} = struct();
					vb{n,r}.w0 = init_w_gmm(x{n}, u);
					[vb{n,r}.w, vb{n,r}.L, vb{n,r}.stat] = ...
						vbem(x{n}, vb{n,r}.w0, u, opts.vbem);
				end
			end
			vb = reshape([vb{:}], [N R]);

			% keep best restart
			L = arrayfun(@(v) v.L(end), vb);
			[mx, idx] = max(L, [], 2);
			idxs = (idx(:)-1)*N + (1:N)';
			vb = vb(idxs);

			% get viterbi paths
			vit = struct();
			for n = 1:N
				[vit(n).z vit(n).x] = viterbi_vb(vb(n).w, x{n});
			end

			% assign outputs
			runs(k).K = K;
			runs(k).u = u;
			runs(k).vb = vb;
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
		day_time = 	datestr(now, 'yymmdd-HH.MM');
		save_name = sprintf('%s-crashdump-%s.mat', opts.save_name, day_time);
		save(save_name);

	    % close matlabpool if necessary
	    if opts.num_cpu > 1
	    	matlabpool('CLOSE');
	    end

		throw(ME);
	end
