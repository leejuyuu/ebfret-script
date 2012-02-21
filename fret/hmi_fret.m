function hmi_fret(save_name, data_files, K_values, restarts, varargin)
    % hmi_fret(save_name, data_files, K_values, restarts, varargin)
    %
    % Runs HMI inference on a set of FRET time series.
    %
    % Inputs
    % ------
    %
    % save_name : string
    %   File name to save results to (without extension)
    %
    % data_files : (1xD) cell
    %   Set of file names to load FRET data from (see load_fret)
    %
    % K_values : (1xR)
    %   Number of states to use for each run
    %
    % restarts : int
    %   Number of VBEM restarts to perform for each trace
    %
    %
    % Variable Inputs
    % ---------------
    %
    % 'work_dir' : string 
    %   Working directory for algorithm
    %
    % 'load_fret' : struct
    %   Any options to pass to load_fret function
    %
    % 'hmi' : struct
    %   Any options to pass to hmi algorithm
    %
    % 'vbem' : struct
    %   Any options to pass to vbem algorithm
    %
    % 'num_cpu' : int (default: 1)
    %   Number of cpu's to use
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
    ip.addParamValue('hmi', struct(), @isstruct);
    ip.addParamValue('vbem', struct(), @isstruct);
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
    opts_hmi = hmi_defaults();
    fnames = fieldnames(opts_hmi);
    for f = 1:length(fnames)
        if ~isfield(opts.hmi, fnames{f})
            opts.hmi.(fnames{f}) = opts_hmi.(fnames{f});
        end
    end
    opts_vbem = vbem_defaults();
    fnames = fieldnames(opts_vbem);
    for f = 1:length(fnames)
        if ~isfield(opts.vbem, fnames{f})
            opts.vbem.(fnames{f}) = opts_vbem.(fnames{f});
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
        data = load_fret(data_files, opts.load_fret);

        % generate set of initial guesses for hyperparameters
        for k = 1:length(opts.K_values)
            K = opts.K_values(k);
            if K > 1
                % set range of minimum separations for FRET levels of states
                % low minimum separation means high randomness
                % high minimum seperation means evenly spread states 
                % on interval [0, 1]
                sep_range = (0.98 * linspace(0, 1, opts.restarts) + 0.02) ...
                            / (opts.K_values(k) - 1);
            end
            for r = 1:opts.restarts
                if K >1
                    u0(k, r) = init_u(K, 'mu_sep', sep_range(r));
                else
                    u0(k, r) = init_u(K);
                end
            end
        end

        % run hmi for every set of hyperparameters
        fret = cat(2, data.fret);
        runs = cell(length(opts.K_values), 1);
        parfor k = 1:length(opts.K_values)
            rn = cell(opts.restarts, 1);
            for r = 1:opts.restarts
                rn{r} = struct();
                [rn{r}.u, rn{r}.L, rn{r}.vb, rn{r}.vit] = ...
                    hmi(fret, u0(k,r), opts.hmi, 'vbem', opts.vbem);
                
                rn{r}.K = opts.K_values(k);
                rn{r}.u0 = u0(k, r);
            end
            runs{k} = rn;
        end
        runs = cat(1, runs{:});
        runs = reshape([runs{:}], [length(opts.K_values), opts.restarts]);

        % keep only best restart for each number of states
        L = arrayfun(@(rn) rn.L(end), runs);
        [mx, idx] = max(L, [], 2);
        runs = runs(:, idx);

        % save results to disk
        save_name = sprintf('%s.mat', opts.save_name);
        save(save_name, 'data', 'opts', 'runs');
    
    catch ME
        % ok something went wrong here, so dump workspace to disk for inspection
        day_time =  datestr(now, 'yymmdd-hh.MM');
        save_name = sprintf('%s-crashdump-%s.mat', opts.save_name, day_time);
        save(save_name);

        % close matlabpool if necessary
        if opts.num_cpu > 1
            matlabpool('CLOSE');
        end

        throw(ME);
    end