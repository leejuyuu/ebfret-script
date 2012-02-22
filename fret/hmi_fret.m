function hmi_fret(save_name, x, K_values, restarts, varargin)
    % hmi_fret(save_name, x, K_values, restarts, varargin)
    %
    % Runs HMI inference on a set of FRET time series.
    %
    % Inputs
    % ------
    %
    % save_name : string
    %   File name to save results to (without extension)
    %
    % x : (1xN) cell
    %   Time series to perform inference on.
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
    % 'num_cpu' : int (default: 1)
    %   Number of cpu's to use
    %
    % 'hmi' : struct
    %   Any options to pass to hmi algorithm
    %
    % 'vbem' : struct
    %   Any options to pass to vbem algorithm
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
    ip.addParamValue('num_cpu', 1, @isscalar);
    ip.addParamValue('hmi', struct(), @isstruct);
    ip.addParamValue('vbem', struct(), @isstruct);
    ip.parse(save_name, x, K_values, restarts, varargin{:});
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

    try
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
        runs = cell(length(opts.K_values), 1);
        parfor k = 1:length(opts.K_values)
            rn = cell(opts.restarts, 1);
            for r = 1:opts.restarts
                rn{r} = struct();
                [rn{r}.u, rn{r}.L, rn{r}.vb, rn{r}.vit] = ...
                    hmi(x, u0(k,r), opts.hmi, 'vbem', opts.vbem);
                
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
        day_time =  datestr(now, 'yymmdd-HH.MM');
        save_name = sprintf('%s-crashdump-%s.mat', opts.save_name, day_time);
        save(save_name);

        % close matlabpool if necessary
        if opts.num_cpu > 1
            matlabpool('CLOSE');
        end

        throw(ME);
    end