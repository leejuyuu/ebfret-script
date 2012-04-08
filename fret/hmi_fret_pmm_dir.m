function runs = hmi_fret_pmm_dir(x, hf_runs, M_values, K, varargin)
    % hmi_fret_pmm_dir(save_name, x, hf_runs, M_values, K, varargin)
    %
    % Runs HMI inference on a set of FRET time series
    %
    %
    % Inputs
    % ------
    %
    % x : (1xN) cell
    %   Time series to perform inference on.
    %
    % hf_runs : (1xR) struct
    %   Results from hmi_fret first pass
    %
    % M_values : (1xM)
    %   Number of subpopulations to use for each second pass
    %
    % K : int (optional)
    %   Number of states to use for second pass. If unspecified,
    %   the maximum evidence run from the previous pass is used
    %
    %
    % Variable Inputs
    % ---------------
    %
    % 'use_majority_states' : boolean (default: false)
    %   If set to true, the majority_states function will be used,
    %   i.e. the max evidence run with number of states K_max will
    %   collapsed down to a K state solution consisting of the 
    %   most populated states.
    %
    % 'max_outliers' : int (default: 1)
    %   Filter out traces with more than specified number of time
    %   points in outlier states when selecting majority state data.
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
    %
    % Outputs
    % -------
    % 
    % Results are saved to 'save_name.mat'

    % parse input
    ip = inputParser();
    ip.StructExpand = true;
    ip.addRequired('x', @iscell);
    ip.addRequired('hf_runs', @isstruct);
    ip.addRequired('M_values', @isnumeric);
    ip.addOptional('K', 0, @isnumeric);
    ip.addParamValue('use_majority_states', false, @isscalar);
    ip.addParamValue('max_outliers', 1, @isscalar);
    ip.addParamValue('num_cpu', 1, @isscalar);
    ip.addParamValue('hmi', struct(), @isstruct);
    ip.addParamValue('vbem', struct(), @isstruct);
    ip.parse(x, hf_runs, M_values, K, varargin{:})
    opts = ip.Results;

    % get required args
    x = opts.x;
    hf_runs = opts.hf_runs;

    try
        % find run index and number of states with max evidence
        L = arrayfun(@(rn) rn.L(end), hf_runs);
        [m, i_max] = max(L(:));
        K_max = hf_runs(i_max).K;

        % if number of states is not specified, use max evidence value
        if ~opts.K
            opts.K = K_max;
        end

        if (opts.K < K_max) & opts.use_majority_states
            % select K most populated states 
            [vb u idxs] = majority_states([hf_runs(i_max).vb], ...
                                          hf_runs(i_max).u, ...
                                          opts.K, opts.max_outliers);
            x = {x{idxs}};
        else
            % get max evidence hyperparameters with K states from previous pass
            idxs = find(opts.K == [hf_runs.K]);
            [m, j_max] = max(L(idxs));
            i = idxs(j_max);
            w = [hf_runs(i).vb.w];
            u = hf_runs(i).u;
        end

        % use pmm_dir function to initialize guesses for the u.A 
        % hyperparameters of each subpopulation
        for r = 1:length(M_values)
            fprintf('Initializing priors for M = %d subpopulations\n', M_values(r))
            runs(r).u0 = init_u_pmm_dir(M_values(r), w, u, 'display', 'all');
            for m = 1:M_values(r)
                runs(r).w0(:,m) = w;
                for n = 1:length(w)
                    runs(r).w0(n,m).A = runs(r).w0(n,m).A - u.A + runs(r).u0(m).A;
                end
            end
        end

        % run second pass inference
        for m = 1:length(runs)
            fprintf('Running HMI for M = %d subpopulations\n', M_values(m))
            
            % run inference on full dataset
            [u, L, vb, vit, omega] = ... 
                hmi(x, runs(m).u0, runs(m).w0, ...
                    opts.hmi, 'vbem', opts.vbem);
            runs(m).u = u;
            runs(m).L = L;
            runs(m).vb = vb;
            runs(m).vit = vit;
            runs(m).omega = omega;
        end

     catch ME
         % ok something went wrong here, so dump workspace to disk for inspection
         day_time =  datestr(now, 'yymmdd-HH.MM');
         save_name = sprintf('crashdump-hmi_fret_pmm_dir-%s.mat', day_time);
         save(save_name);

         % close matlabpool if necessary
         if opts.num_cpu > 1
             matlabpool('CLOSE');
         end

         throw(ME);
     end
