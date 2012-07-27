function runs = hmi_fret_pmm_dir(x, w, u, M_values, varargin)
    % hmi_fret_pmm_dir(x, w, u, M_values, varargin)
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
    % w : (1xN) struct
    %   VBEM variational parameters from first pass
    %
    % u : struct
    %   Prior parameters from first pass
    %
    % M_values : (1xM)
    %   Number of subpopulations to use for each second pass
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
    ip.addRequired('w', @isstruct);
    ip.addRequired('u', @isstruct);
    ip.addRequired('M_values', @isnumeric);
    ip.addParamValue('num_cpu', 1, @isscalar);
    ip.addParamValue('hmi', struct(), @isstruct);
    ip.addParamValue('vbem', struct(), @isstruct);
    ip.parse(x, w, u, M_values, varargin{:})
    opts = ip.Results;

    % get required args
    x = opts.x;
    w = opts.w;
    u = opts.u;
    M_values = opts.M_values;

    try
        % use pmm_dir function to initialize guesses for the u.A 
        % hyperparameters of each subpopulation
        for r = 1:length(M_values)
            fprintf('Initializing priors for M = %d subpopulations\n', M_values(r))
            % initialize priors from mixture model
            runs(r).u0 = init_u_pmm_dir(M_values(r), w, u, 'display', 'all');
            % set equal prior weight
            [runs(r).u0.omega] = deal(1);
            % ensure posterior counts for A are correct
            for m = 1:M_values(r)
                runs(r).w0(:,m) = w;
                for n = 1:length(w)
                    runs(r).w0(n,m).A = runs(r).w0(n,m).A - u.A + runs(r).u0(m).A;
                end
            end
            % assign posterior weight
            [runs(r).w0.omega] = deal(1./M_values(r));
        end

        % run second pass inference
        for m = 1:length(runs)
            fprintf('Running HMI for M = %d subpopulations\n', M_values(m))
            
            % run inference on full dataset
            [u, L, vb, vit, phi] = ... 
                hmi(x, runs(m).u0, runs(m).w0, ...
                    opts.hmi, 'vbem', opts.vbem);
            runs(m).u = u;
            runs(m).L = L;
            runs(m).vb = vb;
            runs(m).vit = vit;
            runs(m).phi = phi;
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
         rethrow(ME);
     end
