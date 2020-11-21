function runs = eb_fret_pmm_dir(x, w, u, M_values, varargin)
    % eb_fret_pmm_dir(x, w, u, M_values, varargin)
    %
    % Runs EB inference on a set of FRET time series
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
    % 'A_counts' : int (optional)
    %   Normalize priors on the transition matrix for each 
    %   subpopulation to the specified number of counts
    %
    % 'num_cpu' : int (default: 1)
    %   Number of cpu's to use
    %
    % 'eb' : struct
    %   Any options to pass to eb algorithm
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
    ip.addParamValue('A_counts', 0, @isscalar);
    ip.addParamValue('num_cpu', 1, @isscalar);
    ip.addParamValue('eb', struct(), @isstruct);
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
            for m = 1:M_values(r)
                % normalize prior on A if necessary
                if opts.A_counts > 0
                    runs(r).u0(m).A = opts.A_counts * normalize_old(runs(r).u0(m).A);
                end
                % ensure posterior counts for A are correct
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
            fprintf('Running EB for M = %d subpopulations\n', M_values(m))
            
            % run inference on full dataset
            [u, L, vb, vit, phi] = ... 
                eb_hmm(x, runs(m).u0, runs(m).w0, ...
                       opts.eb, 'vbem', opts.vbem);
            runs(m).u = u;
            runs(m).L = L;
            runs(m).vb = vb;
            runs(m).vit = vit;
            runs(m).phi = phi;
        end

     catch ME
         % ok something went wrong here, so dump workspace to disk for inspection
         day_time =  datestr(now, 'yymmdd-HH.MM');
         save_name = sprintf('crashdump-eb_fret_pmm_dir-%s.mat', day_time);
         save(save_name);

         % close matlabpool if necessary
         if opts.num_cpu > 1
             matlabpool('CLOSE');
         end
         rethrow(ME);
     end
