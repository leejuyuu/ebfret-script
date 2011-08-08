function hFRET_subpop(data_files, output_file, varargin)
% Runs a hierarchical inference proces on a collection of single 
% molecule FRET time series to determine the levels of the FRET
% states, as well as the transition rates between the states.
%
% In addition to detecting the best parameters for each trace,
% this method also calculates a set of hyperparameters that
% define the distribution of parameters over all traces. 
%
% In brief, this method does the following:
% 1. Infer the maximum evidence parameters for each trace with 
%    Variational Bayes Expectation Maximization (i.e. vbFRET)
% 2. For each configuration of subpopulations, construct a best
%    initial guess for the hyperparameters using VBEM output
% 3. RunhFRET iterations
%    * Use last hyperparameters to rerun VBEM on each trace
%    * Update hyperparameters using VBEM results
%      (using method of moments)
%
%
% Inputs
% ------
%
% data_files (1xD cell)
%   Names of datasets to analyse (e.g. {'file1.dat' 'file2.dat'}). 
%   Datasets are T x 2N, with the donor signal on odd columns
%   and acceptor signal on even columns.
%   
% output_file (string)
%   Name of file to save output to (e.g. 'output.mat')
%
%
% Variable Inputs
% ---------------
%
% 'subdims' (1xS cell)
%   Specifies number of states in subpopulations that hFRET should
%   try to detect. Should have the form {[K1 .. Kn], [K1 .. Km], ...}
%   where each K specifies the number of states in a subpopulation.
%
% 'RemoveBleaching' (boolean)
%   Remove photobleaching from traces
%
% 'MinLength' (integer)
%   Minimum length of traces 
%  
% 'BlackList' (array of indices)
%   Array of trace indices to throw out (e.g. because they contain a
%   photoblinking event or other anomaly)   
%
%
% Outputs (saved to file)
% -----------------------
%
% theta (struct)
%   Maximum likelihood parameters calculated from the 
%   hyperparameters of best run.
%
%   .A (KxK)
%       Transition matrix. A(i, j) gives probability of switching 
%       from state i to state j at each time point
%   .pi (1xK)
%       Initial probability of each state
%   .mu (1xK)
%       FRET levels for each state
%       (i.e. means for Gaussian emissions model)
%   .sigma (1xK)
%       Noise levels for each FRET state
%       (i.e. std dev for Gaussian emissions model)
%   .Wa (KxK)
%       ?? TODO: CHECK THIS ??
%   .Wan (KxK)
%       Normalised version of Wa
%
% u (struct)
%   Hyperparameters for best run.
%
%   .ua
%       Transition matrix (dirchlet distribution)
%   .upi
%       Initial probabilities for each state (dirchlet)
%   .m
%       Levels for states (Gaussian-Gamma)
%   .beta
%       Observation count (Gaussian-Gamma)
%   .W
%       Precision matrix (Gaussian-Gamma) 
%   .v
%       Observation counts (== beta + 1) (Gaussian-Gamme)
%
% data (struct)
%   Datasets on which inference was performed
%
%   .raw (1xN cell)
%       Raw 2D donor/acceptor signals (empty if not provided)
%   .FRET (1xN cell)
%       FRET signals (== acceptor / (donor + acceptor)) 
%
% hmi (struct)
%   All variables associated with the hierarchical inference
%   process
%
%   .theta (RxS cell)
%       Maximum likelihood parameters for each hFRET iteration
%   .u (RxS cell)
%       Hyperparameters for each hFRET iteration.
%   .LP (RxS cell)
%       Evidence for each hFRET iteration.
%   .out (RxS cell)
%       VBEM output of all traces for each hFRET iteration
%   .subdim (1xS cell)
%       Number of states in subpopulations
%
% vbem (struct)
%   .LP (?x?)
%     Lower bound evidence estimate 
%
%   .out (?x?x? cell)
%     Output of VBEM inference process containing posteriors for the 
%     parameters of each trace.
%
%     .Wa (KxK)
%       Dirichlet posterior for each row of transition matrix
%     .Wpi (1xK)
%       Dirichlet posterior for initial state probabilities
%     .m (1xK)
%       Gaussian-Gamma/Wishart posterior for state means 
%     .beta (Kx1)
%       Gaussian-Gamma/Wishart posterior for state occupation count
%     .W (1xK)
%       Gaussian-Gamma/Wishart posterior for state precisions
%     .v (Kx1)
%       Gaussian-Gamma/Wishart posterior for degrees of freedom
%     .F (1xI)
%       Lower bound evidence for each iteration
%     .Nk (1xK)
%       Sufficient statistics: Number of datapoints in each state
%           Nk  =  Σ p(z_t = k)
%     .xbar (1xK)
%       Sufficient statistics: Estimated means of states 
%           E_k[x_t]  =  Σ p(z_t = k) x_t / Σ p(z_t = k) 
%     .S (1x1xK)
%       Sufficient statistics: Estimated variances of states
%           Var_k[x_t]  =  Σ p(z_t = k) (x_t - xbar_k)^2
%
%   .priors (?x? cell)
%   .ua (KxK)
%       Dirichlet prior for each row of transition matrix
%   .upi (1xK)
%       Dirichlet prior for initial state probabilities
%   .mu (1xK)
%       Gaussian-Gamma/Wishart prior for state means 
%   .beta (Kx1)
%       Gaussian-Gamma/Wishart prior for state occupation count 
%   .W (1xK)
%       Gaussian-Gamma/Wishart prior for state precisions
%   .v (Kx1)
%       Gaussian-Gamma/Wishart prior for degrees of freedom
%
%
% Internal Variables
% ------------------
%
% K
%   Max number of states
% R
%   hFRET iterations
% S
%   Number of subpopulation configurations
% I
%   Number of VBEM restarts for each hFRET iteration
% H
%   Number of VBEM restarts during calculation of initial 
%
% Jan-Willem van de Meent
% $Revision: 0.00 $  $Date: 2011/05/04$


% total number of states (total for subpopulations)
K = 5;

% subpopulation sizes 
subpops = {[5],    % i.e. one population with 5 states 
           [4 1],  % i.e. one 4 state one single state population
           [3 2],  % i.e. one 3 state and one two state population
           [4],    
           [3 1],
           [2 2],
           [3],
           [2 1],
           [2],
           [1,1],
           [1]};

% manually remove some traces from analysis
% (e.g. traces with photoblinking events)
bad_traces = [];

% restarts for determination of initial hyperparameter guess
H = 20;

% iterations for hfret loop
R = 10;

% restarts for hfret loop
I = 20;

% min length of traces
Tmin = 30;

data = {'raw_traces.dat'}

% file to save results to
save_name = './hfret.mat';

% add paths for dependencies
addpath(genpath('/proj/jv2403/code/matlab/jwm'));
addpath(genpath('/proj/jv2403/code/matlab/jonbron/vbFRET_June10'));


disp('LOADING TRACES AND REMOVING PHOTOBLEACHING')
disp('')

[data.FRET ...
 data.raw ...
 data.orig] = load_traces(data, ... 
                          'RemoveBleaching', true, ...
                          'MinLength', 100, ...
                          'BlackList', bad_traces);
save(save_name, 'data', '-append');


disp('RUNNING VBEM INFERENCE TO DETERMINE GUESSES FOR HYPER PARAMETERS')
disp('')

vbem = struct('out', cell(K,H,N), ...
              'LP', -inf*ones(K,H,N), ...
              'priors', cell(K,H,N))

vbem.vb_opts = get_hFRET_vbopts();

% transpose data (otherwise VBEM does not run)
for n = 1:N
    if size(FRET{n}, 1) > 1
        FRET{n} = FRET{n}';
    end
end

for k=1:K
    for h=1:H
        % make hyperparameters for the trial
        vbem.priors{k,h} = get_priors(k,h);
           
        for n=1:N
                disp(sprintf('k:%d h:%d n:%d',k,h,n))
                out = VBEM_eb(data.FRET{n}, ...
                              vbem.priors{k,h}, ... 
                              vbem.priors{k,h}, 
                              vbem.vb_opts);
                % not efficiently coded, but fine for now
                if out.F(end) >  vbem.LP(k,h,n)
                    vbem.out{k,h,n} = out;
                    vbem.LP(k,h,n) = out{k,h,n}.F(end);
                end        
        end
    end
end

save(save_name, 'vbem', '-append');


disp('START HIERARCHICAL INFERENCE PROCES')
disp('')

hmi = struct('theta', {}, ...
             'out', {}, ...
             'LP', {}, ...
             'z_hat', {}, ...
             'x_hat', {});

hmi.u0 = get_u0_subpop(subdims, vbem.out, vbem.priors); 
hmi.vb_opts = vbem.vb_opts

save(save_name, 'hmi', '-append');

% loop over HMI iterations
for r = 1:R
    if r == 1
        u_old = hmi.u0;
    else
        u_old = hmi.u{r-1,:};
    end 
    % loop over subpopulation configurations
    for s = 1:S
        % run HMI iteration
        [hmi.u{r,s}, ...
         hmi.theta{r,s}, ... 
         hmi.out{r,s}, ...
         hmi.LP{r,s}, ...
         hmi.z_hat{r,s}, ...
         hmi.x_hat{r,s}] = HMI(data.FRET, u_old{s}, I, vb_opts);
    end
    % save to disk
    save(save_name, 'hmi', '-append');
end