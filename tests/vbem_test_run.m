% datasets
data_sets = {'110801-vbem-testdata-2+2-mu-0.35-0.55-beta-09-W-40-D50k-N0250-T0200-f0.50-r10.mat'}

% file to save results to
save_name = '110801-vbem-output.mat';

% add path for dependencies
addpath(genpath('/Users/janwillem/Research/Columbia/Code/Matlab/jwm/hfret/'))
rmpath(genpath('/Users/janwillem/Research/Columbia/Code/Matlab/jwm/hfret/'))
rmpath(genpath('/proj/jv2403/code/matlab/jonbron/vbFRET_June10'));
addpath(genpath('../'));

% HELPER FUNCS
old_u = @(u) struct('upi', u.pi', ...
                    'ua', u.A, ...
                    'mu', u.mu', ...
                    'beta', u.beta, ...
                    'v', u.nu, ...
                    'W', u.W');

new_u = @(u) struct('pi', u.upi(:)', ...
                    'A', u.ua, ...
                    'mu', u.mu(:)', ...
                    'beta', u.beta(:)', ...
                    'nu', u.v(:)', ...
                    'W', u.W(:)');

out_to_stat = @(o) struct('G', o.Nk(:)', ...
                          'xmean', o.xbar(:)', ...
                          'xvar', permute(o.S, [3 1 2]));

out_to_w = @(o) struct('pi', o.Wpi(:)', ...
                       'A', o.Wa, ...
                       'mu', o.m(:)', ...
                       'beta', o.beta(:)', ...
                       'nu', o.v(:)', ...
                       'W', o.W(:)');

u0_to_w0 = @(u0, T) struct('mu', u0.mu + [0.02; -0.02; -0.02; 0.02], ...
                           'beta', u0.beta + T/4, ...
                           'nu', u0.nu + T/4, ...
                           'W', u0.W + [-10; 10; 10; -10], ...
                           'A', T * u0.A / sum(u0.A(:)), ...
                           'pi', [0.5; 0.5; 0.5; 0.5]);

% LOAD DATA
disp('LOADING TRACES')
disp('')

% load traces and prior parameters from each dataset
FRET = {};
Tmin = 0;
for d = 1:length(data_sets)
    load(data_sets{d}, 'data', 'priors')
    FRETd = {data.FRET};
    % append whilst discarding traces with length <= Tmin
    FRET = {FRET{:} FRETd{cellfun(@(f) length(f)>Tmin, FRETd)}};
    clear dat;
end
N = length(FRET);

% cheat: set initial guess for hyperparameters to real priors
u = priors;
u0 = priors;

% cheat: set first guess for variational parameters from real priors
w0 = u0;

disp('START HFRET LOOP')
disp('')

% run VBEM on each trace with fixed initial guess for w
vb_opts = struct();
vb_opts.maxIter = 100;
vb_opts.threshold = 1e-5;

w0 = cell(N,1);
w = cell(N,1);
L = cell(N,1);
stat = cell(N,1);
x_hat = cell(N,1);
z_hat = cell(N,1);
for n = 1:N
    fprintf('trace n = %d\n', n)
    w0{n} = u0_to_w0(u0, length(FRET{n}));
    for i = 1:20
      %dbstop in vbem at 361
      [w{n}, L{n}, stat{n}] = vbem(FRET{n}, w0{n}, u0, vb_opts);
    end
    [z_hat{n} x_hat{n}] = viterbi(w{n}, FRET{n});
end

% recast from cell to struct array
w0 = [w0{:}];
w = [w{:}];;
stat = [stat{:}];

% save results to disk
save(save_name, 'u', 'w', 'L', 'stat', 'FRET', 'x_hat', 'z_hat', 'w0');


