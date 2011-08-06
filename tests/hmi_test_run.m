% datasets
data_sets = {'110801-vbem-testdata-2+2-mu-0.35-0.55-beta-09-W-40-D50k-N0250-T0200-f0.50-r10.mat'};

% file to save results to
save_name = '110804-hmi-output-mm-cheat.mat';

% add path for dependencies
addpath(genpath('/Users/janwillem/Research/Columbia/Code/Matlab/jwm/hfret/'))
rmpath(genpath('/Users/janwillem/Research/Columbia/Code/Matlab/jwm/hfret/'))
rmpath(genpath('/proj/jv2403/code/matlab/jonbron/vbFRET_June10'));
addpath(genpath('../'));

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

FRET = {FRET{1:250}};
N = length(FRET);

% cheat: set initial guess for hyperparameters to real priors
u0 = priors;

% % decrease strength of prior
% u0.mu = u0.mu + 0.2 * (0.5 - rand(4,1));
% u0.beta(:) = 1;
% u0.nu = u0.beta + 1;
% u0.A = 0.001 * u0.A;
 
% run hmi
[u, L, vb, vit] = hmi(FRET, u0, 'hstep', 'ml');

% save results to disk
save(save_name, 'u', 'L', 'vb', 'vit');


