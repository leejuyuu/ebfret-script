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

% datasets
% format: [2*N T], donor signals: (:,1:2:end), acceptor signals: (:,2:2:end)
data = {'raw_traces.dat'}

% file to save results to
save_name = './hfret.mat';

% add paths for dependencies
addpath(genpath('/proj/jv2403/code/matlab/jwm'));
addpath(genpath('/proj/jv2403/code/matlab/jonbron/vbFRET_June10'));


% LOAD DATA
disp('LOADING TRACES AND REMOVING PHOTOBLEACHING')
disp('')

FRET = {};
labels = {};
for d = 1:length(data)
    dat = load(data{d});
    raw = mat2cell(dat, size(dat,1), 2 * ones(size(dat,2) / 2, 1));

    % mask out bad traces
    mask = ones(length(raw),1);
    mask(bad_traces) = 0;

    % construct FRET signal and remove photobleaching
    FRETd = cell(1, length(raw)); 
    for n = 1:length(raw)
        if mask(n)
            disp(sprintf('d: %d  n: %d', d,n))

            don = raw{n}(2:end, 1);
            acc = raw{n}(2:end, 2);
            fret = acc ./ (don + acc);
            fret(fret<0) = 0;
            fret(fret>1.2) = 1.2;

            id = photobleach_index(don);
            ia = photobleach_index(acc);
            
            % sanity check: donor bleaching should result in acceptor bleaching
            % but we'll allow a few time points tolerance
            if (ia < (id + 5)) & (min(id,ia) >= Tmin)
                FRETd{n} = fret(1:min(id,ia));
            end
        end
    end

    FRET = {FRET{:} FRETd{~cellfun(@isempty, FRETd)}};
    clear dat;
end
N = length(FRET)

% transpose data (otherwise VBEM does not run)
for n = 1:N
    if size(FRET{n}, 1) > 1
        FRET{n} = FRET{n}';
    end
end

vb_opts = get_hFRET_vbopts();

disp('RUNNING VBFRET TO DETERMINE GUESSES FOR HYPER PARAMETERS')
disp('')

% VBHMM preprocessing
% get number of traces in data
out0 = cell(K,H,N);
LP0 = -inf*ones(K,H,N);
priors = cell(K,H);

for k=1:K
    for h=1:H
        % make hyperparameters for the trial
        priors{k,h} = get_priors(k,h);
           
        for n=1:N
                disp(sprintf('k:%d h:%d n:%d',k,h,n))
                out = VBEM_eb(FRET{n}, priors{k,h}, priors{k,h}, vb_opts);
                % not efficiently coded, but fine for now
                if out.F(end) >  LP0(k,h,n)
                    out0{k,h,n} = out;
                    LP0(k,h,n) = out0{k,h,n}.F(end);
                end        
        end
    end
end

save(save_name)




disp('CALCULATE INITIAL GUESS FOR HYPERPARAMETERS')
disp('')

% sum evidence over traces
% note: could normalize evidence by trace length - one reason to not do
% this is that longer traces probably have more accurate inference and,
% consequently should be weighted more heavily
LPkh = sum(LP0,3);


% CONSTRUCT INITIAL GUESSES FOR HYPERPARAMETERS FROM VBEM OUTPUT

% get hyperparameter blocks for all possible subpopulation sizes
u0k = cell(K,1);
out0k = cell(K,1);
hbest = zeros(K,1);

for k=1:K
    % compute posterior hyperparmaters and most probable posterior parameters
    [ig hbest(k)] = max(LPkh(k,:));
    out0k{k} = squeeze(out0(k,hbest(k), :))';
    [u0k{k} theta0k] = get_ML_par(out0k{k}, priors{k,hbest(k)}, squeeze(LP0(k,hbest(k),:))');
end
 %%

% construct hyperparameters for each configuration
S = length(subpops);
u0 = cell(S,1);
for s = 1:S
    us = u0k{subpops{s}(1)};

    % loop over remaining blocks
    for b = 2:length(subpops{s})
        % get best parameters for given subblock size
        ub = u0k{subpops{s}(b)};

        % extend us with ub 
        us.W = [us.W ub.W];
        us.v = [us.v; ub.v];
        us.beta = [us.beta; ub.beta];
        us.mu = [us.mu ub.mu];

        % upi needs to be normalized
        Ks = length(us.upi);
        Kb = length(ub.upi);
        us.upi = [Ks*us.upi Kb*ub.upi] ./ (Ks+Kb);
        
        % transition matrix
        tua = zeros(Ks + Kb);
        tua(1:size(us.ua,1), 1:size(us.ua,2)) = us.ua * Ks / (Ks+Kb);  
        tua(size(us.ua,1)+1:end, size(us.ua,2)+1:end) = ub.ua * Kb / (Ks+Kb);
        tua(tua==0) = 1e-5;
        us.ua = tua;  
    end
    u0{s} = us;
end

save(save_name)

disp('START HFRET LOOP')
disp('')

out = cell(R,S);
LP = cell(R,S);
z_hat = cell(R,S);
x_hat = cell(R,S);
u = cell(R,S);
theta = cell(R,S);

% transpose data (otherwise chmmViterbi does not run)
for n = 1:N
    if size(FRET{n}, 1) > 1
        FRET{n} = FRET{n}';
    end
end

for r = 1:R
    for s = 1:S
        out{r,s} = cell(1,N); 
        LP{r,s} = -inf*ones(1,N);
        z_hat{r,s} = cell(1,N); 
        x_hat{r,s} = cell(1,N);
        
        for n = 1:N
                disp(sprintf('r: %d s: %d n:%d',r,s,n))
                temp_out = cell(I,1);
                
                for i = 1:I
                    if r == 1
                        initM = get_M0(u0{s}, length(FRET{n}));
                        temp_out{i} = VBEM_eb(FRET{n}, initM, u0{s}, vb_opts);
                    else
                        initM = get_M0(u{r-1,s}, length(FRET{n}));
                        temp_out{i} = VBEM_eb(FRET{n}, initM, u{r-1,s}, vb_opts);
                    end
                end 

                for i = 1:I
                    % Only save the iterations with the best out.F
                    if temp_out{i}.F(end) > LP{r,s}(n)
                        LP{r,s}(n) = temp_out{i}.F(end);
                        out{r,s}{n} = temp_out{i};
                    end
                end 
        end

        for n = 1:N
            [z_hat{r,s}{n} x_hat{r,s}{n}] = chmmViterbi_eb(out{r,s}{n},FRET{n});
        end
        % compute posterior hyperparmaters and most probable posterior parameters
        if r ==1
            [u{r,s} theta{r,s}] = get_ML_par(out{r,s}, u0{s}, LP{r,s});
        else
            [u{r,s} theta{r,s}] = get_ML_par(out{r,s}, u{r-1,s}, LP{r,s});
        end
        save(save_name)
    end
end

%matlabpool close force

