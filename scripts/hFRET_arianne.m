if isunix
[mpath ig ig]=fileparts(mfilename('fullpath'));
vbfretpath=[mpath '../../'];
addpath(genpath('vbfretpath/aux/KPMtools'));
addpath(genpath('vbfretpath/aux/netlab'));
addpath(genpath('vbfretpath/aux/vbhmm'));
addpath(genpath('vbfretpath/aux/stats'));
addpath(genpath('vbfretpath/aux/vbemgmm'));
addpath(genpath('vbfretpath/'));
addpath(genpath('vbfretpath/dat/'));
addpath(genpath('vbfretpath/src'));
addpath(genpath('../'));
addpath(genpath('../../'));
addpath(genpath('../gui/src/'));
addpath('./src')
end

dname = '01-13-10-PEC12 MetY-2'

load(dname)
d_t = clock; 
save_name = sprintf('%s_out_D%02d%02d%02d',dname,d_t(2),d_t(3),d_t(1)-2000)

K=5;
%%
H=5;

vb_opts = get_hFRET_vbopts();


% VBHMM preprocessing
% get number of traces in data
N = length(FRET);   
out0 = cell(K,H,N);
LP0 = -inf*ones(K,H,N);
priors = cell(K,H);

for k=1:K
    for h=1:H
        % make hyperparameters for the trial
        priors{k,h} = get_priors_arianne(k,h);
                
        for n=1:N
            if size(FRET{n},1) > 1
                FRET{n} = FRET{n}';
            end
                disp(sprintf('k:%d h:%d n:%d',k,h,n))
                out = VBEM_eb(FRET{n}, priors{k,h}, priors{k,h},vb_opts);
                % not efficiently coded, but fine for now
                if out.F(end) >  LP0(k,h,n)
                    out0{k,h,n} = out;
                    LP0(k,h,n) = out0{k,h,n}.F(end);
                end        
        end
    end
end
save(save_name)

%%%%%%%%%%%%%%%%%%%%%%%% VBHMM postprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
R = 10;
I = 20;
out = cell(R,K);
LP = cell(R,K);
z_hat = cell(R,K);
x_hat = cell(R,K);
u = cell(R,K);
theta = cell(R,K);
Hbest = zeros(1,K);

% sum evidence over traces
% note: could normalize evidence by trace length - one reason to not do
% this is that longer traces probably have more accurate inference and,
% consequently should be weighted more heavily
LPkh = sum(LP0,3);


% find hyperparameters which maximized evidence
% and consolidate posteriors from best prior
%%
for k=1:K
    [ig Hbest(k)] = max(LPkh(k,:));
    out{1,k} = squeeze(out0(k,Hbest(k),:))';
    % compute posterior hyperparmaters and most probable posterior parameters
    [u{1,k} theta{1,k}] = get_ML_par(out{1,k},priors{k,Hbest(k)});
    z_hat{1,k} = cell(1,N); x_hat{1,k} = cell(1,N);
    for n = 1:N
        [z_hat{1,k}{n} x_hat{1,k}{n}] = chmmViterbi_eb(out{1,k}{n},FRET{n});
    end
end
%%
save(save_name)

for r = 2:R
    for k = 1:K
        out{r,k} = cell(1,N); LP{r,k} = -inf*ones(1,N);
        z_hat{r,k} = cell(1,N); x_hat{r,k} = cell(1,N);
        
        for n=1:N
                disp(sprintf('r: %d k: %d n:%d',r,k,n))
            for i = 1:I
                initM = get_M0(u{r-1,k},length(FRET{n}));
                temp_out = VBEM_eb(FRET{n}, initM, u{r-1,k},vb_opts);
                % Only save the iterations with the best out.F
                if temp_out.F(end) > LP{r,k}(n)
                    LP{r,k}(n) = temp_out.F(end);
                    out{r,k}{n} = temp_out;
                end
            end 
        end

        for n = 1:N
            [z_hat{r,k}{n} x_hat{r,k}{n}] = chmmViterbi_eb(out{r,k}{n},FRET{n});
        end
        % compute posterior hyperparmaters and most probable posterior parameters
        [u{r,k} theta{r,k}] = get_ML_par(out{r,k},u{r-1,k});

        save(save_name)
    end
end 
