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
end

% load data
% load smallT_traces_10k.mat
%load smallT_traces_20k_c35.mat
%load smallT_traces_20k_c40.mat
%load smallT_traces_20k_c35_020110.mat
load smallT_traces_20k_c40_020110.mat

% make file name
d_t = clock;
save_name = sprintf('%s_vbFRET_D%02d%02d%02d',dname,d_t(2),d_t(3),d_t(1)-2000);

% analyze data in 1D
D = 1;
K = length(mus1D)+2;
N = length(FRET(:));
numrestarts = 20;

% analyzeFRET program settings
PriorPar.upi = 1;
PriorPar.mu = .5*ones(D,1);
PriorPar.beta = 0.25;
PriorPar.W = 50*eye(D);
PriorPar.v = 5.0;
PriorPar.ua = 1.0;
PriorPar.uad = 0;
%PriorPar.Wa_init = true;


% set the vb_opts for VBEM
% stop after vb_opts iterations if program has not yet converged
vb_opts.maxIter = 100;
% question: should this be a function of the size of the data set??
vb_opts.threshold = 1e-5;
% display graphical analysis
vb_opts.displayFig = 0;
% display nrg after each iteration of forward-back
vb_opts.displayNrg = 0;
% display iteration number after each round of forward-back
vb_opts.displayIter = 0;
% display number of steped needed for convergance
vb_opts.DisplayItersToConverge = 0;


% bestMix = cell(N,K);
bestOut=cell(1,L);
outF=cell(1,L);
best_idx=cell(1,L);
for l=1:L
    N=Nvec(l);
    bestOut{l}=cell(N,K);
    outF{l}=-inf*ones(N,K);
    best_idx{l}=zeros(N,K);
end

for l=1:L
    N=Nvec(l);
    for n=1:N
        for k=1:K
            ncentres = k;
            mu_g = (1:ncentres)/(ncentres+1);
            i=1;
            maxLP = -Inf;
            while i<numrestarts+1
                if k==1 && i > 3
                    break
                end
                if i == 1
                    start_guess = 'guess_no_EMinit';
                elseif i == 2 
                    start_guess = 'guess_no_EMinit';
                else
                    start_guess = 'rand';
                end                
                clear mix out;
                disp(sprintf('l:%d n:%d k:%d i:%d',l,n,k,i))
                % Initialize gaussians
                % Note: x and mix can be saved at this point andused for future
                % experiments or for troubleshooting. try-catch needed because
                % sometimes the K-means algorithm doesn't initialze and the program
                % crashes otherwise when this happens.
                try
                    [mix] = get_gmm_mix(FRET{l}{n},ncentres,D,start_guess,mu_g);
    %                [out] = chmmVBEM(FRET{n}, mix, PriorPar, vb_opts);
                    [out] = vbFRET_VBEM(FRET{l}{n}, mix, PriorPar, vb_opts);
                catch
                    disp('There was an error during, repeating restart.');
                    runError=lasterror;
                    disp(runError.message)
                    continue
                end

                % Only save the iterations with the best out.F
                if out.F(end) > maxLP
                    maxLP = out.F(end);
    %                 bestMix{n,k} = mix;
                    bestOut{l}{n,k} = out;
                    outF{l}(n,k)=out.F(end);
                    best_idx{l}(n,k) = i;
                end
                % save data
                i=i+1;
            end
        end
       % save results
       save(save_name);           

    end
end
save(save_name);
%%%%%%%%%%%%%%%%%%%%%%%% VBHMM postprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%

% analyze accuracy and save analysis
disp('Analyzing results...')

%% Get best restarts and

z_hat=cell(1,L);
x_hat=cell(1,L);
for l=1:L
    N=Nvec(l);
    z_hat{l}=cell(N,K);
    x_hat{l}=cell(N,K);
end

for l=1:L
    N=Nvec(l);
    for n=1:N
        for k=1:K
            [z_hat{l}{n,k} x_hat{l}{n,k}] = chmmViterbi(bestOut{l}{n,k},FRET{l}{n});
        end
    end
end
save(save_name);

% calclate biophysJ 4 probabilities

disp('Calculating 4 probs')

z_hat_star = cell(1,L);
x_hat_star = cell(1,L);
out_star = cell(1,L);
for l=1:L
    N=Nvec(l);
    z_hat_star{l} = cell(N,1);
    x_hat_star{l} = cell(N,1);
    out_star{l} = cell(N,1);
end

for l=1:L
    N=Nvec(l);
    for n=1:N
        [ig k] = max(outF{l}(n,:));
        z_hat_star{l}{n} = z_hat{l}{n,k};
        x_hat_star{l}{n} = x_hat{l}{n,k};
        out_star{l}{n} = bestOut{l}{n,k};
    end
end

save(save_name);
%%
sumc_vb = cell(1,L);
sumv_vb = cell(1,L);

for l=1:L
    tru_par.mu = mus1D;
    tru_par.Fvec = Fcell{l};
    tru_par.A = Aobs{l};
    tru_par.zPath = zPath{l};
    tru_par.xPath = xPath{l};
    % check results quality
    [sumc{l} sumv{l}] = bj_analysis(out_star{l},z_hat_star{l},tru_par);
end

disp('...done w/ analysis') 

save(save_name);  


%% make viewable by vbFRET
Ntot = 0;
for l=1:L
    N=Nvec(l);
    Ntot = Ntot + N;
end

data = cell(Ntot,1);
labels = cell(Ntot,1);
path = cell(Ntot,1);
path_tru = cell(Ntot,1);
labels_tru = cell(Ntot,1);
count = 0;
for l=1:L
    N=Nvec(l);
    for n=1:N;
        count = count + 1;
        data{count} = [1-FRET{l}{n}(:) FRET{l}{n}(:)];
        path{count} = x_hat_star{l}{n};
        labels{count} = sprintf('l%02d n%03d',l,n);
        path_tru{count} = xPath{l}{n};
        labels_tru{count} = sprintf('l%02d n%03d true',l,n);
    end
end

save(sprintf('%s_gui',save_name),'data','path','labels')

path = path_tru;
labels = labels_tru;
save(sprintf('%s_gui_tru',save_name),'data','path','labels')



