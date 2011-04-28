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
% load randMu_traces_25.mat
% load randMu_traces_100.mat
%load randMu_traces_100_cov25
load randMu_traces_100_cov30

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
bestOut=cell(N,K);
outF=-inf*ones(N,K);
best_idx=zeros(N,K);

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
            disp(sprintf('n:%d k:%d i:%d',n,k,i))
            % Initialize gaussians
            % Note: x and mix can be saved at this point andused for future
            % experiments or for troubleshooting. try-catch needed because
            % sometimes the K-means algorithm doesn't initialze and the program
            % crashes otherwise when this happens.
            try
                [mix] = get_gmm_mix(FRET{n},ncentres,D,start_guess,mu_g);
%                [out] = chmmVBEM(FRET{n}, mix, PriorPar, vb_opts);
                [out] = vbFRET_VBEM(FRET{n}, mix, PriorPar, vb_opts);
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
                bestOut{n,k} = out;
                outF(n,k)=out.F(end);
                best_idx(n,k) = i;
            end
            % save data
            i=i+1;
        end
    end
   % save results
   save(save_name);           

end

%%%%%%%%%%%%%%%%%%%%%%%% VBHMM postprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%

% analyze accuracy and save analysis
disp('Analyzing results...')

%% Get best restarts and
z_hat=cell(N,K);
x_hat=cell(N,K);
for n=1:N
    for k=1:K
        [z_hat{n,k} x_hat{n,k}] = chmmViterbi(bestOut{n,k},FRET{n});
    end
end

save(save_name);

% calclate biophysJ 4 probabilities

z_hat_star = cell(N,1);
x_hat_star = cell(N,1);
out_star = cell(N,1);
for n=1:N
    [ig k] = max(outF(n,:));
    z_hat_star{n} = z_hat{n,k};
    x_hat_star{n} = x_hat{n,k};
    out_star{n} = bestOut{n,k};
end


save(save_name);

%% make F by N
z_hat_star = reshape(z_hat_star,size(FRET));
x_hat_star = reshape(x_hat_star,size(FRET));
out_star = reshape(out_star,size(FRET));
[F N] = size(FRET);
sumc_vb = cell(1,F);
sumv_vb = cell(1,F);

for f=1:F
    tru_par.mu = mus1D;
    tru_par.Fvec = Fmtx(f,:);
    tru_par.A = Aobs(f,:);
    tru_par.zPath = zPath(f,:);
    tru_par.xPath = xPath(f,:);
    % check results quality
    [sumc{f} sumv{f}] = bj_analysis(out_star(f,:),z_hat_star(f,:),tru_par);
end

disp('...done w/ analysis') 

save(save_name);  

% make viewable by vbFRET
data = cell(F,N);
labels = cell(F,N);
path = cell(F,N);

for f=1:F
    for n=1:N;
        data{f,n} = [1-FRET{f,n}(:) FRET{f,n}(:)];
        path{f,n} = x_hat_star{f,n};
        labels{f,n} = sprintf('f%02d n%03d',f,n);
    end
end

data = data(:);path=path(:);labels=labels(:);
save(save_name)
