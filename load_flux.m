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


% load randMu_traces_25.mat
% load randMu_traces_100.mat
% load randMu_traces_100_cov25
load randMu_traces_100_cov30

d_t = clock; 

F
for f=1:F
    tru_par.mu = mus1D;
    tru_par.Fvec = Fmtx(f,:);
    tru_par.A = Aobs(f,:);
    tru_par.zPath = zPath(f,:);
    tru_par.xPath = xPath(f,:);
    save_name = sprintf('%s_f%02d_D%02d%02d%02d',dname,f,d_t(2),d_t(3),d_t(1)-2000)

    hFRET(save_name,length(mus1D),FRET(f,:),tru_par);
end 
