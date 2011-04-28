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


load double_decay_traces_T100_100.mat
load double_decay_traces_T300_100.mat
load double_decay_traces_T500_100.mat

d_t = clock; 

F
for f=1:F
    tru_par.mu = mus1D;
    tru_par.A0 = A0{f}; 
    tru_par.A = Aobs(f,:);
    tru_par.zPath = zPath(f,:);
    tru_par.xPath = xPath(f,:);
    save_name = sprintf('%s_f%02d_D%02d%02d%02d',dname,f,d_t(2),d_t(3),d_t(1)-2000)

    %hFRET_double_decay(save_name,length(mus1D),FRET(f,:),tru_par);
    hFRET_double_decay_new(save_name,length(mus1D),FRET(f,:),tru_par);
end 
