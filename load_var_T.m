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


% load smallT_traces_10k.mat
%load smallT_traces_20k_c35.mat
%load smallT_traces_20k_c40.mat
load smallT_traces_20k_c40_020110.mat

d_t = clock; 

L
for l=1:L
    tru_par.mu = mus1D;
    tru_par.Fvec = Fcell{l};
    tru_par.A = Aobs{l};
    tru_par.zPath = zPath{l};
    tru_par.xPath = xPath{l};
    save_name = sprintf('%s_L%02d_D%02d%02d%02d',dname,l,d_t(2),d_t(3),d_t(1)-2000)

    hFRET(save_name,length(mus1D),FRET{l},tru_par);
end 
