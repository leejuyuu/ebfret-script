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


load g2long_traces_d120408.mat
%load g3long_traces_d120308.mat
%load gauss2_traces_d120108.mat
%load gauss3_traces_d110268.mat
%load gauss4_traces_d110278.mat

d_t = clock; 

S
for s=1:S
    FRET = x(s,:);  
    N = length(FRET)
    tru_par.Fvec = Fcell{l};
    tru_par.mu = mus1D;
    tru_par.A = Aobs(s,1:N);
    tru_par.zPath = zPath(s,1:N);
    tru_par.xPath = cell(1,length(tru_par.zPath));
    for q=1:N
        tru_par.xPath{q} = tru_par.mu(tru_par.zPath{q});
        FRET{q} = FRET{q}';
    end 
    save_name = sprintf('%s_s%02d_r0_D%02d%02d%02d',dname,s,d_t(2),d_t(3),d_t(1)-2000)

    hFRET(save_name,length(mus1D),FRET,tru_par);
end 
