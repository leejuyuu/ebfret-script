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

% data_name = 'hFRET_K2_010_traces'
% data_name = 'hFRET_K2_025_traces'   
% data_name = 'hFRET_K2_050_traces'
% data_name = 'hFRET_K2_100_traces'
% data_name = 'hFRET_K2_150_traces'
% data_name = 'hFRET_K2_200_traces'
% data_name = 'hFRET_K3_010_traces'
% data_name = 'hFRET_K3_025_traces'
% data_name = 'hFRET_K3_050_traces'
% data_name = 'hFRET_K3_100_traces'
% data_name = 'hFRET_K3_150_traces'
% data_name = 'hFRET_K3_200_traces'
% data_name = 'hFRET_K4_010_traces'
% data_name = 'hFRET_K4_025_traces'
% data_name = 'hFRET_K4_050_traces'
% data_name = 'hFRET_K4_100_traces'
% data_name = 'hFRET_K4_150_traces'
% data_name = 'hFRET_K4_200_traces'

data_name = 'hFRET_K3close_010_traces'
% data_name = 'hFRET_K3close_025_traces'
% data_name = 'hFRET_K3close_050_traces'
% data_name = 'hFRET_K3close_100_traces'
% data_name = 'hFRET_K3close_150_traces'
% data_name = 'hFRET_K3close_200_traces'

file_done = vbFRET_hComp(data_name)
