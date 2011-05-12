function [FRET raw orig] = load_traces(data_files, varargin)
% Loads traces from a number of data files into a single dataset.
%
% Data must be formatted as a matrix with T rows (time points) and
% 2N columns (donor/acceptor signal for each trace). Donor signals
% are assumed to be located on odd columns (:,1:2:end-1), whereas
% acceptor signals are on even columns (:,2:2:end). 
%
%
% Inputs
% ------
%
% data_files (1xD cell)
%   Names of datasets to analyse (e.g. {'file1.dat' 'file2.dat'}). 
%   Datasets are T x 2N, with the donor signal on odd columns
%   and acceptor signal on even columns.
%
%   TODO: Modify this function to deal with other inputs
%   (e.g. vbFRET saved sessions)
%   
%
% Variable Inputs
% ---------------
%
% 'RemoveBleaching' (boolean)
%   Remove photobleaching from traces
%
% 'MinLength' (integer)
%   Minimum length of traces 
%  
% 'BlackList' (array of indices)
%   Array of trace indices to throw out (e.g. because they contain a
%   photoblinking event or other anomaly)   
%
% 'ShowProgress' (boolean)
%   Display messages to indicate loading progress 
%   (for large datasets where photobleaching removal takes long time)
%
%
% Outputs
% -------
%
% FRET (1xN cell)
%   FRET signals (Tx1 == acceptor / (donor + acceptor)) 
% raw (1xN cell)
%   Raw 2D donor/acceptor signals (Tx2)
% orig (1xN cell)
%   Raw 2D signals without removal of photobleaching
%   or blacklisted/short traces
%   (empty if no photobleaching removal has taken place)
%
% Jan-Willem van de Meent
% $Revision: 1.00 $  $Date: 2011/05/04$


% parse variable arguments
RemoveBleaching = false;
MinLength = 0;
BlackList = [];
ShowProgress = false;
for i = 1:length(varargin)
    if isstr(varargin{i})
        switch lower(varargin{i})
        case {'removebleaching'}
            RemoveBleaching = varargin{i+1};
        case {'minlength'}
            MinLength = varargin{i+1};
        case {'blacklist'}
            BlackList = varargin{i+1};
        case {'showprogress'}
            ShowProgress = varargin{i+1};
        end
    end
end 

FRET = {};
raw = {};
orig = {};
labels = {};
for d = 1:length(data_files)
    if ShowProgress
        disp(sprintf('Loading Dataset: %s', data_files{d}));
    end
    dat = load(data_files{d});
    origd = mat2cell(dat, size(dat,1), 2 * ones(size(dat,2) / 2, 1));

    % mask out bad traces
    mask = ones(length(origd),1);
    mask(BlackList) = 0;


    % construct FRET signal and remove photobleaching
    FRETd = cell(1, length(origd));
    rawd = cell(1, length(origd));
    for n = 1:length(origd)
        if mask(n)
            if ShowProgress
                disp(sprintf('   processing trace: %d', n));
            end

            % every 1st column is assumed to contain donor signal,
            % whereas every 2nd column is assumed to contain acceptor
            don = origd{n}(2:end, 1);
            acc = origd{n}(2:end, 2);
            fret = acc ./ (don + acc);
            fret(fret<0) = 0;
            fret(fret>1.2) = 1.2;

            if RemoveBleaching
                % find photobleaching point in donor and acceptor
                id = photobleach_index(don);
                ia = photobleach_index(acc);
            else
                id = length(don);
                ia = length(acc);
            end
            
            % sanity check: donor bleaching should result in acceptor bleaching
            % but we'll allow a few time points tolerance
            if (ia < (id + 5)) & (min(id,ia) >= MinLength)
                % keep stripped signal
                FRETd{n} = fret(1:min(id,ia));
                rawd{n} = [don(1:min(id,ia)) acc(1:min(id,ia))];
            end
        else
            if ShowProgress
                disp(sprintf('   skipping trace: %d', n));
            end
        end
    end

    FRET = {FRET{:} FRETd{~cellfun(@isempty, FRETd)}};
    raw = {raw{:},  rawd{~cellfun(@isempty, rawd)}};
    orig = {orig{:}, origd};
    clear dat;
end