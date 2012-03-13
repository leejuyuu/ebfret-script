function [data, raw] = load_fret(data_files, varargin)
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
%   Datasets may have one of two possible formats.
%
%   dat : (Tx2N numeric)
%       dat is formatted as a matrix with donor signals on odd columns
%       and acceptor signals on even colums
%
%   dat : (Nx1 cell)
%       Each trace dat{n} is a T(n)x2 matrix with donor on the first
%       and acceptor on the second column.
%
% Variable Inputs
% ---------------
%
% 'variable' (string, default:empty)
%   If specified, assume data is saved as a matlab file,
%   with data stored under the specified variable name.
%
% 'has_labels' (boolean, default:false)
%   Assume first row contains trace labels
%
% 'remove_bleaching' (boolean, default:false)
%   Remove photobleaching from traces
%
% 'min_length' (integer, default:0)
%   Minimum length of traces 
%
% 'clip_range' ([min, max], default: [-0.2, 1.2])
%   Defines lower and upper limits of valid range. Points outside
%   this range are considered outliers.
%
% 'max_outliers' (integer, default:inf)
%	Reject trace if it contains more than a certain number of points
%   which are outside of clip_range
%
% 'strip_first' (boolean, default:false)
%   Discard first time point in each trace
%
% 'blacklist' (array of indices, default:[])
%   Array of trace indices to throw out (e.g. because they contain a
%   photoblinking event or other anomaly)   
%
% 'show_progress' (boolean, default:false)
%   Display messages to indicate loading progress 
%   (for large datasets where photobleaching removal takes long time)
%
%
% Outputs
% -------
%
% data (1xD) struct
%   .fret (1xN cell)
%       FRET signals (Tx1 == acceptor / (donor + acceptor)) 
%
%   .donor (1xN cell)
%       Donor fluorophore signals
%
%   .acceptor (1xN cell)
%       Acceptor fluorophore signals
%
%   .labels (1xN int)
%       Index of each trace
%
%   .idxs (1xN int)
%	  Indices of accepted traces
%
% raw (1xD) cell
%   .donor (1xM cell)
%       Unprocessed donor fluorophore signals
%
%   .acceptor (1xM cell)
%       Unprocessed acceptor fluorophore signals
%
%
% Jan-Willem van de Meent
% $Revision: 1.10 $  $Date: 2012/02/10$

% parse inputs
ip = inputParser();
ip.StructExpand = true;
ip.addRequired('data_files', @(d) iscell(d) | isstr(d));
ip.addParamValue('variable', '', @isstr);
ip.addParamValue('has_labels', true, @isscalar);
ip.addParamValue('remove_bleaching', false, @isscalar);
ip.addParamValue('min_length', 1, @isscalar);
ip.addParamValue('clip_range', [-0.2, 1.2], @isnumeric);
ip.addParamValue('max_outliers', inf, @isscalar);
ip.addParamValue('strip_first', 0, @isscalar);
ip.addParamValue('blacklist', [], @isnumeric);
ip.addParamValue('show_progress', false, @isscalar);
ip.parse(data_files, varargin{:});

% collect inputs
args = ip.Results;
if isstr(args.data_files)
    data_files = {args.data_files};
else
    data_files = args.data_files;
end

for d = 1:length(data_files)
    if args.show_progress
        disp(sprintf('Loading Dataset: %s', data_files{d}));
    end

    % load dataset
    if isempty(args.variable)
        dat = load(data_files{d});
    else
        dat = load(data_files{d}, args.variable);
        dat = dat.(args.variable);
    end

    % check if data is in matrix format
    if isnumeric(dat)
        % convert data into cell array
        raw_data = mat2cell(dat, size(dat,1), 2 * ones(size(dat,2) / 2, 1));
    else
        % assume a cell array with dat{n} = [donor, acceptor]
        raw_data = dat;
    end

    for n = 1:length(raw_data)
        % get trace
        rw = raw_data{n};

        % strip labels if necessary
        if args.has_labels
            labels{n} = rw(1, 1);
            rw = rw(2:end, :);
        else
            labels{n} = n;
        end

        % strip first point (often bad data)
        if args.strip_first
            rw = rw(2:end, :);
        end

        % store trace again
        raw_data{n} = rw;
    end

    % store raw data in output
    raw(d).donor = cellfun(@(rd) rd(:,1), raw_data, 'UniformOutput', false);
    raw(d).acceptor = cellfun(@(rd) rd(:,2), raw_data, 'UniformOutput', false);

    % mask out bad traces
    mask = ones(length(raw_data),1);
    mask(args.blacklist) = 0;

    % construct FRET signal and remove photobleaching
    fret = {};
    don = {};
    acc = {};
    idxs = [];
    for n = 1:length(raw_data)
        if mask(n)
            if args.show_progress
                disp(sprintf('   processing trace: %d', n));
            end

            % every 1st column is assumed to contain donor signal,
            % whereas every 2nd column is assumed to contain acceptor
            dn = raw_data{n}(:, 1);
            ac = raw_data{n}(:, 2);
            fr = ac ./ (dn + ac);
   
   			% clip outlier points 
            fr(fr<args.clip_range(1)) = args.clip_range(1);
            fr(fr>args.clip_range(2)) = args.clip_range(2);

            if args.remove_bleaching
                % find photobleaching point in donor and acceptor
                id = photobleach_index(dn);
                ia = photobleach_index(ac);
            else
                id = length(dn);
                ia = length(ac);
            end
            
            % sanity check: donor bleaching should result in acceptor bleaching
            % but we'll allow a few time points tolerance
			tol = 5;
            if (ia < (id + tol)) & (min(id,ia) >= args.min_length)
				rng = 1:min(id,ia);
				outliers = sum((fr(rng)<=args.clip_range(1)) | (fr(rng)>=args.clip_range(2)));
				if (outliers <= args.max_outliers)
                	% keep stripped signal
                	fret{end+1} = fr(1:min(id,ia));
                	don{end+1} = dn(1:min(id,ia));
                    acc{end+1} = ac(1:min(id,ia));
					idxs(end+1) = n;
			    elseif args.show_progress
                	disp(sprintf('   rejecting trace (too many outlier points): %d', n));
				end
            end
        else
            if args.show_progress
                disp(sprintf('   skipping trace: %d', n));
            end
        end
    end

    data(d).fret = {fret{:}};
    data(d).donor = {don{:}};
    data(d).acceptor = {acc{:}};
    data(d).idxs = [idxs(:)];
    data(d).labels = {labels{idxs}};
    data(d).file_name = data_files{d};
end
