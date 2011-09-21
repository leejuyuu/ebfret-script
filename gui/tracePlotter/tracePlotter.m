function varargout = tracePlotter(FRET, varargin)
% TRACE_PLOTTER Visualise a dataset of single-molecule FRET traces 
% along results ofwith VBEM inference.
%
% Inputs
% ------
%
%   'FRET' (1xN cell)
%       Single-molecule FRET traces
%
% Variable Inputs
% ---------------
%
%   'raw' (1xN cell)
%		Donor/Acceptor signals
%
%   'vb' (1xN struct, optional)
%       VBEM output for each trace
%
%   'vit' (1xN struct, optional)
%       Viterbi paths
%
%   Any other variable arguments are passed to tracePlotterGUI.
%
% Jan-Willem van de Meent
% $Revision: 1.00 $  $Date: 2011/05/16$

% Last Modified by GUIDE v2.5 16-May-2011 15:36:13

% concatenate args into single struct array

N = length(FRET)
for n = 1:N
	data(n).FRET = FRET{n};
end

% loop over variable arguments
i=1;
while i <= length(varargin)
	matched = false;
    if isstr(varargin{i})
        switch lower(varargin{i})
        case {'raw'}
        	matched = true;
        	raw = varargin{i+1};
        	for n = 1:length(raw)
				data(n).don = raw{n}(:,1);
				data(n).acc = raw{n}(:,2);
			end
            
        case {'vb'}
        	matched = true;
        	vb = varargin{i+1}
        	% add vb fields to data struct
			fields = fieldnames(vb);
			for f = 1:length(fields)
				[data.(fields{f})] = vb.(fields{f})
			end
            
        case {'vit'}
        	matched = true;
        	vit = varargin{i+1}
        	% add vit fields to data struct
			fields = fieldnames(vit);
			for f = 1:length(fields)
				[data.(fields{f})] = vit.(fields{f})
			end
		end
		        
        if matched
        	% strip arguments from varargin 
        	% (so they are not passed to tracePlotterGUI)
        	I = length(varargin);
        	msk = ((1:I)~=i) & ((1:I)~=(i+1));
        	varargin = {varargin{find(msk)}};
        else
        	i = i+1;
        end
    end
end

% append data to last position of varargin cell
varargin = {varargin{:}, data};

if nargout
    [varargout{1:nargout}] = tracePlotterGUI(varargin{:});
else
    tracePlotterGUI(varargin{:});
end


