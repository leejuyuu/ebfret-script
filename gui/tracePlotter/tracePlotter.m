function varargout = tracePlotter(FRET, out, x_hat, z_hat, varargin)
% TRACE_PLOTTER Visualise a dataset of single-molecule FRET traces 
% along results ofwith VBEM inference.
%
% Inputs
% ------
%
%   'FRET' (1xN cell)
%       Single-molecule FRET traces
%
%   'out' (1xN cell)
%       VBEM output for each trace
%
%   'x_hat' (1xN cell)
%       Viterbi path FRET level for each timepoint
%
%   'z_hat' (1xN cell)
%       Viterbi path state index for each timepoint
%
%
% Variable Inputs
% ---------------
%   
%   Any variable arguments are passed to tracePlotterGUI.
%
% Jan-Willem van de Meent
% $Revision: 1.00 $  $Date: 2011/05/16$

% Last Modified by GUIDE v2.5 16-May-2011 15:36:13

args = struct('FRET', {FRET{:}}, ...
              'out', {out{:}}, ...
              'x_hat', {x_hat{:}}, ...
              'z_hat', {z_hat{:}});

if nargin > 4
    varargin = {varargin{:}, args};
else
    varargin = {args};
end


if nargout
    [varargout{1:nargout}] = tracePlotterGUI(varargin{:});
else
    tracePlotterGUI(varargin{:});
end


