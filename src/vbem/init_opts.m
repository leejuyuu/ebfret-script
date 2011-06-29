function vbem_opts = init_opts();
% vb_opts = init_opts(varargs)
%
% Initialize default VBEM algorithm options
%
% Output
% ------
%
% 	vb_opts : struct
%		.maxIter
%			Maximum number of VBEM iterations
% 		.threshold
%			Convergence threshold. VBEM iteration will stop when  
%           relative increase in evidence drops below this value.

vbem_opts.maxIter = 100;
vbem_opts.threshold = 1e-5;

% TODO: no longer sure this is still needed
%vb_opts.displayFig = 0;
%vb_opts.DisplayItersToConverge = 1;
%vb_opts.displayNrg = 0;
%vb_opts.displayIter = 0;
