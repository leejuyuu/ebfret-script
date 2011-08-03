function opts = init_opts()
% opts = init_opts()
%
% Initialize default VBEM algorithm options
%
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
%
%
% Jan-Willem van de Meent
% $Revision: 1.00 $  $Date: 2011/08/03$

opts.maxIter = 100;
opts.threshold = 1e-5;