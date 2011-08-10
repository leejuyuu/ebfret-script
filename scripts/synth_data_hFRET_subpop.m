function data = synth_data_hFRET_subpop(u, N, T, varargin)
% Generates a synthetic set of FRET traces from a set of priors
%
% Inputs
% ------
%
% u (struct)
%   .A (KxK)
%       Dirichlet prior for each row of transition matrix
%   .pi (Kx1)
%       Dirichlet prior for initial state probabilities
%   .mu (Kx1)
%       Gaussian-Gamma/Wishart prior for state means 
%   .beta (Kx1)
%       Gaussian-Gamma/Wishart prior for state occupation count 
%   .W (Kx1x1)
%       Gaussian-Gamma/Wishart prior for state precisions
%   .nu (Kx1)
%       Gaussian-Gamma/Wishart prior for degrees of freedom
%
% N
% 	Number of traces to generate
%
% T
%	(Average) number of timepoints per trace
%
%
% Variable Inputs
% ---------------
%
% 'ExpLength' (boolean)
%	Use an exponential distribution for the trace lengths, with 
%   average length T.
%
% Outputs
% -------
%
% data (Nx1 struct)
%   .FRET (Tx1)
%		Synthetically generated FRET trace
%	.z (Tx1)
%		State index at each time point for each trace
%	.x (Tx1)
%		State FRET level at each time point for each trace
% 	.theta (struct)
%		.A (KxK)
%			Transition matrix
%		.pi (1xK)
%			Initial probability for states
%		.m (1xK)
%			Emission FRET levels of states
%		.lambda (1xK)
%			Emission precision (1/variance) for states
%
%    
% Jan-Willem van de Meent
% $Revision: 1.2$  $Date: 2011/08/10$

% parse variable arguments
ExpLength = false;

for i = 1:length(varargin)
    if isstr(varargin{i})
        switch lower(varargin{i})
        case {'explength'}
            ExpLength = varargin{i+1};
        end
    end
end 

% generate trace lengths
if ExpLength
	Tn = ceil(exprnd(T, [N 1]));
else
	Tn = ones([N 1]) .* T;
end

% loop over traces
K = length(u.mu);
for n = 1:N
	% sample parameters from u
	theta{n} = struct('A', zeros(K,K), ...
	                  'pi', zeros(1,K), ...
	                  'm', zeros(1,K), ...
	                  'sigma', zeros(1,K));
	% initial probabilities 
	theta{n}.pi = dirrnd(u.pi');
	% loop over states
	for k = 1:K
		%disp(sprintf('[debug] n: %d, k: %d', n, k))
		% transition matrix row k 
		theta{n}.A(k, :) = dirrnd(u.A(k, :));
		% emission model std dev
		theta{n}.sigma(k) = 1 ./ sqrt(wishrnd(u.W(k), u.nu(k)));
		% emission model state mean
		theta{n}.m(k) = u.mu(k) + randn() ...
		                .* (theta{n}.sigma(k) ./ sqrt(u.beta(k)));
	end

	% generate trace states 
	% (note: z{n}(t) is a K element vector with elements 
	% delta(i, k) if in state i at time t) 
	z{n} = zeros(Tn(n), K);
	% initial state (sample from multinomial)
	z{n}(1, :) = mnrnd(1, theta{n}.pi);
	% loop over time points
	% todo: could vectorise this but prob not worth it
	for t = 2:Tn(n)
		% sample from multinomial using transition matrix 
		% row of prev state
		z{n}(t, :) = mnrnd(1, theta{n}.A(find(z{n}(t-1, :)), :));
	end

	% generate x levels
	x{n} = z{n} * theta{n}.m';

	% generate FRET levels
	FRET{n} = x{n} + ...
			  (randn(size(z{n})) .* z{n}) * theta{n}.sigma';

	% collapse z from state vector to state index
	[z{n}, tidx] = find(z{n}');
end

data = struct('FRET', FRET, ...
              'z', z, ...
              'x', x, ...
              'theta', theta);
