function data = synth_data_fret(u, N, T, varargin)
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
% 'exp_length' (boolean)
%	Use an exponential distribution for the trace lengths, with 
%   average length T.
%
% Outputs
% -------
%
% data (Nx1 struct)
%   .fret (Tx1)
%		Synthetically generated FRET trace
%	.z (Tx1)
%		State index at each time point for each trace
%	.x (Tx1)
%		State FRET level at each time point for each trace
% 	.theta (struct)
%		.A (KxK)
%			Transition matrix
%		.pi (Kx1)
%			Initial probability for states
%		.mu (Kx1)
%			Emission FRET levels of states
%		.Lambda (Kx1)
%			Emission precision (1/variance) for states
%
%    
% Jan-Willem van de Meent
% $Revision: 1.2$  $Date: 2011/08/10$

% parse inputs
ip = inputParser();
ip.StructExpand = true;
ip.addRequired('u', @isstruct);
ip.addRequired('N', @isscalar);
ip.addRequired('T', @isscalar);
ip.addParamValue('exp_length', false, @isscalar);
ip.parse(u, N, T, varargin{:});

% collect inputs
args = ip.Results;
u = args.u;
N = args.N;
T = args.T;

% generate trace lengths
if args.exp_length
	Tn = ceil(exprnd(T, [N 1]));
else
	Tn = ones([N 1]) .* T;
end

% loop over traces
K = length(u.mu);
for n = 1:N
	% sample parameters from u
	theta{n} = struct('A', zeros(K,K), ...
	                  'pi', zeros(K,1), ...
	                  'mu', zeros(K,1), ...
	                  'Lambda', zeros(K,1));
	% initial probabilities 
	theta{n}.pi = dirrnd(u.pi');
	% loop over states
	for k = 1:K
		%disp(sprintf('[debug] n: %d, k: %d', n, k))
		% transition matrix row k 
		theta{n}.A(k, :) = dirrnd(u.A(k, :));
		% emission model std dev
		theta{n}.Lambda(k) = wishrnd(u.W(k), u.nu(k));
		% emission model state mean
		theta{n}.mu(k) = u.mu(k) + randn() ...
		                ./ sqrt(theta{n}.Lambda(k) * u.beta(k));
	end

	% generate trace states 
	% (note: z{n}(t) has the form of an indicator variable
	%  i.e. if z(t) in state k, then z(t,l) = delta(k, l))
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

	% collapse z from state vector to state index
	z{n} = sum(bsxfun(@times, z{n}, 1:K),2);

	% generate x levels
	x{n} = theta{n}.mu(z{n});

	% generate FRET levels
	fret{n} = x{n} + randn(size(z{n})) ./ sqrt(theta{n}.Lambda(z{n}));
end

data = struct('fret', fret, ...
              'z', z, ...
              'x', x, ...
              'theta', theta);
