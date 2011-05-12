function data = synth_data_hFRET_subpop(priors, N, T, varargin)
% Generates a synthetic set of FRET traces from a set of priors
%
% Inputs
% ------
%
% priors (struct)
%   .ua (KxK)
%       Dirichlet prior for each row of transition matrix
%   .upi (1xK)
%       Dirichlet prior for initial state probabilities
%   .mu (1xK)
%       Gaussian-Gamma/Wishart prior for state means 
%   .beta (Kx1)
%       Gaussian-Gamma/Wishart prior for state occupation count 
%   .W (1xK)
%       Gaussian-Gamma/Wishart prior for state precisions
%   .v (Kx1)
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
%	.z_hat (Tx1)
%		State index at each time point for each trace
%	.x_hat (Tx1)
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
% $Revision: 1.00 $  $Date: 2011/05/12$

% parse variable arguments
ExpLength = false;
for i = 1:length(varargin)
    if isstr(varargin{i})
        switch lower(varargin{i})
        case {'ExpLength'}
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
K = length(priors.mu);
for n = 1:N
	% sample parameters from priors
	theta{n} = struct('A', zeros(K,K), ...
	                  'pi', zeros(1,K), ...
	                  'm', zeros(1,K), ...
	                  'sigma', zeros(1,K));
	% initial probabilities 
	theta{n}.pi = dirrnd(priors.upi);
	% loop over states
	for k = 1:K
		%disp(sprintf('[debug] n: %d, k: %d', n, k))
		% transition matrix row k 
		theta{n}.A(k, :) = dirrnd(priors.ua(k, :));
		% emission model std dev
		theta{n}.sigma(k) = 1 ./ sqrt(wishrnd(priors.W(k), priors.v(k)));
		% emission model state mean
		theta{n}.m(k) = priors.mu(k) + randn() ...
		                .* (theta{n}.sigma(k) ./ sqrt(priors.beta(k)));
	end

	% generate trace states 
	% (note: z_hat{n}(t) is a K element vector with elements 
	% delta(i, k) if in state i at time t) 
	z_hat{n} = zeros(Tn(n), K);
	% initial state (sample from multinomial)
	z_hat{n}(1, :) = mnrnd(1, theta{n}.pi);
	% loop over time points
	% todo: could vectorise this but prob not worth it
	for t = 2:Tn(n)
		% sample from multinomial using transition matrix 
		% row of prev state
		z_hat{n}(t, :) = mnrnd(1, theta{n}.A(find(z_hat{n}(t-1, :)), :));
	end

	% generate x_hat levels
	x_hat{n} = z_hat{n} * theta{n}.m';

	% generate FRET levels
	FRET{n} = ones(size(z_hat{n})) * theta{n}.m' + ...
			  randn(size(z_hat{n})) * theta{n}.sigma';

	% collapse z_hat from state vector to state index
	[z_hat{n}, tidx] = find(z_hat{n}');
end

data = struct('FRET', FRET, ...
              'z_hat', z_hat, ...
              'x_hat', x_hat, ...
              'theta', theta);