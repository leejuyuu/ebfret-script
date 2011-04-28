function [index, d] = photobleach_index(signal, sigma, threshold)
% Finds the point in a FRET donor/acceptor signal where 
% photobleaching occurs.
%
% INPUT
%   signal      A 1D donor or acceptor signal 
%   sigma       Optional width (std dev) of gaussian kernel 
%               for smoothing of detection signal (default = 1)
%   threshold   Return 0 if no peak found above a certain threshold
% 
% OUTPUT 
%   index       Index of the bleaching point. Returns length(signal)
%               if no bleaching point is found.
%   d           Detection signal 
% 
%  Jan-Willem van de Meent
%  $Revision: 1.10 $  $Date: 2011/04/27$
%  $Revision: 1.00 $  $Date: 2011/04/15$

T = length(signal);
mf = zeros(T, 1);
sf = zeros(T, 1);
mb = zeros(T, 1);
sb = zeros(T, 1);

for t = 1:length(signal)
    % calculate mean and std dev
    % of signal from start to t 
    mf(t) = mean(signal(1:t));
    sf(t) = std(signal(1:t));
    % calculate mean and std dev
    % of signal from t to end
    mb(t) = mean(signal(t:end));
    sb(t) = std(signal(t:end));
end

% create gaussian for time smoothing
if nargin < 2
    sigma = 1;
end

W = round(3 * sigma);
S = exp(-(-W:W).^2 ./ sigma.^2);
S = S / sum(S);

% avoid division by zero in next step
sf(1) = sf(2);
sb(end) = sb(end-1);

% calculate forward and backward deviation
% (convoluted with Gaussian to smoothe signal)
df = conv((mf - signal)./sf, S);
db = conv((signal - mb)./sb, S); 
d = df + db;

% find maximum
[dmax,index] = max(d);

% check against threshold
if nargin < 3
    threshold = 4;
end

if dmax > threshold
    index = index - W;
else
    index = length(signal);
end