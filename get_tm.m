function [A Ag] = get_tm(mus,xPath)
% this function takes Viterbi paths and returns a transition matrix, A. The
% first and last transition of any given trace are not used in the
% analysis.Ag calculates the transition matrix by assuming the dwell times
% come from a geometric distribution and taking the ML estmate of the
% gemoetric distribution parameter (this method is more accurate, since the
% HMM actually does generate dwell times which are geometrically
% distributed, however, since no one in the field does this it's not a
% helpful comparison. 

K = length(mus);
N = length(xPath);
A = zeros(K);
Ag = zeros(K);
% holds lengths of dwell times for each state
dwells = cell(1,K);
% holds the transition destination
dest = cell(1,K);
% holds cumulative dwell-time histogram info
cum_dwells = cell(1,K);
% hold best fit parameters for exponential decay
fit_par = cell(1,K);

% this will break if any trace has more than 1000 transisions
for k=1:K
    dwells{k} = zeros(1e3,1);
    dest{k} = zeros(1e3,1);
end
count = zeros(1,K);
Avec = zeros(1,K);


for n=1:N;
    xPath{n} = xPath{n}(:);

    % get indices of first time step of each new state
    % xdiff 1 contains the first state transitioned to (i.e. information
    % about which state the trace started in is not used)
    xdiff = find(diff(xPath{n})~=0) + 1;
    % find the closest true hidden state to the inferred hidden state
    [ig kmap] = min(abs(xPath{n}(xdiff)*ones(1,K)-mus(ones(length(xdiff),1),:)),[],2);
    for i=1:(length(xdiff)-1)
        k=kmap(i);
        count(k) = count(k)+1;
        dwells{k}(count(k)) = xdiff(i+1)-xdiff(i);
        dest{k}(count(k)) = kmap(i+1);
    end
end

for k=1:K
    dwells{k}(dwells{k} == 0) = [];
    dest{k}(dest{k} == 0) = [];
    
    M = max(dwells{k});
    cum_dwells{k} = zeros(1,M);
    for m=1:M
        cum_dwells{k}(m) = sum(dwells{k} >= m);
    end
    % f(x) = exp(-k*t), where f(0) = 1
   fx = cum_dwells{k}/cum_dwells{k}(1);
   fit_par{k} = polyfit(0:M-1,log(fx),1);
   % the normalized cumulative histogram can is described by exp(-k*t), but it
   % can also be written as p^t, where p is the probability of remaining in
   % the dwell state at each time step (i.e. the p_k values are the diagonal
   % of the KxK transition matrix). Therefore p = exp(-k), so you can get the
   % diagonal entries for the transition matrix from the k values calculated
   % in fit_par above.
   A(k,k) = exp(fit_par{k}(1));
   Ag(k,k) = 1-1/(mean(dwells{k}));
   
   % for a given row of A, the off diagonal entries should be proportional
   % to the number of transitions to state j (assuming A is indixed with
   % entries Aij) and should sum to 1-Aii.

   for kk=1:K
       Avec(kk) = sum(dest{k}==kk);
   end
%    if Avec(k) ~= 0
%        warning('get_tim is finding self-transitions');
%    end
   Avec = normalise(Avec)*(1-A(k,k));
   A(k,:) = A(k,:) + Avec;
   Avec = normalise(Avec)*(1-Ag(k,k));
   Ag(k,:) = Ag(k,:) + Avec;
end