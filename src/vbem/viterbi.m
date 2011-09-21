function [z_hat x_hat] = viterbi(w, x)

% function z_hat=chmmViterbi(out,x)
%
% function formerly called gmmVbemHmmViterbi
% 
% This program runs the Viterbi algorithm on the trace x using paramters
% from out.
%
% Inputs:
%     pZ0 (1xK) = normalize(out.Wpi) = p(Z(1) = i)
%     A (KxK) = transition matrix. A(i,j) = Pr(Z(t+1)=j | Z(t)=i) =
%       normalize(out.Wa,2)
%     mus (DxK) = out.m = guessed gaussian means
%     W (DxDxK) = out.W = Wishart parameter
%     v (Kx1) = out.v = Wishart parameter 
%
%     covarMtx for state k =  inv(W(:,:,k)) / (v(k)-D-1) = mode of Wishart
%     distribution. See: http://en.wikipedia.org/wiki/Wishart_distribution
%
% Outputs:
%     z_hat (1xT) = vector containin most probable hidden state at time 1:T
%     x_hat (DxT) = vector containing mean of most probable
%       state at time 1:T
%
% Internal variables:
%    omega(TxK): omega(t,k) = max p(x1,...,xn,z1,...zn) taken over
%       (z1,...,zn-1)
%    bestPriorZ (TxK) = bestPriorZ(t,k) = best predecessor state given that
%       we ended up in state k at time t
%
% Likelihood of observations (p(y(t) | Z(t)=i) are calculated from x using
% gauss(mu,variance,X) from netlab. 


% Program based on Kevin Murphy's viterbi_path.m, from Bayes Net Toolbox 
% http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=1341&objectType=FILE  
% and Bishop 13.2.5. Unlike Muprhy's code, but consistient with Bishop, all
% calculations are performed using log(P) rather than probablities.

% In viterbi_path.m Murphy uses the following variable names, several of
% which we have renamed. He makes the following comments in his code:
%
% Inputs:
% prior(i) = Pr(Q(1) = i) = our pZ0
% transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i) = our A
% obslik(i,t) = Pr(y(t) | Q(t)=i). We just calculate this value as needed
%   and never store it in a variable.
%
% Outputs:
% path(t) = q(t), where q1 ... qT is the argmax of the above expression. We
%   call this z_hat.
% delta(j,t) = prob. of the best sequence of length t-1 and then going to
%   state j, and O(1:t). We call this omega to be consistent with Bishop
%   chapter 13.2.5. 
% psi(j,t) = the best predecessor state, given that we ended up in state j
%   at t. We call this bestPriorZ 


% A lot of paths have 0 probablity. Not a problem for the calculation, but
% creates a lot of warning messages.
warning('off','MATLAB:log:logOfZero')

% get dimensions
[K D] = size(w.mu);
T = length(x);

% define gaussian distribution
if D == 1
	gauss = @(x, mu, l) (l/(2*pi))^0.5 * exp(-0.5 * l * (x - mu)^2);
else
	gauss = @(x, mu, L) ...
	   (2*pi).^(-0.5*D) * det(L).^(0.5) ...
	   .* exp(-0.5 .* (x - mu)' * L * (x - mu));
end

% Compute values for timestep 1
% omega(z1) = ln(p(z1)) + ln(p(x1|z1))
% CB 13.69
pZ0 = normalize(w.pi);
omega = zeros(T, K);
for k=1:K
   omega(1, k) = log(pZ0(k)) + log(gauss(x(1, :), w.mu(k, :), w.W(k, :, :) * w.nu(k)));
end

% get most likely posterior transition matrix
A = normalize(w.A, 2);

% stores most likely previous state at each timepoint (dependent on the state)
bestPriorZ = zeros(T, K);

% arbitrary value, since there is no predecessor to t=1
bestPriorZ(1, :) = 0;

% forward pass
% omega(zn)=ln(p(xn|zn))+max{ln(p(zn|zn-1))+omega(zn-)}
% CB 13.68
for t=2:T
    for k=1:K
        [omega(t, k) bestPriorZ(t, k)] = max(log(A(:, k)') + omega(t-1, :));
        omega(t, k) = omega(t, k) + log(gauss(x(t,:), w.mu(k,:), w.W(k,:,:) * w.nu(k)));
    end
end
    
% backward pass
z_hat = zeros(T, 1);
[L z_hat(T)] = max(omega(T,:));
for t=(T-1):-1:1
    z_hat(t) = bestPriorZ(t+1, z_hat(t+1));
end
x_hat = w.mu(z_hat, :);

warning('on','MATLAB:log:logOfZero')
