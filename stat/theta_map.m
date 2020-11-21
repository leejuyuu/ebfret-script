function [theta, sigma] = theta_map(w)
    % [theta, sigma] = theta_map(w)
    % 
    % Returns a MAP estimate of the parameters theta 
    % from a set of variational paremters w
    theta = struct();
    sigma = struct();
    for n = 1:length(w)
        K = length(w(n).pi);
        % calculate expectation values
        theta(n).pi = normalize_old(w(n).pi);
        theta(n).A = normalize_old(w(n).A, 2);
        theta(n).mu = w(n).mu;
        for k = 1:K
            theta(n).Lambda(k, :, :) = w(n).W(k, :, :) .* w(n).nu(k);
        end
        % calculate standard deviations
        sigma(n).pi = sqrt(var_dir(w(n).pi));
        sigma(n).A = sqrt(var_dir(w(n).A));
        sigma(n).mu = sqrt(1 ./ (w(n).beta .* w(n).W .* (w(n).nu - 2)));
        for k = 1:K
            d_W = diag(squeeze(w(n).W(k,:,:)));
            sigma(n).Lambda(k, :, :)  = (w(n).W(k, :, :) + d_W + d_W') .* w(n).nu(k);
        end
    end
end
