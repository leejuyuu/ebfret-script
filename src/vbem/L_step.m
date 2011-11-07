function L = L_step(w, u, E_ln_det_L, ln_Z)
    % L = L_step(w, u)
    %
    % Calculates lower bound L for evidence 
    %
    %   L = ln(p*(x)) - D_kl(q(theta| w) || p(theta | u))
    %   D_kl(q(theta) || p(theta)) = D_kl(q(mu, l) || p(mu, l)) 
    %                                + D_kl(q(A) || p(A)) 
    %                                + D_kl(q(pi) || p(pi))

    % get dimensions
    [K D] = size(w.mu);

    % D_kl(q(pi) || p(pi)) = sum_l (w.pi(l) - u.pi(l)) 
    %                              (psi(w.pi(l)) - psi(u.pi(l)))
    D_kl_pi = kl_dir(w.pi, u.pi);

    % D_KL(q(A) || p(A)) = sum_{k,l} (w.A(k,l) - u.A(k,l)) 
    %                                (psi(w.A(k,l)) - psi(u.A(k,l)))
    D_kl_A = kl_dir(w.A, u.A);

    % Calculate Dkl(q(mu, L | w) || p(mu, L | u)) 

    % pre-compute some terms so calculation can be vectorized
    % (this was done after profiling)
    if D > 1
        log_det_W_u = zeros(K, 1);
        log_det_W_w = zeros(K, 1);
        E_Tr_Winv_L = zeros(K, 1);
        dmWdm = zeros(K, 1);
        for k = 1:K
            log_det_W_u(k) = log(det(u.W(k, :, :)));
            log_det_W_w(k) = log(det(w.W(k, :, :)));
            E_Tr_Winv_L(k) = trace(inv(u.W(k, :, :) ...
                                       * w.W(k, :, :)));
            dmWdm = (w.mu(k,:) - u.mu(k,:))' ...
                    * w.W(k,:,:) ...
                    * (w.mu(k,:) - u.mu(k,:));
        end
    else
        % for the most common D=1 case we don't need 
        % calls to det, inv, mtimes
        log_det_W_u = log(u.W);
        log_det_W_w = log(w.W);
        E_Tr_Winv_L = w.W ./ u.W;
        dmWdm = w.W .* (w.mu-u.mu).^2;
    end 

    % Log norm const Log[B(W, nu)] for Wishart (CB B.79)
    log_B = @(log_det_W, nu) ...
            - (nu / 2) .* log_det_W ...
            - (nu * D / 2) * log(2) ...
            - (D * (D-1) / 4) * log(pi) ...
            - sum(gammaln(0.5 * bsxfun(@minus, nu + 1, (1:D))), 2);

    % E_q[q(mu, L | w)]
    % =
    % 1/2 E_q[log |L|]
    % + D log(w.beta / (2*pi)) 
    % - 1/2 D
    % + log(B(w.W, w.nu))
    % + 1/2 (w.nu - D - 1) * E_q[log |L|]
    % - 1/2 w.nu D
    E_log_NW_w = 0.5 * E_ln_det_L ...        
                 + 0.5 * D * log(w.beta ./ (2*pi)) ...
                 - 0.5 * D ...
                 + log_B(log_det_W_w, w.nu) ...
                 + 0.5 * (w.nu-D-1) .* E_ln_det_L ...
                 - 0.5 * w.nu * D;

    % E_q[log[Norm(mu | u.mu, u.beta L)]
    % =
    %   1/2 D log(u.beta / 2 pi) 
    %   + 1/2 E_q[log |L|]
    %   - 1/2 u.beta (D / w.beta + E_q[(mu-u.mu)^T L (mu-u.mu)])  
    E_log_Norm_u = 0.5 * (D * log(u.beta / (2*pi)) ...
                          + E_ln_det_L ... 
                          - D * u.beta ./ w.beta ...
                          - u.beta .* w.nu .* dmWdm);

    % E_q[log[Wish(L | u.W, u.nu)]
    % =
    % log(B(u.W, u.nu)) 
    % + 1/2 (u.nu - D - 1) E_q[log |L|]
    % - 1/2 w.nu Tr[Inv(u.W) * w.W]
    E_log_Wish_u = log_B(log_det_W_u, u.nu) ...
                   + 0.5 * (u.nu - D - 1) .* E_ln_det_L ...
                   - 0.5 * w.nu .* E_Tr_Winv_L;

    % E_q[p(mu, L | u)]
    % = 
    % E_q[log[Norm(mu | u.mu, u.beta L)]
    % + E_q[log[Wish(L | u.W, u.nu)]
    E_log_NW_u = E_log_Norm_u + E_log_Wish_u;

    % Dkl(q(mu, L | w) || p(mu, L | u)) 
    % =
    % E_q[log q(mu, L | w)] - E_q[log p(mu, L | u)]
    D_kl_mu_L = E_log_NW_w - E_log_NW_u;    

    % L = ln(Z) - D_kl(q(theta | w) || p(theta | u))
    L = ln_Z - sum(D_kl_mu_L) - sum(D_kl_A) - sum(D_kl_pi);