function D_kl = kl_shmm(w, u)
    % D_kl = kl_shmm(w, u)
    %
    % Calculates Kullback-Leiber divergence
    %
    %   D_kl(q(theta) || p(theta)) = D_kl(q(dmu, l) || p(dmu, l)) 
    %                                + D_kl(q(A) || p(A)) 
    %                                + D_kl(q(pi) || p(pi))

    % D_KL(q(A) || p(A)) = sum_{k,l} (w.A(k,l) - u.A(k,l)) 
    %                                (psi(w.A(k,l)) - psi(u.A(k,l)))
    D_kl_A = kl_dir(w.A, u.A);

    % Calculate Dkl(q(dmu, L | w) || p(dmu, L | u)) 
    w.mu = w.dmu;
    u.mu = u.dmu;
    D_kl_mu_L = kl_nw(w, u);

    % D_kl(q(theta | w) || p(theta | u))
    D_kl = sum(D_kl_mu_L) + sum(D_kl_A);