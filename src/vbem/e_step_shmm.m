function [E_ln_A, E_ln_px_z] = e_step_shmm(w, x, mu0)
    % [E_ln_A, E_p_x_z] = e_step_hmm(w)
    % 
    % E-step of VBEM algorithm for Stepping HMM inference.

    % Expectation of log transition matrix A under q(A | w.A) 
    % (MJB 3.70, JCK 42)
    %
    % E_ln_A(k, l)  =  psi(w.A(k,l)) - psi(Sum_l w.A(k,l))
    E_ln_A = bsxfun(@minus, psi(w.A), psi(sum(w.A, 2)));

    % Expectation of log precision and log emission probability
    w.mu = w.dmu + mu0;
    E_ln_px_z = e_step_nw(w, x);