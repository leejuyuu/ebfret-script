function [E_ln_pi, E_ln_A, E_ln_det_L, E_ln_px_z] = e_step(w, x)
    % [E_ln_pi, E_ln_A, E_p_x_z] = e_step(w)
    % 
    % E-step of VBEM algorithm.

    % get dimensions
    [K D] = size(w.mu);

    % Expectation of log intial state priors pi under q(pi | w.pi) 
    % (MJB 3.69, CB 10.66, JCK 41)
    %
    % E[ln(w.pi(k))]  =  Int d pi  Dir(pi | w.pi) ln(pi)
    %                 =  psi(w.pi(k)) - psi(Sum_l w.pi(l)))
    E_ln_pi = psi(w.pi) - psi(sum(w.pi)); 

    % Expectation of log transition matrix A under q(A | w.A) 
    % (MJB 3.70, JCK 42)
    %
    % E_ln_A(k, l)  =  psi(w.A(k,l)) - psi(Sum_l w.A(k,l))
    E_ln_A = bsxfun(@minus, psi(w.A), psi(sum(w.A, 2)));

    % Expectation of log emission precision |Lambda| 
    % under q(W | w.W) (CB 10.65, JKC 44)
    %
    % E_ln_det_L(k)  =  E[ln(|Lambda(k)|)]
    %                =  ln(|w.W|) + D ln(2) 
    %                   + Sum_d psi((w.nu(k) + 1 - d)/2)
    if D>1
        E_ln_det_L = zeros(K, 1);  
        for k=1:K
          E_ln_det_L(k) = log(det(w.W(k, :, :))) + D * log(2) + ...
                          sum(psi((w.nu(k) + 1 - (1:D)) / 2), 2);
        end
    else
        E_ln_det_L = log(w.W) + D * log(2) + ...
                     sum(psi(0.5 * bsxfun(@minus, w.nu + 1, (1:D))), 2);
    end

    % Expectation of Mahalanobis distance Delta^2 under q(theta | w)
    % (10.64, JKC 44)
    %
    % E_Delta2(t, k) 
    %   = E[(x(t,:) - mu(k,:))' * Lambda * (x(t,:) - mu(l,:))]
    %   = D / w.beta(k) 
    %    + w.nu(k) Sum_de dx(t, d, k) W(d, e) dx(t, e, k)
    if D>1
        % dx(d, t, k) = x(t, d) - mu(k, d)
        dx = bsxfun(@minus, x', reshape(w.mu', [D 1 K]));
        % W(d, e, k) = w.W(k, d, e)
        W = permute(w.W, [2 3 1]);
        % dxWdx(t, k) = Sum_de dx(d,t,k) * W(d, e, k) * dx(e, t, k)
        dxWdx = squeeze(mtimesx(reshape(dx, [1 D T K]), ...
                                mtimesx(reshape(W, [D D 1 K]), ...
                                        reshape(dx, [D 1 T K]))));
        % note, the mtimesx function applies matrix multiplication to
        % the first two dimensions of an N-dim array, while using singleton
        % expansion to the remaining dimensions. 
        % 
        % TODO: make mtimesx usage optional? (needs compile on Linux/MacOS)
    else
        % dx(t, k) = x(t) - mu(k)
        dx = bsxfun(@minus, x, w.mu');
        % dxWdx(t, k) = Sum_de dx(t,k) * W(k) * dx(t, k)
        dxWdx = bsxfun(@times, dx, bsxfun(@times, w.W', dx));
    end
    % E_md(t, k) = D / w.beta(k) + w.nu(k) * dxWdx(t,k)
    E_Delta2 = bsxfun(@plus, (D ./ w.beta)', bsxfun(@times, w.nu', dxWdx));

    % Log expectation of p(x | z, theta) under q(theta | w)
    %
    % E_ln_px_z(t, k)
    %   = log(1 / 2 pi) * (D / 2)
    %     + 0.5 * E[ln(|Lambda(k,:,:)]]
    %     - 0.5 * E[Delta(t,k)^2]
    E_ln_px_z = log(2 * pi) * (-D / 2) ...
                + bsxfun(@minus, 0.5 * E_ln_det_L', ...
                                 0.5 * E_Delta2);
                
