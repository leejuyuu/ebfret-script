function w = m_step_shmm(u, x, g, xi, mu0)
    % w = m_step_shmm(u, x, g, xi, mu0)
    %
    % M-step: updates parameters w for q(theta | w)
    %
    % CB 10.60-10.63 and MJB 3.54 (JKC 25), 3.56 (JKC 21). 
    %
    % TODO: untested for D>1

    % get dimensions
    K = length(mu0);
    [T D] = size(x);

    % initialize output
    w = struct();

    % Update for A
    %
    % w.A(k, l) = u.A(k, l) + sum_t xi(t, k, l)
    w.A = u.A + xi;

    % g0(t,k) = g(t,k) / G(k)
    g0 = bsxfun(@rdivide, g, T);

    % dx(t,k,d) = x(t,d) - mu0(k,d)
    dx = bsxfun(@minus, reshape(x,[T 1 D]), mu0(:)');

    % E_dx(d) = sum_tk g0(t,k) dx0(t,k,d)
    E_dx = reshape(sum(sum(bsxfun(@times, g0, dx),2),1), [D 1]);

    % E_dx2(d,e) = sum_tk g0(t,k) dx0(t,k,d) dx0(t,k,e)
    dx_ = permute(dx, [3 1 2]);
    E_dx2 = ...
        sum(sum(bsxfun(@times, ...
                       reshape(g0, [1 1 T K]), ...
                       bsxfun(@times, ...
                              reshape(dx_, [D 1 T K]), ...
                              reshape(dx_, [1 D T K]))), 3), 4);

    % V_dx(d,e) = E_dx2(d,e) - E_dx(d) E_dx(e)
    V_dx = E_dx2 - bsxfun(@times, E_dx, E_dx');

    % xvar0(d,e) = (E_dx(d) - u.dmu(d)) (E_dx(e) - u.dmu(e))
    E_dx0 = E_dx - u.dmu;
    V_dx0 = bsxfun(@times, E_dx0, E_dx0');

    % fprintf('[debug] E_dx : %07.2f\n', E_dx);
    % fprintf('[debug] V_dx : %07.2f\n', V_dx);
    % fprintf('[debug] V_dx0: %07.2f\n', V_dx0);

    % update for beta: add counts in time series
    w.beta = u.beta + T;
    
    % update for w.mu, weighted average of u.mu and u.mu + dx
    w.dmu = u.dmu + T .* E_dx0 ./ w.beta;

    % update for nu: add counts in time series
    w.nu = u.nu + T;

    % TODO: check math
    if D>1
         w.W = inv(inv(u.W(1,:,:)) ... 
                  + T * reshape(V_dx, [1 D D])...
                  + u.beta * T / w.beta * reshape(V_dx0, [1 D D]));
    else
        w.W = 1 ./ (1 ./ u.W + T * V_dx  ...
                    + u.beta * T ./ w.beta .* V_dx0);
    end

    % ensure output w has same field order as input u
    w = orderfields(w, fields(u));