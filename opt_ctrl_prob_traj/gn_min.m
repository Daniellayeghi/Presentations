function [min_val, gn_hess] = gn_min(params)
    [row, col] = size(params.init);
    delta = 1e12; iter = 1;
    min_val = [params.init];
    gn_hess = [params.init];
    reg = eye(row, col) * 0.1;
    
    while abs(delta) > params.tol && iter < params.it_lim
        gn_hess(iter) = (params.g_func(min_val(iter))' * ...
                        params.g_func(min_val(iter))) * 2 + reg;
        delta = -params.g_func(min_val(iter)) * inv(gn_hess(iter));
        
        if (gn_hess(iter) == 0)
            error("Hessian approximation = 0");
        end
        min_val(iter+1) = min_val(iter) + delta*params.alpha;
        iter = iter + 1;
    end
    fprintf("** MIN %f at conv level: %f **\n", min_val(end), delta)
end