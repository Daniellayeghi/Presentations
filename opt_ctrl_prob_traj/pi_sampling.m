function c_input =  pi_sampling(params, nwt_args)
   % Regularisations lambdas
   n_reg = @(n_input, n_mean, n_cov)((n_input - n_mean)' * inv(n_cov) * (n_input - n_mean));
   c_reg = @(n_input, cinput, cov)((n_input - cinput)' * inv(cov) * (n_input - cinput));
   p_reg = @(n_input, cov)((n_input' * inv(cov) * n_input));                              

   func = params.func;
   k = params.imp;
   c_input = [params.input];
   csts = zeros(params.samps, 1);   
   n_cov  = inv(nwt_args.hess) * params.hess_reg; 
   cov = params.cov; n_mean = nwt_args.sol;
   
   for it = 1:params.it
       rng(1)
       r_input = normrnd(0,sqrt(cov),[params.samps, 1]);
       for j = 1:params.samps
          % Perturb initial input 
          n_input = c_input(end) + r_input(j);
          
          % Compute regularisation terms for KLs
          regs = k * n_reg(n_input, n_mean, n_cov) - ...
                 c_reg(n_input, c_input(end), cov) + ...
                 (1-k) * p_reg(n_input, cov);
             
          % Sum reg terms with cst func to compute total cost   
          csts(j, 1) = -1/params.lambda * func(n_input).^2 + (-0.5 * regs);
       end
       
       % Compute softmax
       csts = csts;
       min_cost = max(csts);
       exp_cost = exp(csts - min_cost);
       norm_const = sum(exp_cost);
       weights = exp_cost ./ norm_const;
       c_input(end+1) = c_input(end) +(weights' * r_input);
       
      end
end