function bootstrap_res=bootstrap_res(res,nsamp,processfun)
coder.extrinsic("datasample");
    
    n=length(res);
    % if(res{1}.test_dim<=20000)
    % num_parallel=2;
    % else
    % num_parallel=1;
    % end

    grad_per_ess_arr=zeros(res{1}.test_dim,nsamp);
    for(it=1:nsamp)        
        bind=ceil(rand(1,n)*n);
        grad_per_ess_arr(:,it)=processfun(res,bind);
    end
    %grad_per_ess_arrp=zeros(res{1}.test_dim,num_parallel);
    %resc=parallel.pool.Constant(res);
    %processfunc=parallel.pool.Constant(processfun);
    % for(out_it=1:ceil(nsamp/num_parallel))        
    %     parfor in_it=1:min(nsamp-(out_it-1)*num_parallel,num_parallel)
    %         res2=cell(n,1);
    %         bind=ceil(rand(1,n)*n);
    %         for(it=1:n)
    %             res2{it}=res{bind(it)};
    %         end
    %         grad_per_ess_arrp(:,in_it)=processfun(res2);
    %         %grad_per_ess_arrp(:,in_it)=processfun(resc.Value);
    %     end
    %     grad_per_ess_arr(:,((out_it-1)*num_parallel+1):((out_it-1)*num_parallel+min(nsamp-(out_it-1)*num_parallel,num_parallel)) )=grad_per_ess_arrp(:,1:min(nsamp-(out_it-1)*num_parallel,num_parallel));
    % end
    sd=std(grad_per_ess_arr,0,2);
    
    nbeta=res{1}.nbeta;
    max_grad_per_ess=max(grad_per_ess_arr(1:nbeta,:),[],1);
    sdmax=std(max_grad_per_ess);
    bootstrap_res=struct;
    bootstrap_res.grad_per_ess_arr=grad_per_ess_arr;
    bootstrap_res.sd=sd;
    bootstrap_res.max_grad_per_ess=max_grad_per_ess;
    bootstrap_res.sdmax=sdmax;
    bootstrap_res.ci025=quantile(grad_per_ess_arr',0.025);
    bootstrap_res.ci975=quantile(grad_per_ess_arr',0.975);
    bootstrap_res.ci025_max=quantile(max_grad_per_ess,0.025);   
    bootstrap_res.ci975_max=quantile(max_grad_per_ess,0.975);
end