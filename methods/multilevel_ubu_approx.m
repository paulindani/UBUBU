function res=multilevel_ubu_approx(niter,burnin,rep,h,gam, grad_lpost, test_function, options)
    %#codegen    
    inputs=struct;
    inputs.gam=gam;
    inputs.grad_lpost=grad_lpost;
    inputs.test_function=test_function;
    inputs.nbeta=options.nbeta;
    inputs.test_dim=options.test_dim;    
    inputs.Hprodv=options.Hprodv;    
    inputs.invcholHprodv=options.invcholHprodv;    
    inputs.exp_hM=options.exp_hM;
    inputs.beta_min=options.beta_min;
    inputs.repfullgrad=options.repfullgrad;
    r=options.r;
    c=options.c;   
    nbeta=options.nbeta;
    test_dim=options.test_dim;
    maxlevel=options.maxlevel;
    repfullgrad=options.repfullgrad;
    max_parallel_chain=options.max_parallel_chain;    

    repruns=ones(1,maxlevel+1);
    problevels=zeros(1,maxlevel+1);    
    repruns(1)=rep;
    problevels(1)=rep;
    for lev=1:maxlevel
        problevels(lev+1)=rep*c/(r^(lev-1));
        repruns(lev+1)=ceil(problevels(lev+1));
    end
    
    blockstart=[1,cumsum(repruns(1:maxlevel))+1];
    blockend=cumsum(repruns(1:(maxlevel+1)));    
    maxruns=sum(repruns);  
    detlevels=min(sum(problevels>=1)+1,maxlevel);%>=(1/2-1e-8));
    randlevels=maxlevel-detlevels;
    if(detlevels<maxlevel)
    repruns((detlevels+2):(maxlevel+1))=(rand(1,randlevels)<problevels((detlevels+2):(maxlevel+1)));
    end
    ngradtot=0;%(burnin+niter)*rep;
    %ngradtot=niter*rep;
    for lev=1:detlevels
        levm=lev-1;
        extraburnin2=burnin*(2^lev);
        jointburnin=burnin*(2^levm)*lev*(2+(lev>1));
        ngradtot=ngradtot+repruns(lev+1)*(extraburnin2+jointburnin+niter*2^levm*(2+(lev>1)))/repfullgrad;
    end
    
    for lev=(detlevels+1):maxlevel
        levm=lev-1;
        extraburnin2=burnin*(2^lev);
        jointburnin=burnin*(2^levm)*lev*3;
        ngradtot=ngradtot+problevels(lev+1)*(extraburnin2+jointburnin+niter*2^lev*3)/repfullgrad;
    end   
    

    
    res=struct;
    res.rep=rep;
    res.niter=niter;   
    res.detlevels=detlevels;
    res.randlevels=randlevels;
    res.repruns=repruns;    
    res.maxlevel=maxlevel;
    res.problevels=problevels;
    res.ngradtot=ngradtot;
    res.nbeta=nbeta;
    res.means=zeros_sd(test_dim,maxruns);
    res.squaremeans=zeros_sd(test_dim,maxruns);
    res.blockstart=blockstart;
    res.blockend=blockend;
    res.test_dim=test_dim;
    res.maxruns=maxruns;
    res.rho=1/2;
           
    
    no_batches=ceil(rep/max_parallel_chain);
    for repit=1:no_batches
        if(repit<no_batches)
            par_chain=max_parallel_chain;
        else 
            par_chain=rep-max_parallel_chain*(no_batches-1);
        end
        [res.means(1:test_dim,((repit-1)*max_parallel_chain+1):((repit-1)*max_parallel_chain+par_chain)),res.squaremeans(1:test_dim,((repit-1)*max_parallel_chain+1):((repit-1)*max_parallel_chain+par_chain))]=Gaussian_samples(par_chain,niter,inputs);
    end
    
    

    for(lev=1:(detlevels-1))
        no_batches=ceil(repruns(lev+1)/max_parallel_chain);
        for repit=1:no_batches
            if(repit<no_batches)
                par_chain=max_parallel_chain;
            else
                par_chain=repruns(lev+1)-max_parallel_chain*(no_batches-1);
            end
            levm=lev-1;
            extraburnin2=burnin*(2^levm);
            jointburnin=burnin*(2^levm)*lev;
            if(lev==1)
                [res.means(1:test_dim,(blockstart(lev+1)+(repit-1)*max_parallel_chain):(blockstart(lev+1)+(repit-1)*max_parallel_chain+par_chain-1)),res.squaremeans(1:test_dim,(blockstart(lev+1)+(repit-1)*max_parallel_chain):(blockstart(lev+1)+(repit-1)*max_parallel_chain+par_chain-1))]=doubleMCMC_levels01(par_chain,niter*2^levm,h/(2^levm),extraburnin2, jointburnin,(2^levm),inputs);
            else
                [res.means(1:test_dim,(blockstart(lev+1)+(repit-1)*max_parallel_chain):(blockstart(lev+1)+(repit-1)*max_parallel_chain+par_chain-1)),res.squaremeans(1:test_dim,(blockstart(lev+1)+(repit-1)*max_parallel_chain):(blockstart(lev+1)+(repit-1)*max_parallel_chain+par_chain-1))]=doubleMCMC(par_chain,niter*2^levm,h/(2^levm),extraburnin2, jointburnin,(2^levm),inputs);
            end
        end
    end

    coder.varsize("multilevels",[1 1]);
    multilevelsfind=find(repruns((detlevels+2):(maxlevel+1)),1,'last');
    if(isempty(multilevelsfind))
        multilevels=0;
    else
        multilevels=multilevelsfind(1);
    end
    lev=detlevels;levm=lev-1;
    extraburnin2=burnin*(2^levm);
    jointburnin=burnin*(2^levm)*lev;
    [res.means(1:test_dim,blockstart(lev+1):blockstart(lev+1+multilevels)),res.squaremeans(1:test_dim,blockstart(lev+1):blockstart(lev+1+multilevels))]=...        
    multiMCMC(niter*2^levm,h/(2^levm),extraburnin2, jointburnin, (2^levm), multilevels, repruns((detlevels+1):(maxlevel+1)), inputs);
end
 

function z=zeros_sd(a,b)
   z=zeros(a,b,"double");
end
function z=randn_sd(a,b)
   z=randn(a,b,"double");
end
function hc=hper2const(h,gam)
    hc=struct;
    gam=double(gam);
    h=double(h);
    hc.h=h;
    gh=gam*h;
    s=sqrt(4*expm1(-gh/2)-expm1(-gh)+gh);
    hc.eta=double(exp(-gh/2));
    hc.etam1g=double((-expm1(-gh/2))/gam);
    hc.sqrtetam1=sqrt(-expm1(-gh/2));
    hc.c11=double(s/gam);
    hc.c21=double(exp(-gh)*(expm1(gh/2))^2/s);
    hc.c22=double(sqrt(8*expm1(-gh/2)-4*expm1(-gh)-gh*expm1(-gh))/s);
end

function [mx1,mx2]=xi_h_mx(h,gam)
    hc=hper2const(h,gam);
    h2c=hper2const(h/2,gam);
    Ch=double([hc.c11,0; hc.c21, hc.c22]);
    Ch2=double([h2c.c11,0; h2c.c21, h2c.c22]);
    Mh=double([1, hc.etam1g; 0, hc.eta]);
    Mh2=double([1, h2c.etam1g; 0, h2c.eta]);

    mx1=double(Ch\(Mh2*Ch2));
    mx2=double(Ch\Ch2);
    % Sigmah=zeros(2);
    % Sigmah(1,1)=-(3+exp(-gam*h)-4*exp(-gam*h/2)-gam*h)/(gam^2);
    % Sigmah(1,2)=exp(-gam*h)*(-1+exp(gam*h/2))^2/gam;
    % Sigmah(2,1)=Sigmah(1,2);
    % Sigmah(2,2)=1-exp(-gam*h);
    % err=Ch*Ch'-Sigmah
    % err=Ch*(Ch')-Mh2*Ch2*(Ch2')*(Mh2')-Ch2*(Ch2')
end

function xi_new=transform_xi(xi,h,gam)
    nbeta=size(xi,1);
    n=size(xi,2);
    xi_new=zeros_sd(nbeta,n/2);
    [mx1,mx2]=xi_h_mx(h,gam);
    for it=1:(n/4)
        xi_new(1:nbeta,(it-1)*2+1)=mx1(1,1)*xi(1:nbeta,((it-1)*4+1))+mx1(1,2)*xi(1:nbeta,((it-1)*4+2))+mx2(1,1)*xi(1:nbeta,((it-1)*4+3))+mx2(1,2)*xi(1:nbeta,((it-1)*4+4));
        xi_new(1:nbeta,it*2)=mx1(2,1)*xi(1:nbeta,((it-1)*4+1))+mx1(2,2)*xi(1:nbeta,((it-1)*4+2))+mx2(2,1)*xi(1:nbeta,((it-1)*4+3))+mx2(2,2)*xi(1:nbeta,((it-1)*4+4));
    end
end

function xiarr=gen_xiarr(nbeta,multilevels,h,gam)
    xiarr=zeros_sd(nbeta,8*(2^(multilevels+1)-1));
    start_ind=zeros(1,multilevels+1);
    end_ind=zeros(1,multilevels+1);
    start_ind(1)=1;
    end_ind(1)=8;
    for it=2:(multilevels+1)
        start_ind(it)=end_ind(it-1)+1;
        end_ind(it)=end_ind(it-1)+8*2^(it-1);
    end 
    xiarr(1:nbeta,start_ind(multilevels+1):end_ind(multilevels+1))=randn_sd(nbeta,8*2^(multilevels));
    for it=multilevels:(-1):1
        xiarr(1:nbeta,start_ind(it):end_ind(it))=transform_xi(xiarr(1:nbeta,start_ind(it+1):end_ind(it+1)),h/2^(it-1),gam);
    end
end



% function [xn,vn]=UBU_step(x,v,rep,hper2c,inputs,xi)        
%     [xn,vn]=U(x,v,hper2c,xi(:,1:rep),xi(:,(rep+1):(2*rep)));
%     grad=inputs.grad_lpost(xn);
%     vn=vn-(hper2c.h)*grad;
%     [xn,vn]=U(xn,vn,hper2c,xi(:,(2*rep+1):(3*rep)),xi(:,(3*rep+1):(4*rep)));
% end
% 
% function [xn,vn]=OHO_step(x,v,rep,hper2c,beta_min,exphM,xi)        
%     xn=x;
%     vn=v;
%     vn=vn*hper2c.eta+hper2c.c21*xi(:,1:rep)+hper2c.c22*xi(:,(rep+1):(2*rep));
%     beta_min_mx=beta_min*ones(1,rep);
%     xnvn=exphM*[xn-beta_min_mx;vn];
%     xn=xnvn(1:nbeta,1:rep)+beta_min_mx;
%     vn=xnvn((nbeta+1):(2*nbeta),1:rep);
%     vn=vn*hper2c.eta+hper2c.c21*xi(:,(2*rep+1):(3*rep))+hper2c.c22*xi(:,(3*rep+1):(4*rep));
% end

function grad=grad_lpost_approx(x,beta_star,grad_beta_star,inputs)
    grad=grad_beta_star+inputs.Hprodv(x-beta_star);
end

% function [xn,vn]=OHO_step(x,v,rep,hper2c,inputs,xi)        
%     xn=x;
%     vn=v;
%     vn=vn*hper2c.eta+hper2c.c21*xi(:,1:rep)+hper2c.c22*xi(:,(rep+1):(2*rep));
%     beta_min_mx=inputs.beta_min*ones(1,rep);
%     xnvn=inputs.exp_hM([xn-beta_min_mx;vn],hper4c.h);    
%     xn=xnvn(1:inputs.nbeta,1:rep)+beta_min_mx;
%     vn=xnvn((inputs.nbeta+1):(2*inputs.nbeta),1:rep);
%     vn=vn*hper2c.eta+hper2c.c21*xi(:,(2*rep+1):(3*rep))+hper4c.c22*xi(:,(3*rep+1):(4*rep));
% end
% 
% function [xn,vn]=UBU_approx_step(x,v,beta_star,grad_beta_star,rep,hper2c,inputs,xi)        
%     [xn,vn]=U(x,v,hper2c,xi(:,1:rep),xi(:,(rep+1):(2*rep)));
%     grad=grad_lpost_approx(xn,beta_star,grad_beta_star,inputs);
%     vn=vn-hper2c.h*grad;
%     [xn,vn]=U(xn,vn,hper2c,xi(:,(2*rep+1):(3*rep)),xi(:,(3*rep+1):(4*rep)));
% end

function [xn,vn]=U(x,v,hc,xi1,xi2)
    xn=x+hc.etam1g*v+hc.c11*xi1;
    vn=v*hc.eta+hc.c21*xi1+hc.c22*xi2;
    % eta=exp(-h*gam);
    % Z1=sqrt(h)*xi1;
    % Z2=sqrt((1-eta^2)/(2*gam))*(sqrt((1-eta)*2/((1+eta)*(gam*h)))*xi1+sqrt(1-(1-eta)/(1+eta)*2/(gam*h))*xi2);
    % xn=x+(1-eta)/gam*v+sqrt(2/gam)*(Z1-Z2);
    % vn=v*eta+sqrt(2*gam)*Z2;
end

function [xn,vn,x2n,v2n]=OHO_UBU_approx_step2(x,v,x2,v2,beta_star2,grad_beta_star2,rep,hper4c,inputs,xi)        
    xn=x;
    vn=v;
    vn=vn*hper4c.eta+hper4c.c21*xi(:,1:rep)+hper4c.c22*xi(:,(rep+1):(2*rep));
    vn=vn*hper4c.eta+hper4c.c21*xi(:,(2*rep+1):(3*rep))+hper4c.c22*xi(:,(3*rep+1):(4*rep));
    beta_min_mx=inputs.beta_min*ones(1,rep);
    xnvn=inputs.exp_hM([xn-beta_min_mx;vn],hper4c.h*2);    
    xn=xnvn(1:inputs.nbeta,1:rep)+beta_min_mx;
    vn=xnvn((inputs.nbeta+1):(2*inputs.nbeta),1:rep);
    vn=vn*hper4c.eta+hper4c.c21*xi(:,(4*rep+1):(5*rep))+hper4c.c22*xi(:,(5*rep+1):(6*rep));
    vn=vn*hper4c.eta+hper4c.c21*xi(:,(6*rep+1):(7*rep))+hper4c.c22*xi(:,(7*rep+1):(8*rep));

    [x2n,v2n]=U(x2,v2,hper4c,xi(:,1:rep),xi(:,(rep+1):(2*rep)));
    grad=grad_lpost_approx(x2n,beta_star2,grad_beta_star2,inputs);
    %v2n=v2n-(h/2)*grad;
    v2n=v2n-hper4c.h*grad;
    [x2n,v2n]=U(x2n,v2n,hper4c,xi(:,(2*rep+1):(3*rep)),xi(:,(3*rep+1):(4*rep)));
    [x2n,v2n]=U(x2n,v2n,hper4c,xi(:,(4*rep+1):(5*rep)),xi(:,(5*rep+1):(6*rep)));
    grad=grad_lpost_approx(x2n,beta_star2,grad_beta_star2,inputs);        
    %v2n=v2n-(h/2)*grad;
    v2n=v2n-hper4c.h*grad;
    [x2n,v2n]=U(x2n,v2n,hper4c,xi(:,(6*rep+1):(7*rep)),xi(:,(7*rep+1):(8*rep)));
end

function [xn,vn,x2n,v2n]=UBU_approx_step2(x,v, x2,v2,beta_star1,grad_beta_star1,beta_star2,grad_beta_star2, rep,hper4c,inputs,xi)
    [x2n,v2n]=U(x2,v2,hper4c,xi(:,1:rep),xi(:,(rep+1):(2*rep)));
    grad=grad_lpost_approx(x2n,beta_star2,grad_beta_star2,inputs);
    %v2n=v2n-(h/2)*grad;
    v2n=v2n-hper4c.h*grad;
    [x2n,v2n]=U(x2n,v2n,hper4c,xi(:,(2*rep+1):(3*rep)),xi(:,(3*rep+1):(4*rep)));
    [x2n,v2n]=U(x2n,v2n,hper4c,xi(:,(4*rep+1):(5*rep)),xi(:,(5*rep+1):(6*rep)));
    grad=grad_lpost_approx(x2n,beta_star2,grad_beta_star2,inputs);        
    %v2n=v2n-(h/2)*grad;
    v2n=v2n-hper4c.h*grad;
    [x2n,v2n]=U(x2n,v2n,hper4c,xi(:,(6*rep+1):(7*rep)),xi(:,(7*rep+1):(8*rep)));

    [xn,vn]=U(x,v,hper4c,xi(:,1:rep),xi(:,(rep+1):(2*rep)));
    [xn,vn]=U(xn,vn,hper4c,xi(:,(2*rep+1):(3*rep)),xi(:,(3*rep+1):(4*rep)));
    grad=grad_lpost_approx(xn,beta_star1,grad_beta_star1,inputs);
    %vn=vn-(h)*grad;
    vn=vn-2*hper4c.h*grad;
    [xn,vn]=U(xn,vn,hper4c,xi(:,(4*rep+1):(5*rep)),xi(:,(5*rep+1):(6*rep)));
    [xn,vn]=U(xn,vn,hper4c,xi(:,(6*rep+1):(7*rep)),xi(:,(7*rep+1):(8*rep)));
end


% function [xn,vn]=burnMCMC(x,v,rep,burn,h,inputs)
%         xn=x;vn=v;
%         hper2c=hper2const(h,inputs.gam); 
%         for(it=1:burn)
%             xi=randn_sd(inputs.nbeta,4*rep);
%             [xn,vn]=UBU_step(xn,vn,rep,hper2c,inputs,xi);
%         end
% end

function [xn,vn,x2n,v2n]=burnMCMC2(rep,burn_oho_ubu,burn_ubu,h,inputs)        
        xn=inputs.beta_min*ones(1,rep)+inputs.invcholHprodv(randn_sd(inputs.nbeta,rep));
        vn=randn_sd(inputs.nbeta,rep);
        x2n=xn;v2n=vn;
        
        hper4c=hper2const(h/2,inputs.gam);        
        beta_star1=xn; 
        grad_beta_star1=inputs.grad_lpost(beta_star1);
        beta_star2=xn; 
        grad_beta_star2=grad_beta_star1;


        for(it=1:burn_oho_ubu)
            xi=randn_sd(inputs.nbeta,8*rep);
            [xn,vn,x2n,v2n]=OHO_UBU_approx_step2(xn,vn, x2n,v2n,beta_star2,grad_beta_star2,rep, hper4c,inputs,xi);
            if(mod(2*it,inputs.repfullgrad)==0) beta_star2=x2n; grad_beta_star2=inputs.grad_lpost(beta_star2); end            
    
            % if(mod(it,100)==0)
            %    it
            % end
        end
        for(it=(burn_oho_ubu+1):(burn_oho_ubu+burn_ubu))
            xi=randn_sd(inputs.nbeta,8*rep);
            [xn,vn,x2n,v2n]=UBU_approx_step2(xn,vn, x2n,v2n,beta_star1,grad_beta_star1,beta_star2,grad_beta_star2,rep, hper4c,inputs,xi);
            if(mod(it,inputs.repfullgrad)==0) beta_star1=xn; grad_beta_star1=inputs.grad_lpost(beta_star1); end
            if(mod(2*it,inputs.repfullgrad)==0) beta_star2=x2n; grad_beta_star2=inputs.grad_lpost(beta_star2); end            
        end
end

function [meanx,meanxsquare]=Gaussian_samples(rep,niter,inputs)
    meanx=zeros_sd(inputs.test_dim,rep);
    meanxsquare=zeros_sd(inputs.test_dim,rep);
    for it=1:niter
        % if(mod(it,100)==0)
        % it
        % end
        xn=inputs.beta_min*ones(1,rep)+inputs.invcholHprodv(randn_sd(inputs.nbeta,rep));
        test_vals=inputs.test_function(xn);
        meanx=meanx+test_vals;
        meanxsquare=meanxsquare+test_vals.^2;
    end
    meanx=meanx/niter;
    meanxsquare=meanxsquare/niter;    
end


function [meanx,meanxsquare]=doubleMCMC_levels01(rep,niter,h,extraburnin2, jointburnin,thin,inputs)
    [xn,vn,x2n,v2n]=burnMCMC2(rep,extraburnin2+jointburnin,0,h,inputs);    
    beta_star2=x2n; 
    grad_beta_star2=inputs.grad_lpost(x2n);

        
    test_vals1=zeros_sd(inputs.test_dim,rep);
    test_vals2=zeros_sd(inputs.test_dim,rep);

    meanx=zeros_sd(inputs.test_dim,rep);
    meanxsquare=zeros_sd(inputs.test_dim,rep);

    hper4c=hper2const(h/2,inputs.gam);

    for(it=1:floor(niter/thin))
        for(j=1:thin)
            xi=randn_sd(inputs.nbeta,8*rep);
            [xn,vn,x2n,v2n]=OHO_UBU_approx_step2(xn,vn, x2n,v2n,beta_star2,grad_beta_star2,rep, hper4c,inputs,xi);
            if(mod(((it-1)*thin+j)*2,inputs.repfullgrad)==0) beta_star2=x2n; grad_beta_star2=inputs.grad_lpost(beta_star2); end            
        end
        test_vals1=inputs.test_function(xn);
        test_vals2=inputs.test_function(x2n);
        
        meanx=meanx+test_vals2-test_vals1;
        meanxsquare=meanxsquare+test_vals2.^2-test_vals1.^2;
    end
    meanx=meanx/floor(niter/thin);
    meanxsquare=meanxsquare/floor(niter/thin);
end

function [meanx,meanxsquare]=doubleMCMC(rep,niter,h,extraburnin2, jointburnin,thin,inputs)
    [xn,vn,x2n,v2n]=burnMCMC2(rep,extraburnin2,jointburnin,h,inputs);    
    beta_star1=xn; 
    grad_beta_star1=inputs.grad_lpost(beta_star1);
    beta_star2=x2n; 
    grad_beta_star2=inputs.grad_lpost(beta_star2);

        
    test_vals1=zeros_sd(inputs.test_dim,rep);
    test_vals2=zeros_sd(inputs.test_dim,rep);

    meanx=zeros_sd(inputs.test_dim,rep);
    meanxsquare=zeros_sd(inputs.test_dim,rep);

    hper4c=hper2const(h/2,inputs.gam);

    for(it=1:floor(niter/thin))
        for(j=1:thin)
            xi=randn_sd(inputs.nbeta,8*rep);
            [xn,vn,x2n,v2n]=UBU_approx_step2(xn,vn, x2n,v2n,beta_star1,grad_beta_star1,beta_star2,grad_beta_star2,rep, hper4c,inputs,xi);
            if(mod((it-1)*thin+j,inputs.repfullgrad)==0) beta_star1=xn; grad_beta_star1=inputs.grad_lpost(beta_star1); end
            if(mod(((it-1)*thin+j)*2,inputs.repfullgrad)==0) beta_star2=x2n; grad_beta_star2=inputs.grad_lpost(beta_star2); end            
        end
        test_vals1=inputs.test_function(xn);
        test_vals2=inputs.test_function(x2n);
        
        meanx=meanx+test_vals2-test_vals1;
        meanxsquare=meanxsquare+test_vals2.^2-test_vals1.^2;
    end
    meanx=meanx/floor(niter/thin);
    meanxsquare=meanxsquare/floor(niter/thin);
end

function [meanx,meanxsquare]=multiMCMC(niter,h,extraburnin2, jointburnin, thin, multilevels, repruns, inputs)

    xnarr=zeros_sd(inputs.nbeta,multilevels+1);
    vnarr=zeros_sd(inputs.nbeta,multilevels+1);
    x2narr=zeros_sd(inputs.nbeta,multilevels+1);
    v2narr=zeros_sd(inputs.nbeta,multilevels+1);

    xnarr(:,multilevels+1)=inputs.beta_min+inputs.invcholHprodv(randn_sd(inputs.nbeta,1));
    vnarr(:,multilevels+1)=randn_sd(inputs.nbeta,1);
    x2narr(:,multilevels+1)=xnarr(:,multilevels+1);v2narr(:,multilevels+1)=vnarr(:,multilevels+1);    

    beta_star1_arr=zeros_sd(inputs.nbeta,multilevels+1);
    grad_beta_star1_arr=zeros_sd(inputs.nbeta,multilevels+1);
    beta_star2_arr=zeros_sd(inputs.nbeta,multilevels+1);
    grad_beta_star2_arr=zeros_sd(inputs.nbeta,multilevels+1);
    beta_star1_arr(:,multilevels+1)=xnarr(:,multilevels+1);
    grad_beta_star1_arr(:,multilevels+1)=inputs.grad_lpost(xnarr(:,multilevels+1));
    beta_star2_arr(:,multilevels+1)=beta_star1_arr(:,multilevels+1);
    grad_beta_star2_arr(:,multilevels+1)=grad_beta_star1_arr(:,multilevels+1);

    meanx=zeros_sd(inputs.test_dim,multilevels+1);
    meanxsquare=zeros_sd(inputs.test_dim,multilevels+1);

    start_ind=zeros(1,multilevels+1);
    end_ind=zeros(1,multilevels+1);
    start_ind(1)=1;
    end_ind(1)=8; % 8 is because 8 Gaussians are needed to simulate each coupled UBU step between step sizes h and h/2
    for(it=2:(multilevels+1))
        start_ind(it)=end_ind(it-1)+1;
        end_ind(it)=start_ind(it)+8*2^(it-1)-1;
    end

    for it=1:(jointburnin+extraburnin2*(multilevels+1))
        xiarr=gen_xiarr(inputs.nbeta,multilevels,h,inputs.gam); 

        for mit=(multilevels+1):(-1):1

            hper4c=hper2const(h/2^(mit),inputs.gam);                                
            if(it>(multilevels+2-mit)*extraburnin2)
                for it2=1:(2^(mit-1))
                    % [xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit)]=...
                    %     UBU_step2(xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit),1,hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)):(start_ind(mit)+8*it2-1)));           
                    [xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit)]=...
                    UBU_approx_step2(xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit),beta_star1_arr(:,mit),grad_beta_star1_arr(:,mit),beta_star2_arr(:,mit),grad_beta_star2_arr(:,mit),1, hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)):(start_ind(mit)+8*it2-1)));            
                    if(mod((it-1)*2^(mit-1)+it2,inputs.repfullgrad)==0) beta_star1_arr(:,mit)=xnarr(:,mit); grad_beta_star1_arr(:,mit)=inputs.grad_lpost(xnarr(:,mit)); end
                    if(mod(((it-1)*2^(mit-1)+it2)*2,inputs.repfullgrad)==0) beta_star2_arr(:,mit)=x2narr(:,mit); grad_beta_star2_arr(:,mit)=inputs.grad_lpost(x2narr(:,mit)); end                               
                end
            elseif(it>(multilevels+1-mit)*extraburnin2) 
                for it2=1:(2^(mit-1))
                    [xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit)]=...
                    OHO_UBU_approx_step2(xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit),beta_star2_arr(:,mit),grad_beta_star2_arr(:,mit),1, hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)):(start_ind(mit)+8*it2-1)));            
                    if(mod(((it-1)*2^(mit-1)+it2)*2,inputs.repfullgrad)==0) beta_star2_arr(:,mit)=x2narr(:,mit); grad_beta_star2_arr(:,mit)=inputs.grad_lpost(x2narr(:,mit)); end
                    % [x2narr(:,mit),v2narr(:,mit)]=...
                    % UBU_step(x2narr(:,mit),v2narr(:,mit),1,hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)):(start_ind(mit)+8*(it2-1)+3)));
                    % [x2narr(:,mit),v2narr(:,mit)]=...
                    % UBU_step(x2narr(:,mit),v2narr(:,mit),1,hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)+4):(start_ind(mit)+8*it2-1)));
                end
            elseif(it==(multilevels+1-mit)*extraburnin2)
                xnarr(:,mit)=xnarr(:,mit+1);
                vnarr(:,mit)=vnarr(:,mit+1);
                x2narr(:,mit)=xnarr(:,mit+1);
                v2narr(:,mit)=vnarr(:,mit+1);
                beta_star1_arr(:,mit)=xnarr(:,mit); grad_beta_star1_arr(:,mit)=inputs.grad_lpost(xnarr(:,mit));
                beta_star2_arr(:,mit)=beta_star1_arr(:,mit);grad_beta_star2_arr(:,mit)=grad_beta_star1_arr(:,mit);
            end            
        end
    end

    for mit=(multilevels+1):(-1):1
        if(repruns(mit)==1)    
            beta_star1_arr(:,mit)=xnarr(:,mit); grad_beta_star1_arr(:,mit)=inputs.grad_lpost(xnarr(:,mit)); 
            beta_star2_arr(:,mit)=x2narr(:,mit); grad_beta_star2_arr(:,mit)=inputs.grad_lpost(x2narr(:,mit));
        end
    end

    for(it=1:floor(niter/thin))
        for(j=1:thin)
            xiarr=gen_xiarr(inputs.nbeta,multilevels,h,inputs.gam);
            for mit=(multilevels+1):(-1):1
            if(repruns(mit)==1)
                for it2=1:(2^(mit-1))
                    hper4c=hper2const(h/2^(mit),inputs.gam);
                    %[xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit)]=...
                    %UBU_step2(xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit),1,hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)):(start_ind(mit)+8*it2-1)));

                    [xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit)]=...
                    UBU_approx_step2(xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit),beta_star1_arr(:,mit),grad_beta_star1_arr(:,mit),beta_star2_arr(:,mit),grad_beta_star2_arr(:,mit),1, hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)):(start_ind(mit)+8*it2-1)));            
                    if(mod(((it-1)*thin+j-1)*2^(mit-1)+it2,inputs.repfullgrad)==0) beta_star1_arr(:,mit)=xnarr(:,mit); grad_beta_star1_arr(:,mit)=inputs.grad_lpost(xnarr(:,mit)); end
                    if(mod((((it-1)*thin+j-1)*2^(mit-1)+it2)*2,inputs.repfullgrad)==0) beta_star2_arr(:,mit)=x2narr(:,mit); grad_beta_star2_arr(:,mit)=inputs.grad_lpost(x2narr(:,mit)); end
                end
            end
            end
        end

        for mit=(multilevels+1):(-1):1
        if(repruns(mit)==1)
            test_vals1=inputs.test_function(xnarr(:,mit));
            test_vals2=inputs.test_function(x2narr(:,mit));
            meanx(:,mit)=meanx(:,mit)+test_vals2-test_vals1;
            meanxsquare(:,mit)=meanxsquare(:,mit)+test_vals2.^2-test_vals1.^2;
        end
        end
    end
    meanx=meanx/floor(niter/thin);
    meanxsquare=meanxsquare/floor(niter/thin);
end


% function [meanx,meanxsquare]=multiMCMC(x,v,niter,h,extraburnin2, jointburnin, thin, multilevels, repruns, inputs)
% 
%     xnarr=x*ones(1,multilevels+1);
%     vnarr=v*ones(1,multilevels+1);
%     x2narr=x*ones(1,multilevels+1);
%     v2narr=v*ones(1,multilevels+1);
% 
%     meanx=zeros_sd(inputs.test_dim,multilevels+1);
%     meanxsquare=zeros_sd(inputs.test_dim,multilevels+1);
% 
%     start_ind=zeros(1,multilevels+1);
%     end_ind=zeros(1,multilevels+1);
%     start_ind(1)=1;
%     end_ind(1)=8;
%     for(it=2:(multilevels+1))
%         start_ind(it)=end_ind(it-1)+1;
%         end_ind(it)=start_ind(it)+8*2^(it-1)-1;
%     end
% 
% 
%     for it=1:(jointburnin+extraburnin2*(multilevels+1))
%         xiarr=gen_xiarr(inputs.nbeta,multilevels,h,inputs.gam); 
% 
%         for mit=(multilevels+1):(-1):1
% 
%             if(repruns(mit)==1)               
%                 hper4c=hper2const(h/2^(mit),inputs.gam);                                
%                 if(it>(multilevels+2-mit)*extraburnin2)
%                     for it2=1:(2^(mit-1))
%                         [xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit)]=UBU_step2(xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit),1,hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)):(start_ind(mit)+8*it2-1)));           
%                     end
%                 elseif(it>(multilevels+1-mit)*extraburnin2)
%                     for it2=1:(2^(mit-1))
%                         [x2narr(:,mit),v2narr(:,mit)]=UBU_step(x2narr(:,mit),v2narr(:,mit),1,hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)):(start_ind(mit)+8*(it2-1)+3)));
%                         [x2narr(:,mit),v2narr(:,mit)]=UBU_step(x2narr(:,mit),v2narr(:,mit),1,hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)+4):(start_ind(mit)+8*it2-1)));
%                     end
%                 end
%             end
%         end
%     end
% 
%     for(it=1:floor(niter/thin))
%         for(j=1:thin)
%             xiarr=gen_xiarr(inputs.nbeta,multilevels,h,inputs.gam);
%             for mit=(multilevels+1):(-1):1
%             if(repruns(mit)==1)
%                 for it2=1:(2^(mit-1))
%                         hper4c=hper2const(h/2^(mit),inputs.gam);
%                         [xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit)]=UBU_step2(xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit),1,hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)):(start_ind(mit)+8*it2-1)));
%                 end
%             end
%             end
%         end
% 
%         for mit=(multilevels+1):(-1):1
%         if(repruns(mit)==1)
%             test_vals1=inputs.test_function(xnarr(:,mit));
%             test_vals2=inputs.test_function(x2narr(:,mit));
%             meanx(:,mit)=meanx(:,mit)+test_vals2-test_vals1;
%             meanxsquare(:,mit)=meanxsquare(:,mit)+test_vals2.^2-test_vals1.^2;
%         end
%         end
%     end
%     meanx=meanx/floor(niter/thin);
%     meanxsquare=meanxsquare/floor(niter/thin);
% end





% function res=multilevel_ubu_approx(niter,burnin,rep,h,gam, grad_lpost, grad_lpost_approx, test_function,  options)
%     %#codegen
% 
%     data_type_const=options.data_type_const;
%     to_data_type=options.to_data_type;
% 
%     r=options.r;
%     c=options.c;
%     n=options.n;
% 
%     test_dim=options.test_dim;
%     beta_min=options.beta_min;
%     lev_no_fullgrad=options.lev_no_fullgrad;
%     repfullgrad=options.repfullgrad;
% 
%     grad_beta_min=grad_lpost(beta_min);
% 
% 
%     maxlevel=options.maxlevel;
%     max_parallel_chain=options.max_parallel_chain;
% 
%     nbeta=length(beta_min);
% 
%     repruns=ones(1,maxlevel+1);
%     repruns(1)=rep;
%     problevels=zeros(1,maxlevel);
% 
% 
%     for(lev=1:maxlevel)
%         problevels(lev)=rep*c/(r^(lev-1));
%         repruns(lev+1)=ceil(problevels(lev));
%     end
%     detlevels=sum(problevels>=1/2);
%     randlevels=maxlevel-detlevels;
%     probrandlevels=zeros(1,randlevels);
%     probrandlevels=problevels((detlevels+1):maxlevel);
%     maxruns=sum(repruns);
% 
% 
%     blockstart=[1,cumsum(repruns(1:maxlevel))+1];
%     blockend=cumsum(repruns(1:(maxlevel+1)));
% 
% 
%     res=struct;
% 
%     res.detlevels=detlevels;
%     res.ngradtot=0;
%     res.niter=niter;
%     res.repruns=zeros(maxlevel+1,wholerep);
%     res.wholerep=wholerep;
%     res.maxlevel=maxlevel;
%     res.detlevels=detlevels;
%     res.randlevels=randlevels;
%     res.probrandlevels=probrandlevels;
%     res.means=zeros(test_dim,maxruns,wholerep);
%     res.squaremeans=zeros(test_dim,maxruns,wholerep);
%     res.blockstart=blockstart;
%     res.blockend=blockend;
%     res.test_dim=test_dim;
%     res.lev_no_fullgrad=lev_no_fullgrad;
% 
% 
%     ngradtot=0;
%     if(lev_no_fullgrad<0)
%     ngradtot=(burnin+niter)*rep;
%     end
%     for(lev=1:detlevels)
%         levm=lev-1;
%         extraburnin2=burnin*(2^lev)*(lev>lev_no_fullgrad);
%         jointburnin=burnin*(2^levm)*lev*((levm>lev_no_fullgrad)+2*(lev>lev_no_fullgrad));
%         ngradtot=ngradtot+repruns(lev+1)*(extraburnin2+jointburnin+niter*2^levm*((levm>lev_no_fullgrad)+2*(lev>lev_no_fullgrad)));
%     end
% 
%     for(lev=(detlevels+1):maxlevel)
%         levm=lev-1;
%         extraburnin2=burnin*(2^lev)*(lev>lev_no_fullgrad);
%         jointburnin=burnin*(2^levm)*lev*((levm>lev_no_fullgrad)+2*(lev>lev_no_fullgrad));
%         ngradtot=ngradtot+probrandlevels(lev-detlevels)*(extraburnin2+jointburnin+niter*2^levm*((levm>lev_no_fullgrad)+2*(lev>lev_no_fullgrad)));
%     end
%     res.ngradtot=ngradtot;
% 
%     for(wholeit=1:wholerep)
%     wholeit
%     no_batches=ceil(rep/max_parallel_chain);
%     for repit=1:no_batches
%         if(repit<no_batches)
%             par_chain=max_parallel_chain;
%         else 
%             par_chain=rep-max_parallel_chain*(no_batches-1);
%         end
%         x=to_data_type(beta_min*ones(1,par_chain));
%         v=randn(nbeta,par_chain,data_type_const());
% 
%         [res.means(1:test_dim,((repit-1)*max_parallel_chain+1):((repit-1)*max_parallel_chain+par_chain),wholeit),res.squaremeans(1:test_dim,((repit-1)*max_parallel_chain+1):((repit-1)*max_parallel_chain+par_chain),wholeit)]=singleMCMC(x,v,par_chain,niter,h,burnin,repfullgrad);
%     end
% 
%     repruns((detlevels+2):(maxlevel+1))=rand(1,randlevels)<probrandlevels;
%     res.repruns(:,wholeit)=repruns;
% 
%     for(lev=1:maxlevel)
%         no_batches=ceil(repruns(lev+1)/max_parallel_chain);
% 
%         for repit=1:no_batches
%             if(repit<no_batches)
%                 par_chain=max_parallel_chain;
%             else
%                 par_chain=repruns(lev+1)-max_parallel_chain*(no_batches-1);
%             end
%             x=to_data_type(beta_min*ones(1,par_chain));
%             v=randn(nbeta,par_chain,data_type_const());
%             x2=to_data_type(beta_min*ones(1,par_chain));
%             v2=v;
%             levm=lev-1;
%             extraburnin2=burnin*(2^levm);
%             jointburnin=burnin*(2^levm)*lev;
%             repfullgrad1=(levm>lev_no_fullgrad)*repfullgrad;
%             repfullgrad2=(lev>lev_no_fullgrad)*repfullgrad;
%             [res.means(1:test_dim,(blockstart(lev+1)+(repit-1)*max_parallel_chain):(blockstart(lev+1)+(repit-1)*max_parallel_chain+par_chain-1),wholeit),res.squaremeans(1:test_dim,(blockstart(lev+1)+(repit-1)*max_parallel_chain):(blockstart(lev+1)+(repit-1)*max_parallel_chain+par_chain-1),wholeit)]=doubleMCMC(x,v,x2,v2,par_chain,niter*2^levm,h/(2^levm),extraburnin2, jointburnin,(2^levm),repfullgrad1,repfullgrad2);
%         end
%     end
% 
%     end
% 
% 
% 
%     function [xn,vn]=U(x,v,h,xi1,xi2)
%         hg=h*gam;
%         eta=exp(-hg);
%         one_minus_eta=-expm1(-hg);
%         Cxi2=realsqrt((hg^2/6+hg^3/12+hg^4/40+hg^5/180+hg^6/1008+hg^7/6720+hg^8/51840+hg^9/453600)/(1+exp(hg)));
%         Z1=realsqrt(h)*xi1;      
%         Z2=realsqrt((1+eta)*one_minus_eta/(2*gam))*(realsqrt(one_minus_eta*2/((1+eta)*(hg)))*xi1+Cxi2*xi2);
%         xn=x+one_minus_eta/gam*v+realsqrt(2/gam)*(Z1-Z2);
%         vn=v*eta+realsqrt(2*gam)*Z2;
%     end    
%     % function [xn,vn]=U(x,v,h,xi1,xi2)
%     %     eta=exp(-h*gam);
%     %     Z1=sqrt(h)*xi1;
%     %     Z2=(sqrt((1-eta^2)/(2*gam))*(sqrt((1-eta)*2/((1+eta)*(gam*h))))*xi1+sqrt(1-(1-eta)/(1+eta)*2/(gam*h))*xi2);
%     %     xn=x+((1-eta)/gam)*v+sqrt(2/gam)*(Z1-Z2);
%     %     vn=v*eta+sqrt(2*gam)*Z2;
%     % end
% 
% 
%     function [xn,vn]=UBU_approx_step(x,v,rep,h,beta_star,grad_beta_star)        
%         xi=randn(nbeta,4*rep,data_type_const());
%         [xn,vn]=U(x,v,h/2,xi(:,1:rep),xi(:,(rep+1):(2*rep)));
% 
% 
%         grad=grad_lpost_approx(xn,beta_star,grad_beta_star);        
% 
%         vn=vn-(h)*grad;
%         [xn,vn]=U(xn,vn,h/2,xi(:,(2*rep+1):(3*rep)),xi(:,(3*rep+1):(4*rep)));
%     end
% 
%     function [xn,vn,x2n,v2n]=UBU_approx_step2(x,v, x2,v2,rep,h,beta_star1,grad_beta_star1,beta_star2,grad_beta_star2)
%         xi=randn(nbeta,8*rep,data_type_const());
%         [x2n,v2n]=U(x2,v2,h/4,xi(:,1:rep),xi(:,(rep+1):(2*rep)));
% 
%         grad=grad_lpost_approx(x2n,beta_star2,grad_beta_star2);
% 
%         v2n=v2n-(h/2)*grad;
%         [x2n,v2n]=U(x2n,v2n,h/4,xi(:,(2*rep+1):(3*rep)),xi(:,(3*rep+1):(4*rep)));
%         [x2n,v2n]=U(x2n,v2n,h/4,xi(:,(4*rep+1):(5*rep)),xi(:,(5*rep+1):(6*rep)));
% 
% 
%         grad=grad_lpost_approx(x2n,beta_star2,grad_beta_star2);        
% 
%         v2n=v2n-(h/2)*grad;
%         [x2n,v2n]=U(x2n,v2n,h/4,xi(:,(6*rep+1):(7*rep)),xi(:,(7*rep+1):(8*rep)));
% 
%         [xn,vn]=U(x,v,h/4,xi(:,1:rep),xi(:,(rep+1):(2*rep)));
%         [xn,vn]=U(xn,vn,h/4,xi(:,(2*rep+1):(3*rep)),xi(:,(3*rep+1):(4*rep)));
% 
%         grad=grad_lpost_approx(xn,beta_star1,grad_beta_star1);
% 
%         vn=vn-(h)*grad;
%         [xn,vn]=U(xn,vn,h/4,xi(:,(4*rep+1):(5*rep)),xi(:,(5*rep+1):(6*rep)));
%         [xn,vn]=U(xn,vn,h/4,xi(:,(6*rep+1):(7*rep)),xi(:,(7*rep+1):(8*rep)));
%     end
% 
% 
%     function [xn,vn]=burnMCMC(x,v,rep,burn,h,repfullgrad)
%             xn=x;vn=v;
% 
%             if(repfullgrad>0)
%                 beta_star=xn; 
%                 grad_beta_star=grad_lpost(beta_star);
%             else
%                 beta_star=beta_min*ones(1,rep);
%                 grad_beta_star=grad_beta_min*ones(1,rep);
%             end
% 
%             for(it=1:burn)
%                 [xn,vn]=UBU_approx_step(xn,vn,rep,h,beta_star,grad_beta_star);
%                 if(mod(it,repfullgrad)==0)
%                     beta_star=xn;
%                     grad_beta_star=grad_lpost(beta_star);
%                 end
%             end
%     end
% 
% 
% 
%     function [xn,vn,x2n,v2n]=burnMCMC2(x,v,x2,v2,rep,burn,h,repfullgrad1,repfullgrad2)
%             xn=x;vn=v;x2n=x2;v2n=v2;
% 
%             if(repfullgrad1>0)
%                 beta_star1=xn; 
%                 grad_beta_star1=grad_lpost(beta_star1);
%             else
%                 beta_star1=beta_min*ones(1,rep);
%                 grad_beta_star1=grad_beta_min*ones(1,rep);
%             end
% 
%             if(repfullgrad2>0)
%                 beta_star2=xn; 
%                 grad_beta_star2=grad_lpost(beta_star2);
%             else
%                 beta_star2=beta_min*ones(1,rep);
%                 grad_beta_star2=grad_beta_min*ones(1,rep);
%             end            
% 
%             for(it=1:burn)
%                 [xn,vn,x2n,v2n]=UBU_approx_step2(xn,vn, x2n,v2n,rep, h,beta_star1,grad_beta_star1,beta_star2,grad_beta_star2);
%                 if(mod(it,repfullgrad1)==0) beta_star1=xn; grad_beta_star1=grad_lpost(beta_star1); end
%                 if(mod(it,repfullgrad2/2)==0) beta_star2=x2n; grad_beta_star2=grad_lpost(beta_star2); end
%             end
%     end
% 
% 
% 
%     function [meanx,meanxsquare]=singleMCMC(x,v,rep,niter,h,burn,repfullgrad)
%         [xn,vn]=burnMCMC(x,v,rep,burn,h,repfullgrad);
%         test_vals=zeros(test_dim,rep,data_type_const());
%         meanx=zeros(test_dim,rep,data_type_const());
%         meanxsquare=zeros(test_dim,rep,data_type_const());
% 
%         if(repfullgrad>0)
%             beta_star=xn; 
%             grad_beta_star=grad_lpost(beta_star);
%         else
%             beta_star=beta_min*ones(1,rep);
%             grad_beta_star=grad_beta_min*ones(1,rep);
%         end
% 
%         for (it=1:niter)
%             [xn,vn]=UBU_approx_step(xn,vn,rep,h,beta_star,grad_beta_star);
%             if(mod(it,repfullgrad)==0) beta_star=xn; grad_beta_star=grad_lpost(beta_star); end           
%             test_vals=test_function(xn);
%             meanx=meanx+test_vals;
%             meanxsquare=meanxsquare+test_vals.^2;
%         end
%         meanx=meanx/niter;
%         meanxsquare=meanxsquare/niter;
%     end
% 
%     function [meanx,meanxsquare]=doubleMCMC(x,v,x2,v2,rep,niter,h,extraburnin2, jointburnin,thin,repfullgrad1,repfullgrad2)
%         [x2n,v2n]=burnMCMC(x2,v2,rep,extraburnin2*2,h/2,repfullgrad2);
%         [xn,vn,x2n,v2n]=burnMCMC2(x,v,x2n,v2n,rep,jointburnin,h,repfullgrad1,repfullgrad2);
%         test_vals1=zeros(test_dim,rep,data_type_const());
%         test_vals2=zeros(test_dim,rep,data_type_const());
% 
%         meanx=zeros(test_dim,rep,data_type_const());
%         meanxsquare=zeros(test_dim,rep,data_type_const());
% 
% 
%         if(repfullgrad1>0)
%         beta_star1=xn; 
%         else
%         beta_star1=beta_min*ones(1,rep);
%         end
% 
%         if(repfullgrad2>0)
%         beta_star2=x2n; 
%         else
%         beta_star2=beta_min*ones(1,rep);
%         end
% 
%         grad_beta_star1=grad_lpost(beta_star1);
%         grad_beta_star2=grad_lpost(beta_star2);
% 
%         for(it=1:floor(niter/thin))
%             for(j=1:thin)
%                 [xn,vn,x2n,v2n]=UBU_approx_step2(xn,vn, x2n,v2n,rep, h,beta_star1,grad_beta_star1,beta_star2,grad_beta_star2);
%             end
%             if(mod(it,repfullgrad1)==0) beta_star1=xn; grad_beta_star1=grad_lpost(beta_star1); end
%             if(mod(it,repfullgrad2/2)==0) beta_star2=x2n; grad_beta_star2=grad_lpost(beta_star2); end
% 
%             test_vals1=test_function(xn);
%             test_vals2=test_function(x2n);
% 
%             meanx=meanx+test_vals2-test_vals1;
%             meanxsquare=meanxsquare+test_vals2.^2-test_vals1.^2;
%         end
%         meanx=meanx/floor(niter/thin);
%         meanxsquare=meanxsquare/floor(niter/thin);
%     end
% 
% end







