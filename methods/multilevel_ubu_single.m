function res=multilevel_ubu_single(niter,burnin,rep,h,gam, grad_lpost, test_function, options)
    %#codegen
    %coder.gpu.kernelfun;
    
    no_batches=0;
    inputs=struct;
    inputs.gam=gam;
    inputs.grad_lpost=grad_lpost;
    inputs.test_function=test_function;
    inputs.nbeta=options.nbeta;
    inputs.test_dim=options.test_dim;    

    r=options.r;
    c=options.c;
    nbeta=options.nbeta;
    test_dim=options.test_dim;
    beta_min=options.beta_min;
    maxlevel=options.maxlevel;
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
    ngradtot=(burnin+niter)*rep;

    for lev=1:detlevels
        levm=lev-1;
        extraburnin2=burnin*(2^lev);
        jointburnin=burnin*(2^levm)*lev*3;
        ngradtot=ngradtot+repruns(lev+1)*(extraburnin2+jointburnin+niter*2^levm*3);
    end
    
    for lev=(detlevels+1):maxlevel
        levm=lev-1;
        extraburnin2=burnin*(2^lev);
        jointburnin=burnin*(2^levm)*lev*3;
        ngradtot=ngradtot+problevels(lev+1)*(extraburnin2+jointburnin+niter*2^lev*3);
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
           
    
    no_batches=ceil(rep/max_parallel_chain);
    for repit=1:no_batches
        if(repit<no_batches)
            par_chain=max_parallel_chain;
        else 
            par_chain=rep-max_parallel_chain*(no_batches-1);
        end
        x=beta_min*ones(1,par_chain);
        v=randn_sd(nbeta,par_chain);
        [res.means(1:test_dim,((repit-1)*max_parallel_chain+1):((repit-1)*max_parallel_chain+par_chain)),res.squaremeans(1:test_dim,((repit-1)*max_parallel_chain+1):((repit-1)*max_parallel_chain+par_chain))]=singleMCMC(x,v,par_chain,niter,h,burnin,inputs);
    end
    
    

    for(lev=1:(detlevels-1))
        no_batches=ceil(repruns(lev+1)/max_parallel_chain);
        for repit=1:no_batches
            if(repit<no_batches)
                par_chain=max_parallel_chain;
            else
                par_chain=repruns(lev+1)-max_parallel_chain*(no_batches-1);
            end
            x=beta_min*ones(1,par_chain);
            v=randn_sd(nbeta,par_chain);
            x2=beta_min*ones(1,par_chain);
            v2=v;
            levm=lev-1;
            extraburnin2=burnin*(2^levm);
            jointburnin=burnin*(2^levm)*lev;
            [res.means(1:test_dim,(blockstart(lev+1)+(repit-1)*max_parallel_chain):(blockstart(lev+1)+(repit-1)*max_parallel_chain+par_chain-1)),res.squaremeans(1:test_dim,(blockstart(lev+1)+(repit-1)*max_parallel_chain):(blockstart(lev+1)+(repit-1)*max_parallel_chain+par_chain-1))]=doubleMCMC(x,v,x2,v2,par_chain,niter*2^levm,h/(2^levm),extraburnin2, jointburnin,(2^levm),inputs);
        end
    end

    coder.varsize("multilevels",[1 1]);
    multilevelsfind=find(repruns((detlevels+2):(maxlevel+1)),1,'last');
    if(isempty(multilevelsfind))
        multilevels=0;
    else
        multilevels=multilevelsfind(1);
    end

    x=beta_min;
    v=randn_sd(nbeta,1);
    lev=detlevels;levm=lev-1;
    extraburnin2=burnin*(2^levm);
    jointburnin=burnin*(2^levm)*lev;
    [res.means(1:test_dim,blockstart(lev+1):blockstart(lev+1+multilevels)),res.squaremeans(1:test_dim,blockstart(lev+1):blockstart(lev+1+multilevels))]=...        
    multiMCMC(x,v,niter*2^levm,h/(2^levm),extraburnin2, jointburnin, (2^levm), multilevels, repruns((detlevels+1):(maxlevel+1)), inputs);
end
 

function z=zeros_sd(a,b)
%    z=zeros(a,b,"double");
    z=zeros(a,b,"single");    
end
function z=randn_sd(a,b)
%    z=randn(a,b,"double");    
    z=randn(a,b,"single");    
end
function hc=hper2const(h,gam)
    hc=struct;
    hc.h=h;
    gh=double(gam)*double(h);
    s=sqrt(4*expm1(-gh/2)-expm1(-gh)+gh);

    hc.eta=double(exp(-gh/2));
    hc.etam1g=double((-expm1(-gh/2))/gam);
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
        end_ind(it)=start_ind(it)+8*2^(it-1)-1;
    end 
    xiarr(1:nbeta,start_ind(multilevels+1):end_ind(multilevels+1))=randn_sd(nbeta,8*2^(multilevels));
    for it=multilevels:(-1):1
        xiarr(1:nbeta,start_ind(it):end_ind(it))=transform_xi(xiarr(1:nbeta,start_ind(it+1):end_ind(it+1)),h/2^(it+1),gam);
    end
end

function [xn,vn]=U(x,v,hc,xi1,xi2)
    xn=x+hc.etam1g*v+hc.c11*xi1;
    vn=v*hc.eta+hc.c21*xi1+hc.c22*xi2;
    % eta=exp(-h*gam);
    % Z1=sqrt(h)*xi1;
    % Z2=sqrt((1-eta^2)/(2*gam))*(sqrt((1-eta)*2/((1+eta)*(gam*h)))*xi1+sqrt(1-(1-eta)/(1+eta)*2/(gam*h))*xi2);
    % xn=x+(1-eta)/gam*v+sqrt(2/gam)*(Z1-Z2);
    % vn=v*eta+sqrt(2*gam)*Z2;
end

function [xn,vn]=UBU_step(x,v,rep,hper2c,inputs,xi)        
    [xn,vn]=U(x,v,hper2c,xi(:,1:rep),xi(:,(rep+1):(2*rep)));
    grad=inputs.grad_lpost(xn);
    vn=vn-(hper2c.h)*grad;
    [xn,vn]=U(xn,vn,hper2c,xi(:,(2*rep+1):(3*rep)),xi(:,(3*rep+1):(4*rep)));
end

function [xn,vn,x2n,v2n]=UBU_step2(x,v, x2,v2,rep,hper4c,inputs,xi)
    [x2n,v2n]=U(x2,v2,hper4c,xi(:,1:rep),xi(:,(rep+1):(2*rep)));
    grad=inputs.grad_lpost(x2n);
    v2n=v2n-hper4c.h*grad;
    [x2n,v2n]=U(x2n,v2n,hper4c,xi(:,(2*rep+1):(3*rep)),xi(:,(3*rep+1):(4*rep)));
    [x2n,v2n]=U(x2n,v2n,hper4c,xi(:,(4*rep+1):(5*rep)),xi(:,(5*rep+1):(6*rep)));
    grad=inputs.grad_lpost(x2n);
    v2n=v2n-hper4c.h*grad;
    [x2n,v2n]=U(x2n,v2n,hper4c,xi(:,(6*rep+1):(7*rep)),xi(:,(7*rep+1):(8*rep)));

    [xn,vn]=U(x,v,hper4c,xi(:,1:rep),xi(:,(rep+1):(2*rep)));
    [xn,vn]=U(xn,vn,hper4c,xi(:,(2*rep+1):(3*rep)),xi(:,(3*rep+1):(4*rep)));
    grad=inputs.grad_lpost(xn);
    vn=vn-hper4c.h*2*grad;
    [xn,vn]=U(xn,vn,hper4c,xi(:,(4*rep+1):(5*rep)),xi(:,(5*rep+1):(6*rep)));
    [xn,vn]=U(xn,vn,hper4c,xi(:,(6*rep+1):(7*rep)),xi(:,(7*rep+1):(8*rep)));
end



function [xn,vn]=burnMCMC(x,v,rep,burn,h,inputs)
        xn=x;vn=v;
        hper2c=hper2const(h,inputs.gam);
        for(it=1:burn)
            xi=randn_sd(inputs.nbeta,4*rep);
            [xn,vn]=UBU_step(xn,vn,rep,hper2c,inputs,xi);
        end
end

function [xn,vn,x2n,v2n]=burnMCMC2(x,v,x2,v2,rep,burn,h,inputs)        
        hper4c=hper2const(h/2,inputs.gam);        
        xn=x;vn=v;x2n=x2;v2n=v2;
        for(it=1:burn)
            xi=randn_sd(inputs.nbeta,8*rep);
            [xn,vn,x2n,v2n]=UBU_step2(xn,vn, x2n,v2n,rep, hper4c,inputs,xi);
        end
end

function [meanx,meanxsquare]=singleMCMC(x,v,rep,niter,h,burn,inputs)
    [xn,vn]=burnMCMC(x,v,rep,burn,h,inputs);
    meanx=zeros_sd(inputs.test_dim,rep);
    meanxsquare=zeros_sd(inputs.test_dim,rep);

    hper2c=hper2const(h,inputs.gam);

    for it=1:niter
        xi=randn_sd(inputs.nbeta,4*rep);
        [xn,vn]=UBU_step(xn,vn,rep,hper2c,inputs,xi);
        test_vals=inputs.test_function(xn);
        % if (mod(it,1)==0)
        %     disp(it)
        %     disp(norm(test_vals(:,1)))
        % end        
        meanx=meanx+test_vals;
        meanxsquare=meanxsquare+test_vals.^2;
    end
    meanx=meanx/niter;
    meanxsquare=meanxsquare/niter;
end

function [meanx,meanxsquare]=doubleMCMC(x,v,x2,v2,rep,niter,h,extraburnin2, jointburnin,thin,inputs)
    [x2n,v2n]=burnMCMC(x2,v2,rep,extraburnin2*2,h/2,inputs);
    [xn,vn,x2n,v2n]=burnMCMC2(x,v,x2n,v2n,rep,jointburnin,h,inputs);
    test_vals1=zeros_sd(inputs.test_dim,rep);
    test_vals2=zeros_sd(inputs.test_dim,rep);

    meanx=zeros_sd(inputs.test_dim,rep);
    meanxsquare=zeros_sd(inputs.test_dim,rep);

    hper2c=hper2const(h,inputs.gam);
    hper4c=hper2const(h/2,inputs.gam);

    for(it=1:floor(niter/thin))
        for(j=1:thin)
            xi=randn_sd(inputs.nbeta,8*rep);
            [xn,vn,x2n,v2n]=UBU_step2(xn,vn, x2n,v2n,rep,hper4c,inputs,xi);
        end
        test_vals1=inputs.test_function(xn);
        test_vals2=inputs.test_function(x2n);
        
        meanx=meanx+test_vals2-test_vals1;
        meanxsquare=meanxsquare+test_vals2.^2-test_vals1.^2;
    end
    meanx=meanx/floor(niter/thin);
    meanxsquare=meanxsquare/floor(niter/thin);
end

function [meanx,meanxsquare]=multiMCMC(x,v,niter,h,extraburnin2, jointburnin, thin, multilevels, repruns, inputs)
    %xnarr=zeros(inputs.nbeta,multilevels+1);
    %vnarr=zeros(inputs.nbeta,multilevels+1);
    %x2narr=zeros(inputs.nbeta,multilevels+1);
    %v2narr=zeros(inputs.nbeta,multilevels+1);
    %coder.varsize("xnarr");
    xnarr=x*ones(1,multilevels+1);
    vnarr=v*ones(1,multilevels+1);
    x2narr=x*ones(1,multilevels+1);
    v2narr=v*ones(1,multilevels+1);

    meanx=zeros_sd(inputs.test_dim,multilevels+1);
    meanxsquare=zeros_sd(inputs.test_dim,multilevels+1);
        
    start_ind=zeros(1,multilevels+1);
    end_ind=zeros(1,multilevels+1);
    start_ind(1)=1;
    end_ind(1)=8;
    for(it=2:(multilevels+1))
        start_ind(it)=end_ind(it-1)+1;
        end_ind(it)=start_ind(it)+8*2^(it-1)-1;
    end


    for it=1:(jointburnin+extraburnin2*(multilevels+1))
        xiarr=gen_xiarr(inputs.nbeta,multilevels,h,inputs.gam); 

        for mit=(multilevels+1):(-1):1

            if(repruns(mit)==1)               
                hper4c=hper2const(h/2^(mit),inputs.gam);                                
                if(it>(multilevels+2-mit)*extraburnin2)
                    for it2=1:(2^(mit-1))
                        [xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit)]=UBU_step2(xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit),1,hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)):(start_ind(mit)+8*it2-1)));           
                    end
                elseif(it>(multilevels+1-mit)*extraburnin2)
                    for it2=1:(2^(mit-1))
                        [x2narr(:,mit),v2narr(:,mit)]=UBU_step(x2narr(:,mit),v2narr(:,mit),1,hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)):(start_ind(mit)+8*(it2-1)+3)));
                        [x2narr(:,mit),v2narr(:,mit)]=UBU_step(x2narr(:,mit),v2narr(:,mit),1,hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)+4):(start_ind(mit)+8*it2-1)));
                    end
                end
            end
        end
    end

    for(it=1:floor(niter/thin))
        for(j=1:thin)
            xiarr=gen_xiarr(inputs.nbeta,multilevels,h,inputs.gam);
            for mit=(multilevels+1):(-1):1
            if(repruns(mit)==1)
                for it2=1:(2^(mit-1))
                        hper4c=hper2const(h/2^(mit),inputs.gam);
                        [xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit)]=UBU_step2(xnarr(:,mit),vnarr(:,mit),x2narr(:,mit),v2narr(:,mit),1,hper4c,inputs,xiarr(:,(start_ind(mit)+8*(it2-1)):(start_ind(mit)+8*it2-1)));
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
