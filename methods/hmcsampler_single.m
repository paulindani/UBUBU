function res=hmcsampler(niter,burnin,numsteps,partial,h,lpost,grad_lpost,test_function,x0)
    
    
    tot_acc=0;
    nbeta=size(x0,1);

    v0=randn(nbeta,1,"single");
    xx=zeros(nbeta,1,"single");
    vv=zeros(nbeta,1,"single");
    xx=x0;vv=v0;

    res=struct;
    test_dim=length(test_function(zeros(nbeta,1)));
    res.test_dim=test_dim;
    res.means=zeros(test_dim,1,"single");
    res.squaremeans=zeros(test_dim,1,"single");
    res.mean_acc=0;
    res.nbeta=nbeta;
    res.ngradtot=(niter)*numsteps;
    
    for(hmc_it=1:burnin)
        [xx,vv,acc]=hmc_step(xx,vv,numsteps,partial);
        %hmc_it
        %av_acc
        tot_acc=tot_acc+acc;
    end
    for(hmc_it=1:niter)
        [xx,vv,acc]=hmc_step(xx,vv,numsteps,partial);
        %hmc_it
        %av_acc
        txx=test_function(xx);
        res.means=res.means+txx;
        res.squaremeans=res.squaremeans+(txx).^2;
        tot_acc=tot_acc+acc;
    end
    mean_acc=tot_acc/(niter+burnin);
    res.mean_acc=mean_acc;
    res.means=res.means/niter;
    res.squaremeans=res.squaremeans/niter;


    function [xn,vn,acc]=hmc_step(x,v,numsteps,partial)
        xn=zeros(nbeta,1,"single");
        vn=zeros(nbeta,1,"single");
        acc=zeros(1,1);
        u=log(rand);
               
        vo=v;xo=x;
        vn=vo;xn=xo;

        vn=vn-(h/2)*grad_lpost(xn);

        nstep=1+geornd(1/(numsteps));
        
        for(leapfrog_it=1:(nstep-1))
            xn=xn+h*vn;
            vn = vn-h*grad_lpost(xn);
        end        
        xn=xn+h*vn;
        vn=vn-(h/2)*grad_lpost(xn);
        vn=-vn;
        acc=(u<(lpost(xo)-lpost(xn)+sum(vo.^2)/2-sum(vn.^2)/2) );
        xn=xn*acc+xo*(1-acc);
        vn=vn*acc+vo*(1-acc);
        
        vn=-vn;
        vn=partial*vn+sqrt(1-partial^2)*randn(nbeta,1,"single");       
    end

end
