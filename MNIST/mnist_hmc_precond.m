function [res,grad_per_ess,bootstrap_grad_per_ess]=mnist_hmc_precond(niter,burnin,rep,numsteps,hrat,bootstrapsamp)
%#codegen
coder.gpu.kernelfun;


to_data_type=@(x)(single(x));


v=load("C:\Matlab\MNIST\mnist.mat");
beta_data=load("C:\Matlab\MNIST\beta_min10_precond.mat");

training=v.training;
test=v.test;
n=length(training.images);  
nbeta=(28*28+1)*10;
npix=28*28+1;

Xd=to_data_type([ones(1,n);reshape(training.images,28*28,n)]');
ndata=n;
X=Xd(1:ndata,:);
n=ndata;

Xp=X';
yd=double(training.labels);
y=yd(1:ndata);
yp=y';
reg=to_data_type(10);

npred=1000;
images2=test.images(:,:,1:npred);
labels2=test.labels(1:npred);
Xpred=to_data_type([ones(1,npred);reshape(images2,28*28,npred)]');
ypred=to_data_type(labels2);
Xpredp=Xpred';

beta_min=to_data_type(beta_data.beta_min);
m=to_data_type(beta_data.m)
M=to_data_type(beta_data.M)
Rinvmx=to_data_type(beta_data.Rinvmx);


pr = prob(beta_min);
ind=(pr>0.05).*(pr<0.95);
nind=sum(ind);



h=to_data_type(hrat/realsqrt(M));
partial=0.7;

test_function=@(x)([x;prob_ind(x,ind)]);

%test_function=@(x)(x);

tic
res=cell(1,rep);

for pit=1:rep
res{pit}=hmcsampler_single(niter,burnin,numsteps, partial,h,@lpost,@grad_lpost,test_function, beta_min);
end
mean_acc=0;
for pit=1:rep
mean_acc=mean_acc+res{pit}.mean_acc;
end
mean_acc=mean_acc/rep


grad_per_ess=process_res_hmc(res);
bootstrap_grad_per_ess=bootstrap_res(res,bootstrapsamp,@process_res_hmc);
toc


    function J = lprior(beta)
        beta=Rinvmx*beta;
        J=reg/2*sum(beta.^2,1);
    end


    function J = llik(beta)
        reps=size(beta,2);
        beta_arr=(reshape(Rinvmx*beta,npix,reps*10))';

        betaX=reshape((beta_arr)*(Xp),10,n*reps);    
        maxbetaX=ones(10,1)*max(betaX,[],1);
        betaX=betaX-maxbetaX;
        expbetaX=exp(betaX);

        s=(reshape(sum(reshape(expbetaX,10,n*reps),1),reps,n));
        yprep=ones(reps,1)*yp;
        idx=1+10*(0:(n*reps-1))+(yprep(:))';
        betayX=(reshape(betaX(idx),reps,n));
        J=(sum(log(s),2)-sum(betayX,2))';
    end

    % function J = llik(beta)
    %     beta=Rinvmx*beta;
    %     reps=size(beta,2);
    %     beta_arr=(reshape(beta,npix,reps*10))';
    %     betaX=beta_arr*Xp;
    %     expbetaX=exp(betaX);
    %     s=(reshape(sum(reshape(expbetaX,10,n*reps),1),reps,n));
    %     yprep=ones(reps,1)*yp;
    %     idx=1+10*(0:(n*reps-1))+(yprep(:))';
    %     betayX=(reshape(betaX(idx),reps,n));
    %     J=(sum(log(s),2)-sum(betayX,2))';
    % end



    function J = lpost(beta)
        J=lprior(beta)+llik(beta);
    end

    function grad=grad_llik(beta)
    beta=Rinvmx*beta;
    reps=size(beta,2);
    beta_arr=(reshape(beta,npix,reps*10))';
    % betaX=beta_arr*Xp;
    % expbetaX=exp(betaX);
    % s10=(reshape(ones(10,1)*sum(reshape(expbetaX,10,n*reps),1),reps*10,n));
    % rat=(expbetaX./s10);
    betaX=reshape(beta_arr*Xp,10,n*reps);    
    maxbetaX=ones(10,1)*max(betaX,[],1);
    betaX=betaX-maxbetaX;
    expbetaX=exp(betaX);
    s10=reshape(ones(10,1)*sum(expbetaX,1),reps*10,n);    
    rat=reshape(expbetaX,reps*10,n)./s10;
    yprep=ones(reps,1)*yp;
    idx=1+10*(0:(n*reps-1))+(yprep(:))';
    rat(idx)=rat(idx)-1;
    grad=reshape(Xp*rat',nbeta,reps);
    grad=Rinvmx*grad;    
    end

    function grad = grad_lprior(beta)
        beta=Rinvmx*beta;
        grad=reg*beta;
        grad=Rinvmx*grad;
    end
    
    function grad = grad_lpost(beta)
        grad=grad_lprior(beta)+grad_llik(beta);
    end


    function pr = prob(beta)       
        beta=Rinvmx*beta;
        reps=size(beta,2);
        npred=size(Xpred,1);       
        beta_arr=(reshape(beta,npix,reps*10))';
        betaX=reshape(beta_arr*Xpredp,10,npred*reps);    
        maxbetaX=ones(10,1)*max(betaX,[],1);
        expbetaX=exp(betaX-maxbetaX);
        s10=ones(10,1)*sum(expbetaX,1);
        p=reshape(expbetaX./s10,10,reps,npred);
        pr=reshape(permute(p,[1,3,2]),10*npred,reps);
    end

    function pr_ind = prob_ind(beta,ind)       
        pr_all=prob(beta);
        pr_ind=pr_all(ind==1,:);
    end
end






