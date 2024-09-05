function grad_per_ess=mnist_check
%#codegen
coder.gpu.kernelfun;
%addpath("C:\Matlab\methods\")
%ubu_svrg2_multi_mex(2000,400,128,0.5,1000,'n',-1);

data_type_const=@()("double");
to_data_type=@(x)(double(x));


v=load("C:\Matlab\MNIST\mnist.mat");
beta_data=load("C:\Matlab\MNIST\beta_min10.mat");

beta_data0=load("C:\Matlab\MNIST\beta_min10n.mat");
%beta_data=load("C:\Matlab\MNIST\beta_min10_precond.mat");
hessian=beta_data0.hessian;

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
yd=to_data_type(training.labels);
y=yd(1:ndata);
yp=y';
reg=to_data_type(10);

npred=length(test.images);  
Xpred=to_data_type([ones(1,npred);reshape(test.images,28*28,npred)]');
ypred=to_data_type(test.labels);
Xpredp=Xpred';

beta_min=to_data_type(beta_data.beta_min);
% hess = hess_lpost(beta_min);
% hess=(hess+hess')/2;
%[vH,dH]=eig((hessian+hessian')/2);
%eH=diag(real(dH))
%m=min(eH)
%M=max(eH)

% der3=zeros(1,length(eH));
% for(k=1:length(eH))
%     ep=0.01;
%     xplus=beta_min+vH(:,k)*ep;
%     xplus2=beta_min+2*vH(:,k)*ep;
%     xminus=beta_min-vH(:,k)*ep;
%     xminus2=beta_min-2*vH(:,k)*ep;
%     der3(k)=(1/(2*ep^3))*(lpost(xplus2)-lpost(xminus2) - 2* lpost(xplus) + 2 * lpost(xminus))/(eH(k)^(3/2));
% end
% [max3,ind3]=max(abs(der3));
% der3(ind3)
% ind3
% return
% der3=zeros(1,nbeta);
% vH=eye(nbeta);
% for(k=1:nbeta)
%     ep=0.01;
%     xplus=beta_min+vH(:,k)*ep;
%     xplus2=beta_min+2*vH(:,k)*ep;
%     xminus=beta_min-vH(:,k)*ep;
%     xminus2=beta_min-2*vH(:,k)*ep;
%     der3(k)=(1/(2*ep^3))*(lpost(xplus2)-lpost(xminus2) - 2* lpost(xplus) + 2 * lpost(xminus))/(hessian(k,k)^(3/2));
% end
% [max3,ind3]=max(abs(der3));
% der3(ind3)
% ind3
% return

%k=7819;
%eH(k)
k=7491;
eH=diag(hessian);
vH=eye(nbeta);
%sum(eH<=eH(k))
%norm(hessian*vH(:,k)-eH(k)*vH(:,k))

nplot=200;
t=linspace(-3/sqrt(eH(k)),3/sqrt(eH(k)),nplot);
xt=beta_min*ones(1,nplot)+vH(:,k)*t;

figure, plot(t,lpost(xt)-lpost(beta_min),"Color","red");
fontsize(12,"points")

hold on;
plot(t,t.^2*eH(k)/2,"Color","blue", ...
    "LineStyle", "--");

title("Comparison of potential and Gaussian approximation - MNIST")

legend('Potential','Gaussian','FontSize',12,'Location','SouthWest');
%h=annotation('textbox',[0.5 0.6 0.1 0.1]);
%        set(h,'String',{'Potential', 'Gaussian'},'FontSize',16);
xlabel('t:[-3 * sd of Gaussian , 3*sd of Gaussian]')
ylabel('Value of potential/approximation')

figure, plot(t,lpost(beta_min)+t.^2*eH(k)/2-lpost(xt))
fontsize(12,"points")
xlabel('t:[-3 * sd of Gaussian , 3*sd of Gaussian]')
ylabel('Value of difference')
title("Difference of potential and Gaussian approximation - MNIST")

%figure, histogram(eH)
return;
%m=to_data_type(reg)
%M=to_data_type(beta_data.M)



    function J = lprior(beta)
        J=reg/2*sum(beta.^2,1);
    end


    function J = llik(beta)
        reps=size(beta,2);
        beta_arr=(reshape(beta,npix,reps*10))';
        betaX=beta_arr*Xp;
        expbetaX=exp(betaX);
        s=(reshape(sum(reshape(expbetaX,10,n*reps),1),reps,n));
        yprep=ones(reps,1)*yp;
        idx=1+10*(0:(n*reps-1))+(yprep(:))';
        betayX=(reshape(betaX(idx),reps,n));
        J=(sum(log(s),2)-sum(betayX,2))';
    end



    function J = lpost(beta)
        J=lprior(beta)+llik(beta);
    end

    function grad=grad_llik(beta)       
    reps=size(beta,2);
    beta_arr=(reshape(beta,npix,reps*10))';
    betaX=beta_arr*Xp;
    expbetaX=exp(betaX);
    s10=(reshape(ones(10,1)*sum(reshape(expbetaX,10,n*reps),1),reps*10,n));
    rat=(expbetaX./s10);
    yprep=ones(reps,1)*yp;
    idx=1+10*(0:(n*reps-1))+(yprep(:))';
    rat(idx)=rat(idx)-1;
    grad=reshape(Xp*rat',nbeta,reps);
    end

    function grad = grad_lprior(beta)
        grad=reg*beta;
    end
    
    function grad = grad_lpost(beta)
        grad=grad_lprior(beta)+grad_llik(beta);
    end

    % function hess = hess_lpost(beta)
    % betaarr=(reshape(beta,npix,10))';
    % betaX=betaarr*Xp;
    % expbetaX=exp(betaX);
    % sum_expbetaX=sum(expbetaX,1);
    % hess=zeros(nbeta);
    % for(l=1:10)
    %     l
    %     hess(((l-1)*npix+1):(l*npix),((l-1)*npix+1):(l*npix))=X'*diag(expbetaX(l,:)./(sum_expbetaX))*X;
    % end
    % for(l=1:10)
    %     l
    %     for(k=1:10)
    %         hess(((l-1)*npix+1):(l*npix),((k-1)*npix+1):(k*npix))=hess(((l-1)*npix+1):(l*npix),((k-1)*npix+1):(k*npix))-X'*diag(expbetaX(l,:).*expbetaX(k,:)./(sum_expbetaX.^2))*X;
    %     end
    % end
    % hess=hess+reg*eye(nbeta);
    % end

end






