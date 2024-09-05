function [res,grad_per_ess,bootstrap_grad_per_ess]=poisson_multi_approx(niter,burnin,rep,wholerep,hrat,gamrat,repfullgrad,c,bootstrapsamp)
%function [res,grad_per_ess,bootstrap_grad_per_ess]=poisson_multi(niter,burnin,rep,wholerep,hrat,gamrat,bootstrapsamp)

%addpath("C:\Matlab\methods\")


beta_data=load("C:\Matlab\Poisson\beta_min.mat");
reg=beta_data.reg;
reg0=beta_data.reg0;
G=beta_data.G;
beta_min=beta_data.beta_min;
X=beta_data.X;
y=beta_data.y;
numteams=beta_data.numteams;
weeks=beta_data.weeks;
nbeta=2*numteams*weeks;

ind_attack=sub2ind([numteams,weeks],X.attacker,X.week);
ind_defense=sub2ind([numteams,weeks],X.defender,X.week);
attack_map=zeros(numteams*weeks,3);
defense_map=zeros(numteams*weeks,3);
for(it=1:(2*G))
    X_ind=X.attacker(it)+numteams*(X.week(it)-1);
    if(attack_map(X_ind,1)==0)
        attack_map(X_ind,1)=it;
    elseif(attack_map(X_ind,2)==0)
        attack_map(X_ind,2)=it;
    else
        attack_map(X_ind,3)=it;
    end
end
for(it=1:(2*G))
    X_ind=X.defender(it)+numteams*(X.week(it)-1);
    if(defense_map(X_ind,1)==0)
        defense_map(X_ind,1)=it;
    elseif(defense_map(X_ind,2)==0)
        defense_map(X_ind,2)=it;
    else
        defense_map(X_ind,3)=it;
    end
end
attack_map=attack_map+(attack_map==0)*(2*G+1);
defense_map=defense_map+(defense_map==0)*(2*G+1);

Hprior=zeros(weeks,weeks);
for(Hit=1:(weeks-1))
    Hprior(Hit:(Hit+1),Hit:(Hit+1))=Hprior(Hit:(Hit+1),Hit:(Hit+1))+[1,-1;-1,1];
end

pars=struct;
pars.Hprior=Hprior;
pars.numteams=numteams;
pars.weeks=weeks;
pars.reg0=reg0;
pars.reg=reg;
pars.y=y;
pars.ind_defense=ind_defense;
pars.ind_attack=ind_attack;
pars.attack_map=attack_map;
pars.defense_map=defense_map;
pars.nbeta=nbeta;
pars.hessian=sparse(nbeta,nbeta);
pars.G=G;

m=reg0
hessian_prior=hess_lprior(beta_min,pars);
pars.hessian_prior=hessian_prior;
hessian_llik=hess_llik(beta_min,pars);
hessian=hessian_prior+hessian_llik;
[R,FLAG,P] = chol(hessian); %generating Gaussian with precision matrix hessian: P*(R\randn(nbeta,rep));
Mhessian=[sparse(nbeta,nbeta), speye(nbeta,nbeta); -hessian, sparse(nbeta,nbeta)];

M=normest_new(hessian)


h=hrat/realsqrt(M);

gam=realsqrt(m)*gamrat;

options=cell(1,wholerep);
hessian_w=cell(1,wholerep);
Mhessian_w=cell(1,wholerep);
P_w=cell(1,wholerep);
R_w=cell(1,wholerep);
pars_w=cell(1,wholerep);
for wit=1:wholerep
    hessian_w{wit}=hessian;
    Mhessian_w{wit}=Mhessian;
    P_w{wit}=P;
    R_w{wit}=R;
    pars_w{wit}=pars;
    options{wit}=struct;
    options{wit}.r=2*sqrt(2);
    options{wit}.c=c;
    options{wit}.maxlevel=8;
    options{wit}.max_parallel_chain=64;
    options{wit}.repfullgrad=repfullgrad;
    options{wit}.beta_min=beta_min;
    options{wit}.test_dim=nbeta;
    options{wit}.nbeta=nbeta;
    options{wit}.Hprodv=@(x)(hessian_w{wit}*x);
    options{wit}.invcholHprodv=@(x)(P_w{wit}*(R_w{wit}\x));
    options{wit}.exp_hM=@(x,h)(exphMvec(h,Mhessian_w{wit},x,6));
end
    % options=struct;
    % options.r=2*sqrt(2);
    % options.c=c;
    % options.maxlevel=8;
    % options.max_parallel_chain=64;
    % options.repfullgrad=repfullgrad;
    % options.beta_min=beta_min;
    % options.test_dim=nbeta;
    % options.nbeta=nbeta;
    % options.Hprodv=@(x)(hessian*x);
    % options.invcholHprodv=@(x)(P*(R\x));
    % options.exp_hM=@(x,h)(exphMvec(h,Mhessian,x,6));


test_function=@(x)(x);



tic
res=cell(wholerep,1);
parfor it=1:wholerep
    res{it}=multilevel_ubu_approx(niter,burnin,rep,h,gam, @(x)(grad_lpost(x,pars_w{it})),test_function, options{it});
end
%grad_per_ess=0;bootstrap_grad_per_ess=0;
grad_per_ess=process_res_multi(res,1:wholerep,true);
bootstrap_grad_per_ess=bootstrap_res(res,bootstrapsamp,@process_res_multi);
toc

end



    function res=pospart(x)
        res=x.*(x>0);
    end
    function res=softplus(x)
        xl0=(x<0);
        res=(xl0).*log(1+exp(-pospart(-x)))+(1-xl0).*(x+log(1+exp(-pospart(x))));
    end

    function res=softplusder(x)
        res=1./(1+exp(-x));
    end
    function res=softplusder2(x)
        res=1./(2+exp(-x)+exp(x));
    end

    function J = lprior(beta,pars)
        numteams=pars.numteams;
        weeks=pars.weeks;
        reg0=pars.reg0;

        J=reg0/2*sum(beta.^2,1);
        reps=size(beta,2);
        attack=reshape(beta(1:(numteams*weeks),1:reps),[numteams,weeks,reps]);
        defense=reshape(beta((numteams*weeks+1):(2*numteams*weeks),1:reps),[numteams,weeks,reps]);
        J=J+reshape(reg/2*(sum((attack(:,2:weeks,:)-attack(:,1:(weeks-1),:)).^2,[1, 2])+sum((defense(:,2:weeks,:)-defense(:,1:(weeks-1),:)).^2,[1, 2])),[1,reps]);
    end

    function grad = grad_lprior(beta,pars)
        grad=pars.hessian_prior*beta;
        % reps=size(beta,2);
        % attack=reshape(beta(1:(numteams*weeks),1:reps),[numteams,weeks,reps]);
        % defense=reshape(beta((numteams*weeks+1):(2*numteams*weeks),1:reps),[numteams,weeks,reps]);
        % 
        % grad_attack=pagemtimes(attack,Hprior);
        % grad_defense=pagemtimes(defense,Hprior);
        % grad=reg0*beta;
        % grad(1:(numteams*weeks),1:reps)=grad(1:(numteams*weeks),1:reps)+reg*reshape(grad_attack,[numteams*weeks,reps]);
        % grad((numteams*weeks+1):(numteams*weeks*2),1:reps)=grad((numteams*weeks+1):(numteams*weeks*2),1:reps)+reg*reshape(grad_defense,[numteams*weeks,reps]);
    end
    

    function J = llik(beta,pars)
        reps=size(beta,2);
        numteams=pars.numteams;
        weeks=pars.weeks;
        reg0=pars.reg0;
        y=pars.y;

        attack=beta(1:(numteams*weeks),1:reps);
        defense=beta((numteams*weeks+1):(2*numteams*weeks),1:reps);
        xvar=attack(ind_attack,1:reps)+defense(ind_defense,1:reps);
        lambda=softplus(xvar);
        J=sum(-(y*ones(1,reps)).*log(lambda(1:(2*G),1))+lambda(1:(2*G),1:reps),1);
    end

    function grad=grad_llik(beta,pars)
        numteams=pars.numteams;
        weeks=pars.weeks;
        y=pars.y;
        ind_defense=pars.ind_defense;
        ind_attack=pars.ind_attack;
        attack_map=pars.attack_map;
        defense_map=pars.defense_map;

        reps=size(beta,2);
        attack=beta(1:(numteams*weeks),1:reps);
        defense=beta((numteams*weeks+1):(2*numteams*weeks),1:reps);
        xvar=attack(ind_attack,1:reps)+defense(ind_defense,1:reps);
        lambda=softplus(xvar);

        grad_terms=[(-(y*ones(1,reps))./lambda+1).*softplusder(xvar);zeros(1,reps)];
        grad2_attack=reshape(grad_terms(attack_map(:),1:reps),numteams*weeks,3,reps);
        grad2_defense=reshape(grad_terms(defense_map(:),1:reps),numteams*weeks,3,reps);
        
        %grad=[sum(grad2_attack,2);sum(grad2_defense,2)];
        s_attack=squeeze(sum(grad2_attack,2));
        s_defense=squeeze(sum(grad2_defense,2));

        grad=[s_attack;s_defense];

    end

    function hess=hess_lprior(beta,pars)

        numteams=pars.numteams;
        weeks=pars.weeks;
        reg0=pars.reg0;
        nbeta=pars.nbeta;
        Hprior=pars.Hprior;
        reg=pars.reg;

        hess=reg0*speye(nbeta);

        for(teamit=1:(numteams))
            hess(teamit:numteams:(numteams*weeks),teamit:numteams:(numteams*weeks))=...
            hess(teamit:numteams:(numteams*weeks),teamit:numteams:(numteams*weeks))+reg*Hprior;

            hess((numteams*weeks+teamit):numteams:(2*numteams*weeks),(numteams*weeks+teamit):numteams:(2*numteams*weeks))=...
            hess((numteams*weeks+teamit):numteams:(2*numteams*weeks),(numteams*weeks+teamit):numteams:(2*numteams*weeks))+reg*Hprior;

            %hess(((teamit-1)*weeks+1):(teamit*weeks),((teamit-1)*weeks+1):(teamit*weeks))=...
            %hess(((teamit-1)*weeks+1):(teamit*weeks),((teamit-1)*weeks+1):(teamit*weeks))+reg*Hprior;
        end
    end

    function hess=hess_llik(beta,pars)
        numteams=pars.numteams;
        weeks=pars.weeks;
        y=pars.y;
        nbeta=pars.nbeta;
        ind_defense=pars.ind_defense;
        ind_attack=pars.ind_attack;
        G=pars.G;

        hess=sparse(nbeta,nbeta);

        attack=beta(1:(numteams*weeks));
        defense=beta((numteams*weeks+1):(2*numteams*weeks));
        xvar=attack(ind_attack)+defense(ind_defense);
        lambda=softplus(xvar);

        %hess_terms=[(-y./lambda+1).*softplusder2(xvar)+(y./(lambda.^2)).*((softplusder(xvar)).^2);0];
        hess_terms=(-y./lambda+1).*softplusder2(xvar)+(y./(lambda.^2)).*((softplusder(xvar)).^2);
        %hess2_attack=reshape(hess_terms(attack_map(:)),numteams*weeks,3);
        %hess2_defense=reshape(hess_terms(defense_map(:)),numteams*weeks,3);
        
        %s_attack=sum(hess2_attack,2);
        %s_defense=sum(hess2_defense,2);
        ind_defense2=numteams*weeks+ind_defense;
        for(git=1:(2*G))
            hess(ind_attack(git),ind_attack(git))=hess(ind_attack(git),ind_attack(git))+hess_terms(git);
            hess(ind_attack(git),ind_defense2(git))=hess(ind_attack(git),ind_defense2(git))+hess_terms(git);
            hess(ind_defense2(git),ind_attack(git))=hess(ind_defense2(git),ind_attack(git))+hess_terms(git);           
            hess(ind_defense2(git),ind_defense2(git))=hess(ind_defense2(git),ind_defense2(git))+hess_terms(git);                   
        end
        
    end

    function hess=hess_lpost(beta,pars)
        hess=hess_lprior(beta,pars)+hess_llik(beta,pars);
    end

    % function J = lpost(beta)
    %     J=lprior(beta)+llik(beta);
    % end



    function grad = grad_lpost(beta,pars)
        grad=grad_lprior(beta,pars)+grad_llik(beta,pars);
    end

    function r=exphMvec(h,Mx,v,mxiter)
        r=v;
        Mxv=v;
        for(it=1:mxiter)
           Mxv=Mx*Mxv;
           r=r+Mxv*(h^it)/factorial(it);
        end
    end









% 
% %#codegen
% %coder.gpu.kernelfun;
% %addpath("C:\Matlab\methods\")
% %ubu_svrg2_multi_mex(2000,400,128,0.5,1000,'n',-1);
% data_type_const=@()("double");
% to_data_type=@(x)(double(x));
% 
% 
% beta_data=load("C:\Matlab\Poisson\beta_min.mat");
% %beta_data2=coder.load("C:\Matlab\Poisson\beta_hessian.mat");
% reg=to_data_type(beta_data.reg);
% reg0=to_data_type(beta_data.reg0);
% G=to_data_type(beta_data.G);
% beta_min=to_data_type(beta_data.beta_min);
% X=beta_data.X;
% y=to_data_type(beta_data.y);
% numteams=beta_data.numteams;
% weeks=beta_data.weeks;
% nbeta=2*numteams*weeks;
% %hessian=to_data_type(beta_data2.hessian);
% 
% 
% %lambda=eigs(hessian,10,'largestreal','MaxIterations',1000)
% %return;
% %C=to_data_type(beta_data2.C);
% 
% ind_attack=sub2ind([numteams,weeks],X.attacker,X.week);
% ind_defense=sub2ind([numteams,weeks],X.defender,X.week);
% attack_map=to_data_type(zeros(numteams*weeks,3));
% defense_map=to_data_type(zeros(numteams*weeks,3));
% for(it=1:(2*G))
%     X_ind=X.attacker(it)+numteams*(X.week(it)-1);
%     if(attack_map(X_ind,1)==0)
%         attack_map(X_ind,1)=it;
%     elseif(attack_map(X_ind,2)==0)
%         attack_map(X_ind,2)=it;
%     else
%         attack_map(X_ind,3)=it;
%     end
% end
% for(it=1:(2*G))
%     X_ind=X.defender(it)+numteams*(X.week(it)-1);
%     if(defense_map(X_ind,1)==0)
%         defense_map(X_ind,1)=it;
%     elseif(defense_map(X_ind,2)==0)
%         defense_map(X_ind,2)=it;
%     else
%         defense_map(X_ind,3)=it;
%     end
% end
% attack_map=attack_map+(attack_map==0)*(2*G+1);
% defense_map=defense_map+(defense_map==0)*(2*G+1);
% 
% Hprior=zeros(weeks,weeks);
% for(Hit=1:(weeks-1))
%     Hprior(Hit:(Hit+1),Hit:(Hit+1))=Hprior(Hit:(Hit+1),Hit:(Hit+1))+[1,-1;-1,1];
% end
% % 
% % sqrtHprior=sqrtm(reg*Hprior+reg0*eye(weeks));
% % C=sqrthess_lprior(beta_min);
% % beta_min=C*beta_min;
% hessian_prior=hess_lprior(beta_min);
% hessian_llik=hess_llik(beta_min);
% hessian=hessian_prior+hessian_llik;
% 
% 
% % % 
% %  tic
% %  grad2=zeros(nbeta,1000);
% %  for(it=1:1)
% %  grad2=grad2+grad_llik(beta_min+ones(nbeta,1000));
% %  end
% %  sum(sum(abs(grad2)))
% %  toc
% % % 
% % % 
% %  tic
% %  grad3=zeros(nbeta,1000);
% %  for(it=1:1)
% %  grad3=grad3+hessian_prior*ones(nbeta,1000);
% %  end
% %  sum(sum(abs(grad3)))
% %  toc
% % 
% %  tic
% %  grad4=zeros(nbeta,1000);
% %  for(it=1:1)
% %  grad4=grad4+hessian_llik*ones(nbeta,1000);
% %  end
% %  sum(sum(abs(grad4)))
% %  toc
% %  grad_per_ess=0;
% %  return;
% % d = symrcm(hessian);
% % C=chol(hessian(d,d));
% % %sum(sum(abs(C'*(C)-hessian(d,d))))
% % dinv=zeros(1,nbeta);
% % dinv(d)=1:nbeta;
% % C2=C(dinv,dinv);
% % beta_min=C2*beta_min;
% 
% % v=randn(nbeta,10);
% % v1=C2\v;
% % v2=C\v(d,:);v2=v2(dinv,:);
% % err=sum(sum(abs(v1-v2)))
% %return;
% % %sum(sum(abs(C'*(C)-hessian)))
% % save beta_hessian.mat hessian C m M
% 
% % dir=randn(nbeta,1);dir=dir/norm(dir);
% % ep=1e-5;
% % diff1=(lprior(beta_min+ep*dir)-lprior(beta_min-ep*dir))/(2*ep);
% % diff2=sum(dir.*grad_lprior(beta_min));
% % err=abs(diff1-diff2)/abs(diff1)
% % diff1=(grad_lpost(beta_min+ep*dir)-grad_lpost(beta_min-ep*dir))/(2*ep);
% % diff2=(hessian*dir);
% % err=norm(diff2-diff1)/norm(diff1)
% % eH=eig(Hprior*reg+reg0*eye(weeks));
% % min(eH)
% % max(eH)
% 
% % figure, hist(eH)
% 
% %grad_beta_min=grad_llik(beta_min);
% %M=to_data_type(8*reg);
% %hessian=hess_lpost(beta_min);
% m=to_data_type(reg0)
% M=normest_new(hessian)
% 
% h=to_data_type(hrat/realsqrt(M));
% 
% gam=to_data_type(realsqrt(m)*gamrat);
% 
% n=2*G;
% 
% 
% options=struct;
% options.r=2*sqrt(2);
% options.c=c;
% options.maxlevel=8;
% options.max_parallel_chain=16;
% options.nbeta=nbeta;
% options.repfullgrad=repfullgrad;
% options.test_dim=nbeta;
% options.beta_min=beta_min;
% test_function=@(x)(x);
% 
% 
% tic
% res=multilevel_ubu_approx(niter,burnin,rep,wholerep,h,gam, @grad_lpost,@grad_lpost_approx,test_function, options);
% grad_per_ess=process_res_multi(res);
% toc
% 
% 
%     function res=pospart(x)
%         res=x.*(x>0);
%     end
%     function res=softplus(x)
%         xl0=(x<0);
%         res=(xl0).*log(1+exp(-pospart(-x)))+(1-xl0).*(x+log(1+exp(-pospart(x))));
%         
%     end
% 
%     function res=softplusder(x)
%         res=1./(1+exp(-x));
%     end
%     function res=softplusder2(x)
%         res=1./(2+exp(-x)+exp(x));
%     end
% 
%     % function J = lprior(beta)
%     %     J=reg0/2*sum(beta.^2,1);
%     %     reps=size(beta,2);
%     %     attack=reshape(beta(1:(numteams*weeks),1:reps),[numteams,weeks,reps]);
%     %     defense=reshape(beta((numteams*weeks+1):(2*numteams*weeks),1:reps),[numteams,weeks,reps]);
%     %     J=J+reshape(reg/2*(sum((attack(:,2:weeks,:)-attack(:,1:(weeks-1),:)).^2,[1, 2])+sum((defense(:,2:weeks,:)-defense(:,1:(weeks-1),:)).^2,[1, 2])),[1,reps]);
%     % end
% 
%     function grad = grad_lprior(beta)
%         grad=hessian_prior*beta;
%         % reps=size(beta,2);
%         % attack=reshape(beta(1:(numteams*weeks),1:reps),[numteams,weeks,reps]);
%         % defense=reshape(beta((numteams*weeks+1):(2*numteams*weeks),1:reps),[numteams,weeks,reps]);
%         % 
%         % grad_attack=pagemtimes(attack,Hprior);
%         % grad_defense=pagemtimes(defense,Hprior);
%         % grad=reg0*beta;
%         % grad(1:(numteams*weeks),1:reps)=grad(1:(numteams*weeks),1:reps)+reg*reshape(grad_attack,[numteams*weeks,reps]);
%         % grad((numteams*weeks+1):(numteams*weeks*2),1:reps)=grad((numteams*weeks+1):(numteams*weeks*2),1:reps)+reg*reshape(grad_defense,[numteams*weeks,reps]);
%     end
% 
%     % 
%     % function J = llik(beta)        
%     %     reps=size(beta,2);
%     %     attack=beta(1:(numteams*weeks),1:reps);
%     %     defense=beta((numteams*weeks+1):(2*numteams*weeks),1:reps);
%     %     xvar=attack(ind_attack,1:reps)+defense(ind_defense,1:reps);
%     %     lambda=softplus(xvar);
%     %     J=sum(-(y*ones(1,reps)).*log(lambda(1:(2*G),1))+lambda(1:(2*G),1:reps),1);
%     % end
% 
%     function grad=grad_llik(beta)
%         reps=size(beta,2);
%         attack=beta(1:(numteams*weeks),1:reps);
%         defense=beta((numteams*weeks+1):(2*numteams*weeks),1:reps);
%         xvar=attack(ind_attack,1:reps)+defense(ind_defense,1:reps);
%         lambda=softplus(xvar);
% 
%         grad_terms=[(-(y*ones(1,reps))./lambda+1).*softplusder(xvar);zeros(1,reps)];
%         grad2_attack=reshape(grad_terms(attack_map(:),1:reps),numteams*weeks,3,reps);
%         grad2_defense=reshape(grad_terms(defense_map(:),1:reps),numteams*weeks,3,reps);
% 
%         s_attack=squeeze(sum(grad2_attack,2));
%         s_defense=squeeze(sum(grad2_defense,2));
% 
%         grad=[s_attack;s_defense];
%     end
% 
%     function grad = grad_lpost_approx(beta,beta_star,grad_beta_star)
%     grad = grad_beta_star+(hessian)*(beta-beta_star);
% 
%     %truegrad=grad_lpost(beta);
%     %diffgrad=sum(sum(abs(grad-truegrad)))/sum(sum(abs(truegrad)))
% 
%     end 
% 
%     function hess=hess_lprior(beta)
%         hess=reg0*speye(nbeta);
% 
%         for(teamit=1:(numteams))
%             hess(teamit:numteams:(numteams*weeks),teamit:numteams:(numteams*weeks))=...
%             hess(teamit:numteams:(numteams*weeks),teamit:numteams:(numteams*weeks))+reg*Hprior;
% 
%             hess((numteams*weeks+teamit):numteams:(2*numteams*weeks),(numteams*weeks+teamit):numteams:(2*numteams*weeks))=...
%             hess((numteams*weeks+teamit):numteams:(2*numteams*weeks),(numteams*weeks+teamit):numteams:(2*numteams*weeks))+reg*Hprior;
% 
%             %hess(((teamit-1)*weeks+1):(teamit*weeks),((teamit-1)*weeks+1):(teamit*weeks))=...
%             %hess(((teamit-1)*weeks+1):(teamit*weeks),((teamit-1)*weeks+1):(teamit*weeks))+reg*Hprior;
%         end
%     end
% 
% 
%     function hess=hess_llik(beta)
%         hess=sparse(nbeta,nbeta);
% 
%         attack=beta(1:(numteams*weeks));
%         defense=beta((numteams*weeks+1):(2*numteams*weeks));
%         xvar=attack(ind_attack)+defense(ind_defense);
%         lambda=softplus(xvar);
% 
%         hess_terms=(-y./lambda+1).*softplusder2(xvar)+(y./(lambda.^2)).*((softplusder(xvar)).^2);
% 
%         ind_defense2=numteams*weeks+ind_defense;
%         for(git=1:(2*G))
%             hess(ind_attack(git),ind_attack(git))=hess(ind_attack(git),ind_attack(git))+hess_terms(git);
%             hess(ind_attack(git),ind_defense2(git))=hess(ind_attack(git),ind_defense2(git))+hess_terms(git);
%             hess(ind_defense2(git),ind_attack(git))=hess(ind_defense2(git),ind_attack(git))+hess_terms(git);           
%             hess(ind_defense2(git),ind_defense2(git))=hess(ind_defense2(git),ind_defense2(git))+hess_terms(git);                   
%         end
% 
%     end
% 
%     function hess=hess_lpost(beta)
%         hess=hess_lprior(beta)+hess_llik(beta);
%     end
% 
% 
%     function grad = grad_lpost(beta)
%         grad=grad_lprior(beta)+grad_llik(beta);
%     end
% 
% 
% 
% end
% 
% 
% 
% 
% 
% 
