function plothist(grad_per_ess, name_of_title,xpos,ypos)
figure('Position',[20 20 820 500]), histogram(grad_per_ess)
fontsize(16,"points")
title(name_of_title)
minlabel=sprintf('Min grads/ESS -- %.2f', min(grad_per_ess));
maxlabel=sprintf('Max grads/ESS -- %.2f', max(grad_per_ess));
% Create the textbox
if(nargin==2)
    xpos=0.6;
end
if(nargin<=3)
    ypos=0.75;
end
h=annotation('textbox',[xpos ypos 0.1 0.1]);
        set(h,'String',{minlabel, maxlabel},'FontSize',16);
xlabel('Gradients/ESS amongst all coordinates')
ylabel('Number of coordinates')
end