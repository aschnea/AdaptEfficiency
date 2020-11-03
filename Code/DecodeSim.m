%% Decode
clear

load('awake_combined_tuning.mat','tune_u6_r')
prefs=0:2.5:177.5;
oris=prefs;
steps=1:3:length(prefs);
steps_fine=1:0.5:100;
ori_set=[170 10 30;10 30 50;30 50 70;50 70 90;70 90 110;90 110 130;110 130 150;130 150 170;150 170 10];
% ori_set=[0 10 20;20 30 40;40 50 60;60 70 80;80 90 100;100 110 120;120 130 140;140 150 160;160 170 0];
% ori_set=[5 10 15;25 30 35;45 50 55;65 70 75;85 90 95;105 110 115;125 130 135;145 150 155;165 170 175];
% ori_set=[5 10; 10 15; 25 30; 30 35; 45 50; 50 55; 65 70; 70 75; 85 90; 90 95;...
%     105 110; 110 115; 125 130; 130 135; 145 150; 150 155; 165 170; 170 175];
pop_size=2:3:length(prefs);
classtest=[zeros(100,1); ones(100,1); 2*ones(100,1)];
classtrain=[zeros(900,1); ones(900,1); 2*ones(900,1)];
% classtest=[zeros(100,1); ones(100,1)];
% classtrain=[zeros(900,1); ones(900,1)];

perf_p=nan*zeros(9,length(pop_size),5,5,25);
perf_s=nan*zeros(9,length(pop_size),5,5,25);
perf_n=nan*zeros(9,length(pop_size),5,5,25);
perf_b=nan*zeros(9,length(pop_size),5,5,25);
% prob_p=nan*zeros(18,length(pop_size),10,10,25,length(classtest));
% prob_s=nan*zeros(18,length(pop_size),10,10,25,length(classtest));
% prob_n=nan*zeros(18,length(pop_size),10,10,25,length(classtest));
% prob_b=nan*zeros(18,length(pop_size),10,10,25,length(classtest));
% tic


% for a1=[1 10]
%     for s1=[.5 2]
        for b1=3 %[1 3 4]
            for k1=3 %[2 3 5]
                for k2=0%[-3 3]
                    for g1=2.2%[1 2.2]
%                         for g2=[.5 2]
                                
for r=1:10
    clearvars -except prefs oris steps* tune_u6_r r perf* ori_set pop_size class* prob*...
        a1 s1 b1 k1 k2 g1 g2

    tmp=randperm(size(tune_u6_r,1),length(prefs));
%     tmp=1:length(prefs);
    amp=max(tune_u6_r(tmp,:),[],2); % *10 
    spont=min(tune_u6_r(tmp,:),[],2); % +5
    bw=0.2*rand(length(prefs),1);
%     bw2=.33*rand(length(prefs),1);%+0.3*rand(length(prefs),1);

    amp2=amp*1.1;%*a1;
    spont2=spont+1;%s1;%spont+5;
    bw2=bw*b1;
    ks=k1;
    kn=k2;
    gs=g1;
    gn=1;%g2;
        adapt_stim=1-(0.1047*gs)*exp((5.862+ks)*(cos(deg2rad(90-oris))-1)); %+0.0117
        adapt_neur=1-(0.1267*gn)*exp((3.4912+kn)*(cos(deg2rad(90-prefs))-1));%+0.0229
        adapt_both=(1-(0.0631*gn)*exp((3.6366+kn)*(cos(deg2rad(90-prefs))-1)))'.*(1-(0.092*gs)*exp((7.413+ks)*(cos(deg2rad(90-oris))-1)));
    
    % matched kernels:
    % adapt_stim=1-0.2*exp(4*(cos(deg2rad(oris))-1)); %+0.02
    % adapt_neur=1-0.2*exp(4*(cos(deg2rad(prefs))-1));%+0.02
    % adapt_both=(1-0.2*exp(4*(cos(deg2rad(prefs))-1)))'.*(1-0.2*exp(4*(cos(deg2rad(oris))-1)));%+0.02
    
% %     % awake physiology kernels:
%     adapt_stim=1-0.1047*exp(5.862*(cos(deg2rad(90-oris))-1)); %+0.0117
%     adapt_neur=1-0.1267*exp(3.4912*(cos(deg2rad(90-prefs))-1));%+0.0229
% %     adapt_both=(1-0.0997*exp(3.5773*(cos(deg2rad(90-prefs))-1)))'.*(1-0.0608*exp(7.834*(cos(deg2rad(90-oris))-1)));%+0.0254
% % new awake fit - same R2 but stronger stim and weaker neur gain
%     adapt_both=(1-0.0631*exp(3.6366*(cos(deg2rad(90-prefs))-1)))'.*(1-0.092*exp(7.413*(cos(deg2rad(90-oris))-1)));%+0.022
% % Did I reverse the kernels for stim and neuron?:
% %     adapt_both=(1-0.0997*exp(3.5773*(cos(deg2rad(90-oris))-1)))'.*(1-0.0608*exp(7.834*(cos(deg2rad(90-prefs))-1)));%+0.0254
%     % anesthetized physiology kernels:
% %     adapt_stim=1-0.221*exp(6.077*(cos(deg2rad(90-oris))-1)); %+0.0117
% %     adapt_neur=1-0.219*exp(5.011*(cos(deg2rad(90-prefs))-1));%+0.0229
% %     adapt_both=(1-.161*exp(5.247*(cos(deg2rad(90-prefs))-1)))'.*(1-0.162*exp(21*(cos(deg2rad(90-oris))-1)));%+0.0254
    
    tune_pre=zeros(length(prefs),length(oris));
    tune_stim=zeros(length(prefs),length(oris));
    tune_neur=zeros(length(prefs),length(oris));
    resp_pre=zeros(length(prefs),3000);
    resp_stim=zeros(length(prefs),3000);
    resp_neur=zeros(length(prefs),3000);
    resp_both=zeros(length(prefs),3000);
    
    for i=1:length(prefs)
%         tune_pre(i,:)=spont(i)+(amp(i)-spont(i))*exp(bw(i)*(cos(2*deg2rad(oris-prefs(i)))-1));
%         tune_pre(i,:)=(amp(i)-spont(i))*exp(bw(i)*(cos(2*deg2rad(oris-prefs(i)))-1));
        tune_pre(i,:)=exp(bw2(i)*(cos(2*deg2rad(oris-prefs(i)))-1));

%         tune_stim(i,:)=tune_pre(i,:).*adapt_stim+0.0117;
%         tune_neur(i,:)=tune_pre(i,:).*adapt_neur(i)+0.0229;
    end
        tmp=min(tune_pre,[],2);
        tune_pre=tune_pre-tmp; tune_pre=tune_pre.*amp2+spont2;
        tune_both=tune_pre.*adapt_both+0.0254; % awake
%         tune_both=tune_pre.*adapt_both+0.036;  % anesthetized
%         tune_both=tune_pre.*adapt_both+0.022; 
    
    for i=1:length(prefs)
        tune_stim(i,:)=tune_pre(i,:).*adapt_stim+0.0117;
        tune_neur(i,:)=tune_pre(i,:).*adapt_neur(i)+0.0229;
    end
    
    for o=1:size(ori_set,1) % select orientation classes
        o1=find(oris==ori_set(o,1));
        o2=find(oris==ori_set(o,2));
        o3=find(oris==ori_set(o,3));

        for a=1:size(tune_pre,1)    % generate responses to selected oris
            resp_pre(a,:)=[poissrnd(tune_pre(a,o1),1,1000)...
                poissrnd(tune_pre(a,o2),1,1000) poissrnd(tune_pre(a,o3),1,1000)];
            resp_stim(a,:)=[poissrnd(tune_stim(a,o1),1,1000)...
                poissrnd(tune_stim(a,o2),1,1000) poissrnd(tune_stim(a,o3),1,1000)];
            resp_neur(a,:)=[poissrnd(tune_neur(a,o1),1,1000)...
                poissrnd(tune_neur(a,o2),1,1000) poissrnd(tune_neur(a,o3),1,1000)];
            resp_both(a,:)=[poissrnd(tune_both(a,o1),1,1000)...
                poissrnd(tune_both(a,o2),1,1000) poissrnd(tune_both(a,o3),1,1000)];
                
            % 2 class analysis:
%             resp_pre(a,:)=[poissrnd(tune_pre(a,o1),1,1000) poissrnd(tune_pre(a,o2),1,1000)];
%             resp_stim(a,:)=[poissrnd(tune_stim(a,o1),1,1000) poissrnd(tune_stim(a,o2),1,1000)];
%             resp_neur(a,:)=[poissrnd(tune_neur(a,o1),1,1000) poissrnd(tune_neur(a,o2),1,1000)];
%             resp_both(a,:)=[poissrnd(tune_both(a,o1),1,1000) poissrnd(tune_both(a,o2),1,1000)];
        end
        for p=1:length(pop_size)    % select population
            for q=1:4              % # of times pick random cells
                tmp=randperm(length(prefs),pop_size(p));

                for n=1:4          % select x-validation
                    train=[randperm(1000,900) randperm(1000,900)+1000 randperm(1000,900)+2000];
                    test=setdiff(1:3000,train);
%                     train=[randperm(1000,900) randperm(1000,900)+1000];
%                     test=setdiff(1:2000,train);
                    
                    Xpre=resp_pre(tmp,train)';
                    Xs=resp_stim(tmp,train)';
                    Xn=resp_neur(tmp,train)';
                    Xb=resp_both(tmp,train)';
                    obj_p=fitcdiscr(Xpre,classtrain);
                    obj_s=fitcdiscr(Xs,classtrain);
                    obj_n=fitcdiscr(Xn,classtrain);
                    obj_b=fitcdiscr(Xb,classtrain);

                    label_p=predict(obj_p,resp_pre(tmp,test)');
                    label_s=predict(obj_s,resp_stim(tmp,test)');
                    label_n=predict(obj_n,resp_neur(tmp,test)');
                    label_b=predict(obj_b,resp_both(tmp,test)');

                    temp=abs(label_p - classtest);
                    perf_p(o,p,q,n,r)=length(find(temp==0))/length(temp);
                    temp=abs(label_s - classtest);
                    perf_s(o,p,q,n,r)=length(find(temp==0))/length(temp);
                    temp=abs(label_n - classtest);
                    perf_n(o,p,q,n,r)=length(find(temp==0))/length(temp);
                    temp=abs(label_b - classtest);
                    perf_b(o,p,q,n,r)=length(find(temp==0))/length(temp);
                    
%                     [lda_p,err_p,tmp_p]=classify(resp_pre(tmp,test)',Xpre,classtrain);
%                     [lda_s,err_s,tmp_s]=classify(resp_stim(tmp,test)',Xs,classtrain);
%                     [lda_n,err_n,tmp_n]=classify(resp_neur(tmp,test)',Xn,classtrain);
%                     [lda_b,err_b,tmp_b]=classify(resp_both(tmp,test)',Xb,classtrain);
%                     tmpp=find(lda_p'==classtest);
%                     tmps=find(lda_s'==classtest);
%                     tmpn=find(lda_n'==classtest);
%                     tmpb=find(lda_b'==classtest);
%                     perf_p(o,p,q,n,r)=length(tmpp)/length(classtest);
%                     perf_s(o,p,q,n,r)=length(tmps)/length(classtest);
%                     perf_n(o,p,q,n,r)=length(tmpn)/length(classtest);
%                     perf_b(o,p,q,n,r)=length(tmpb)/length(classtest);
%                     prob_p(o,p,q,n,r,:)=tmp_p(:,1);
%                     prob_s(o,p,q,n,r,:)=tmp_s(:,1);
%                     prob_n(o,p,q,n,r,:)=tmp_n(:,1);
%                     prob_b(o,p,q,n,r,:)=tmp_b(:,1);
                end
            end
%             disp([o p])
        end
    end
    disp(r)
end
% AVG over population repeats; AVG over trial x-val; AVG over random neurons w/in population
p=squeeze(nanmean(perf_p,5)); p=squeeze(nanmean(p,4)); p=squeeze(nanmean(p,3));
s=squeeze(nanmean(perf_s,5)); s=squeeze(nanmean(s,4)); s=squeeze(nanmean(s,3));
n=squeeze(nanmean(perf_n,5)); n=squeeze(nanmean(n,4)); n=squeeze(nanmean(n,3));
b=squeeze(nanmean(perf_b,5)); b=squeeze(nanmean(b,4)); b=squeeze(nanmean(b,3));

% 2-class STOPPED HERE - BEGIN WORK HERE

% toc
%             end
%         end
%     end

for e=1:size(ori_set,1)
    [par_u(e,:), gof_u(e,:)]=lda_pnf_fit([5 0.6 0.5],p(e,:),[0.1 0.1 0.1],[]);
    [par_s(e,:), gof_s(e,:)]=lda_pnf_fit([5 0.6 0.5],s(e,:),[0.1 0.1 0.1],[]);
    [par_n(e,:), gof_n(e,:)]=lda_pnf_fit([5 0.6 0.5],n(e,:),[0.1 0.1 0.1],[]);
    [par_b(e,:), gof_b(e,:)]=lda_pnf_fit([5 0.6 0.5],b(e,:),[0.1 0.1 0.1],[]);
%     [par_sh(e,:), gof_sh(e,:)]=lda_pnf_fit([5 0.6 0.5],pnf_sh_train(e,:),[0.1 0.1 0.1],[]);
%     [par_nh(e,:), gof_nh(e,:)]=lda_pnf_fit([5 0.6 0.5],pnf_gh_train(e,:),[0.1 0.1 0.1],[]);
    
    fit_u(e,:)=1-par_u(e,3)*exp(-(steps_fine/par_u(e,1)).^par_u(e,2));
    fit_s(e,:)=1-par_s(e,3)*exp(-(steps_fine/par_s(e,1)).^par_s(e,2));
    fit_n(e,:)=1-par_n(e,3)*exp(-(steps_fine/par_n(e,1)).^par_n(e,2));
    fit_b(e,:)=1-par_b(e,3)*exp(-(steps_fine/par_b(e,1)).^par_b(e,2));
%     fit_sh(e,:)=1-par_sh(e,3)*exp(-(steps_fine/par_sh(e,1)).^par_sh(e,2));
%     fit_nh(e,:)=1-par_nh(e,3)*exp(-(steps_fine/par_nh(e,1)).^par_nh(e,2));
end

% figure; supertitle('stim adapt vs pre')
% % supertitle(num2str([e d c]))
% %data
% for i=1:size(p,1)
%     subplot(3,3,i)
%     plot(steps_fine,fit_s(i,:),'r')       %biased
%     hold on
%     box off
%     plot(steps_fine,fit_u(i,:),'k')
%     axis([0 75 0.3 1])
%     plot(0.33*ones(75,1),':k')
%     title(['Ori: ' num2str(ori_set(i,2))])
% end
% figure; supertitle('neuron adapt vs pre')
% % supertitle(num2str([e d c]))
% %data
% for i=1:size(p,1)
%     subplot(3,3,i)
%     plot(steps_fine,fit_n(i,:),'r')       %biased
%     hold on
%     box off
%     plot(steps_fine,fit_u(i,:),'k')
%     axis([0 75 0.3 1])
%     plot(0.33*ones(75,1),':k')
%     title(['Ori: ' num2str(ori_set(i,2))])
% end
figure; supertitle('both adapt vs pre')
% supertitle(num2str([e d c]))
% %data
for i=1:size(p,1)
    subplot(3,3,i)
    plot(steps_fine,fit_b(i,:),'r')       %biased
    hold on
    box off
    plot(steps_fine,fit_u(i,:),'k')
    axis([0 75 0.3 1])
    plot(0.33*ones(75,1),':k')
    title(['Ori: ' num2str(ori_set(i,2))])
end

for a=1:9
    tmpp(a)=trapz(steps_fine./steps_fine(end),fit_u(a,:));
    tmps(a)=trapz(steps_fine./steps_fine(end),fit_s(a,:));
    tmpn(a)=trapz(steps_fine./steps_fine(end),fit_n(a,:));
    tmpf(a)=trapz(steps_fine./steps_fine(end),fit_b(a,:));
end
AUCp=tmpp;%[tmpp(5) mean([tmpp(4) tmpp(6)]) mean([tmpp(3) tmpp(7)]) mean([tmpp(2) tmpp(8)]) mean([tmpp(1) tmpp(9)])];
AUCs=tmps;%[tmps(5) mean([tmps(4) tmps(6)]) mean([tmps(3) tmps(7)]) mean([tmps(2) tmps(8)]) mean([tmps(1) tmps(9)])];
AUCn=tmpn;%[tmpn(5) mean([tmpn(4) tmpn(6)]) mean([tmpn(3) tmpn(7)]) mean([tmpn(2) tmpn(8)]) mean([tmpn(1) tmpn(9)])];
AUCf=tmpf;%[tmpf(5) mean([tmpf(4) tmpf(6)]) mean([tmpf(3) tmpf(7)]) mean([tmpf(2) tmpf(8)]) mean([tmpf(1) tmpf(9)])];

figure; hold on
plot(AUCs-AUCp,'b')
plot(AUCn-AUCp,'r')
plot(AUCf-AUCp,'g')
plot([0 10],[0 0],'k:')
ylim([-0.03 0.06])
% tmp=[a1, s1, b1];
% tmp=[k1, k2, g1, g2];
% tmp=[1.1, 1, b1, k1, g1];
% title('spont x2')
axis square
set(gca,'TickDir','out','XTick',1:2:9,'XTickLabel',{'-80','-40','0','40','80'})
%                         end
                    end
                end
            end
        end
%     end
% end
stop
clearvars -except prefs oris steps* tune_u6_r r perf* ori_set pop_size class* p s n b par* fit* prob* AUC*
save('GainModels_Decode_3class_fix2')
% 50 reps ~ 12 hours
stop

%% variability of decoder based on each population
% clear; load('GainModels_Decode_3class')
p2=squeeze(nanmean(perf_p,4)); p2=squeeze(nanmean(p2,3));
% s2=squeeze(nanmean(perf_s,4)); s2=squeeze(nanmean(s2,3));
% n2=squeeze(nanmean(perf_n,4)); n2=squeeze(nanmean(n2,3));
b2=squeeze(nanmean(perf_b,4)); b2=squeeze(nanmean(b2,3));

for d=1:size(p2,3)
    for e=1:size(ori_set,1)
        [par_u2(d,e,:), gof_u2(d,e,:)]=lda_pnf_fit([5 0.6 0.5],p2(e,:,d),[0.1 0.1 0.1],[]);
%         [par_s2(d,e,:), gof_s2(d,e,:)]=lda_pnf_fit([5 0.6 0.5],s2(e,:,d),[0.1 0.1 0.1],[]);
%         [par_n2(d,e,:), gof_n2(d,e,:)]=lda_pnf_fit([5 0.6 0.5],n2(e,:,d),[0.1 0.1 0.1],[]);
        [par_b2(d,e,:), gof_b2(d,e,:)]=lda_pnf_fit([5 0.6 0.5],b2(e,:,d),[0.1 0.1 0.1],[]);
        
        fit_u2(d,e,:)=1-par_u2(d,e,3)*exp(-(steps_fine/par_u2(d,e,1)).^par_u2(d,e,2));
%         fit_s2(d,e,:)=1-par_s2(d,e,3)*exp(-(steps_fine/par_s2(d,e,1)).^par_s2(d,e,2));
%         fit_n2(d,e,:)=1-par_n2(d,e,3)*exp(-(steps_fine/par_n2(d,e,1)).^par_n2(d,e,2));
        fit_b2(d,e,:)=1-par_b2(d,e,3)*exp(-(steps_fine/par_b2(d,e,1)).^par_b2(d,e,2));
    end
end
for d=1:size(par_u2,1)
    for a=1:9
        tmpp(d,a)=trapz(steps_fine./steps_fine(end),squeeze(fit_u2(d,a,:)));
%         tmps(d,a)=trapz(steps_fine./steps_fine(end),squeeze(fit_s2(d,a,:)));
%         tmpn(d,a)=trapz(steps_fine./steps_fine(end),squeeze(fit_n2(d,a,:)));
        tmpf(d,a)=trapz(steps_fine./steps_fine(end),squeeze(fit_b2(d,a,:)));
    end
end
AUCp2=tmpp;%[tmpp(5) mean([tmpp(4) tmpp(6)]) mean([tmpp(3) tmpp(7)]) mean([tmpp(2) tmpp(8)]) mean([tmpp(1) tmpp(9)])];
% AUCs2=tmps;%[tmps(5) mean([tmps(4) tmps(6)]) mean([tmps(3) tmps(7)]) mean([tmps(2) tmps(8)]) mean([tmps(1) tmps(9)])];
% AUCn2=tmpn;%[tmpn(5) mean([tmpn(4) tmpn(6)]) mean([tmpn(3) tmpn(7)]) mean([tmpn(2) tmpn(8)]) mean([tmpn(1) tmpn(9)])];
AUCf2=tmpf;%[tmpf(5) mean([tmpf(4) tmpf(6)]) mean([tmpf(3) tmpf(7)]) mean([tmpf(2) tmpf(8)]) mean([tmpf(1) tmpf(9)])];

figure; 
% subplot(131); hold on; supertitle('models AUC variance (50 populations)')
% plot(AUCs2'-AUCp2','b')
% ylim([-0.05 0.1])
% plot([0 10],[0 0],'k:')
% axis square
% subplot(132); hold on
% plot(AUCn2'-AUCp2','r')
% plot([0 10],[0 0],'k:')
% ylim([-0.05 0.1])
% axis square
% subplot(133); 
hold on
plot(AUCf2'-AUCp2','g')
plot([0 10],[0 0],'k:')
ylim([-0.05 0.1])
axis square

save('GainModels_Decode_3class_acute_narrowBW')
%% decode figures

figure; supertitle('Gain Model: LDA PNF (fit)')
% supertitle(num2str([e d c]))
%data
for i=1:size(fit_u,1)
    subplot(3,3,i); hold on
    plot(steps_fine,fit_u(i,:),'k')
    plot(steps_fine,fit_s(i,:),'r')
    plot(steps_fine,fit_n(i,:),'b')
    plot(steps_fine,fit_b(i,:),'g')
    axis square; box off
    axis([0 91 0.3 1])
    plot(0.33*ones(91,1),':k')
    title(['Ori: ' num2str(ori_set(i,2))])
end
xlabel('# of units')
ylabel('Predicted % correct')

figure; supertitle('Weibull parameters - 20deg')
subplot(3,3,1)
hold on; axis square
title('Stimulus gain')
histogram(par_u(:,3),0.75:0.01:0.85,'FaceColor','k')
histogram(par_s(:,3),0.75:0.01:0.85,'FaceColor','r')
plot(mean(par_u(:,3)),5,'kv')
plot(mean(par_s(:,3)),5,'rv')
legend({'Pre','Stimulus gain'},'Location','north')
ylabel('# of cases')
xlabel('Coefficient value')
subplot(3,3,2)
hold on; axis square
title('Stimulus gain')
histogram(par_u(:,1),9:0.2:13,'FaceColor','k')
histogram(par_s(:,1),9:0.2:13,'FaceColor','r')
plot(mean(par_u(:,1)),3,'kv')
plot(mean(par_s(:,1)),3,'rv')
xlabel('denominator value')
subplot(3,3,3)
hold on; axis square
title('Stimulus gain')
histogram(par_u(:,2),0.5:0.01:0.6,'FaceColor','k')
histogram(par_s(:,2),0.5:0.01:0.6,'FaceColor','r')
plot(mean(par_u(:,2)),8,'kv')
plot(mean(par_s(:,2)),8,'rv')
xlabel('exponent value')

subplot(3,3,4)
hold on; axis square
title('Neuron gain')
histogram(par_u(:,3),0.75:0.01:0.85,'FaceColor','k')
histogram(par_n(:,3),0.75:0.01:0.85,'FaceColor','b')
plot(mean(par_u(:,3)),5,'kv')
plot(mean(par_n(:,3)),5,'bv')
legend({'Pre','Neuron gain'},'Location','north')
ylabel('# of cases')
xlabel('Coefficient value')
subplot(3,3,5)
hold on; axis square
title('Neuron gain')
histogram(par_u(:,1),9:0.2:13,'FaceColor','k')
histogram(par_n(:,1),9:0.2:13,'FaceColor','b')
plot(mean(par_u(:,1)),3,'kv')
plot(mean(par_n(:,1)),3,'bv')
xlabel('denominator value')
subplot(3,3,6)
hold on; axis square
title('Neuron gain')
histogram(par_u(:,2),0.5:0.01:0.6,'FaceColor','k')
histogram(par_n(:,2),0.5:0.01:0.6,'FaceColor','b')
plot(mean(par_u(:,2)),8,'kv')
plot(mean(par_n(:,2)),8,'bv')
xlabel('exponent value')

subplot(3,3,7)
hold on; axis square
title('Neuron and Stimulus')
histogram(par_u(:,3),0.75:0.01:0.85,'FaceColor','k')
histogram(par_b(:,3),0.75:0.01:0.85,'FaceColor','g')
plot(mean(par_u(:,3)),5,'kv')
plot(mean(par_b(:,3)),5,'gv')
legend({'Pre','Stimulus and Neuron'},'Location','north')
ylabel('# of cases')
xlabel('Coefficient value')
subplot(3,3,8)
hold on; axis square
title('Neuron and Stimulus')
histogram(par_u(:,1),9:0.2:13,'FaceColor','k')
histogram(par_b(:,1),9:0.2:13,'FaceColor','g')
plot(mean(par_u(:,1)),3,'kv')
plot(mean(par_b(:,1)),3,'gv')
xlabel('denominator value')
subplot(3,3,9)
hold on; axis square
title('Neuron and Stimulus')
histogram(par_u(:,2),0.5:0.01:0.6,'FaceColor','k')
histogram(par_b(:,2),0.5:0.01:0.6,'FaceColor','g')
plot(mean(par_u(:,2)),8,'kv')
plot(mean(par_b(:,2)),8,'gv')
xlabel('exponent value')

% subplot(5,3,10)
% hold on; axis square
% title('stim half')
% histogram(par_u_test(:,3),0.6:0.02:0.8)
% histogram(par_sh_test(:,3),0.6:0.02:0.8)
% ylabel('# of cases')
% xlabel('Coefficient value')
% subplot(5,3,11)
% hold on; axis square
% title('stim half')
% histogram(par_u_test(:,1),6:0.2:10)
% histogram(par_sh_test(:,1),6:0.2:10)
% xlabel('denominator value')
% subplot(5,3,12)
% hold on; axis square
% title('stim half')
% histogram(par_u_test(:,2),0.6:0.02:0.9)
% histogram(par_sh_test(:,2),0.6:0.02:0.9)
% xlabel('exponent value')
% 
% subplot(5,3,13)
% hold on; axis square
% title('neuron half')
% histogram(par_u_test(:,3),0.6:0.02:0.8)
% histogram(par_gh_test(:,3),0.6:0.02:0.8)
% ylabel('# of cases')
% xlabel('Coefficient value')
% subplot(5,3,14)
% hold on; axis square
% title('neuron half')
% histogram(par_u_test(:,1),6:0.2:10) % 20 - 2.8:0.2:4.2
% histogram(par_gh_test(:,1),6:0.2:10) % 10 - 6:0.2:10
% xlabel('denominator value')         % 5 - 26:0.5:32
% subplot(5,3,15)
% hold on; axis square
% title('neuron half')
% histogram(par_u_test(:,2),0.6:0.02:0.9)     % 20/10 - 0.6:0.02:0.9
% histogram(par_gh_test(:,2),0.6:0.02:0.9)    % 5 - 0.5:0.02:0.7
% xlabel('exponent value')
STOP

% figure
% supertitle('Model: Weibull params')
% subplot(331)
% plot(par_u(:,3),par_s(:,3),'.b')
% ylabel('Stimulus gain value')
% refline(1,0)
% axis square; box off
% title('Coefficient')
% subplot(332)
% plot(par_u(:,1),par_s(:,1),'.b')
% refline(1,0)
% title('Denominator')
% axis square; box off
% subplot(333)
% plot(par_u(:,2),par_s(:,2),'.b')
% refline(1,0)
% title('Exponent')
% axis square; box off
% 
% subplot(334)
% plot(par_u(:,3),par_n(:,3),'.r')
% ylabel('Neuron gain value')
% refline(1,0)
% axis square; box off
% subplot(335)
% plot(par_u(:,1),par_n(:,1),'.r')
% refline(1,0)
% axis square; box off
% subplot(336)
% plot(par_u(:,2),par_n(:,2),'.r')
% refline(1,0)
% axis square; box off
% 
% subplot(337)
% plot(par_u(:,3),par_b(:,3),'.g')
% ylabel('Dual gain value')
% refline(1,0)
% axis square; box off
% subplot(338)
% plot(par_u(:,1),par_b(:,1),'.g')
% xlabel('Baseline value')
% refline(1,0)
% axis square; box off
% subplot(339)
% plot(par_u(:,2),par_b(:,2),'.g')
% refline(1,0)
% axis square; box off
% % % % % % % % % % % % % % % % % % % % % 
figure
supertitle('Model: Weibull params')
subplot(331); hold on
plot(par_s(:,3)./par_u(:,3),'.b')
% plot([5 5],[],':k')
ylim([0.95 1.05])
ylabel('Value: Stimulus gain/Pre')
refline(0,1)
axis square; box off
set(gca,'XTick',[1 5 9],'XTickLabel',{'-80','0','80'},'TickDir','out')
title('Coefficient')
subplot(334)
plot(par_s(:,1)./par_u(:,1),'.b')
ylim([0.8 1.1])
refline(0,1)
title('Denominator')
axis square; box off
set(gca,'XTick',[1 5 9],'XTickLabel',{'-80','0','80'},'TickDir','out')
subplot(337)
plot(par_s(:,2)./par_u(:,2),'.b')
ylim([0.95 1.05])
refline(0,1)
title('Exponent')
axis square; box off
set(gca,'XTick',[1 5 9],'XTickLabel',{'-80','0','80'},'TickDir','out')

subplot(332)
plot(par_n(:,3)./par_u(:,3),'.r')
ylabel('Value: Neuron gain/Pre')
ylim([0.95 1.05])
refline(0,1)
axis square; box off
set(gca,'XTick',[1 5 9],'XTickLabel',{'-80','0','80'},'TickDir','out')
subplot(335)
plot(par_n(:,1)./par_u(:,1),'.r')
ylim([0.8 1.1])
refline(0,1)
axis square; box off
set(gca,'XTick',[1 5 9],'XTickLabel',{'-80','0','80'},'TickDir','out')
subplot(338)
plot(par_n(:,2)./par_u(:,2),'.r')
ylim([0.95 1.05])
refline(0,1)
axis square; box off
set(gca,'XTick',[1 5 9],'XTickLabel',{'-80','0','80'},'TickDir','out')

subplot(333)
plot(par_b(:,3)./par_u(:,3),'.g')
ylabel('Value: Dual gain/Pre')
ylim([0.95 1.05])
refline(0,1)
axis square; box off
set(gca,'XTick',[1 5 9],'XTickLabel',{'-80','0','80'},'TickDir','out')
subplot(336)
plot(par_b(:,1)./par_u(:,1),'.g')
ylim([0.8 1.1])
xlabel('Center stimulus orientation')
refline(0,1)
axis square; box off
set(gca,'XTick',[1 5 9],'XTickLabel',{'-80','0','80'},'TickDir','out')
subplot(339)
plot(par_b(:,2)./par_u(:,2),'.g')
ylim([0.95 1.05])
refline(0,1)
axis square; box off
set(gca,'XTick',[1 5 9],'XTickLabel',{'-80','0','80'},'TickDir','out')
%% correlations
clear
load('awake_combined_tuning.mat','tune_u6_r')
% prefs=0:2.5:177.5;
prefs=0:2.5:180;
oris=prefs;
stim=[zeros(1,1000) 20*ones(1,1000) 40*ones(1,1000) 60*ones(1,1000)...
    80*ones(1,1000) 100*ones(1,1000) 120*ones(1,1000) 140*ones(1,1000)...
    160*ones(1,1000)];

tune_pre=zeros(length(prefs),length(oris));
tune_stim=zeros(length(prefs),length(oris));
tune_neur=zeros(length(prefs),length(oris));
tune_sh=zeros(length(prefs),length(oris));
tune_nh=zeros(length(prefs),length(oris));
resp_pre=zeros(length(prefs),length(stim));
resp_stim=zeros(length(prefs),length(stim));
resp_neur=zeros(length(prefs),length(stim));
resp_both=zeros(length(prefs),length(stim));
resp_sh=zeros(length(prefs),length(stim));
resp_nh=zeros(length(prefs),length(stim));
corr_u=nan*zeros(100,length(prefs),length(prefs));
corr_gain=nan*zeros(100,length(prefs),length(prefs));
corr_ssa=nan*zeros(100,length(prefs),length(prefs));
corr_full=nan*zeros(100,length(prefs),length(prefs));
corr_nh=nan*zeros(100,length(prefs),length(prefs));
corr_sh=nan*zeros(100,length(prefs),length(prefs));
corr_sig=nan*zeros(100,length(prefs),length(prefs));
corr_sig_gain=nan*zeros(100,length(prefs),length(prefs));
corr_sig_ssa=nan*zeros(100,length(prefs),length(prefs));
corr_sig_full=nan*zeros(100,length(prefs),length(prefs));
corr_sig_nh=nan*zeros(100,length(prefs),length(prefs));
corr_sig_sh=nan*zeros(100,length(prefs),length(prefs));
    
tic
for n=1:75
    clearvars -except prefs oris tune_u6_r stim corr* n tune* resp*
    %   generate parameters for tuning:
    tmp=randperm(size(tune_u6_r,1),length(prefs));
%     amp=tune_u6_r(tmp,:); amp=3.9*max(amp,[],2)+0.1;
%     spont=tune_u6_r(tmp,:); spont=min(spont,[],2);
    amp=max(tune_u6_r(tmp,:),[],2)*1.1; 
    spont=min(tune_u6_r(tmp,:),[],2)+1;
%     bw=0.33*rand(length(prefs),1)+0.02;
    bw=0.6*rand(length(prefs),1);

    %   generate baseline tuning curves
    for i=1:length(prefs)
        tune_pre(i,:)=spont(i)+(amp(i)-spont(i))*exp(bw(i)*(cos(2*deg2rad(oris-prefs(i)))-1));
    end
    %   generate adapted tuning curves
    adapt_stim=1-0.1047*exp(5.862*(cos(deg2rad(90-oris))-1)); %+0.0117
    adapt_neur=1-0.1267*exp(3.4912*(cos(deg2rad(90-prefs))-1));%+0.0229
%     adapt_both=(1-0.0997*exp(3.5773*(cos(deg2rad(90-prefs))-1)))'.*(1-0.0608*exp(7.834*(cos(deg2rad(90-oris))-1)));%+0.0254
    adapt_both=(1-0.0631*exp(3.6366*(cos(deg2rad(90-prefs))-1)))'.*(1-0.092*exp(7.413*(cos(deg2rad(90-oris))-1)));
    
    adapt_nh=1-0.0997*exp(3.5773*(cos(deg2rad(90-prefs))-1));
    adapt_sh=1-0.0608*exp(7.834*(cos(deg2rad(90-oris))-1));
    for i=1:length(prefs)
        tune_stim(i,:)=tune_pre(i,:).*adapt_stim+0.0117;
        tune_neur(i,:)=tune_pre(i,:).*adapt_neur(i)+0.0229;
        tune_sh(i,:)=tune_pre(i,:).*adapt_sh+0.0254;
        tune_nh(i,:)=tune_pre(i,:).*adapt_nh(i)+0.0254;
    end
    tune_both=tune_pre.*adapt_both+0.0254;
    %   generate responses:
    for a=1:size(tune_pre,1)
        for b=1:length(stim)
            resp_pre(a,b)=poissrnd(tune_pre(a,find(oris==stim(b))));
            resp_stim(a,b)=poissrnd(tune_stim(a,find(oris==stim(b))));
            resp_neur(a,b)=poissrnd(tune_neur(a,find(oris==stim(b))));
            resp_both(a,b)=poissrnd(tune_both(a,find(oris==stim(b))));
            resp_nh(a,b)=poissrnd(tune_nh(a,find(oris==stim(b))));
            resp_sh(a,b)=poissrnd(tune_sh(a,find(oris==stim(b))));
        end
    end

    %   calculate correlations:
    for i = 1:size(resp_pre,1)
        for j=i:size(resp_pre,1)
            corr_u(n,i,j)=akcorrcoef(resp_pre(i,:)',resp_pre(j,:)');
            corr_gain(n,i,j)=akcorrcoef(resp_neur(i,:)',resp_neur(j,:)');
            corr_ssa(n,i,j)=akcorrcoef(resp_stim(i,:)',resp_stim(j,:)');
            corr_full(n,i,j)=akcorrcoef(resp_both(i,:)',resp_both(j,:)');
            corr_nh(n,i,j)=akcorrcoef(resp_nh(i,:)',resp_nh(j,:)');
            corr_sh(n,i,j)=akcorrcoef(resp_sh(i,:)',resp_sh(j,:)');

            corr_sig(n,i,j)=akcorrcoef(tune_pre(i,:)',tune_pre(j,:)');
            corr_sig_gain(n,i,j)=akcorrcoef(tune_neur(i,:)',tune_neur(j,:)');
            corr_sig_ssa(n,i,j)=akcorrcoef(tune_stim(i,:)',tune_stim(j,:)');
            corr_sig_full(n,i,j)=akcorrcoef(tune_both(i,:)',tune_both(j,:)');
            corr_sig_nh(n,i,j)=akcorrcoef(tune_nh(i,:)',tune_nh(j,:)');
            corr_sig_sh(n,i,j)=akcorrcoef(tune_sh(i,:)',tune_sh(j,:)');
        end
    end
    disp(n)
end
toc

%   calculate mean for each ori pref pair
corr_u_m=squeeze(nanmean(corr_u,1));
corr_gain_m=squeeze(nanmean(corr_gain,1));
corr_ssa_m=squeeze(nanmean(corr_ssa,1));
corr_full_m=squeeze(nanmean(corr_full,1));
corr_nh_m=squeeze(nanmean(corr_nh,1));
corr_sh_m=squeeze(nanmean(corr_sh,1));
corr_sig_m=squeeze(nanmean(corr_sig,1));
corr_sig_gain_m=squeeze(nanmean(corr_sig_gain,1));
corr_sig_ssa_m=squeeze(nanmean(corr_sig_ssa,1));
corr_sig_full_m=squeeze(nanmean(corr_sig_full,1));
corr_sig_nh_m=squeeze(nanmean(corr_sig_nh,1));
corr_sig_sh_m=squeeze(nanmean(corr_sig_sh,1));

%   reflect for full matrix:
for i = 1:size(corr_u,2)
    for j=1:size(corr_u,2)
        if i>j
            corr_u_m(i,j)=corr_u_m(j,i);
            corr_gain_m(i,j)=corr_gain_m(j,i);
            corr_ssa_m(i,j)=corr_ssa_m(j,i);
            corr_full_m(i,j)=corr_full_m(j,i);
            corr_nh_m(i,j)=corr_nh_m(j,i);
            corr_sh_m(i,j)=corr_sh_m(j,i);

            corr_sig_m(i,j)=corr_sig_m(j,i);
            corr_sig_gain_m(i,j)=corr_sig_gain_m(j,i);
            corr_sig_ssa_m(i,j)=corr_sig_ssa_m(j,i);
            corr_sig_full_m(i,j)=corr_sig_full_m(j,i);
            corr_sig_nh_m(i,j)=corr_sig_nh_m(j,i);
            corr_sig_sh_m(i,j)=corr_sig_sh_m(j,i);
        end
    end
end

cnv=[0.1 0.3 0.1;0.3 0.5 0.3;0.1 0.3 0.1];
cnv=cnv/sum(cnv(:));

clearvars -except corr* prefs oris stim tune_u6_r cnv
% save('GainModels_corr_temp')
save('GainModels_corr_newBW')
% ~3hr run time
%% correlations figures
figure; supertitle('Gain Models correlation differences')
subplot(4,3,1)
imagesc(corr_gain_m-corr_u_m,[-0.025 0.025])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('Neuron gain - pre')
subplot(4,3,2)
imagesc(corr_ssa_m-corr_u_m,[-0.025 0.025])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('Stimulus gain - pre')
subplot(4,3,3)
imagesc(corr_full_m-corr_u_m,[-0.025 0.025])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('Both - pre')
subplot(4,3,4)
imagesc(corr_nh_m-corr_u_m,[-0.025 0.025])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('nh-pre')
subplot(4,3,5)
imagesc(corr_sh_m-corr_u_m,[-0.025 0.025])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
colorbar
axis square; box off
title('sh-pre')
% signal correlations:
subplot(4,3,7)
imagesc(corr_sig_gain_m-corr_sig_m,[-1 1])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('gain-pre')
subplot(4,3,8)
imagesc(corr_sig_ssa_m-corr_sig_m,[-1 1])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('ssa-pre')
subplot(4,3,9)
imagesc(corr_sig_full_m-corr_sig_m,[-1 1])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('full-pre')
subplot(4,3,10)
imagesc(corr_sig_nh_m-corr_sig_m,[-1 1])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
title('nh-pre')
subplot(4,3,11)
imagesc(corr_sig_sh_m-corr_sig_m,[-1 1])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
colorbar
axis square; box off
title('sh-pre')

% same but convolved:
figure; supertitle('Gain Models correlation differences')
subplot(4,3,1)
imagesc(conv2(corr_gain_m-corr_u_m,cnv,'same'),[-0.02 0.02])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('Neuron gain - pre')
subplot(4,3,2)
imagesc(conv2(corr_ssa_m-corr_u_m,cnv,'same'),[-0.02 0.02])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('Stimulus gain - pre')
subplot(4,3,3)
imagesc(conv2(corr_full_m-corr_u_m,cnv,'same'),[-0.02 0.02])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('Both - pre')
subplot(4,3,4)
imagesc(conv2(corr_nh_m-corr_u_m,cnv,'same'),[-0.02 0.02])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('nh-pre')
subplot(4,3,5)
imagesc(conv2(corr_sh_m-corr_u_m,cnv,'same'),[-0.02 0.02])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
colorbar
axis square; box off
title('sh-pre')
% signal correlations:
subplot(4,3,7)
imagesc(conv2(corr_sig_gain_m-corr_sig_m,cnv,'same'),[-1 1])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('gain-pre')
subplot(4,3,8)
imagesc(conv2(corr_sig_ssa_m-corr_sig_m,cnv,'same'),[-1 1])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('ssa-pre')
subplot(4,3,9)
imagesc(conv2(corr_sig_full_m-corr_sig_m,cnv,'same'),[-1 1])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('full-pre')
subplot(4,3,10)
imagesc(conv2(corr_sig_nh_m-corr_sig_m,cnv,'same'),[-1 1])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
title('nh-pre')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
subplot(4,3,11)
imagesc(conv2(corr_sig_sh_m-corr_sig_m,cnv,'same'),[-1 1])
set(gca,'XTick',[1 37 73],'YTick',[1 37 73],'XTickLabel',{'-90','0','90'},...
    'YTickLabel',{'-90','0','90'},'TickDir','out')
colorbar
axis square; box off
title('sh-pre')

%% tuning curves

clear
load('awake_combined_tuning.mat','tune_u6_r')
% prefs=0:2.5:177.5;
prefs=0:2.5:180;
oris=prefs;

tune_pre=zeros(length(prefs),length(oris));
tune_stim=zeros(length(prefs),length(oris));
tune_neur=zeros(length(prefs),length(oris));
tune_sh=zeros(length(prefs),length(oris));
tune_nh=zeros(length(prefs),length(oris));

%   generate parameters for tuning:
tmp=randperm(size(tune_u6_r,1),length(prefs));
% amp=tune_u6_r(tmp,:); amp=3.9*max(amp,[],2)+0.1;
% spont=tune_u6_r(tmp,:); spont=min(spont,[],2);
% bw=0.33*rand(length(prefs),1)+0.02;
bw=0.6*rand(length(prefs),1);
amp=max(tune_u6_r(tmp,:),[],2)*1.1; 
spont=min(tune_u6_r(tmp,:),[],2);

%   generate baseline tuning curves
for i=1:length(prefs)
    tune_pre(i,:)=spont(i)+(amp(i)-spont(i))*exp(bw(i)*(cos(2*deg2rad(oris-prefs(i)))-1));
end
%   generate adapted tuning curves
adapt_stim=1-0.1047*exp(5.862*(cos(deg2rad(90-oris))-1)); %+0.0117
adapt_neur=1-0.1267*exp(3.4912*(cos(deg2rad(90-prefs))-1));%+0.0229
adapt_both=(1-0.0997*exp(3.5773*(cos(deg2rad(90-prefs))-1)))'.*(1-0.0608*exp(7.834*(cos(deg2rad(90-oris))-1)));%+0.0254
adapt_nh=1-0.0997*exp(3.5773*(cos(deg2rad(90-prefs))-1));
adapt_sh=1-0.0608*exp(7.834*(cos(deg2rad(90-oris))-1));
for i=1:length(prefs)
    tune_stim(i,:)=tune_pre(i,:).*adapt_stim+0.0117;
    tune_neur(i,:)=tune_pre(i,:).*adapt_neur(i)+0.0229;
    tune_sh(i,:)=tune_pre(i,:).*adapt_sh+0.0254;
    tune_nh(i,:)=tune_pre(i,:).*adapt_nh(i)+0.0254;
end
tune_both=tune_pre.*adapt_both+0.0254;
    
figure; supertitle('Example tuning curves and kernels')
subplot(231)
hold on
plot(tune_pre(2:5:end,:)','k')
plot(tune_stim(2:5:end,:)','r')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
xlabel('Stimulus (°)')
ylabel('Response (sp/s)')
title('Stimulus gain')
subplot(232)
hold on
plot(tune_pre(2:5:end,:)','k')
plot(tune_neur(2:5:end,:)','b')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
xlabel('Stimulus (°)')
title('Neuron gain')
subplot(233)
hold on
plot(tune_pre(2:5:end,:)','k')
plot(tune_both(2:5:end,:)','g')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out')
axis square; box off
xlabel('Stimulus (°)')
title('Neuron & stimulus')
subplot(234)
plot(adapt_stim,'r')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out')
hold on; plot([1 73],[1 1],'--k')
ylim([0.8 1.05])
axis square; box off
ylabel('Gain')
xlabel('Stimulus (°)')
subplot(235)
plot(adapt_neur,'b')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out')
hold on; plot([1 73],[1 1],'--k')
ylim([0.8 1.05])
axis square; box off
xlabel('Orientation preference (°)')
subplot(236)
imagesc(adapt_both)
axis square; box off
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out',...
    'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlabel('Stimulus (°)')
ylabel('Orientation preference (°)')

%% Mutual Information/Redundancy

clear
load('awake_combined_tuning.mat','tune_u6_r')
% prefs=0:2.5:177.5;
prefs=0:2.5:180;
oris=prefs;
stim=[zeros(1,1000) 20*ones(1,1000) 40*ones(1,1000) 60*ones(1,1000)...
    80*ones(1,1000) 100*ones(1,1000) 120*ones(1,1000) 140*ones(1,1000)...
    160*ones(1,1000)];

tune_pre=zeros(length(prefs),length(oris));
tune_stim=zeros(length(prefs),length(oris));
tune_neur=zeros(length(prefs),length(oris));
% tune_sh=zeros(length(prefs),length(oris));
% tune_nh=zeros(length(prefs),length(oris));
resp_pre=zeros(length(prefs),length(stim));
resp_stim=zeros(length(prefs),length(stim));
resp_neur=zeros(length(prefs),length(stim));
resp_both=zeros(length(prefs),length(stim));
% resp_sh=zeros(length(prefs),length(stim));
% resp_nh=zeros(length(prefs),length(stim));
MIu=zeros(50,length(prefs));
MIn=zeros(50,length(prefs));
MIs=zeros(50,length(prefs));
MIf=zeros(50,length(prefs));
% MInh=zeros(100,length(prefs));
% MIsh=zeros(100,length(prefs));
MIu_sp=zeros(50,length(prefs));
MIn_sp=zeros(50,length(prefs));
MIs_sp=zeros(50,length(prefs));
MIf_sp=zeros(50,length(prefs));
% MInh_sp=zeros(100,length(prefs));
% MIsh_sp=zeros(100,length(prefs));
MIju=zeros(50,length(prefs)-1,length(prefs));
MIjn=zeros(50,length(prefs)-1,length(prefs));
MIjs=zeros(50,length(prefs)-1,length(prefs));
MIjf=zeros(50,length(prefs)-1,length(prefs));
% MIjnh=zeros(100,length(prefs)-1,length(prefs));
% MIjsh=zeros(100,length(prefs)-1,length(prefs));
redundancy=zeros(50,length(prefs)-1,length(prefs));
redundancy_gain=zeros(50,length(prefs)-1,length(prefs));
redundancy_ssa=zeros(50,length(prefs)-1,length(prefs));
redundancy_full=zeros(50,length(prefs)-1,length(prefs));
% redundancy_gh=zeros(100,length(prefs)-1,length(prefs));
% redundancy_sh=zeros(100,length(prefs)-1,length(prefs));
    

tic
for n=1:50
    clearvars -except prefs oris tune_u6_r stim corr* n redun* MI* tune*
    %   parameters for tuning curves:
    tmp=randperm(size(tune_u6_r,1),length(prefs));
%     amp=tune_u6_r(tmp,:); amp=3.9*max(amp,[],2)+0.1;
%     spont=tune_u6_r(tmp,:); spont=min(spont,[],2);
%     bw=0.33*rand(length(prefs),1)+0.02;
    bw=0.6*rand(length(prefs),1);
    amp=max(tune_u6_r(tmp,:),[],2)*1.1; 
    spont=min(tune_u6_r(tmp,:),[],2)+1;

    
    % matched kernels:
%     adapt_stim=1-0.2*exp(4*(cos(deg2rad(oris))-1)); %+0.02
%     adapt_neur=1-0.2*exp(4*(cos(deg2rad(prefs))-1));%+0.02
%     adapt_both=(1-0.2*exp(4*(cos(deg2rad(prefs))-1)))'.*(1-0.2*exp(4*(cos(deg2rad(oris))-1)));%+0.02
    % awake:
    adapt_stim=1-0.1047*exp(5.862*(cos(deg2rad(90-oris))-1)); %+0.0117
    adapt_neur=1-0.1267*exp(3.4912*(cos(deg2rad(90-prefs))-1));%+0.0229
%     adapt_both=(1-0.0997*exp(3.5773*(cos(deg2rad(90-prefs))-1)))'.*(1-0.0608*exp(7.834*(cos(deg2rad(90-oris))-1)));%+0.0254
    adapt_both=(1-0.0631*exp(3.6366*(cos(deg2rad(90-prefs))-1)))'.*(1-0.092*exp(7.413*(cos(deg2rad(90-oris))-1)));%+0.022
% %     adapt_nh=1-0.0997*exp(3.5773*(cos(deg2rad(90-prefs))-1));
% %     adapt_sh=1-0.0608*exp(7.834*(cos(deg2rad(90-oris))-1));
    % anesthetized:
%     adapt_stim=1-0.22*exp(5*(cos(deg2rad(90-oris))-1)); %+0.0117
%     adapt_neur=1-0.22*exp(6*(cos(deg2rad(90-prefs))-1));%+0.0229
%     adapt_both=(1-0.16*exp(5*(cos(deg2rad(90-prefs))-1)))'.*(1-0.16*exp(21*(cos(deg2rad(90-oris))-1)));%+0.0254
    
    %   generate baseline tuning curves
    for i=1:length(prefs)
        tune_pre(i,:)=spont(i)+(amp(i)-spont(i))*exp(bw(i)*(cos(2*deg2rad(oris-prefs(i)))-1));
        
        tune_stim(i,:)=tune_pre(i,:).*adapt_stim+0.0117;
        tune_neur(i,:)=tune_pre(i,:).*adapt_neur(i)+0.0229;
%         tune_sh(i,:)=tune_pre(i,:).*adapt_sh+0.0254;
%         tune_nh(i,:)=tune_pre(i,:).*adapt_nh(i)+0.0254;
    end
%     tune_both=tune_pre.*adapt_both+0.0254;
    tune_both=tune_pre.*adapt_both+0.036;
    
    %   generate responses from tuning curves for selected stim
    for a=1:size(tune_pre,1)    
        for b=1:length(stim)
            resp_pre(a,b)=poissrnd(tune_pre(a,find(oris==stim(b))));
            resp_stim(a,b)=poissrnd(tune_stim(a,find(oris==stim(b))));
            resp_neur(a,b)=poissrnd(tune_neur(a,find(oris==stim(b))));
            resp_both(a,b)=poissrnd(tune_both(a,find(oris==stim(b))));
%             resp_nh(a,b)=poissrnd(tune_nh(a,find(oris==stim(b))));
%             resp_sh(a,b)=poissrnd(tune_sh(a,find(oris==stim(b))));
        end
    end
    %   calc MI for each neuron
    for i=1:size(tune_pre,1)
        clear tmp* bin r_ent* cond_ent* p pgain psssa pfull pgh psh x*
        bin=0:1:max([resp_pre(i,:) resp_neur(i,:) resp_stim(i,:) resp_both(i,:)]);% resp_nh(i,:) resp_sh(i,:)]);
        % 1d probability matrices for each unit
        p=histc(resp_pre(i,:),bin)/size(resp_pre,2);
        pgain=histc(resp_neur(i,:),bin)/size(resp_pre,2);
        pssa=histc(resp_stim(i,:),bin)/size(resp_pre,2);
        pfull=histc(resp_both(i,:),bin)/size(resp_pre,2);
%         pgh=histc(resp_nh(i,:),bin)/size(resp_pre,2);
%         psh=histc(resp_sh(i,:),bin)/size(resp_pre,2);
        % calculate response entropy
        for j=1:length(bin)
            r_ent(j)=p(j)*log2(p(j));
            r_ent_gain(j)=pgain(j)*log2(pgain(j));
            r_ent_ssa(j)=pssa(j)*log2(pssa(j));
            r_ent_full(j)=pfull(j)*log2(pfull(j));
%             r_ent_gh(j)=pgh(j)*log2(pgh(j));
%             r_ent_sh(j)=psh(j)*log2(psh(j));
        % store entropy 
        E(i)=-1*nansum(r_ent);
        Egain(i)=-1*nansum(r_ent_gain);
        Essa(i)=-1*nansum(r_ent_ssa);
        Efull(i)=-1*nansum(r_ent_full);
%         Egh(i)=-1*nansum(r_ent_gh);
%         Esh(i)=-1*nansum(r_ent_sh);
        end
        % calculate conditional entropy
        tmp=unique(stim);
        for j=1:length(tmp)
            x=find(stim==tmp(j));
            xu=histc(resp_pre(i,x),bin)/length(x);
            xgain=histc(resp_neur(i,x),bin)/length(x);
            xssa=histc(resp_stim(i,x),bin)/length(x);
            xfull=histc(resp_both(i,x),bin)/length(x);
%             xgh=histc(resp_nh(i,x),bin)/length(x);
%             xsh=histc(resp_sh(i,x),bin)/length(x);
            p_stim(j)=length(x)/length(stim);
            for k=1:length(bin)
                cond_ent(j,k)=xu(k)*log2(xu(k));
                cond_ent_gain(j,k)=xgain(k)*log2(xgain(k));
                cond_ent_ssa(j,k)=xssa(k)*log2(xssa(k));
                cond_ent_full(j,k)=xfull(k)*log2(xfull(k));
%                 cond_ent_gh(j,k)=xgh(k)*log2(xgh(k));
%                 cond_ent_sh(j,k)=xsh(k)*log2(xsh(k));
            end
        end
        % store conditional entropy in neuron pref domain:
        CE(i)=sum(p_stim.*nansum(cond_ent,2)');
        CEgain(i)=sum(p_stim.*nansum(cond_ent_gain,2)');
        CEssa(i)=sum(p_stim.*nansum(cond_ent_ssa,2)');
        CEfull(i)=sum(p_stim.*nansum(cond_ent_full,2)');
%         CEgh(i)=sum(p_stim.*nansum(cond_ent_gh,2)');
%         CEsh(i)=sum(p_stim.*nansum(cond_ent_sh,2)');
    end
    MIu(n,:)=E+CE;
    MIn(n,:)=Egain+CEgain;
    MIs(n,:)=Essa+CEssa;
    MIf(n,:)=Efull+CEfull;
%     MInh(n,:)=Egh+CEgh;
%     MIsh(n,:)=Esh+CEsh;
        
    % MI/spike
    MIu_sp(n,:)=MIu(n,:)./mean(resp_pre,2)';
    MIn_sp(n,:)=MIn(n,:)./mean(resp_neur,2)';
    MIs_sp(n,:)=MIs(n,:)./mean(resp_stim,2)';
    MIf_sp(n,:)=MIf(n,:)./mean(resp_both,2)';
%     MInh_sp(n,:)=MInh(n,:)./mean(resp_nh,2)';
%     MIsh_sp(n,:)=MIsh(n,:)./mean(resp_sh,2)';
    
    clear tmp* bin r_ent* cond_ent* p pgain psssa pfull pgh psh x*
    
    %   calculate redundancy of each pair
    for i=1:size(tune_pre,1)-1
        clear tmp* bin* 
        bin=0:1:max([resp_pre(i,:) resp_neur(i,:) resp_stim(i,:) resp_both(i,:)]);% resp_nh(i,:) resp_sh(i,:)]);
        for j=i+1:size(tune_pre,1)
            clear cond_ent* p pgain pssa pgh psh x*
            bin2=0:1:max([resp_pre(j,:) resp_neur(j,:) resp_stim(j,:) resp_both(j,:)]);% resp_nh(j,:) resp_sh(j,:)]);
            
            % joint probability matrix P(x,y)
            p=histcounts2(resp_pre(i,:),resp_pre(j,:),bin,bin2,'Normalization','Probability');
            pgain=histcounts2(resp_neur(i,:),resp_neur(j,:),bin,bin2,'Normalization','Probability');
            pssa=histcounts2(resp_stim(i,:),resp_stim(j,:),bin,bin2,'Normalization','Probability');
            pfull=histcounts2(resp_both(i,:),resp_both(j,:),bin,bin2,'Normalization','Probability');
%             pgh=histcounts2(resp_nh(i,:),resp_nh(j,:),bin,bin2,'Normalization','Probability');
%             psh=histcounts2(resp_sh(i,:),resp_sh(j,:),bin,bin2,'Normalization','Probability');
            
            % calculate joint response entropy
            Ej(i,j)=-1*(nansum(nansum(p.*log2(p))));
            Ej_gain(i,j)=-1*(nansum(nansum(pgain.*log2(pgain))));
            Ej_ssa(i,j)=-1*(nansum(nansum(pssa.*log2(pssa))));
            Ej_full(i,j)=-1*(nansum(nansum(pfull.*log2(pfull))));
%             Ej_gh(i,j)=-1*(nansum(nansum(pgh.*log2(pgh))));
%             Ej_sh(i,j)=-1*(nansum(nansum(psh.*log2(psh))));
            
            % calculate joint conditional entropy
            tmp=unique(stim);
            for k=1:length(tmp)
                x=find(stim==tmp(k));
                p_stim(k)=length(x)/length(stim);
                % joint probability matrix of units i,j for stim k
                xu=histcounts2(resp_pre(i,x),resp_pre(j,x),bin,bin2,'Normalization','Probability');
                xgain=histcounts2(resp_neur(i,x),resp_neur(j,x),bin,bin2,'Normalization','Probability');
                xssa=histcounts2(resp_stim(i,x),resp_stim(j,x),bin,bin2,'Normalization','Probability');
                xfull=histcounts2(resp_both(i,x),resp_both(j,x),bin,bin2,'Normalization','Probability');
%                 xgh=histcounts2(resp_nh(i,x),resp_nh(j,x),bin,bin2,'Normalization','Probability');
%                 xsh=histcounts2(resp_sh(i,x),resp_sh(j,x),bin,bin2,'Normalization','Probability');

                cond_ent(k)=(nansum(nansum(xu.*log2(xu))));
                cond_ent_gain(k)=(nansum(nansum(xgain.*log2(xgain))));
                cond_ent_ssa(k)=(nansum(nansum(xssa.*log2(xssa))));
                cond_ent_full(k)=(nansum(nansum(xfull.*log2(xfull))));
%                 cond_ent_gh(k)=(nansum(nansum(xgh.*log2(xgh))));
%                 cond_ent_sh(k)=(nansum(nansum(xsh.*log2(xsh))));
            end
            CEj(i,j)=nansum(p_stim.*cond_ent);
            CEj_gain(i,j)=nansum(p_stim.*cond_ent_gain);
            CEj_ssa(i,j)=nansum(p_stim.*cond_ent_ssa);
            CEj_full(i,j)=nansum(p_stim.*cond_ent_full);
%             CEj_gh(i,j)=nansum(p_stim.*cond_ent_gh);
%             CEj_sh(i,j)=nansum(p_stim.*cond_ent_sh);

            MIju(n,i,j)=Ej(i,j)+CEj(i,j);
            MIjn(n,i,j)=Ej_gain(i,j)+CEj_gain(i,j);
            MIjs(n,i,j)=Ej_ssa(i,j)+CEj_ssa(i,j);
            MIjf(n,i,j)=Ej_full(i,j)+CEj_full(i,j);
%             MIjnh(n,i,j)=Ej_gh(i,j)+CEj_gh(i,j);
%             MIjsh(n,i,j)=Ej_sh(i,j)+CEj_sh(i,j);
            
            redundancy(n,i,j)=MIju(n,i,j)-(MIu(n,i)+MIu(n,j));
            redundancy_gain(n,i,j)=MIjn(n,i,j)-(MIn(n,i)+MIn(n,j));
            redundancy_ssa(n,i,j)=MIjs(n,i,j)-(MIs(n,i)+MIs(n,j));
            redundancy_full(n,i,j)=MIjf(n,i,j)-(MIf(n,i)+MIf(n,j));
%             redundancy_gh(n,i,j)=MIjnh(n,i,j)-(MInh(n,i)+MInh(n,j));
%             redundancy_sh(n,i,j)=MIjsh(n,i,j)-(MIsh(n,i)+MIsh(n,j));
        end
    end
    
    clearvars -except redundancy* MI* n prefs oris tune_u6_r stim corr* n
    disp(n)
end
toc

% average MI over ori pref
MIu_avg=mean(MIu,1); MIu_std=std(MIu,1);
MIn_avg=mean(MIn,1); MIn_std=std(MIn,1);
MIs_avg=mean(MIs,1); MIs_std=std(MIs,1);
MIf_avg=mean(MIf,1); MIf_std=std(MIf,1);
% MIgh_avg=mean(MInh,1); MIgh_std=std(MInh,1);
% MIsh_avg=mean(MIsh,1); MIsh_std=std(MIsh,1);

% average MI/sp over ori pref
MIu_sp_avg=mean(MIu_sp,1); MIu_sp_std=std(MIu_sp,1);
MIn_sp_avg=mean(MIn_sp,1); MIn_sp_std=std(MIn_sp,1);
MIs_sp_avg=mean(MIs_sp,1); MIs_sp_std=std(MIs_sp,1);
MIf_sp_avg=mean(MIf_sp,1); MIf_sp_std=std(MIf_sp,1);
% MInh_sp_avg=mean(MInh_sp,1); MInh_sp_std=std(MInh_sp,1);
% MIsh_sp_avg=mean(MIsh_sp,1); MIsh_sp_std=std(MIsh_sp,1);

%   Joint MI/Redundancy
% average MI over ori pref
MIju_avg=squeeze(mean(MIju,1));
MIjn_avg=squeeze(mean(MIjn,1)); 
MIjs_avg=squeeze(mean(MIjs,1)); 
MIjf_avg=squeeze(mean(MIjf,1)); 
% MIjnh_avg=squeeze(mean(MIjnh,1)); 
% MIjsh_avg=squeeze(mean(MIjsh,1));
% Redundancy
R_avg=squeeze(mean(redundancy,1));
Rn_avg=squeeze(mean(redundancy_gain,1));
Rs_avg=squeeze(mean(redundancy_ssa,1));
Rf_avg=squeeze(mean(redundancy_full,1));
% Rnh_avg=squeeze(mean(redundancy_gh,1));
% Rsh_avg=squeeze(mean(redundancy_sh,1));
% reflect matrices:
for i=1:length(MIju_avg)
    for j=1:length(MIju_avg)
        if i>j
            MIju_avg(i,j)=MIju_avg(j,i);
            MIjn_avg(i,j)=MIjn_avg(j,i);
            MIjs_avg(i,j)=MIjs_avg(j,i);
            MIjf_avg(i,j)=MIjf_avg(j,i);
%             MIjnh_avg(i,j)=MIjnh_avg(j,i);
%             MIjsh_avg(i,j)=MIjsh_avg(j,i);
            R_avg(i,j)=R_avg(j,i);
            Rn_avg(i,j)=Rn_avg(j,i);
            Rs_avg(i,j)=Rs_avg(j,i);
            Rf_avg(i,j)=Rf_avg(j,i);
%             Rnh_avg(i,j)=Rnh_avg(j,i);
%             Rsh_avg(i,j)=Rsh_avg(j,i);
        end
    end
end

cnv=[0.1 0.3 0.1;0.3 0.5 0.3;0.1 0.3 0.1];
cnv=cnv/sum(cnv(:));
clearvars -except cnv R* MI* prefs oris tune_u6_r stim

save('GainModels_MI_newBW')

% MI and Redudancy figures
figure; supertitle('Gain Models: MI')
subplot(221)
hold on; axis square; box off
plot(MIn_avg-MIu_avg,'b')
plot(MIs_avg-MIu_avg,'r')
plot(MIf_avg-MIu_avg,'g')
% plot(MIgh_avg-MIu_avg,'m')
% plot(MIsh_avg-MIu_avg,'c')
plot(1:length(MIu_avg),zeros(1,length(MIu_avg)),'k--')
legend({'Neuron','Stim','Full','NH','SH'})
xlabel('Orientation preference (°)')
ylabel('Change in MI (adapt-pre)')
title('Single unit MI')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out')
xlim([0 74])

subplot(222)
hold on; axis square; box off
plot(((MIn_avg-MIu_avg)./MIu_avg)*100,'b')
plot(((MIs_avg-MIu_avg)./MIu_avg)*100,'r')
plot(((MIf_avg-MIu_avg)./MIu_avg)*100,'g')
% plot(((MIgh_avg-MIu_avg)./MIu_avg)*100,'m')
% plot(((MIsh_avg-MIu_avg)./MIu_avg)*100,'c')
plot(1:length(MIu_avg),zeros(1,length(MIu_avg)),'k--')
xlabel('Orientation preference (°)')
ylabel('% Change in MI')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out')
xlim([0 74])

% MI/spike
subplot(223)
hold on; axis square; box off
plot(MIn_sp_avg-MIu_sp_avg,'b')
plot(MIs_sp_avg-MIu_sp_avg,'r')
plot(MIf_sp_avg-MIu_sp_avg,'g')
% plot(MInh_sp_avg-MIu_sp_avg,'m')
% plot(MIsh_sp_avg-MIu_sp_avg,'c')
plot(1:length(MIu_sp_avg),zeros(1,length(MIu_sp_avg)),'k--')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out')
xlim([0 74])
legend({'Gain','Ssa','Full','GH','SH'})
xlabel('Orientation preference (°)')
ylabel('Change in MI_s_p_i_k_e (adapt-pre)')
title('neuron-domain')

subplot(224)
hold on; axis square; box off
plot(((MIn_sp_avg-MIu_sp_avg)./MIu_sp_avg)*100,'b')
plot(((MIs_sp_avg-MIu_sp_avg)./MIu_sp_avg)*100,'r')
plot(((MIf_sp_avg-MIu_sp_avg)./MIu_sp_avg)*100,'g')
% plot(((MInh_sp_avg-MIu_sp_avg)./MIu_sp_avg)*100,'m')
% plot(((MIsh_sp_avg-MIu_sp_avg)./MIu_sp_avg)*100,'c')
plot(1:length(MIu_sp_avg),zeros(1,length(MIu_sp_avg)),'k--')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out')
xlim([0 74])
xlabel('Orientation preference (°)')
ylabel('% Change in MI_s_p_i_k_e')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
figure; supertitle('Gain Models: MI_j_o_i_n_t')
subplot(231); hold on; axis square; box off
imagesc(MIju_avg); colorbar;
title('Unadapted');
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(232); hold on; axis square; box off
imagesc(MIjn_avg); colorbar;
title('Neuron gain')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(233); hold on; axis square; box off
imagesc(MIjs_avg); colorbar;
title('Stimulus gain')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(234); hold on; axis square; box off
imagesc(MIjf_avg); colorbar;
title('Neuron and Stimulus')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
% subplot(235); hold on; axis square; box off
% imagesc(MIjnh_avg); colorbar;
% title('Neuron half gain')
% xlabel('Preference, unit 1 (°)')
% ylabel('Preference, unit 2 (°)')
% set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
%     ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
% xlim([0 74]); ylim([0 74])
% subplot(236); hold on; axis square; box off
% imagesc(MIjsh_avg); colorbar;
% title('Stimulus half gain')
% xlabel('Preference, unit 1 (°)')
% ylabel('Preference, unit 2 (°)')
% set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
%     ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
% xlim([0 74]); ylim([0 74])
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
figure; supertitle('Gain Models: MI_j_o_i_n_t diff')
subplot(231); hold on; axis square; box off
imagesc(((MIjn_avg-MIju_avg)./MIju_avg)*100); colorbar;
title('Neuron gain');
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(232); hold on; axis square; box off
imagesc(((MIjs_avg-MIju_avg)./MIju_avg)*100); colorbar;
title('Stimulus gain')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(233); hold on; axis square; box off
imagesc(((MIjf_avg-MIju_avg)./MIju_avg)*100); colorbar;
title('Neuron and stimulus')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
% subplot(234); hold on; axis square; box off
% imagesc(((MIjnh_avg-MIju_avg)./MIju_avg)*100); colorbar;
% title('Neurong half gain')
% xlabel('Preference, unit 1 (°)')
% ylabel('Preference, unit 2 (°)')
% set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
%     ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
% xlim([0 74]); ylim([0 74])
% subplot(235); hold on; axis square; box off
% imagesc(((MIjsh_avg-MIju_avg)./MIju_avg)*100); colorbar;
% title('Stimulus half gain')
% xlabel('Preference, unit 1 (°)')
% ylabel('Preference, unit 2 (°)')
% set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
%     ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
% xlim([0 74]); ylim([0 74])
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
figure; supertitle('Gain Models: MI_j_o_i_n_t diff (convolve)')
subplot(231); hold on; axis square; box off
imagesc(conv2(((MIjn_avg-MIju_avg)./MIju_avg)*100,cnv,'same')); colorbar;
title('Neuron gain');
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(232); hold on; axis square; box off
imagesc(conv2(((MIjs_avg-MIju_avg)./MIju_avg)*100,cnv,'same')); colorbar;
title('Stimulus gain')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(233); hold on; axis square; box off
imagesc(conv2(((MIjf_avg-MIju_avg)./MIju_avg)*100,cnv,'same')); colorbar;
title('Neuron and stimulus')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
% subplot(234); hold on; axis square; box off
% imagesc(conv2(((MIjnh_avg-MIju_avg)./MIju_avg)*100,cnv,'same')); colorbar;
% title('Neurong half gain')
% xlabel('Preference, unit 1 (°)')
% ylabel('Preference, unit 2 (°)')
% set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
%     ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
% xlim([0 74]); ylim([0 74])
% subplot(235); hold on; axis square; box off
% imagesc(conv2(((MIjsh_avg-MIju_avg)./MIju_avg)*100,cnv,'same')); colorbar;
% title('Stimulus half gain')
% xlabel('Preference, unit 1 (°)')
% ylabel('Preference, unit 2 (°)')
% set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
%     ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
% xlim([0 74]); ylim([0 74])
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
figure; supertitle('Gain Models: Synergy')
subplot(231); hold on; axis square; box off
imagesc(R_avg); colorbar;
title('Unadapted')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(232); hold on; axis square; box off
imagesc(Rn_avg); colorbar;
title('Neuron gain')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(233); hold on; axis square; box off
imagesc(Rs_avg); colorbar;
title('Stimulus gain')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(234); hold on; axis square; box off
imagesc(Rf_avg); colorbar;
title('Neuron and stimulus')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
% subplot(235); hold on; axis square; box off
% imagesc(Rnh_avg); colorbar;
% title('Neuron half gain')
% xlabel('Preference, unit 1 (°)')
% ylabel('Preference, unit 2 (°)')
% set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
%     ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
% xlim([0 74]); ylim([0 74])
% subplot(236); hold on; axis square; box off
% imagesc(Rsh_avg); colorbar;
% title('Stimulus half gain')
% xlabel('Preference, unit 1 (°)')
% ylabel('Preference, unit 2 (°)')
% set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
%     ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
% xlim([0 74]); ylim([0 74])
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
figure; supertitle('Gain Models: Syngery diff')
subplot(132); hold on; axis square; box off
imagesc(((Rn_avg-R_avg)./R_avg)*100);% colorbar;
title('Neuron gain');
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(131); hold on; axis square; box off
imagesc(((Rs_avg-R_avg)./R_avg)*100);% colorbar;
title('Stimulus gain')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(133); hold on; axis square; box off
imagesc(((Rf_avg-R_avg)./R_avg)*100);% colorbar;
title('Neuron and stimulus')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
% subplot(234); hold on; axis square; box off
% imagesc(((Rnh_avg-R_avg)./R_avg)*100); colorbar;
% title('Neuron half gain')
% xlabel('Preference, unit 1 (°)')
% ylabel('Preference, unit 2 (°)')
% set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
%     ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
% xlim([0 74]); ylim([0 74])
% subplot(235); hold on; axis square; box off
% imagesc(((Rsh_avg-R_avg)./R_avg)*100); colorbar;
% title('Stimulus half gain')
% xlabel('Preference, unit 1 (°)')
% ylabel('Preference, unit 2 (°)')
% set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
%     ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
% xlim([0 74]); ylim([0 74])
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
figure; supertitle('Gain Models: Syngery diff (convolve)')
subplot(132); hold on; axis square; box off
imagesc(conv2(((Rn_avg-R_avg)./R_avg)*100,cnv,'same'));% colorbar;
title('Neuron gain');
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(131); hold on; axis square; box off
imagesc(conv2(((Rs_avg-R_avg)./R_avg)*100,cnv,'same'));% colorbar;
title('Stimulus gain')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
subplot(133); hold on; axis square; box off
imagesc(conv2(((Rf_avg-R_avg)./R_avg)*100,cnv,'same'));% colorbar;
title('Neuron and stimulus')
xlabel('Preference, unit 1 (°)')
ylabel('Preference, unit 2 (°)')
set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
    ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
xlim([0 74]); ylim([0 74])
% subplot(234); hold on; axis square; box off
% imagesc(conv2(((Rnh_avg-R_avg)./R_avg)*100,cnv,'same')); colorbar;
% title('Neuron half gain')
% xlabel('Preference, unit 1 (°)')
% ylabel('Preference, unit 2 (°)')
% set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
%     ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
% xlim([0 74]); ylim([0 74])
% subplot(235); hold on; axis square; box off
% imagesc(conv2(((Rsh_avg-R_avg)./R_avg)*100,cnv,'same')); colorbar;
% title('Stimulus half gain')
% xlabel('Preference, unit 1 (°)')
% ylabel('Preference, unit 2 (°)')
% set(gca,'XTick',[1 37 73],'XTickLabel',{'-90','0','90'},'TickDir','out'...
%     ,'YTick',[1 37 73],'YTickLabel',{'-90','0','90'})
% xlim([0 74]); ylim([0 74])