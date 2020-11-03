%% Area Under Receiver Operating Characteristic Curve
% for decodign analyses
%% awake data - 6:1
clear

rocAUCu_tmp=nan*zeros(6,9,100); %100 is more than any file population size but each file size is different.
rocAUCb_tmp=nan*zeros(6,9,100);
rocAUCus_tmp=nan*zeros(6,9,100);
rocAUCbs_tmp=nan*zeros(6,9,100);
tic
for a =1:6
    clearvars -except a roc*
    if a==1
        load('cadetv1p384_lda_pnf_2class','pop*','class*','trainsize','testsize','prob*')
    elseif a==2
        load('cadetv1p385_lda_pnf_2class','pop*','class*','trainsize','testsize','prob*')
    elseif a==3
        load('cadetv1p403_lda_pnf_2class','pop*','class*','trainsize','testsize','prob*')
    elseif a==4
        load('cadetv1p432_lda_pnf_2class','pop*','class*','trainsize','testsize','prob*')
    elseif a==5
        load('cadetv1p460_lda_pnf_2class','pop*','class*','trainsize','testsize','prob*')
    elseif a==6
        load('cadetv1p468_lda_pnf_2class','pop*','class*','trainsize','testsize','prob*')
    end
    
    clear pop*
    ori_base=0:20:160;
    x_u=nan*zeros(9,size(class_up_test,2),2,101);%2*testsize*size(class_up_test,3));
    y_u=nan*zeros(9,size(class_up_test,2),2,101);
    t_u=nan*zeros(9,size(class_up_test,2),2,101);
    auc_u=nan*zeros(9,size(class_up_test,2),2);
    x_b=nan*zeros(9,size(class_up_test,2),2,101);
    y_b=nan*zeros(9,size(class_up_test,2),2,101);
    t_b=nan*zeros(9,size(class_up_test,2),2,101);
    auc_b=nan*zeros(9,size(class_up_test,2),2);
    x_us=nan*zeros(9,size(class_up_test,2),2,101);
    y_us=nan*zeros(9,size(class_up_test,2),2,101);
    t_us=nan*zeros(9,size(class_up_test,2),2,101);
    auc_us=nan*zeros(9,size(class_up_test,2),2);
    x_bs=nan*zeros(9,size(class_up_test,2),2,101);
    y_bs=nan*zeros(9,size(class_up_test,2),2,101);
    t_bs=nan*zeros(9,size(class_up_test,2),2,101);
    auc_bs=nan*zeros(9,size(class_up_test,2),2);
    
    for o=1:length(ori_base)
        for s=1:size(class_up_test,2)   % population size
            tmpu=[];
            tmpb=[];
            tmpus=[];
            tmpbs=[];
            labels=[];
            
            labels_tmp=unique(squeeze(class_up_test(o,s,end,:)));
            for r=1:size(class_up_test,3)   % CV repeats (should be 200)
                tmpu=[tmpu; squeeze(prob_up_test(o,s,r,:))];
                tmpb=[tmpb; squeeze(prob_bp_test(o,s,r,:))];
                tmpus=[tmpus; squeeze(prob_up_test_shuf(o,s,r,:))];
                tmpbs=[tmpbs; squeeze(prob_bp_test_shuf(o,s,r,:))];
                labels=[labels; ones(testsize,1)*labels_tmp(1); ones(testsize,1)*labels_tmp(2)];
                
            end
            
            for p=1:2
                if p==1
                    id=o;
                    posclass=min(labels_tmp);
                else
                    id=o+1;
                    if id>length(ori_base)
                        id=1;
                    end
                    posclass=max(labels_tmp);
                    tmpu=1-tmpu;
                    tmpb=1-tmpb;
                    tmpus=1-tmpus;
                    tmpbs=1-tmpbs;
                end
                [x_u(id,s,p,:),y_u(id,s,p,:),t_u(id,s,p,:),auc_u(id,s,p)]=perfcurve(labels,tmpu,posclass,'TVals',0:0.01:1,'UseNearest','off');
                [x_b(id,s,p,:),y_b(id,s,p,:),t_b(id,s,p,:),auc_b(id,s,p)]=perfcurve(labels,tmpb,posclass,'TVals',0:0.01:1,'UseNearest','off');
                [x_us(id,s,p,:),y_us(id,s,p,:),t_us(id,s,p,:),auc_us(id,s,p)]=perfcurve(labels,tmpus,posclass,'TVals',0:0.01:1,'UseNearest','off');
                [x_bs(id,s,p,:),y_bs(id,s,p,:),t_bs(id,s,p,:),auc_bs(id,s,p)]=perfcurve(labels,tmpbs,posclass,'TVals',0:0.01:1,'UseNearest','off');
                % structure: ori, pop size, low neighbor and high neighbor, values
            end
            rocAUCu_tmp(a,o,s)=nanmean(squeeze(auc_u(o,s,:)),'all');
            rocAUCb_tmp(a,o,s)=nanmean(squeeze(auc_b(o,s,:)),'all');
            rocAUCus_tmp(a,o,s)=nanmean(squeeze(auc_us(o,s,:)),'all');
            rocAUCbs_tmp(a,o,s)=nanmean(squeeze(auc_bs(o,s,:)),'all');
            % structure: mean of session x ori x popsize
        end
    end
end
toc
% average auc over all sessions
for o=1:9
    for s=1:100
        rocAUCu(o,s)=nanmean(squeeze(rocAUCu_tmp(:,o,s)),'all');
        rocAUCb(o,s)=nanmean(squeeze(rocAUCb_tmp(:,o,s)),'all');
        rocAUCus(o,s)=nanmean(squeeze(rocAUCus_tmp(:,o,s)),'all');
        rocAUCbs(o,s)=nanmean(squeeze(rocAUCbs_tmp(:,o,s)),'all');
    end
end

figure
supertitle('mean AUROC as a function of population size and orientation')
for e=1:9
    v=e+4;
    if v>9
        v=v-9;
    end
    subplot(3,3,v); hold on; box off
    plot(rocAUCu(e,:),'k')
    plot(rocAUCb(e,:),'r')
    axis square
    title(['Ori: ' num2str(ori_base(e))])
end
xlabel('Population size')
ylabel('AUROC')
%% anesthetized data - 6:1 150ms
for a =1:5
    if a==1
        continue
%         load('141r001p007_awaketime6_lda_pnf_2class','pop*','class*','trainsize','testsize','ori_base','prob')
    elseif a==2
        load('141r001p025_awaketime6_lda_pnf_2class','pop*','class*','trainsize','testsize','ori_base','prob')
    elseif a==3
        load('141r001p039_awaketime6_lda_pnf_2class','pop*','class*','trainsize','testsize','ori_base','prob')
    elseif a==4
        load('142l001p006_awaketime6_lda_pnf_2class','pop*','class*','trainsize','testsize','ori_base','prob')
    elseif a==5
        load('142l001p010_awaketime6_lda_pnf_2class','pop*','class*','trainsize','testsize','ori_base','prob')
           
    end
end