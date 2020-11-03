%% pseudopopulation LDA OSI breakdown
    
%% store responses and OSI. 
% Create matched trial distributions between awake and anesthetized
clear

oribias_w_all=[];   %OSI
oripref_w_all=[];   %ori pref
oribias_c_all=[];
oripref_c_all=[];
resp_u_w=[];        % resp matrix w/ matched trials of each stim
resp_b_w=[];
resp_u_c=[];
resp_b_c=[];

% Acute files:
for n=1:5
    if n==1
%         load('142l001p006_awaketime6_entropy','resp*','keep*','base','bias','ori*');
        load('142l001p006_awaketime6_preprocess','resp_raw*','base','bias','ori*');
    elseif n==2
%         load('142l001p010_awaketime6_entropy','resp*','keep*','base','bias','ori*');
        load('142l001p010_awaketime6_preprocess','resp_raw*','base','bias','ori*');
    elseif n==3
%         load('141r001p039_awaketime6_entropy','resp*','keep*','base','bias','ori*');
        load('141r001p039_awaketime6_preprocess','resp_raw*','base','bias','ori*');
    elseif n==4
%         load('141r001p025_awaketime6_entropy','resp*','keep*','base','bias','ori*');
        load('141r001p025_awaketime6_preprocess','resp_raw*','base','bias','ori*');
    elseif n==5
%         load('141r001p007_awaketime6_entropy','resp*','keep*','base','bias','ori*');
        load('141r001p007_awaketime6_preprocess','resp_raw*','base','bias','ori*');
    end
    
    clearvars -except n ori* keep* resp* stimrefs* base bias
    % exclude unresponsive units:
    tmp=sum(resp_raw_base,2)>75;
    tmp2=sum(resp_raw_bias,2)>75;
    tmp3=find(tmp==1 & tmp2==1);
    resp_raw_base=resp_raw_base(tmp3,:);
    resp_raw_bias=resp_raw_bias(tmp3,:);
    oribias=oribias(tmp3);
    oripref=oripref(tmp3);
    
    % store OSI and ori pref:
    oribias_c_all=[oribias_c_all oribias];
    oripref_c_all=[oripref_c_all oripref];
    
    % find bias ori
    val_mode=mode(bias);
    % align biased ori to first column
    tmp=find(val_mode==ori_base);
    val_shift=1-tmp;
    oris=circshift(ori_base(1:9),[0,val_shift]);
    
    %   make a uniform response distribution in both environments:
    a=histc(bias,ori_base);
    min_count=min(a);   % Trials of each ori to create distribution
    for j=1:10
        tmp1=[];
        tmp2=[];
        for i=1:length(ori_base)-1
            % trials to keep from uniform
            tmp=find(base==ori_base(i));
            tmp1=[tmp1;tmp(randperm(min_count,451))']; %451 is pre-determined minimum # of trials across all recordings
            % trials to keep from bias
            temp=find(bias==ori_base(i));
            tmp2=[tmp2;temp(randperm(min_count,451))'];
        end
        keep(:,j)=tmp1;  % distribution for uniform trials
        keep2(:,j)=tmp2; % same distribution for bias trials
        tmp_u(:,:,j)=resp_raw_base(:,keep(:,j));
        tmp_b(:,:,j)=resp_raw_bias(:,keep2(:,j));
    end
    resp_u_c=cat(1,tmp_u,resp_u_c);
    resp_b_c=cat(1,tmp_b,resp_b_c);
    
%     % keep same # of trials of each class/ori across all recordigns
%     k1=[];
%     k2=[];
%     tmp=length(keep)/9; % # of trials of each class/ori in current loop
%     for b=1:length(ori_base)-1
%         a=find(oris==ori_base(b));
%         tmp2=(a-1)*tmp+1;
%         k1=[k1; keep(tmp2:tmp2+450)]; 
%         k2=[k2; keep2(tmp2:tmp2+450)]; 
%         % these are the indices ^
%     end
%     resp_u_c=[resp_u_c; resp_raw_base(:,k1)]; % resp matrix w/ matched trials of each stim
%     resp_b_c=[resp_b_c; resp_raw_bias(:,k2)];
    
    if n==1
        for b=1:length(ori_base)-1
            stimrefs_c(b,:)=ori_base(b)+zeros(1,451);
        end
    end
end
clear tmp*
stimrefs_c_v=[stimrefs_c(1,:) stimrefs_c(2,:) stimrefs_c(3,:) stimrefs_c(4,:)...
    stimrefs_c(5,:) stimrefs_c(6,:) stimrefs_c(7,:) stimrefs_c(8,:) stimrefs_c(9,:)];

% Awake files:
for n=1:6
    if n==1
        load('cadetv1p384_tuning','oris_u','oris_b','orip*','orib*','resp*');
%         load('cadetv1p384_entropy');
    elseif n==2
        load('cadetv1p385_tuning','oris_u','oris_b','orip*','orib*','resp*');
%         load('cadetv1p385_entropy');
    elseif n==3
        load('cadetv1p403_tuning','oris_u','oris_b','orip*','orib*','resp*');
%         load('cadetv1p403_entropy');
    elseif n==4
        load('cadetv1p432_tuning','oris_u','oris_b','orip*','orib*','resp*');
%         load('cadetv1p432_entropy');
    elseif n==5
%         load('cadetv1p437_entropy)';
        load('cadetv1p460_tuning','oris_u','oris_b','orip*','orib*','resp*');
%         load('cadetv1p460_entropy');
    elseif n==6
        load('cadetv1p468_tuning','oris_u','oris_b','orip*','orib*','resp*');
%         load('cadetv1p468_entropy');
    end

    clearvars -except n ori* keep* resp* stimrefs*
    % exclude unresponsive units:
    tmp=sum(resp_uniform,2)>75;
    tmp2=sum(resp_bias,2)>75;
    tmp3=find(tmp==1 & tmp2==1);
    resp_uniform=resp_uniform(tmp3,:);
    resp_bias=resp_bias(tmp3,:);
    oribias_u=oribias_u(tmp3);
    oripref_u=oripref_u(tmp3);
    
    % store OSI and ori pref:
    oribias_w_all=[oribias_w_all oribias_u];
    oripref_w_all=[oripref_w_all oripref_u];
    
    % find bias ori
    val_mode=mode(oris_b);
    % align biased ori to first column
    tmp=find(val_mode==ori_base);
    val_shift=1-tmp;
    oris=circshift(ori_base(1:9),[0,val_shift]);
    
    %   make a uniform response distribution in both environments:
    a=histc(oris_b,ori_base(1:9));
    min_count=min(a);   % Trials of each ori to create distribution
    for j=1:10
        tmp1=[];
        tmp2=[];
        for i=1:length(ori_base)-1
            % trials to keep from uniform
            tmp=find(oris_u==ori_base(i));
            tmp1=[tmp1;tmp(randperm(min_count,144))']; %451 is pre-determined minimum # of trials across all recordings
            % trials to keep from bias
            temp=find(oris_b==ori_base(i));
            tmp2=[tmp2;temp(randperm(min_count,144))'];
        end
        keep_w(:,j)=tmp1;  % distribution for uniform trials
        keep2_w(:,j)=tmp2; % same distribution for bias trials
        tmp_u(:,:,j)=resp_uniform(:,keep_w(:,j));
        tmp_b(:,:,j)=resp_bias(:,keep2_w(:,j));
    end
    resp_u_w=cat(1,tmp_u,resp_u_w);
    resp_b_w=cat(1,tmp_b,resp_b_w);
    
%     % keep same # of trials of each class/ori across all recordigns
%     k1=[];
%     k2=[];
%     tmp=length(keep)/9; % # of trials of each class/ori in current loop
%     for b=1:length(oris)
%         tmp2=(b-1)*tmp+1;
%         k1=[k1; keep(tmp2:tmp2+143)]; %144 is pre-determined minimum # of trials across all recordings
%         k2=[k2; keep2(tmp2:tmp2+143)]; 
%         % these are the indices ^
%     end
%     resp_u_w=[resp_u_w; resp_uniform(:,k1)]; % resp matrix w/ matched trials of each stim
%     resp_b_w=[resp_b_w; resp_bias(:,k2)];
    if n==1
        for b=1:length(oris)
            stimrefs_w(b,:)=oris(b)+zeros(1,144);
        end
    end
    
end
stimrefs_w_v=[stimrefs_w(1,:) stimrefs_w(2,:) stimrefs_w(3,:) stimrefs_w(4,:)...
    stimrefs_w(5,:) stimrefs_w(6,:) stimrefs_w(7,:) stimrefs_w(8,:) stimrefs_w(9,:)];

clearvars -except stimrefs_w* resp_u_w resp_b_w oribias_w_all oripref_w_all oris...
    resp_u_c resp_b_c oribias_c_all oripref_c_all stimrefs_c*

%% bins
bins=0:0.05:1;
acute_bins=histc(oribias_c_all,bins);
awake_bins=histc(oribias_w_all,bins);
% fixed bins
fbins=[0:0.05:0.25;...
    0.25:0.05:0.50;...
    0.50:0.05:0.75;...
    0.75:0.05:1];

% sliding bins
sbins=[0.1:0.05:0.35;...
    0.2:0.05:0.45;...
    0.3:0.05:0.55;...
    0.4:0.05:0.65;...
    0.5:0.05:0.75;...
    0.6:0.05:0.85;...
    0.7:0.05:0.95];

% adaptive bins (same # of units in each OSI bin) 
% acute: 65 units per 1/4
% awake: 106 units per 1/4
% tmp=[histc(oribias_c_all,bins);histc(oribias_w_all,bins)];
% tmp2=min(tmp);
% tmp3=sum(tmp2);
% abins

%% pdf plot
figure; hold on
plot(1:21,awake_bins./length(oribias_w_all))
plot(1:21,acute_bins./length(oribias_c_all))
axis square
xlabel('OSI')
ylabel('proportion of units')
legend('awake','anesthetized')
title('OSI pdf')
set(gca,'XTick',[1 6 11 16 21],'XTickLabel',{'0','0.25','0.5','0.75','1'},'TickDir','out')

%% run LDA
% Shuffle trials so no correlations within recordings.
% separate units by OSI. 3 different ways (fixed bins, sliding bins, adaptive bins)

trainsize_w=round(0.85*size(stimrefs_w,2));
testsize_w=size(stimrefs_w,2)-trainsize_w;
trainsize_c=round(0.85*size(stimrefs_c,2));
testsize_c=size(stimrefs_c,2)-trainsize_c;
name='lda_psuedopop';

%             % responses of those units:
%             r_u_w=resp_u_w(osi_keep_w,:);
%             r_b_w=resp_b_w(osi_keep_w,:);
%             r_u_c=resp_u_c(osi_keep_c,:);
%             r_b_c=resp_b_c(osi_keep_c,:);

for w=1:2 % 3 ways of breaking up OSI
    if w==1 % FIXED BINS, no overlap
        osi_tmp=fbins;
        for b=1:size(osi_tmp,1)
            clearvars -except bins fbins sbins abins resp* acute_bins awake_bins...
                stimrefs_w* resp_u_w resp_b_w oribias_w_all oripref_w_all oris...
                resp_u_c resp_b_c oribias_c_all oripref_c_all stimrefs_c* w b osi_tmp...
                trainsize* testsize* name par* fit*
            tic
            % units within osi range:
            osi_keep_w=find(oribias_w_all>=osi_tmp(b,1) & oribias_w_all<=osi_tmp(b,end));
            osi_keep_c=find(oribias_c_all>=osi_tmp(b,1) & oribias_c_all<=osi_tmp(b,end));
            % # of units in each bin:
            w_bins=histc(oribias_w_all(osi_keep_w),osi_tmp(b,:)); 
            c_bins=histc(oribias_c_all(osi_keep_c),osi_tmp(b,:));
            % minimum # of units in each bin btwn awake and anesthetized:
            min_count=min([c_bins;w_bins]); 
            
            % # of units to keep btwn awake/anesthetized to have matched distribution:
            n_units_f=sum(min_count); 
            steps=2:1:n_units_f;
            steps_fine=2:.25:100;
            
            % work for decoder:
            pop_lda_up_train_w=nan*zeros(9,length(steps),200);
            pop_lda_bp_train_w=nan*zeros(9,length(steps),200);
            pop_lda_up_test_w=nan*zeros(9,length(steps),200);
            pop_lda_bp_test_w=nan*zeros(9,length(steps),200);
            class_up_test_w=nan*zeros(9,length(steps),200,testsize_w*3);
            class_bp_test_w=nan*zeros(9,length(steps),200,testsize_w*3);
            prob_up_test_w=nan*zeros(9,length(steps),200,testsize_w*3);
            prob_bp_test_w=nan*zeros(9,length(steps),200,testsize_w*3);
            
            pop_lda_up_train_c=nan*zeros(9,length(steps),200);
            pop_lda_bp_train_c=nan*zeros(9,length(steps),200);
            pop_lda_up_test_c=nan*zeros(9,length(steps),200);
            pop_lda_bp_test_c=nan*zeros(9,length(steps),200);
            class_up_test_c=nan*zeros(9,length(steps),200,testsize_c*3);
            class_bp_test_c=nan*zeros(9,length(steps),200,testsize_c*3);
            prob_up_test_c=nan*zeros(9,length(steps),200,testsize_c*3);
            prob_bp_test_c=nan*zeros(9,length(steps),200,testsize_c*3);
            
            for o=1:length(oris)
                % select 3 oris for comparison:
                if o==1
                    n1=length(oris);
                    n2=o+1;
                elseif o==length(oris)
                    n1=o-1;
                    n2=1;
                else
                    n1=o-1;
                    n2=o+1;
                end
                indexo_w=find(stimrefs_w_v==oris(o));
                indexn1_w=find(stimrefs_w_v==oris(n1));
                indexn2_w=find(stimrefs_w_v==oris(n2));
                indexo_c=find(stimrefs_c_v==oris(o));
                indexn1_c=find(stimrefs_c_v==oris(n1));
                indexn2_c=find(stimrefs_c_v==oris(n2));
                
                % stimulus identity of train/test trials:
                train_w=[oris(o)*ones(1,trainsize_w) oris(n1)*ones(1,trainsize_w) oris(n2)*ones(1,trainsize_w)];
                train_c=[oris(o)*ones(1,trainsize_c) oris(n1)*ones(1,trainsize_c) oris(n2)*ones(1,trainsize_c)];
                test_w=[oris(o)*ones(1,testsize_w) oris(n1)*ones(1,testsize_w) oris(n2)*ones(1,testsize_w)];
                test_c=[oris(o)*ones(1,testsize_c) oris(n1)*ones(1,testsize_c) oris(n2)*ones(1,testsize_c)];
                idx=1;  % index for population size
                
                
                for e=steps % loop over population size
                    id1=1; % index for loop over neuron samples and trials (20x neurons & 10x trials = 200);
                    % loop over random neurons selected:
                    for c=1:20 % index random populations
                        z=c;
                        if z>10
                            z=z-10;
                        end
                        % select units and their responses:
                        r_u_w=[];
                        r_b_w=[];
                        r_u_c=[];
                        r_b_c=[];
                        for d=1:length(min_count)-1
                            tmpw=randperm(w_bins(d),min_count(d));
                            tmpc=randperm(c_bins(d),min_count(d));
                            osi_keep_w=find(oribias_w_all>=osi_tmp(b,d) & oribias_w_all<=osi_tmp(b,d+1));   % mean(osi_keep_w) ~= mean(osi_keep_c)
                            osi_keep_c=find(oribias_c_all>=osi_tmp(b,d) & oribias_c_all<=osi_tmp(b,d+1));
                            r_u_w=[r_u_w; squeeze(resp_u_w(osi_keep_w(tmpw),:,z))];
                            r_b_w=[r_b_w; squeeze(resp_b_w(osi_keep_w(tmpw),:,z))];
                            r_u_c=[r_u_c; squeeze(resp_u_c(osi_keep_c(tmpc),:,z))];
                            r_b_c=[r_b_c; squeeze(resp_b_c(osi_keep_c(tmpc),:,z))];
                        end

                        
                        % select random units and appropriate stim trials:
                        j=randperm(size(r_u_w,1),e);
                        units_up_w=r_u_w(j,[indexo_w indexn1_w indexn2_w]);
                        units_bp_w=r_b_w(j,[indexo_w indexn1_w indexn2_w]);
                        j=randperm(size(r_u_c,1),e);
                        units_up_c=r_u_c(j,[indexo_c indexn1_c indexn2_c]);
                        units_bp_c=r_b_c(j,[indexo_c indexn1_c indexn2_c]);
                        
                        % loop over random trials selected:
                        for j = 1:10
                            % train and test trials (awake):
                            % shuffle trials within each class
                            for k=1:size(units_up_w,1)
                                tmpushuf=[randperm(size(stimrefs_w,2)) randperm(size(stimrefs_w,2))+size(stimrefs_w,2) randperm(size(stimrefs_w,2))+size(stimrefs_w,2)*2];
                                tmpbshuf=[randperm(size(stimrefs_w,2)) randperm(size(stimrefs_w,2))+size(stimrefs_w,2) randperm(size(stimrefs_w,2))+size(stimrefs_w,2)*2];
                                units_up_w(k,:)=units_up_w(k,tmpushuf);
                                units_bp_w(k,:)=units_bp_w(k,tmpbshuf);
                            end
                            tmp=randperm(size(stimrefs_w,2),trainsize_w);
                            tmp2=[tmp tmp+size(stimrefs_w,2) tmp+2*size(stimrefs_w,2)];
                            x=setdiff(1:size(units_up_w,2),tmp2);
                            
                            trainu_w=units_up_w(:,tmp2);
                            trainb_w=units_bp_w(:,tmp2);
                            testu_w=units_up_w(:,x);
                            testb_w=units_bp_w(:,x);
                            
                            % try to catch units that will creater error in Training data
                            temp=sum(trainu_w,2);
                            temp2=sum(trainb_w,2);
                            zerou=ismember(0,temp);
                            zerob=ismember(0,temp2);
                            if zerou==1 || zerob==1
                                tmp=randperm(size(stimrefs_w,2),trainsize_w);
                                tmp2=[tmp tmp+size(stimrefs_w,2) tmp+2*size(stimrefs_w,2)];
                                x=setdiff(1:size(units_up_w,2),tmp2);
                                trainu_w=units_up_w(:,tmp2);
                                trainb_w=units_bp_w(:,tmp2);
                                testu_w=units_up_w(:,x);
                                testb_w=units_bp_w(:,x);
                                
                                temp=sum(trainu_w,2);
                                temp2=sum(trainb_w,2);
                                zerou=ismember(0,temp);
                                zerob=ismember(0,temp2);
                                if zerou==1 || zerob==1
                                    tmp=randperm(size(stimrefs_w,2),trainsize_w);
                                    tmp2=[tmp tmp+size(stimrefs_w,2) tmp+2*size(stimrefs_w,2)];
                                    x=setdiff(1:size(units_up_w,2),tmp2);
                                    trainu_w=units_up_w(:,tmp2);
                                    trainb_w=units_bp_w(:,tmp2);
                                    testu_w=units_up_w(:,x);
                                    testb_w=units_bp_w(:,x);
                                    
                                    temp=sum(trainu_w,2);
                                    temp2=sum(trainb_w,2);
                                    zerou=ismember(0,temp);
                                    zerob=ismember(0,temp2);
                                    if zerou==1 || zerob==1
                                        tmp=randperm(size(stimrefs_w,2),trainsize_w);
                                        tmp2=[tmp tmp+size(stimrefs_w,2) tmp+2*size(stimrefs_w,2)];
                                        x=setdiff(1:size(units_up_w,2),tmp2);
                                        trainu_w=units_up_w(:,tmp2);
                                        trainb_w=units_bp_w(:,tmp2);
                                        testu_w=units_up_w(:,x);
                                        testb_w=units_bp_w(:,x);
                                        
                                        temp=sum(trainu_w,2);
                                        temp2=sum(trainb_w,2);
                                        zerou=ismember(0,temp);
                                        zerob=ismember(0,temp2);
                                    end
                                end
                            end
                            
                            % run classifier:
                            [tmp_lda_up,tmp_err_up,tmp_prob_u]=classify(testu_w',trainu_w',train_w');
                            [tmp_lda_bp,tmp_err_bp,tmp_prob_b]=classify(testb_w',trainb_w',train_w');
                            tmp5=find(tmp_lda_up'==test_w); % percent correct on test trials
                            tmp6=find(tmp_lda_bp'==test_w);
                            
                            % store output
                            pop_lda_up_train_w(o,idx,id1)=1-tmp_err_up;
                            pop_lda_bp_train_w(o,idx,id1)=1-tmp_err_bp;
                            pop_lda_up_test_w(o,idx,id1)=length(tmp5)/length(tmp_lda_up);
                            pop_lda_bp_test_w(o,idx,id1)=length(tmp6)/length(tmp_lda_bp);
                            
                            % store classifications:
                            class_up_test_w(o,idx,id1,:)=tmp_lda_up';
                            class_bp_test_w(o,idx,id1,:)=tmp_lda_bp';
                            
                            % store posterior probabilities
                            prob_up_test_w(o,idx,id1,:)=tmp_prob_u(:,1)'; % prob in class 1
                            prob_bp_test_w(o,idx,id1,:)=tmp_prob_b(:,1)';
                            % structure: ori center, population size, [sampled neurons x trials: train/test]
                            
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
                            
                            % train and test trials (acute)
                            % shuffle trials within each class
                            for k=1:size(units_up_c,1)
                                tmpushuf=[randperm(size(stimrefs_c,2)) randperm(size(stimrefs_c,2))+size(stimrefs_c,2) randperm(size(stimrefs_c,2))+size(stimrefs_c,2)*2];
                                tmpbshuf=[randperm(size(stimrefs_c,2)) randperm(size(stimrefs_c,2))+size(stimrefs_c,2) randperm(size(stimrefs_c,2))+size(stimrefs_c,2)*2];
                                units_up_c(k,:)=units_up_c(k,tmpushuf);
                                units_bp_c(k,:)=units_bp_c(k,tmpbshuf);
                            end
                            tmp=randperm(size(stimrefs_c,2),trainsize_c);
                            tmp2=[tmp tmp+size(stimrefs_c,2) tmp+2*size(stimrefs_c,2)];
                            x=setdiff(1:size(units_up_c,2),tmp2);
                            
                            trainu_c=units_up_c(:,tmp2);
                            trainb_c=units_bp_c(:,tmp2);
                            testu_c=units_up_c(:,x);
                            testb_c=units_bp_c(:,x);
                            
                            % try to catch units that will creater error in Training data
                            temp=sum(trainu_c,2);
                            temp2=sum(trainb_c,2);
                            zerou=ismember(0,temp);
                            zerob=ismember(0,temp2);
                            if zerou==1 || zerob==1
                                tmp=randperm(size(stimrefs_c,2),trainsize_c);
                                tmp2=[tmp tmp+size(stimrefs_c,2) tmp+2*size(stimrefs_c,2)];
                                x=setdiff(1:size(units_up_c,2),tmp2);
                                trainu_c=units_up_c(:,tmp2);
                                trainb_c=units_bp_c(:,tmp2);
                                testu_c=units_up_c(:,x);
                                testb_c=units_bp_c(:,x);
                                
                                temp=sum(trainu_c,2);
                                temp2=sum(trainb_c,2);
                                zerou=ismember(0,temp);
                                zerob=ismember(0,temp2);
                                if zerou==1 || zerob==1
                                    tmp=randperm(size(stimrefs_c,2),trainsize_c);
                                    tmp2=[tmp tmp+size(stimrefs_c,2) tmp+2*size(stimrefs_c,2)];
                                    x=setdiff(1:size(units_up_c,2),tmp2);
                                    trainu_c=units_up_c(:,tmp2);
                                    trainb_c=units_bp_c(:,tmp2);
                                    testu_c=units_up_c(:,x);
                                    testb_c=units_bp_c(:,x);
                                    
                                    temp=sum(trainu_c,2);
                                    temp2=sum(trainb_c,2);
                                    zerou=ismember(0,temp);
                                    zerob=ismember(0,temp2);
                                    if zerou==1 || zerob==1
                                        tmp=randperm(size(stimrefs_c,2),trainsize_c);
                                        tmp2=[tmp tmp+size(stimrefs_c,2) tmp+2*size(stimrefs_c,2)];
                                        x=setdiff(1:size(units_up_c,2),tmp2);
                                        trainu_c=units_up_c(:,tmp2);
                                        trainb_c=units_bp_c(:,tmp2);
                                        testu_c=units_up_c(:,x);
                                        testb_c=units_bp_c(:,x);
                                    end
                                end
                            end
                            
                            % run classifier:
                            [tmp_lda_up,tmp_err_up,tmp_prob_u]=classify(testu_c',trainu_c',train_c');
                            [tmp_lda_bp,tmp_err_bp,tmp_prob_b]=classify(testb_c',trainb_c',train_c');
                            tmp5=find(tmp_lda_up'==test_c); % percent correct on test trials
                            tmp6=find(tmp_lda_bp'==test_c);
                            
                            % store output
                            pop_lda_up_train_c(o,idx,id1)=1-tmp_err_up;
                            pop_lda_bp_train_c(o,idx,id1)=1-tmp_err_bp;
                            pop_lda_up_test_c(o,idx,id1)=length(tmp5)/length(tmp_lda_up);
                            pop_lda_bp_test_c(o,idx,id1)=length(tmp6)/length(tmp_lda_bp);
                            
                            % store classifications:
                            class_up_test_c(o,idx,id1,:)=tmp_lda_up';
                            class_bp_test_c(o,idx,id1,:)=tmp_lda_bp';
                            
                            % store posterior probabilities
                            prob_up_test_c(o,idx,id1,:)=tmp_prob_u(:,1)'; % prob in class 1
                            prob_bp_test_c(o,idx,id1,:)=tmp_prob_b(:,1)';
                            % structure: ori center, population size, [sampled neurons x trials: train/test]
                            id1=id1+1;
                            clear tmp*
                        end
                    end
                    idx=idx+1;
                end
                disp('3ori done')
            end
            
            pop_lda_up_train_mean_w=squeeze(nanmean(pop_lda_up_train_w(:,:,:),3));
            pop_lda_bp_train_mean_w=squeeze(nanmean(pop_lda_bp_train_w(:,:,:),3));
            pop_lda_up_test_mean_w=squeeze(nanmean(pop_lda_up_test_w(:,:,:),3));
            pop_lda_bp_test_mean_w=squeeze(nanmean(pop_lda_bp_test_w(:,:,:),3));
            
            pop_lda_up_train_mean_c=squeeze(nanmean(pop_lda_up_train_c(:,:,:),3));
            pop_lda_bp_train_mean_c=squeeze(nanmean(pop_lda_bp_train_c(:,:,:),3));
            pop_lda_up_test_mean_c=squeeze(nanmean(pop_lda_up_test_c(:,:,:),3));
            pop_lda_bp_test_mean_c=squeeze(nanmean(pop_lda_bp_test_c(:,:,:),3));
            
            % fit performance data to Weibull
            % (center ori, pop size, repeats (10x10=100), performance: train & test)
            for e=1:length(oris)
                % Xdata = # of steps (pop size); Ydata = % correct at that size
                % only doing this for matched distribution
                [par_up_train_w(e,:), gof_up_train_w(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean_w(e,:),[0.1 0.1 0.1],[]);
                [par_bp_train_w(e,:), gof_bp_train_w(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean_w(e,:),[0.1 0.1 0.1],[]);
                [par_up_test_w(e,:), gof_up_test_w(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean_w(e,:),[0.1 0.1 0.1],[]);
                [par_bp_test_w(e,:), gof_bp_test_w(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean_w(e,:),[0.1 0.1 0.1],[]);
                
                fit_up_train_w(e,:)=1-par_up_train_w(e,3)*exp(-(steps_fine/par_up_train_w(e,1)).^par_up_train_w(e,2));
                fit_bp_train_w(e,:)=1-par_bp_train_w(e,3)*exp(-(steps_fine/par_bp_train_w(e,1)).^par_bp_train_w(e,2));
                fit_up_test_w(e,:)=1-par_up_test_w(e,3)*exp(-(steps_fine/par_up_test_w(e,1)).^par_up_test_w(e,2));
                fit_bp_test_w(e,:)=1-par_bp_test_w(e,3)*exp(-(steps_fine/par_bp_test_w(e,1)).^par_bp_test_w(e,2));
                
                [par_up_train_c(e,:), gof_up_train_c(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean_c(e,:),[0.1 0.1 0.1],[]);
                [par_bp_train_c(e,:), gof_bp_train_c(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean_c(e,:),[0.1 0.1 0.1],[]);
                [par_up_test_c(e,:), gof_up_test_c(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean_c(e,:),[0.1 0.1 0.1],[]);
                [par_bp_test_c(e,:), gof_bp_test_c(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean_c(e,:),[0.1 0.1 0.1],[]);
                
                fit_up_train_c(e,:)=1-par_up_train_c(e,3)*exp(-(steps_fine/par_up_train_c(e,1)).^par_up_train_c(e,2));
                fit_bp_train_c(e,:)=1-par_bp_train_c(e,3)*exp(-(steps_fine/par_bp_train_c(e,1)).^par_bp_train_c(e,2));
                fit_up_test_c(e,:)=1-par_up_test_c(e,3)*exp(-(steps_fine/par_up_test_c(e,1)).^par_up_test_c(e,2));
                fit_bp_test_c(e,:)=1-par_bp_test_c(e,3)*exp(-(steps_fine/par_bp_test_c(e,1)).^par_bp_test_c(e,2));
            end
            toc

            clearvars -except bins fbins sbins abins resp* acute_bins awake_bins...
                stimrefs_w* resp_u_w resp_b_w oribias_w_all oripref_w_all oris...
                resp_u_c resp_b_c oribias_c_all oripref_c_all stimrefs_c* w b osi_tmp...
                trainsize* testsize* name par* fit* n_units*
            disp('bin done')
            save(sprintf('%s_f%d_pseudopop',name,b))
        end
        
        % SLIDING BINS:
    elseif w==2
        osi_tmp=sbins;
        for b=1:size(osi_tmp,1)
            clearvars -except bins fbins sbins abins resp* acute_bins awake_bins...
                stimrefs_w* resp_u_w resp_b_w oribias_w_all oripref_w_all oris...
                resp_u_c resp_b_c oribias_c_all oripref_c_all stimrefs_c* w b osi_tmp...
                trainsize* testsize* name par* fit*
            tic
            % units within osi range:
            osi_keep_w=find(oribias_w_all>=osi_tmp(b,1) & oribias_w_all<=osi_tmp(b,end));
            osi_keep_c=find(oribias_c_all>=osi_tmp(b,1) & oribias_c_all<=osi_tmp(b,end));
            % # of units in each bin:
            w_bins=histc(oribias_w_all(osi_keep_w),osi_tmp(b,:)); 
            c_bins=histc(oribias_c_all(osi_keep_c),osi_tmp(b,:));
            % minimum # of units in each bin btwn awake and anesthetized:
            min_count=min([c_bins;w_bins]); 
            
            % # of units to keep btwn awake/anesthetized to have matched distribution:
            n_units_s=sum(min_count);%min([length(osi_keep_w) length(osi_keep_c)]);
            steps=2:1:n_units_s;
            steps_fine=2:.25:100;
            
            % work for decoder:
            pop_lda_up_train_w=nan*zeros(9,length(steps),200);
            pop_lda_bp_train_w=nan*zeros(9,length(steps),200);
            pop_lda_up_test_w=nan*zeros(9,length(steps),200);
            pop_lda_bp_test_w=nan*zeros(9,length(steps),200);
            class_up_test_w=nan*zeros(9,length(steps),200,testsize_w*3);
            class_bp_test_w=nan*zeros(9,length(steps),200,testsize_w*3);
            prob_up_test_w=nan*zeros(9,length(steps),200,testsize_w*3);
            prob_bp_test_w=nan*zeros(9,length(steps),200,testsize_w*3);
            
            pop_lda_up_train_c=nan*zeros(9,length(steps),200);
            pop_lda_bp_train_c=nan*zeros(9,length(steps),200);
            pop_lda_up_test_c=nan*zeros(9,length(steps),200);
            pop_lda_bp_test_c=nan*zeros(9,length(steps),200);
            class_up_test_c=nan*zeros(9,length(steps),200,testsize_c*3);
            class_bp_test_c=nan*zeros(9,length(steps),200,testsize_c*3);
            prob_up_test_c=nan*zeros(9,length(steps),200,testsize_c*3);
            prob_bp_test_c=nan*zeros(9,length(steps),200,testsize_c*3);
            
            for o=1:length(oris)
                % select 3 oris for comparison:
                if o==1
                    n1=length(oris);
                    n2=o+1;
                elseif o==length(oris)
                    n1=o-1;
                    n2=1;
                else
                    n1=o-1;
                    n2=o+1;
                end
                indexo_w=find(stimrefs_w_v==oris(o));
                indexn1_w=find(stimrefs_w_v==oris(n1));
                indexn2_w=find(stimrefs_w_v==oris(n2));
                indexo_c=find(stimrefs_c_v==oris(o));
                indexn1_c=find(stimrefs_c_v==oris(n1));
                indexn2_c=find(stimrefs_c_v==oris(n2));
                
                % stimulus identity of train/test trials:
                train_w=[oris(o)*ones(1,trainsize_w) oris(n1)*ones(1,trainsize_w) oris(n2)*ones(1,trainsize_w)];
                train_c=[oris(o)*ones(1,trainsize_c) oris(n1)*ones(1,trainsize_c) oris(n2)*ones(1,trainsize_c)];
                test_w=[oris(o)*ones(1,testsize_w) oris(n1)*ones(1,testsize_w) oris(n2)*ones(1,testsize_w)];
                test_c=[oris(o)*ones(1,testsize_c) oris(n1)*ones(1,testsize_c) oris(n2)*ones(1,testsize_c)];
                
                idx=1;  % index for population size
                % loop over population size:
                for e=steps
                    id1=1; % index for loop over neuron samples and trials (20x neurons & 10x trials = 200);
                    % loop over random neurons selected:
                    for c=1:20 % index random populations
                        z=c;
                        if z>10
                            z=z-10;
                        end
                        % select units:
                        r_u_w=[];
                        r_b_w=[];
                        r_u_c=[];
                        r_b_c=[];
                        for d=1:length(min_count)-1
                            tmpw=randperm(w_bins(d),min_count(d));
                            tmpc=randperm(c_bins(d),min_count(d));
                            osi_keep_w=find(oribias_w_all>=osi_tmp(b,d) & oribias_w_all<=osi_tmp(b,d+1));
                            osi_keep_c=find(oribias_c_all>=osi_tmp(b,d) & oribias_c_all<=osi_tmp(b,d+1));
                            r_u_w=[r_u_w; squeeze(resp_u_w(osi_keep_w(tmpw),:,z))];
                            r_b_w=[r_b_w; squeeze(resp_b_w(osi_keep_w(tmpw),:,z))];
                            r_u_c=[r_u_c; squeeze(resp_u_c(osi_keep_c(tmpc),:,z))];
                            r_b_c=[r_b_c; squeeze(resp_b_c(osi_keep_c(tmpc),:,z))];
                        end
                        
                        % select random units
                        j=randperm(size(r_u_w,1),e);
                        units_up_w=r_u_w(j,[indexo_w indexn1_w indexn2_w]);
                        units_bp_w=r_b_w(j,[indexo_w indexn1_w indexn2_w]);
                        j=randperm(size(r_u_c,1),e);
                        units_up_c=r_u_c(j,[indexo_c indexn1_c indexn2_c]);
                        units_bp_c=r_b_c(j,[indexo_c indexn1_c indexn2_c]);
                        
                        % loop over random trials selected:
                        for j = 1:10
                            % train and test trials (awake):
                            % shuffle trials within each class
                            for k=1:size(units_up_w,1)
                                tmpushuf=[randperm(size(stimrefs_w,2)) randperm(size(stimrefs_w,2))+size(stimrefs_w,2) randperm(size(stimrefs_w,2))+size(stimrefs_w,2)*2];
                                tmpbshuf=[randperm(size(stimrefs_w,2)) randperm(size(stimrefs_w,2))+size(stimrefs_w,2) randperm(size(stimrefs_w,2))+size(stimrefs_w,2)*2];
                                units_up_w(k,:)=units_up_w(k,tmpushuf);
                                units_bp_w(k,:)=units_bp_w(k,tmpbshuf);
                            end
                            tmp=randperm(size(stimrefs_w,2),trainsize_w);
                            tmp2=[tmp tmp+size(stimrefs_w,2) tmp+2*size(stimrefs_w,2)];
                            x=setdiff(1:size(units_up_w,2),tmp2);
                            
                            trainu_w=units_up_w(:,tmp2);
                            trainb_w=units_bp_w(:,tmp2);
                            testu_w=units_up_w(:,x);
                            testb_w=units_bp_w(:,x);
                            
                            % try to catch units that will creater error in Training data
                            temp=sum(trainu_w,2);
                            temp2=sum(trainb_w,2);
                            zerou=ismember(0,temp);
                            zerob=ismember(0,temp2);
                            if zerou==1 || zerob==1
                                tmp=randperm(size(stimrefs_w,2),trainsize_w);
                                tmp2=[tmp tmp+size(stimrefs_w,2) tmp+2*size(stimrefs_w,2)];
                                x=setdiff(1:size(units_up_w,2),tmp2);
                                trainu_w=units_up_w(:,tmp2);
                                trainb_w=units_bp_w(:,tmp2);
                                testu_w=units_up_w(:,x);
                                testb_w=units_bp_w(:,x);
                                
                                temp=sum(trainu_w,2);
                                temp2=sum(trainb_w,2);
                                zerou=ismember(0,temp);
                                zerob=ismember(0,temp2);
                                if zerou==1 || zerob==1
                                    tmp=randperm(size(stimrefs_w,2),trainsize_w);
                                    tmp2=[tmp tmp+size(stimrefs_w,2) tmp+2*size(stimrefs_w,2)];
                                    x=setdiff(1:size(units_up_w,2),tmp2);
                                    trainu_w=units_up_w(:,tmp2);
                                    trainb_w=units_bp_w(:,tmp2);
                                    testu_w=units_up_w(:,x);
                                    testb_w=units_bp_w(:,x);
                                    
                                    temp=sum(trainu_w,2);
                                    temp2=sum(trainb_w,2);
                                    zerou=ismember(0,temp);
                                    zerob=ismember(0,temp2);
                                    if zerou==1 || zerob==1
                                        tmp=randperm(size(stimrefs_w,2),trainsize_w);
                                        tmp2=[tmp tmp+size(stimrefs_w,2) tmp+2*size(stimrefs_w,2)];
                                        x=setdiff(1:size(units_up_w,2),tmp2);
                                        trainu_w=units_up_w(:,tmp2);
                                        trainb_w=units_bp_w(:,tmp2);
                                        testu_w=units_up_w(:,x);
                                        testb_w=units_bp_w(:,x);
                                        
                                        temp=sum(trainu_w,2);
                                        temp2=sum(trainb_w,2);
                                        zerou=ismember(0,temp);
                                        zerob=ismember(0,temp2);
                                    end
                                end
                            end
                            
                            % run classifier:
                            [tmp_lda_up,tmp_err_up,tmp_prob_u]=classify(testu_w',trainu_w',train_w');
                            [tmp_lda_bp,tmp_err_bp,tmp_prob_b]=classify(testb_w',trainb_w',train_w');
                            tmp5=find(tmp_lda_up'==test_w); % percent correct on test trials
                            tmp6=find(tmp_lda_bp'==test_w);
                            
                            % store output
                            pop_lda_up_train_w(o,idx,id1)=1-tmp_err_up;
                            pop_lda_bp_train_w(o,idx,id1)=1-tmp_err_bp;
                            pop_lda_up_test_w(o,idx,id1)=length(tmp5)/length(tmp_lda_up);
                            pop_lda_bp_test_w(o,idx,id1)=length(tmp6)/length(tmp_lda_bp);
                            
                            % store classifications:
                            class_up_test_w(o,idx,id1,:)=tmp_lda_up';
                            class_bp_test_w(o,idx,id1,:)=tmp_lda_bp';
                            
                            % store posterior probabilities
                            prob_up_test_w(o,idx,id1,:)=tmp_prob_u(:,1)'; % prob in class 1
                            prob_bp_test_w(o,idx,id1,:)=tmp_prob_b(:,1)';
                            % structure: ori center, population size, [sampled neurons x trials: train/test]
                            
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
                            
                            % train and test trials (acute)
                            % shuffle trials within each class
                            for k=1:size(units_up_c,1)
                                tmpushuf=[randperm(size(stimrefs_c,2)) randperm(size(stimrefs_c,2))+size(stimrefs_c,2) randperm(size(stimrefs_c,2))+size(stimrefs_c,2)*2];
                                tmpbshuf=[randperm(size(stimrefs_c,2)) randperm(size(stimrefs_c,2))+size(stimrefs_c,2) randperm(size(stimrefs_c,2))+size(stimrefs_c,2)*2];
                                units_up_c(k,:)=units_up_c(k,tmpushuf);
                                units_bp_c(k,:)=units_bp_c(k,tmpbshuf);
                            end
                            tmp=randperm(size(stimrefs_c,2),trainsize_c);
                            tmp2=[tmp tmp+size(stimrefs_c,2) tmp+2*size(stimrefs_c,2)];
                            x=setdiff(1:size(units_up_c,2),tmp2);
                            
                            trainu_c=units_up_c(:,tmp2);
                            trainb_c=units_bp_c(:,tmp2);
                            testu_c=units_up_c(:,x);
                            testb_c=units_bp_c(:,x);
                            
                            % try to catch units that will creater error in Training data
                            temp=sum(trainu_c,2);
                            temp2=sum(trainb_c,2);
                            zerou=ismember(0,temp);
                            zerob=ismember(0,temp2);
                            if zerou==1 || zerob==1
                                tmp=randperm(size(stimrefs_c,2),trainsize_c);
                                tmp2=[tmp tmp+size(stimrefs_c,2) tmp+2*size(stimrefs_c,2)];
                                x=setdiff(1:size(units_up_c,2),tmp2);
                                trainu_c=units_up_c(:,tmp2);
                                trainb_c=units_bp_c(:,tmp2);
                                testu_c=units_up_c(:,x);
                                testb_c=units_bp_c(:,x);
                                
                                temp=sum(trainu_c,2);
                                temp2=sum(trainb_c,2);
                                zerou=ismember(0,temp);
                                zerob=ismember(0,temp2);
                                if zerou==1 || zerob==1
                                    tmp=randperm(size(stimrefs_c,2),trainsize_c);
                                    tmp2=[tmp tmp+size(stimrefs_c,2) tmp+2*size(stimrefs_c,2)];
                                    x=setdiff(1:size(units_up_c,2),tmp2);
                                    trainu_c=units_up_c(:,tmp2);
                                    trainb_c=units_bp_c(:,tmp2);
                                    testu_c=units_up_c(:,x);
                                    testb_c=units_bp_c(:,x);
                                    
                                    temp=sum(trainu_c,2);
                                    temp2=sum(trainb_c,2);
                                    zerou=ismember(0,temp);
                                    zerob=ismember(0,temp2);
                                    if zerou==1 || zerob==1
                                        tmp=randperm(size(stimrefs_c,2),trainsize_c);
                                        tmp2=[tmp tmp+size(stimrefs_c,2) tmp+2*size(stimrefs_c,2)];
                                        x=setdiff(1:size(units_up_c,2),tmp2);
                                        trainu_c=units_up_c(:,tmp2);
                                        trainb_c=units_bp_c(:,tmp2);
                                        testu_c=units_up_c(:,x);
                                        testb_c=units_bp_c(:,x);
                                    end
                                end
                            end
                            
                            % run classifier:
                            [tmp_lda_up,tmp_err_up,tmp_prob_u]=classify(testu_c',trainu_c',train_c');
                            [tmp_lda_bp,tmp_err_bp,tmp_prob_b]=classify(testb_c',trainb_c',train_c');
                            tmp5=find(tmp_lda_up'==test_c); % percent correct on test trials
                            tmp6=find(tmp_lda_bp'==test_c);
                            
                            % store output
                            pop_lda_up_train_c(o,idx,id1)=1-tmp_err_up;
                            pop_lda_bp_train_c(o,idx,id1)=1-tmp_err_bp;
                            pop_lda_up_test_c(o,idx,id1)=length(tmp5)/length(tmp_lda_up);
                            pop_lda_bp_test_c(o,idx,id1)=length(tmp6)/length(tmp_lda_bp);
                            
                            % store classifications:
                            class_up_test_c(o,idx,id1,:)=tmp_lda_up';
                            class_bp_test_c(o,idx,id1,:)=tmp_lda_bp';
                            
                            % store posterior probabilities
                            prob_up_test_c(o,idx,id1,:)=tmp_prob_u(:,1)'; % prob in class 1
                            prob_bp_test_c(o,idx,id1,:)=tmp_prob_b(:,1)';
                            % structure: ori center, population size, [sampled neurons x trials: train/test]
                            id1=id1+1;
                            clear tmp*
                        end
                    end
                    idx=idx+1;
                end
                disp('3ori done')
            end
            
            pop_lda_up_train_mean_w=squeeze(nanmean(pop_lda_up_train_w(:,:,:),3));
            pop_lda_bp_train_mean_w=squeeze(nanmean(pop_lda_bp_train_w(:,:,:),3));
            pop_lda_up_test_mean_w=squeeze(nanmean(pop_lda_up_test_w(:,:,:),3));
            pop_lda_bp_test_mean_w=squeeze(nanmean(pop_lda_bp_test_w(:,:,:),3));
            
            pop_lda_up_train_mean_c=squeeze(nanmean(pop_lda_up_train_c(:,:,:),3));
            pop_lda_bp_train_mean_c=squeeze(nanmean(pop_lda_bp_train_c(:,:,:),3));
            pop_lda_up_test_mean_c=squeeze(nanmean(pop_lda_up_test_c(:,:,:),3));
            pop_lda_bp_test_mean_c=squeeze(nanmean(pop_lda_bp_test_c(:,:,:),3));
            
            % fit performance data to Weibull
            % (center ori, pop size, repeats (10x10=100), performance: train & test)
            for e=1:length(oris)
                % Xdata = # of steps (pop size); Ydata = % correct at that size
                % only doing this for matched distribution
                [par_up_train_w(e,:), gof_up_train_w(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean_w(e,:),[0.1 0.1 0.1],[]);
                [par_bp_train_w(e,:), gof_bp_train_w(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean_w(e,:),[0.1 0.1 0.1],[]);
                [par_up_test_w(e,:), gof_up_test_w(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean_w(e,:),[0.1 0.1 0.1],[]);
                [par_bp_test_w(e,:), gof_bp_test_w(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean_w(e,:),[0.1 0.1 0.1],[]);
                
                fit_up_train_w(e,:)=1-par_up_train_w(e,3)*exp(-(steps_fine/par_up_train_w(e,1)).^par_up_train_w(e,2));
                fit_bp_train_w(e,:)=1-par_bp_train_w(e,3)*exp(-(steps_fine/par_bp_train_w(e,1)).^par_bp_train_w(e,2));
                fit_up_test_w(e,:)=1-par_up_test_w(e,3)*exp(-(steps_fine/par_up_test_w(e,1)).^par_up_test_w(e,2));
                fit_bp_test_w(e,:)=1-par_bp_test_w(e,3)*exp(-(steps_fine/par_bp_test_w(e,1)).^par_bp_test_w(e,2));
                
                [par_up_train_c(e,:), gof_up_train_c(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean_c(e,:),[0.1 0.1 0.1],[]);
                [par_bp_train_c(e,:), gof_bp_train_c(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean_c(e,:),[0.1 0.1 0.1],[]);
                [par_up_test_c(e,:), gof_up_test_c(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean_c(e,:),[0.1 0.1 0.1],[]);
                [par_bp_test_c(e,:), gof_bp_test_c(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean_c(e,:),[0.1 0.1 0.1],[]);
                
                fit_up_train_c(e,:)=1-par_up_train_c(e,3)*exp(-(steps_fine/par_up_train_c(e,1)).^par_up_train_c(e,2));
                fit_bp_train_c(e,:)=1-par_bp_train_c(e,3)*exp(-(steps_fine/par_bp_train_c(e,1)).^par_bp_train_c(e,2));
                fit_up_test_c(e,:)=1-par_up_test_c(e,3)*exp(-(steps_fine/par_up_test_c(e,1)).^par_up_test_c(e,2));
                fit_bp_test_c(e,:)=1-par_bp_test_c(e,3)*exp(-(steps_fine/par_bp_test_c(e,1)).^par_bp_test_c(e,2));
            end
            toc

            clearvars -except bins fbins sbins abins resp* acute_bins awake_bins...
                stimrefs_w* resp_u_w resp_b_w oribias_w_all oripref_w_all oris...
                resp_u_c resp_b_c oribias_c_all oripref_c_all stimrefs_c* w b osi_tmp...
                trainsize* testsize* name par* fit* n_units*
            disp('bin done')
            save(sprintf('%s_s%d_pseudopop',name,b))
        end
    
        % ADAPTIVE BINS:
%     elseif w==3
% %         not sure how to do this at the moment
%         
%             clearvars -except bins fbins sbins abins resp* acute_bins awake_bins...
%                 stimrefs_w* resp_u_w resp_b_w oribias_w_all oripref_w_all oris...
%                 resp_u_c resp_b_c oribias_c_all oripref_c_all stimrefs_c* w b osi_tmp...
%                 trainsize* testsize* name par* fit* n_units*
%             disp('bin done')
%             save(sprintf('%s_p%d_pseudopop',name,b))
    end
end
stop
%% combine data and plot


fbins=[0:0.05:0.25;...
    0.25:0.05:0.50;...
    0.50:0.05:0.75;...
    0.75:0.05:1];

% sliding bins
sbins=[0.1:0.05:0.35;...
    0.2:0.05:0.45;...
    0.3:0.05:0.55;...
    0.4:0.05:0.65;...
    0.5:0.05:0.75;...
    0.6:0.05:0.85;...
    0.7:0.05:0.95];

ori_base=0:20:160;
for a=1:2
    % FIXED BINS
    if a==1
        for b=1:size(fbins,1)
            name='lda_pseudopop';
            load(sprintf('%s_f%d.mat',name,b))
            
            clearvars -except bins fbins sbins abins resp* resp_u_w resp_b_w oribias_w_all oripref_w_all oris...
                resp_u_c resp_b_c oribias_c_all oripref_c_all a b name par* fit* n_units* ori_base steps* AUC*
            
            load(sprintf('%s_f%d.mat',name,b))
            
            units_f(b)=n_units_f;
            
            for n=1:length(ori_base)
                AUCu_f_w(b,n)=trapz(steps_fine./steps_fine(end),fit_up_test_w(n,:));
                AUCb_f_w(b,n)=trapz(steps_fine./steps_fine(end),fit_bp_test_w(n,:));
                
                AUCu_f_c(b,n)=trapz(steps_fine./steps_fine(end),fit_up_test_c(n,:));
                AUCb_f_c(b,n)=trapz(steps_fine./steps_fine(end),fit_bp_test_c(n,:));
            end
            
        end
        
        % SLIDING bins
    elseif a==2
        for b=1:size(sbins,1)
            name='lda_pseudopop';
            load(sprintf('%s_f%d.mat',name,b))
            
            clearvars -except bins fbins sbins abins resp* resp_u_w resp_b_w oribias_w_all oripref_w_all oris...
                resp_u_c resp_b_c oribias_c_all oripref_c_all a b name par* fit* n_units* ori_base steps* AUC*
            
            load(sprintf('%s_f%d.mat',name,b))
            
            units_s(b)=n_units_s;
            
            for n=1:length(ori_base)
                AUCu_s_w(b,n)=trapz(steps_fine./steps_fine(end),fit_up_test_s(n,:));
                AUCb_s_w(b,n)=trapz(steps_fine./steps_fine(end),fit_bp_test_s(n,:));
                
                AUCu_s_c(b,n)=trapz(steps_fine./steps_fine(end),fit_up_test_s(n,:));
                AUCb_s_c(b,n)=trapz(steps_fine./steps_fine(end),fit_bp_test_s(n,:));
            end
            
        end
    end
end

stop
%%  plot
% Plot:
figure
subplot(231); supertitle('awake data')
hold on;
tmp=AUCb_f_w-AUCu_f_w;
for b=1:size(tmp,1)
    plot(1+0.05*(b-1):5+0.05*(b-1),[tmp(b,1) mean(tmp(b,[2 9])) mean(tmp(b,[3 8])) mean(tmp(b,[4 7])) mean(tmp(b,[5 6]))])
end
plot([0 6],[0 0],'k:')
% ylim([-0.6 0.6])
axis square; box off;
set(gca,'XTick',[1 3 5],'XTickLabel',{0','+-40','+-80'},'TickDir','out')
xlabel('center ori')
ylabel('change in decoder performance')
title('fixed OSI bins, Awake')
legend({'0.25','0.5','0.75','1'})
subplot(232); hold on
tmp=AUCb_f_c-AUCu_f_c;
for b=1:size(tmp,1)
    plot(1+0.05*(b-1):5+0.05*(b-1),[tmp(b,1) mean(tmp(b,[2 9])) mean(tmp(b,[3 8])) mean(tmp(b,[4 7])) mean(tmp(b,[5 6]))])
end
plot([0 6],[0 0],'k:')
% ylim([-0.2 0.2])
axis square; box off;
set(gca,'XTick',[1 3 5],'XTickLabel',{0','+-40','+-80'},'TickDir','out')
title('fixed OSI bins, Acute')

subplot(233); hold on
plot(1:4,units_f)
xlabel('bin order')
ylabel('max population size')
axis square; box off;

subplot(234); supertitle('awake data')
hold on;
tmp=AUCb_s_w-AUCu_s_w;
for b=1:size(tmp,1)
    plot(1+0.05*(b-1):5+0.05*(b-1),[tmp(b,1) mean(tmp(b,[2 9])) mean(tmp(b,[3 8])) mean(tmp(b,[4 7])) mean(tmp(b,[5 6]))])
end
plot([0 6],[0 0],'k:')
% ylim([-0.6 0.6])
axis square; box off;
set(gca,'XTick',[1 3 5],'XTickLabel',{0','+-40','+-80'},'TickDir','out')
xlabel('center ori')
ylabel('change in decoder performance')
title('Sliding OSI bins, Awake')
legend({'.1-.35','.2-.45','.3-.55','.4-.65','.5-.75','.6-.85','.7-.95'})
subplot(235); hold on
tmp=AUCb_s_c-AUCu_s_c;
for b=1:size(tmp,1)
    plot(1+0.05*(b-1):5+0.05*(b-1),[tmp(b,1) mean(tmp(b,[2 9])) mean(tmp(b,[3 8])) mean(tmp(b,[4 7])) mean(tmp(b,[5 6]))])
end
plot([0 6],[0 0],'k:')
% ylim([-0.2 0.2])
axis square; box off;
set(gca,'XTick',[1 3 5],'XTickLabel',{0','+-40','+-80'},'TickDir','out')
title('Sliding OSI bins, Acute')

subplot(236); hold on
plot(1:7,units_s)
xlabel('bin order')
ylabel('max population size')
axis square; box off;

