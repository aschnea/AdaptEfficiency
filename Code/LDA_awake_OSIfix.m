%% AWAKE LDA individual files
clear
oris=0:20:160;
%     elseif a==22
%         load('cadetv1p404_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p404_tuning','ori*','resp*','spont','tune*');
%     elseif a==23
%         load('cadetv1p405_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p405_tuning','ori*','resp*','spont','tune*');
%     elseif a==24
%         load('cadetv1p418_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p418_tuning','ori*','resp*','spont','tune*');
for a=[17 18 20 22 25 27] % 6:1 files
% for a=15:27 % all awake files
    clearvars -except a oris
    if a==1
        load('cadetv1p194_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
        load('cadetv1p194_tuning','ori*','resp*','spont','tune*');
%     elseif a==2
%         load('cadetv1p195_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p195_tuning','ori*','resp*','spont','tune*');
%     elseif a==3
%         load('cadetv1p245_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p245_tuning','ori*','resp*','spont','tune*');
%     elseif a==4
%         load('cadetv1p246_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p246_tuning','ori*','resp*','spont','tune*');
%     elseif a==5
%         load('cadetv1p345_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p345_tuning','ori*','resp*','spont','tune*');
%     elseif a==6
%         load('cadetv1p346_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p346_tuning','ori*','resp*','spont','tune*');
%     elseif a==7
%         load('cadetv1p347_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p347_tuning','ori*','resp*','spont','tune*');
%     elseif a==8
%         load('cadetv1p348_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p348_tuning','ori*','resp*','spont','tune*');
%     elseif a==9
%         load('cadetv1p349_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p349_tuning','ori*','resp*','spont','tune*');
%     elseif a==10
%         load('cadetv1p350_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p350_tuning','ori*','resp*','spont','tune*');
%     elseif a==11
%         load('cadetv1p351_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p351_tuning','ori*','resp*','spont','tune*');
%     elseif a==12
%         load('cadetv1p352_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p352_tuning','ori*','resp*','spont','tune*');
%     elseif a==13
%         load('cadetv1p353_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p368_tuning','ori*','resp*','spont','tune*');
%     elseif a==14
%         load('cadetv1p355_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p355_tuning','ori*','resp*','spont','tune*');
%     elseif a==15
% %         load('cadetv1p366_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%         load('cadetv1p366_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%     elseif a==16
% %         load('cadetv1p371_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%         load('cadetv1p371_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==17
%         load('cadetv1p384_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p384_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==18
%         load('cadetv1p385_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p385_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%     elseif a==19
% %         load('cadetv1p392_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%         load('cadetv1p392_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==20
%         load('cadetv1p403_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p403_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%     elseif a==21
% %         load('cadetv1p419_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%         load('cadetv1p419_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==22
%         load('cadetv1p432_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p432_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%     elseif a==23
% %         load('cadetv1p437_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%         load('cadetv1p437_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%     elseif a==24
% %         load('cadetv1p438_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%         load('cadetv1p438_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==25
%         load('cadetv1p460_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p460_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%     elseif a==26
% %         load('cadetv1p467_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%         load('cadetv1p467_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==27
%         load('cadetv1p468_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p468_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
%     elseif a==28
%         % before I made changes:
%         load('cadetv1p422_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
%         load('cadetv1p422_tuning','ori*','resp*','spont','tune*');
    end
    %%  preprocessing    
    
%     % select tuning criteria:
%     osi_tmp=[0 0.35 1];
%     for t=1:2
%         osi_keep=find(oribias_u_u>=osi_tmp(t) & oribias_u_u<=osi_tmp(t+1));
%         if t==1
%             savename=sprintf('%s_lda_pnf2_lowOSI',filename);
%         elseif t==2
%             savename=sprintf('%s_lda_pnf2_hiOSI',filename);
%         end
% %     end
% clearvars -except osi_tmp osi_keep t savename filename keep* ori* resp_bias resp_uniform
% %     num_units=length(osi_keep);
    

    %% % % % % % % % % % % performance w/ different # of neurons
    % population neurometric function
    
    % subsampled trials:
    base_keep=oris_u(keep);
    bias_keep=oris_b(keep2);

%     num_units=size(resp_raw_base_sub,1);
    % steps of units for analysis
%     steps=2:1:num_units;
    steps_fine=2:.25:110;
    % find and store index of each stim to call for training and test
    % (should be equal number of each if distribution created correctly)
    for b=1:length(oris)
        stimrefsu(b,:)=find(base_keep==oris(b));
        stimrefsb(b,:)=find(bias_keep==oris(b));
    end
    trainsize=round(0.9*size(stimrefsu,2));
    testsize=size(stimrefsu,2)-trainsize;
    name=sprintf('%s_lda_3class',filename);

    % select tuning criteria and run LDA:
    for w=1:3 % 3 ways of breaking up OSI
        if w==1 % fixed bins, no overlap
            osi_tmp=0.01:0.33:1;
            for b=1:length(osi_tmp)-1
                clearvars -except shiftori base_keep bias_keep steps* stimrefs*...
                    trainsize testsize resp_uniform resp_bias keep* ori*...
                    base bias name b w osi_tmp n responsive

                osi_keep=find(oribias_u>=osi_tmp(b) & oribias_u<=osi_tmp(b+1));
                
                % keep units of chosen OSI and matched trials:
                resp_raw_base_sub=resp_uniform(osi_keep,keep);
                resp_raw_bias_sub=resp_bias(osi_keep,keep2);
                
                % exclude 'unresponsive' units for LDA training
                resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_bias_sub,2)>55,:);
                resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_bias_sub,2)>55,:);
                resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_base_sub,2)>55,:);
                resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_base_sub,2)>55,:);
                
                % store # of units in each bin
                n_units_f=size(resp_raw_base_sub,1); 
                steps=2:1:n_units_f;
                
                pop_lda_up_train=nan*zeros(9,length(steps),200);
                pop_lda_bp_train=nan*zeros(9,length(steps),200);
                pop_lda_up_test=nan*zeros(9,length(steps),200);
                pop_lda_bp_test=nan*zeros(9,length(steps),200);
                class_up_test=nan*zeros(9,length(steps),200,testsize*3);
                class_bp_test=nan*zeros(9,length(steps),200,testsize*3);
                prob_up_test=nan*zeros(9,length(steps),200,testsize*3);
                prob_bp_test=nan*zeros(9,length(steps),200,testsize*3);
                tic
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
                    %         % select neighbor ori for 2 class:
                    %         if o==length(oris)
                    %             n1=1;   % last ori w/ first ori (160-0)
                    %         else
                    %             n1=o+1; % next ori (e.g. 0-20, 20-40...)
                    %         end
                    
                    idx=1;  % index for population size
                    % loop over population size:
                    for e=steps
                        id1=1; % index for loop over neuron samples and trials (20x neurons & 10x trials =200);
                        % loop over random neurons selected:
                        for c=1:20 % index for train/test trials
                            j=randperm(size(resp_raw_base_sub,1),e);
                            %             units_u=resp_raw_base(j,:);         % all trials
                            %             units_b=resp_raw_bias(j,:);
                            units_up=resp_raw_base_sub(j,:);    % matched distribution trials
                            units_bp=resp_raw_bias_sub(j,:);
                            
                            % loop over random trials selected:
                            for j = 1:10
                                % train and test trials:
                                tmp=randperm(size(stimrefsu,2),trainsize);
                                % for 3 class:
                                tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp) stimrefsu(n2,tmp)];
                                tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp) stimrefsb(n2,tmp)];
                                x=setdiff([stimrefsu(o,:) stimrefsu(n1,:) stimrefsu(n2,:)],tmp2);
                                y=setdiff([stimrefsb(o,:) stimrefsb(n1,:) stimrefsb(n2,:)],tmp3);
                                xshuf=length(x)/3;
                                % for 2 class:
                                %                     tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                                %                     tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                                %                     x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                                %                     y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                                %                     xshuf=length(x)/2;
                                
                                trainu=units_up(:,tmp2);
                                trainb=units_bp(:,tmp3);
                                testu=units_up(:,x);
                                testb=units_bp(:,y);
                                
                                % try to catch units that will creater error in Training data
                                temp=sum(trainu,2);
                                temp2=sum(trainb,2);
                                zerou=ismember(0,temp);
                                zerob=ismember(0,temp2);
                                if zerou==1 || zerob==1
                                    tmp=randperm(size(stimrefsu,2),trainsize);
                                    tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                                    tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                                    x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                                    y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                                    trainu=units_up(:,tmp2);
                                    trainb=units_bp(:,tmp3);
                                    testu=units_up(:,x);
                                    testb=units_bp(:,y);
                                    zerou=ismember(0,temp);
                                    zerob=ismember(0,temp2);
                                    if zerou==1 || zerob==1
                                        tmp=randperm(size(stimrefsu,2),trainsize);
                                        tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                                        tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                                        x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                                        y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                                        trainu=units_up(:,tmp2);
                                        trainb=units_bp(:,tmp3);
                                        testu=units_up(:,x);
                                        testb=units_bp(:,y);
                                        zerou=ismember(0,temp);
                                        zerob=ismember(0,temp2);
                                        if zerou==1 || zerob==1
                                            tmp=randperm(size(stimrefsu,2),trainsize);
                                            tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                                            tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                                            x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                                            y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                                            trainu=units_up(:,tmp2);
                                            trainb=units_bp(:,tmp3);
                                            testu=units_up(:,x);
                                            testb=units_bp(:,y);
                                            
                                        end
                                    end
                                end
                                
                                % run classifier:
                                [tmp_lda_up,tmp_err_up,tmp_prob_u]=classify(testu',trainu',base_keep(tmp2)');
                                [tmp_lda_bp,tmp_err_bp,tmp_prob_b]=classify(testb',trainb',bias_keep(tmp3)');
                                tmp5=find(tmp_lda_up'==base_keep(x)); % percent correct on test trials
                                tmp6=find(tmp_lda_bp'==bias_keep(y));
                                
                                
                                % store output
                                pop_lda_up_train(o,idx,id1)=1-tmp_err_up;
                                pop_lda_bp_train(o,idx,id1)=1-tmp_err_bp;
                                pop_lda_up_test(o,idx,id1)=length(tmp5)/length(tmp_lda_up);
                                pop_lda_bp_test(o,idx,id1)=length(tmp6)/length(tmp_lda_bp);
                                
                                % store classifications:
                                class_up_test(o,idx,id1,:)=tmp_lda_up';
                                class_bp_test(o,idx,id1,:)=tmp_lda_bp';
                                
                                % store posterior probabilities
                                prob_up_test(o,idx,id1,:)=tmp_prob_u(:,1)'; % prob in class 1
                                prob_bp_test(o,idx,id1,:)=tmp_prob_b(:,1)';
                                % structure: ori center, population size, [sampled neurons x trials: train/test]
                                id1=id1+1;
                                clear tmp* trainu* trainb* testu* testb*
                            end
                        end
                        idx=idx+1;
                    end
                    disp('ori done')
                end
                toc
                
                pop_lda_up_train_mean=squeeze(nanmean(pop_lda_up_train(:,:,:),3));
                pop_lda_bp_train_mean=squeeze(nanmean(pop_lda_bp_train(:,:,:),3));
                pop_lda_up_test_mean=squeeze(nanmean(pop_lda_up_test(:,:,:),3));
                pop_lda_bp_test_mean=squeeze(nanmean(pop_lda_bp_test(:,:,:),3));
                
                % fit performance data to Weibull
                % (center ori, pop size, repeats (10x10=100), performance: train & test)
                for e=1:length(oris)
                    % Xdata = # of steps (pop size); Ydata = % correct at that size
                    % only doing this for matched distribution
                    [par_up_train(e,:), gof_up_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean(e,:),[0.1 0.1 0.1],[]);
                    [par_bp_train(e,:), gof_bp_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean(e,:),[0.1 0.1 0.1],[]);
                    [par_up_test(e,:), gof_up_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean(e,:),[0.1 0.1 0.1],[]);
                    [par_bp_test(e,:), gof_bp_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean(e,:),[0.1 0.1 0.1],[]);
                    
                    fit_up_train(e,:)=1-par_up_train(e,3)*exp(-(steps_fine/par_up_train(e,1)).^par_up_train(e,2));
                    fit_bp_train(e,:)=1-par_bp_train(e,3)*exp(-(steps_fine/par_bp_train(e,1)).^par_bp_train(e,2));
                    fit_up_test(e,:)=1-par_up_test(e,3)*exp(-(steps_fine/par_up_test(e,1)).^par_up_test(e,2));
                    fit_bp_test(e,:)=1-par_bp_test(e,3)*exp(-(steps_fine/par_bp_test(e,1)).^par_bp_test(e,2));
                end
                disp('pnf done')
                clearvars -except shiftori base_keep bias_keep steps* stimrefs*...
                    trainsize testsize resp_uniform resp_bias keep* ori*...
                    base bias name b w osi_tmp n responsive fit* pop* class* prob* n_units* par*
                save(sprintf('%s_f%d',name,b))
            end
                
        elseif w==2 % fixed bins, with overlap
            osi_tmp=[0.1 0.4; 0.2 0.5; 0.3 0.6; 0.4 0.7; 0.5 0.8; 0.6 0.9; 0.7 1];
            for b=1:size(osi_tmp,1)
                clearvars -except shiftori base_keep bias_keep steps* stimrefs*...
                    trainsize testsize resp_uniform resp_bias keep* ori*...
                    base bias name b w osi_tmp n responsive

                osi_keep=find(oribias_u>=osi_tmp(b,1) & oribias_u<=osi_tmp(b,2));
                
                % keep units of chosen OSI and matched trials:
                resp_raw_base_sub=resp_uniform(osi_keep,keep);
                resp_raw_bias_sub=resp_bias(osi_keep,keep2);
                
                % exclude 'unresponsive' units for LDA training
                resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_bias_sub,2)>55,:);
                resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_bias_sub,2)>55,:);
                resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_base_sub,2)>55,:);
                resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_base_sub,2)>55,:);
                % store # of units in each bin
                n_units_s=size(resp_raw_base_sub,1); 
                steps=2:1:n_units_s;
                
                pop_lda_up_train=nan*zeros(9,length(steps),200);
                pop_lda_bp_train=nan*zeros(9,length(steps),200);
                pop_lda_up_test=nan*zeros(9,length(steps),200);
                pop_lda_bp_test=nan*zeros(9,length(steps),200);
                class_up_test=nan*zeros(9,length(steps),200,testsize*3);
                class_bp_test=nan*zeros(9,length(steps),200,testsize*3);
                prob_up_test=nan*zeros(9,length(steps),200,testsize*3);
                prob_bp_test=nan*zeros(9,length(steps),200,testsize*3);
                tic
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
                    %         % select neighbor ori for 2 class:
                    %         if o==length(oris)
                    %             n1=1;   % last ori w/ first ori (160-0)
                    %         else
                    %             n1=o+1; % next ori (e.g. 0-20, 20-40...)
                    %         end
                    
                    idx=1;  % index for population size
                    % loop over population size:
                    for e=steps
                        id1=1; % index for loop over neuron samples and trials (20x neurons & 10x trials =200);
                        % loop over random neurons selected:
                        for c=1:20 % index for train/test trials
                            j=randperm(size(resp_raw_base_sub,1),e);
                            %             units_u=resp_raw_base(j,:);         % all trials
                            %             units_b=resp_raw_bias(j,:);
                            units_up=resp_raw_base_sub(j,:);    % matched distribution trials
                            units_bp=resp_raw_bias_sub(j,:);
                            
                            % loop over random trials selected:
                            for j = 1:10
                                % train and test trials:
                                tmp=randperm(size(stimrefsu,2),trainsize);
                                % for 3 class:
                                tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp) stimrefsu(n2,tmp)];
                                tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp) stimrefsb(n2,tmp)];
                                x=setdiff([stimrefsu(o,:) stimrefsu(n1,:) stimrefsu(n2,:)],tmp2);
                                y=setdiff([stimrefsb(o,:) stimrefsb(n1,:) stimrefsb(n2,:)],tmp3);
                                xshuf=length(x)/3;
                                % for 2 class:
                                %                     tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                                %                     tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                                %                     x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                                %                     y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                                %                     xshuf=length(x)/2;
                                
                                trainu=units_up(:,tmp2);
                                trainb=units_bp(:,tmp3);
                                testu=units_up(:,x);
                                testb=units_bp(:,y);
                                
                                % try to catch units that will creater error in Training data
                                temp=sum(trainu,2);
                                temp2=sum(trainb,2);
                                zerou=ismember(0,temp);
                                zerob=ismember(0,temp2);
                                if zerou==1 || zerob==1
                                    tmp=randperm(size(stimrefsu,2),trainsize);
                                    tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                                    tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                                    x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                                    y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                                    trainu=units_up(:,tmp2);
                                    trainb=units_bp(:,tmp3);
                                    testu=units_up(:,x);
                                    testb=units_bp(:,y);
                                    zerou=ismember(0,temp);
                                    zerob=ismember(0,temp2);
                                    if zerou==1 || zerob==1
                                        tmp=randperm(size(stimrefsu,2),trainsize);
                                        tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                                        tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                                        x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                                        y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                                        trainu=units_up(:,tmp2);
                                        trainb=units_bp(:,tmp3);
                                        testu=units_up(:,x);
                                        testb=units_bp(:,y);
                                        zerou=ismember(0,temp);
                                        zerob=ismember(0,temp2);
                                        if zerou==1 || zerob==1
                                            tmp=randperm(size(stimrefsu,2),trainsize);
                                            tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                                            tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                                            x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                                            y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                                            trainu=units_up(:,tmp2);
                                            trainb=units_bp(:,tmp3);
                                            testu=units_up(:,x);
                                            testb=units_bp(:,y);
                                            
                                        end
                                    end
                                end
                                
                                % run classifier:
                                [tmp_lda_up,tmp_err_up,tmp_prob_u]=classify(testu',trainu',base_keep(tmp2)');
                                [tmp_lda_bp,tmp_err_bp,tmp_prob_b]=classify(testb',trainb',bias_keep(tmp3)');
                                tmp5=find(tmp_lda_up'==base_keep(x)); % percent correct on test trials
                                tmp6=find(tmp_lda_bp'==bias_keep(y));
                                
                                
                                % store output
                                pop_lda_up_train(o,idx,id1)=1-tmp_err_up;
                                pop_lda_bp_train(o,idx,id1)=1-tmp_err_bp;
                                pop_lda_up_test(o,idx,id1)=length(tmp5)/length(tmp_lda_up);
                                pop_lda_bp_test(o,idx,id1)=length(tmp6)/length(tmp_lda_bp);
                                
                                % store classifications:
                                class_up_test(o,idx,id1,:)=tmp_lda_up';
                                class_bp_test(o,idx,id1,:)=tmp_lda_bp';
                                
                                % store posterior probabilities
                                prob_up_test(o,idx,id1,:)=tmp_prob_u(:,1)'; % prob in class 1
                                prob_bp_test(o,idx,id1,:)=tmp_prob_b(:,1)';
                                % structure: ori center, population size, [sampled neurons x trials: train/test]
                                id1=id1+1;
                                clear tmp* trainu* trainb* testu* testb*
                            end
                        end
                        idx=idx+1;
                    end
                    disp('ori done')
                end
                toc
                
                pop_lda_up_train_mean=squeeze(nanmean(pop_lda_up_train(:,:,:),3));
                pop_lda_bp_train_mean=squeeze(nanmean(pop_lda_bp_train(:,:,:),3));
                pop_lda_up_test_mean=squeeze(nanmean(pop_lda_up_test(:,:,:),3));
                pop_lda_bp_test_mean=squeeze(nanmean(pop_lda_bp_test(:,:,:),3));
                
                % fit performance data to Weibull
                % (center ori, pop size, repeats (10x10=100), performance: train & test)
                for e=1:length(oris)
                    % Xdata = # of steps (pop size); Ydata = % correct at that size
                    % only doing this for matched distribution
                    [par_up_train(e,:), gof_up_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean(e,:),[0.1 0.1 0.1],[]);
                    [par_bp_train(e,:), gof_bp_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean(e,:),[0.1 0.1 0.1],[]);
                    [par_up_test(e,:), gof_up_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean(e,:),[0.1 0.1 0.1],[]);
                    [par_bp_test(e,:), gof_bp_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean(e,:),[0.1 0.1 0.1],[]);
                    
                    fit_up_train(e,:)=1-par_up_train(e,3)*exp(-(steps_fine/par_up_train(e,1)).^par_up_train(e,2));
                    fit_bp_train(e,:)=1-par_bp_train(e,3)*exp(-(steps_fine/par_bp_train(e,1)).^par_bp_train(e,2));
                    fit_up_test(e,:)=1-par_up_test(e,3)*exp(-(steps_fine/par_up_test(e,1)).^par_up_test(e,2));
                    fit_bp_test(e,:)=1-par_bp_test(e,3)*exp(-(steps_fine/par_bp_test(e,1)).^par_bp_test(e,2));
                end
                disp('pnf done')
                clearvars -except base_keep bias_keep steps* stimrefs*...
                    trainsize testsize resp_uniform resp_bias keep* ori*...
                    base bias name b w osi_tmp n responsive fit* pop* class* prob* n_units* par*
                save(sprintf('%s_s%d',name,b))
            end
        
        elseif w==3 % variable bins with equal units per bin
            [R,I]=sort(oribias_u);
            L=round(linspace(1,length(oribias_u),4));
            for b=1:length(L)-1
                clearvars -except base_keep bias_keep steps* stimrefs*...
                    trainsize testsize resp_uniform resp_bias keep* ori*...
                    base bias name b w osi_tmp n responsive L I

                osi_keep=(I(L(b):L(b+1)));
                
                % keep units of chosen OSI and matched trials:
                resp_raw_base_sub=resp_uniform(osi_keep,keep);
                resp_raw_bias_sub=resp_bias(osi_keep,keep2);
                
                % exclude 'unresponsive' units for LDA training
                resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_bias_sub,2)>55,:);
                resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_bias_sub,2)>55,:);
                resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_base_sub,2)>55,:);
                resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_base_sub,2)>55,:);
                % store # of units in each bin
                n_units_p=size(resp_raw_base_sub,1); 
                steps=2:1:n_units_p;
                
                pop_lda_up_train=nan*zeros(9,length(steps),200);
                pop_lda_bp_train=nan*zeros(9,length(steps),200);
                pop_lda_up_test=nan*zeros(9,length(steps),200);
                pop_lda_bp_test=nan*zeros(9,length(steps),200);
                class_up_test=nan*zeros(9,length(steps),200,testsize*3);
                class_bp_test=nan*zeros(9,length(steps),200,testsize*3);
                prob_up_test=nan*zeros(9,length(steps),200,testsize*3);
                prob_bp_test=nan*zeros(9,length(steps),200,testsize*3);
                tic
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
                    %         % select neighbor ori for 2 class:
                    %         if o==length(oris)
                    %             n1=1;   % last ori w/ first ori (160-0)
                    %         else
                    %             n1=o+1; % next ori (e.g. 0-20, 20-40...)
                    %         end
                    
                    idx=1;  % index for population size
                    % loop over population size:
                    for e=steps
                        id1=1; % index for loop over neuron samples and trials (20x neurons & 10x trials =200);
                        % loop over random neurons selected:
                        for c=1:20 % index for train/test trials
                            j=randperm(size(resp_raw_base_sub,1),e);
                            %             units_u=resp_raw_base(j,:);         % all trials
                            %             units_b=resp_raw_bias(j,:);
                            units_up=resp_raw_base_sub(j,:);    % matched distribution trials
                            units_bp=resp_raw_bias_sub(j,:);
                            
                            % loop over random trials selected:
                            for j = 1:10
                                % train and test trials:
                                tmp=randperm(size(stimrefsu,2),trainsize);
                                % for 3 class:
                                tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp) stimrefsu(n2,tmp)];
                                tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp) stimrefsb(n2,tmp)];
                                x=setdiff([stimrefsu(o,:) stimrefsu(n1,:) stimrefsu(n2,:)],tmp2);
                                y=setdiff([stimrefsb(o,:) stimrefsb(n1,:) stimrefsb(n2,:)],tmp3);
                                xshuf=length(x)/3;
                                % for 2 class:
                                %                     tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                                %                     tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                                %                     x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                                %                     y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                                %                     xshuf=length(x)/2;
                                
                                trainu=units_up(:,tmp2);
                                trainb=units_bp(:,tmp3);
                                testu=units_up(:,x);
                                testb=units_bp(:,y);
                                
                                % try to catch units that will creater error in Training data
                                temp=sum(trainu,2);
                                temp2=sum(trainb,2);
                                zerou=ismember(0,temp);
                                zerob=ismember(0,temp2);
                                if zerou==1 || zerob==1
                                    tmp=randperm(size(stimrefsu,2),trainsize);
                                    tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                                    tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                                    x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                                    y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                                    trainu=units_up(:,tmp2);
                                    trainb=units_bp(:,tmp3);
                                    testu=units_up(:,x);
                                    testb=units_bp(:,y);
                                    zerou=ismember(0,temp);
                                    zerob=ismember(0,temp2);
                                    if zerou==1 || zerob==1
                                        tmp=randperm(size(stimrefsu,2),trainsize);
                                        tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                                        tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                                        x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                                        y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                                        trainu=units_up(:,tmp2);
                                        trainb=units_bp(:,tmp3);
                                        testu=units_up(:,x);
                                        testb=units_bp(:,y);
                                        zerou=ismember(0,temp);
                                        zerob=ismember(0,temp2);
                                        if zerou==1 || zerob==1
                                            tmp=randperm(size(stimrefsu,2),trainsize);
                                            tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                                            tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                                            x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                                            y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                                            trainu=units_up(:,tmp2);
                                            trainb=units_bp(:,tmp3);
                                            testu=units_up(:,x);
                                            testb=units_bp(:,y);
                                            
                                        end
                                    end
                                end
                                
                                % run classifier:
                                [tmp_lda_up,tmp_err_up,tmp_prob_u]=classify(testu',trainu',base_keep(tmp2)');
                                [tmp_lda_bp,tmp_err_bp,tmp_prob_b]=classify(testb',trainb',bias_keep(tmp3)');
                                tmp5=find(tmp_lda_up'==base_keep(x)); % percent correct on test trials
                                tmp6=find(tmp_lda_bp'==bias_keep(y));
                                
                                
                                % store output
                                pop_lda_up_train(o,idx,id1)=1-tmp_err_up;
                                pop_lda_bp_train(o,idx,id1)=1-tmp_err_bp;
                                pop_lda_up_test(o,idx,id1)=length(tmp5)/length(tmp_lda_up);
                                pop_lda_bp_test(o,idx,id1)=length(tmp6)/length(tmp_lda_bp);
                                
                                % store classifications:
                                class_up_test(o,idx,id1,:)=tmp_lda_up';
                                class_bp_test(o,idx,id1,:)=tmp_lda_bp';
                                
                                % store posterior probabilities
                                prob_up_test(o,idx,id1,:)=tmp_prob_u(:,1)'; % prob in class 1
                                prob_bp_test(o,idx,id1,:)=tmp_prob_b(:,1)';
                                % structure: ori center, population size, [sampled neurons x trials: train/test]
                                id1=id1+1;
                                clear tmp* trainu* trainb* testu* testb*
                            end
                        end
                        idx=idx+1;
                    end
                    disp('ori done')
                end
                toc
                
                pop_lda_up_train_mean=squeeze(nanmean(pop_lda_up_train(:,:,:),3));
                pop_lda_bp_train_mean=squeeze(nanmean(pop_lda_bp_train(:,:,:),3));
                pop_lda_up_test_mean=squeeze(nanmean(pop_lda_up_test(:,:,:),3));
                pop_lda_bp_test_mean=squeeze(nanmean(pop_lda_bp_test(:,:,:),3));
                
                % fit performance data to Weibull
                % (center ori, pop size, repeats (10x10=100), performance: train & test)
                for e=1:length(oris)
                    % Xdata = # of steps (pop size); Ydata = % correct at that size
                    % only doing this for matched distribution
                    [par_up_train(e,:), gof_up_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean(e,:),[0.1 0.1 0.1],[]);
                    [par_bp_train(e,:), gof_bp_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean(e,:),[0.1 0.1 0.1],[]);
                    [par_up_test(e,:), gof_up_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean(e,:),[0.1 0.1 0.1],[]);
                    [par_bp_test(e,:), gof_bp_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean(e,:),[0.1 0.1 0.1],[]);
                    
                    fit_up_train(e,:)=1-par_up_train(e,3)*exp(-(steps_fine/par_up_train(e,1)).^par_up_train(e,2));
                    fit_bp_train(e,:)=1-par_bp_train(e,3)*exp(-(steps_fine/par_bp_train(e,1)).^par_bp_train(e,2));
                    fit_up_test(e,:)=1-par_up_test(e,3)*exp(-(steps_fine/par_up_test(e,1)).^par_up_test(e,2));
                    fit_bp_test(e,:)=1-par_bp_test(e,3)*exp(-(steps_fine/par_bp_test(e,1)).^par_bp_test(e,2));
                end
                disp('pnf done')
                clearvars -except base_keep bias_keep steps* stimrefs*...
                    trainsize testsize resp_uniform resp_bias keep* ori* L I...
                    base bias name b w osi_tmp n responsive fit* pop* class* prob* n_units* par*
                save(sprintf('%s_p%d',name,b))
            end
                
        end
    end  
end
stop
    
%     
% %     pop_lda_u=nan*zeros(length(steps),100,2);
% %     pop_lda_b=nan*zeros(length(steps),100,2);
%     pop_lda_up_train=nan*zeros(9,length(steps),200);
%     pop_lda_bp_train=nan*zeros(9,length(steps),200);
%     pop_lda_up_test=nan*zeros(9,length(steps),200);
%     pop_lda_bp_test=nan*zeros(9,length(steps),200);
%     pop_lda_up_train_shuf=nan*zeros(9,length(steps),200);
%     pop_lda_bp_train_shuf=nan*zeros(9,length(steps),200);
%     pop_lda_up_test_shuf=nan*zeros(9,length(steps),200);
%     pop_lda_bp_test_shuf=nan*zeros(9,length(steps),200);
% %     class_up_train=nan*zeros(9,length(steps),200,trainsize*2);
% %     class_bp_train=nan*zeros(9,length(steps),200,trainsize*2);
%     class_up_test=nan*zeros(9,length(steps),200,testsize*2);
%     class_bp_test=nan*zeros(9,length(steps),200,testsize*2);
% %     class_up_train_shuf=nan*zeros(9,length(steps),200,trainsize*2);
% %     class_bp_train_shuf=nan*zeros(9,length(steps),200,trainsize*2);
%     class_up_test_shuf=nan*zeros(9,length(steps),200,testsize*2);
%     class_bp_test_shuf=nan*zeros(9,length(steps),200,testsize*2);
% %     prob_up_train=nan*zeros(9,length(steps),200,trainsize*2);
% %     prob_bp_train=nan*zeros(9,length(steps),200,trainsize*2);
%     prob_up_test=nan*zeros(9,length(steps),200,testsize*2);
%     prob_bp_test=nan*zeros(9,length(steps),200,testsize*2);
% %     prob_up_train_shuf=nan*zeros(9,length(steps),200,trainsize*2);
% %     prob_bp_train_shuf=nan*zeros(9,length(steps),200,trainsize*2);
%     prob_up_test_shuf=nan*zeros(9,length(steps),200,testsize*2);
%     prob_bp_test_shuf=nan*zeros(9,length(steps),200,testsize*2);
%     
%     tic
%     % loop over orientations (center)
%     for o=1:length(oris)
%         % select neighbor oris for 3 class:
% %         if o==1
% %             n1=length(oris);
% %             n2=o+1;
% %         elseif o==length(oris)
% %             n1=o-1;
% %             n2=1;
% %         else
% %             n1=o-1;
% %             n2=o+1;
% %         end
%         % select neighbor ori for 2 class:
%         if o==length(oris)
%             n1=1;   % last ori w/ first ori (160-0)
%         else
%             n1=o+1; % next ori (e.g. 0-20, 20-40...)
%         end
%         
%         idx=1;  % index for population size 
%         % loop over population size:
%         for e=steps
%             id1=1; % index for loop over neuron samples and trials (20x neurons & 10x trials =200);
%             % loop over random neurons selected:
%             for b=1:20 % index for train/test trials
%                 j=randperm(size(resp_raw_base_sub,1),e);
%     %             units_u=resp_raw_base(j,:);         % all trials
%     %             units_b=resp_raw_bias(j,:);
%                 units_up=resp_raw_base_sub(j,:);    % matched distribution trials
%                 units_bp=resp_raw_bias_sub(j,:);
%                 
%                 % loop over random trials selected:
%                 for j = 1:10
%                     % train and test trials:
%                     tmp=randperm(size(stimrefsu,2),trainsize);
%                     % for 3 class:
% %                     tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp) stimrefsu(n2,tmp)];
% %                     tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp) stimrefsb(n2,tmp)];
% %                     x=setdiff([stimrefsu(o,:) stimrefsu(n1,:) stimrefsu(n2,:)],tmp2);
% %                     y=setdiff([stimrefsb(o,:) stimrefsb(n1,:) stimrefsb(n2,:)],tmp3);
% %                     xshuf=length(x)/3;
%                     % for 2 class:
%                     tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
%                     tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
%                     x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
%                     y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
%                     xshuf=length(x)/2;
%                     
%                     trainu=units_up(:,tmp2);
%                     trainb=units_bp(:,tmp3);
%                     testu=units_up(:,x);
%                     testb=units_bp(:,y);
%                     
%                     % shuffle train and test trials independently for each unit (row):
%                     for k=1:size(trainu,1)
%                         tmpu1shuf=randperm(trainsize);
%                         tmpb1shuf=randperm(trainsize);
%                         tmpu2shuf=randperm(trainsize);
%                         tmpb2shuf=randperm(trainsize);
% %                         tmpu3shuf=randperm(trainsize);    % only need these for 3 class
% %                         tmpb3shuf=randperm(trainsize);
% %                         trainu_shuf(k,:)=units_up(k,[stimrefsu(o,tmp(tmpu1shuf)) stimrefsu(n1,tmp(tmpu2shuf)) stimrefsu(n2,tmp(tmpu3shuf))]);
% %                         trainb_shuf(k,:)=units_bp(k,[stimrefsb(o,tmp(tmpb1shuf)) stimrefsb(n1,tmp(tmpb2shuf)) stimrefsb(n2,tmp(tmpb3shuf))]);
%                         trainu_shuf(k,:)=units_up(k,[stimrefsu(o,tmp(tmpu1shuf)) stimrefsu(n1,tmp(tmpu2shuf))]);
%                         trainb_shuf(k,:)=units_bp(k,[stimrefsb(o,tmp(tmpb1shuf)) stimrefsb(n1,tmp(tmpb2shuf))]);
%                         
%                         tshufu1=randperm(xshuf);
%                         tshufb1=randperm(xshuf);
%                         tshufu2=randperm(xshuf)+xshuf;
%                         tshufb2=randperm(xshuf)+xshuf;
% %                         tshufu3=randperm(xshuf)+xshuf*2;  % only need these for 3 class
% %                         tshufb3=randperm(xshuf)+xshuf*2;
% %                         testu_shuf(k,:)=testu(k,[tshufu1 tshufu2 tshufu3]);
% %                         testb_shuf(k,:)=testb(k,[tshufb1 tshufb2 tshufb3]);
%                         testu_shuf(k,:)=testu(k,[tshufu1 tshufu2]);
%                         testb_shuf(k,:)=testb(k,[tshufb1 tshufb2]);
%                     end
% 
%                     % run classifier:
%     %                 [tmp_lda_u,tmp_err_u]=classify(units_u(:,testu)',units_u(:,trainu)',base2(trainu)');
%     %                 [tmp_lda_b,tmp_err_b]=classify(units_b(:,testb)',units_b(:,trainb)',bias2(trainb)');
%                     [tmp_lda_up,tmp_err_up,tmp_prob_u]=classify(testu',trainu',base_keep(tmp2)');
%                     [tmp_lda_bp,tmp_err_bp,tmp_prob_b]=classify(testb',trainb',bias_keep(tmp3)');
%                     % percent correct on test trials
%                     tmp5=find(tmp_lda_up'==base_keep(x));
%                     tmp6=find(tmp_lda_bp'==bias_keep(y));
%                     % shuffled data:
%                     [tmp_lda_up_shuf,tmp_err_up_shuf,tmp_prob_u_shuf]=classify(testu_shuf',trainu_shuf',base_keep(tmp2)');
%                     [tmp_lda_bp_shuf,tmp_err_bp_shuf,tmp_prob_b_shuf]=classify(testb_shuf',trainb_shuf',bias_keep(tmp3)');
%                     % percent correct on test trials
%                     tmp7=find(tmp_lda_up_shuf'==base_keep(x));
%                     tmp8=find(tmp_lda_bp_shuf'==bias_keep(y));
% 
%                     % store performance output
%     %                 pop_lda_u(idx,id,:)=[1-tmp_err_u length(tmp5)/length(tmp_lda_u)];
%     %                 pop_lda_b(idx,id,:)=[1-tmp_err_b length(tmp6)/length(tmp_lda_b)];
%                     pop_lda_up_train(o,idx,id1)=1-tmp_err_up;
%                     pop_lda_bp_train(o,idx,id1)=1-tmp_err_bp;
%                     pop_lda_up_test(o,idx,id1)=length(tmp5)/length(tmp_lda_up);
%                     pop_lda_bp_test(o,idx,id1)=length(tmp6)/length(tmp_lda_bp);
%                     pop_lda_up_train_shuf(o,idx,id1)=1-tmp_err_up_shuf;
%                     pop_lda_bp_train_shuf(o,idx,id1)=1-tmp_err_bp_shuf;
%                     pop_lda_up_test_shuf(o,idx,id1)=length(tmp7)/length(tmp_lda_up_shuf);
%                     pop_lda_bp_test_shuf(o,idx,id1)=length(tmp8)/length(tmp_lda_bp_shuf);
%                     % store classifications:
%                     class_up_test(o,idx,id1,:)=tmp_lda_up';
%                     class_bp_test(o,idx,id1,:)=tmp_lda_bp';
%                     class_up_test_shuf(o,idx,id1,:)=tmp_lda_up_shuf';
%                     class_bp_test_shuf(o,idx,id1,:)=tmp_lda_bp_shuf';
%                     % store posterior probabilities
%                     prob_up_test(o,idx,id1,:)=tmp_prob_u(:,1)'; % prob in class 1
%                     prob_bp_test(o,idx,id1,:)=tmp_prob_b(:,1)';
%                     prob_up_test_shuf(o,idx,id1,:)=tmp_prob_u_shuf(:,1)';
%                     prob_bp_test_shuf(o,idx,id1,:)=tmp_prob_b_shuf(:,1)';
%                     % structure: ori center, population size, [sampled neurons x trials: train/test]
%                     id1=id1+1;
%                     clear tmp* trainu* trainb* testu* testb*
%                 end
%             end
%             idx=idx+1;
%         end
%         disp('ori comparison done')
%     end
%     clear o e b j k tmp* id1 idx
%     toc
%     
%     pop_lda_up_train_mean=squeeze(nanmean(pop_lda_up_train(:,:,:),3));
%     pop_lda_bp_train_mean=squeeze(nanmean(pop_lda_bp_train(:,:,:),3));
%     pop_lda_up_test_mean=squeeze(nanmean(pop_lda_up_test(:,:,:),3));
%     pop_lda_bp_test_mean=squeeze(nanmean(pop_lda_bp_test(:,:,:),3));
%     pop_lda_up_train_mean_shuf=squeeze(nanmean(pop_lda_up_train_shuf(:,:,:),3));
%     pop_lda_bp_train_mean_shuf=squeeze(nanmean(pop_lda_bp_train_shuf(:,:,:),3));
%     pop_lda_up_test_mean_shuf=squeeze(nanmean(pop_lda_up_test_shuf(:,:,:),3));
%     pop_lda_bp_test_mean_shuf=squeeze(nanmean(pop_lda_bp_test_shuf(:,:,:),3));
%     
%     
%     % fit performance data and find 75% performance mark
%     % (center ori, pop size, repeats (10x10=100), performance: train & test)
%     for e=1:length(oris)
%         % Xdata = # of steps (pop size); Ydata = % correct at that size
%         % only doing this for matched distribution
%         % previously used SigmoidFit - now using weibel function
%         [par_up_train(e,:), gof_up_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean(e,:),[0.1 0.1 0.1],[]);
%         [par_bp_train(e,:), gof_bp_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean(e,:),[0.1 0.1 0.1],[]);
%         [par_up_test(e,:), gof_up_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean(e,:),[0.1 0.1 0.1],[]);
%         [par_bp_test(e,:), gof_bp_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean(e,:),[0.1 0.1 0.1],[]);
%         
%         [par_up_train_shuf(e,:), gof_up_train_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean_shuf(e,:),[0.1 0.1 0.1],[]);
%         [par_bp_train_shuf(e,:), gof_bp_train_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean_shuf(e,:),[0.1 0.1 0.1],[]);
%         [par_up_test_shuf(e,:), gof_up_test_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean_shuf(e,:),[0.1 0.1 0.1],[]);
%         [par_bp_test_shuf(e,:), gof_bp_test_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean_shuf(e,:),[0.1 0.1 0.1],[]);
%         
%         fit_up_train(e,:)=1-par_up_train(e,3)*exp(-(steps_fine/par_up_train(e,1)).^par_up_train(e,2));
%         fit_bp_train(e,:)=1-par_bp_train(e,3)*exp(-(steps_fine/par_bp_train(e,1)).^par_bp_train(e,2));
%         fit_up_test(e,:)=1-par_up_test(e,3)*exp(-(steps_fine/par_up_test(e,1)).^par_up_test(e,2));
%         fit_bp_test(e,:)=1-par_bp_test(e,3)*exp(-(steps_fine/par_bp_test(e,1)).^par_bp_test(e,2));
%         
%         fit_up_train_shuf(e,:)=1-par_up_train_shuf(e,3)*exp(-(steps_fine/par_up_train_shuf(e,1)).^par_up_train_shuf(e,2));
%         fit_bp_train_shuf(e,:)=1-par_bp_train_shuf(e,3)*exp(-(steps_fine/par_bp_train_shuf(e,1)).^par_bp_train_shuf(e,2));
%         fit_up_test_shuf(e,:)=1-par_up_test_shuf(e,3)*exp(-(steps_fine/par_up_test_shuf(e,1)).^par_up_test_shuf(e,2));
%         fit_bp_test_shuf(e,:)=1-par_bp_test_shuf(e,3)*exp(-(steps_fine/par_bp_test_shuf(e,1)).^par_bp_test_shuf(e,2));
%         
%         % don't need this threshold info anymore:
% %         if max(fit_up_train(e,:))<0.75 || max(fit_up_test(e,:))<0.75
% %             thresh_u_train(e)=find(fit_up_train(e,:)>=.75*max(fit_up_train(e,:)),1,'first');
% %             thresh_u_test(e)=find(fit_up_test(e,:)>=.75*max(fit_up_test(e,:)),1,'first');
% %         else
% %             thresh_u_train(e)=find(fit_up_train(e,:)>=.75,1,'first');
% %             thresh_u_test(e)=find(fit_up_test(e,:)>=.75,1,'first');
% %         end
% %         if max(fit_bp_train(e,:))<0.75 || max(fit_bp_test(e,:))<0.75
% %             thresh_b_train(e)=find(fit_bp_train(e,:)>=.75*max(fit_bp_train(e,:)),1,'first');
% %             thresh_b_test(e)=find(fit_bp_test(e,:)>=.75*max(fit_bp_test(e,:)),1,'first');
% %         else
% %             thresh_b_train(e)=find(fit_bp_train(e,:)>=.75,1,'first');
% %             thresh_b_test(e)=find(fit_bp_test(e,:)>=.75,1,'first');
% %         end
% %         if max(fit_up_train_shuf(e,:))<0.75 || max(fit_up_test_shuf(e,:))<0.75
% %             thresh_u_train_shuf(e)=find(fit_up_train_shuf(e,:)>=.75*max(fit_up_train_shuf(e,:)),1,'first');
% %             thresh_u_test_shuf(e)=find(fit_up_test_shuf(e,:)>=.75*max(fit_up_test_shuf(e,:)),1,'first');
% %         else
% %             thresh_u_train_shuf(e)=find(fit_up_train_shuf(e,:)>=.75,1,'first');
% %             thresh_u_test_shuf(e)=find(fit_up_test_shuf(e,:)>=.75,1,'first');
% %         end
% %         if max(fit_bp_train_shuf(e,:))<0.75 || max(fit_bp_test_shuf(e,:))<0.75
% %             thresh_b_train_shuf(e)=find(fit_bp_train_shuf(e,:)>=.75*max(fit_bp_train_shuf(e,:)),1,'first');
% %             thresh_b_test_shuf(e)=find(fit_bp_test_shuf(e,:)>=.75*max(fit_bp_test_shuf(e,:)),1,'first');
% %         else
% %             thresh_b_train_shuf(e)=find(fit_bp_train_shuf(e,:)>=.75,1,'first');
% %             thresh_b_test_shuf(e)=find(fit_bp_test_shuf(e,:)>=.75,1,'first');
% %         end
%         
%         % previous code
% %         if max(fit_up_train{e}(steps_fine)) <0.75 || max(fit_up_test{e}(steps_fine)) <0.75 
% %             thresh_u_train(e)=steps_fine(find(fit_up_train{e}(steps_fine)>=.75*max(fit_up_train{e}(steps_fine)),1,'first'));
% %             thresh_u_test(e)=steps_fine(find(fit_up_test{e}(steps_fine)>=.75*max(fit_up_test{e}(steps_fine)),1,'first'));
% %         else
% %             thresh_u_train(e)=steps_fine(find(fit_up_train{e}(steps_fine)>=.75,1,'first'));
% %             thresh_u_test(e)=steps_fine(find(fit_up_test{e}(steps_fine)>=.75,1,'first'));
% %         end
% %         if max(fit_bp_train{e}(steps_fine)) <0.75 || max(fit_bp_test{e}(steps_fine)) <0.75
% %             thresh_b_train(e)=steps_fine(find(fit_bp_train{e}(steps_fine)>=.75*max(fit_bp_train{e}(steps_fine)),1,'first'));
% %             thresh_b_test(e)=steps_fine(find(fit_bp_test{e}(steps_fine)>=.75*max(fit_bp_test{e}(steps_fine)),1,'first'));
% %         else
% %             thresh_b_train(e)=steps_fine(find(fit_bp_train{e}(steps_fine)>=.75,1,'first'));
% %             thresh_b_test(e)=steps_fine(find(fit_bp_test{e}(steps_fine)>=.75,1,'first'));
% %         end
%     end
%     
%     % plot LDA performance, separated by center ori (3-class)
%     figure
%     ori_labels=0:20:160;
%     supertitle('Awake, LDA performance vs population size')
%     for e=1:length(ori_labels)
%         subplot(3,3,e); hold on
%         title(sprintf('ori %g',ori_labels(e)))
%         hold on
% %         errorline(1+0.4:1:size(pop_lda_up_train,2)+0.4,pop_lda_up_train_mean(e,:),nanstd(squeeze(pop_lda_up_train(e,:,:))'),'k--')
% %         errorline(1+0.4:1:size(pop_lda_bp_train,2)+0.4,pop_lda_bp_train_mean(e,:),nanstd(squeeze(pop_lda_bp_train(e,:,:))'),'r--')
% %         errorline(1:size(pop_lda_up_test,2),pop_lda_up_test_mean(e,:),nanstd(squeeze(pop_lda_up_train(e,:,:))'),'k.')
% %         errorline(1:size(pop_lda_bp_test,2),pop_lda_bp_test_mean(e,:),nanstd(squeeze(pop_lda_bp_train(e,:,:))'),'r.')
%         
% %         errorline(1+0.4:1:size(pop_lda_up_train_shuf,2)+0.4,pop_lda_up_train_mean_shuf(e,:),nanstd(squeeze(pop_lda_up_train_shuf(e,:,:))'),'b--')
% %         errorline(1+0.4:1:size(pop_lda_bp_train_shuf,2)+0.4,pop_lda_bp_train_mean_shuf(e,:),nanstd(squeeze(pop_lda_bp_train_shuf(e,:,:))'),'m--')
% %         errorline(1:size(pop_lda_up_test_shuf,2),pop_lda_up_test_mean_shuf(e,:),nanstd(squeeze(pop_lda_up_train_shuf(e,:,:))'),'b.')
% %         errorline(1:size(pop_lda_bp_test_shuf,2),pop_lda_bp_test_mean_shuf(e,:),nanstd(squeeze(pop_lda_bp_train_shuf(e,:,:))'),'m.')
%         
%     %     errorline(2:size(resp_raw_base,1),1-mean(pop_err_u,1),std(pop_err_u,1),'k')
%     %     errorline(2:size(resp_raw_base,1),1-mean(pop_err_b,1),std(pop_err_b,1),'r')
%     
%         plot(steps_fine,fit_up_test(e,:),'k')
%         plot(steps_fine,fit_bp_test(e,:),'r')
%         plot(steps_fine,fit_up_test_shuf(e,:),'b')
%         plot(steps_fine,fit_bp_test_shuf(e,:),'m')
%         
%         axis([0 100 0.3 1])
%         plot(0.33*ones(100,1),':k')
%         axis square; box off
% %         if e==1
% %             legend({'unif training','bias training','unif test','bias test'},'Location','southeast')
% %         end
%         xlabel('# of units')
%         ylabel('% correct')
% %     set(gca,'XTick',...
%     end
%     
% %     figure
% %     oris=0:20:160;
% %     supertitle('awake pnf fits')
% %     for e=1:length(oris)
% %         subplot(3,3,e); hold on
% %         title(sprintf('ori %g',e))
% %         plot(steps_fine,fit_up_train(e,:),'k')
% %         plot(steps_fine,fit_up_test(e,:),'k--')
% %         plot(steps_fine,fit_bp_train(e,:),'r')
% %         plot(steps_fine,fit_bp_test(e,:),'r--')
% %         if e==1
% %             legend({'unif train','unif test','bias train','bias test'},'Location','southeast')
% %             ylabel('% correct')
% %             xlabel('# of units')
% %         end
% %         axis square
% %         ylim([0.3 1])
% %     end
%     
% %     figure
% %     if shiftori==1
% %         plot(circshift(thresh_b_train./thresh_u_train,0),'k')
% %         plot(circshift(thresh_b_test./thresh_u_test,0),'r')
% %     else
% %         plot(circshift(thresh_b_train./thresh_u_train,4),'k')
% %         plot(circshift(thresh_b_test./thresh_u_test,4),'r')
% %     end
% %     axis square; box off
% %     xlabel('center ori of 3-class')
% %     ylabel('relative change in 75% threshold')
% %     title('LDA Neurometric Threshold')
%     
%     disp('file done')
%     clearvars -except thresh* pop* num_units ori_base oripref resp* stim*...
%         oribias_u_u oris_u oris_b oris steps* shiftori fit* gof* filename savename...
%         a par* class* trainsize testsize prob* t osi_tmp keep*
%     save(savename)
%     end
% %     stop
% end