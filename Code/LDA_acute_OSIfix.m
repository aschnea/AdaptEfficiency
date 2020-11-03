clear
% for n=[19 22 25 30 33]%1:33
for n=[30 33]
    clearvars -except n
    if n==1
        load('129r001p173_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='129r001p173_lda_pnf';
%     elseif n==2
%         load('130l001p169_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='130l001p169_lda_pnf';
%     elseif n==3
%         load('140l001p107_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='140l001p107_lda_pnf';
%     elseif n==4
%         load('140l001p122_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='140l001p122_lda_pnf';
%     elseif n==5
%         load('140r001p105_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='140r001p105_lda_pnf';
%     elseif n==6
%         load('140r001p122_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='140r001p122_lda_pnf';
%     elseif n==7
%         load('130l001p170_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='130l001p170_lda_pnf';
%     elseif n==8
%         load('140l001p108_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='140l001p108_lda_pnf';
%     elseif n==9
%         load('140l001p110_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='140l001p110_lda_pnf';
%     elseif n==10
%         load('140r001p107_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='140r001p107_lda_pnf';
%     elseif n==11
%         load('140r001p109_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='140r001p109_lda_pnf';
%     elseif n==12
%         load('lowcon114_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='lowcon114_lda_pnf';
%     elseif n==13
%         load('lowcon115_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='lowcon115_lda_pnf';
%     elseif n==14
%         load('lowcon116_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='lowcon116_lda_pnf';
%     elseif n==15
%         load('lowcon117_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='lowcon117_lda_pnf';
%     elseif n==16
%         load('140l113_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='140l113_awaketime_lda_pnf';
%     elseif n==17
%         load('140r113_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
%         name='140r113_awaketime_lda_pnf';
%     
%     elseif n==18 % start of experiment 141 files
%         load('141r001p006_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
%         name='141r001p006_awaketime_lda_pnf';
    elseif n==19
        load('141r001p007_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p007_awaketime6_lda_pnf_2class';
%     elseif n==20
%         load('141r001p009_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
%         name='141r001p009_awaketime_fine_lda_pnf';
%     elseif n==21 % rotated AT 4:1 (80°)
%         load('141r001p024_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
%         name='141r001p024_awaketime_lda_pnf';
    elseif n==22 % rotated AT 6:1 (80°)
        load('141r001p025_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p025_awaketime6_lda_pnf_2class';
%     elseif n==23 % rotated AT fineori (90°??)
%         load('141r001p027_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
%         name='141r001p027_awaketime_fine_lda_pnf';
%     elseif n==24 % rotated fineori (40°)
%         load('141r001p038_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
%         name='141r001p038_awaketime_fine_lda_pnf';
    elseif n==25 % rotated 6:1 (120°)
        load('141r001p039_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p039_awaketime6_lda_pnf_2class';
%     elseif n==26 % rotated awaketime 4:1 (120°)
%         load('141r001p041_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
%         name='141r001p041_awaketime_lda_pnf';
%     elseif n==27
%         load('141r001p114_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
%         name='141r001p114_lda_pnf';
%         
%     elseif n==28
%         load('142l001p002_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
%         name='142l001p002_awaketime_lda_pnf';
%     elseif n==29
%         load('142l001p004_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
%         name='142l001p004_awaketime_fine_lda_pnf';
    elseif n==30
        load('142l001p006_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p006_awaketime6_lda_pnf_3class';
%     elseif n==31
%         load('142l001p007_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
%         name='142l001p007_awaketime_lda_pnf';
%     elseif n==32
%         load('142l001p009_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
%         name='142l001p009_awaketime_fine_lda_pnf';
    elseif n==33
        load('142l001p010_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p010_awaketime6_lda_pnf_3class';
    end
    
    %%  preprocessing
    % identify bias ori and align biased ori to first column
    val_mode=mode(bias);   
    if val_mode==0
        shiftori=0;
    else
        shiftori=1;
    end
    % remove trials with blank stimulus
    [tmp]=find(base(keep)~=200);
    [tmp2]=find(bias(keep2)~=200);
    keep=keep(tmp);
    keep2=keep2(tmp2);
    base_keep=base(keep);
    bias_keep=bias(keep2);
    
    % steps of units for analysis
%     steps=2:1:num_units;
    steps_fine=2:.25:100;
    % find and store cases of each stim to call for training and test
    % (should be equal number of each if distribution created correctly)
    for b=1:length(ori_base)-1
        stimrefsu(b,:)=find(base_keep==ori_base(b));
        stimrefsb(b,:)=find(bias_keep==ori_base(b));
    end
    trainsize=round(0.9*size(stimrefsu,2));
    testsize=size(stimrefsu,2)-trainsize;

    % select tuning criteria and run LDA:
    for w=1:3 % 3 ways of breaking up OSI
        if w==1 % fixed bins, no overlap
            osi_tmp=0.01:0.33:1;
            for b=1:length(osi_tmp)-1
                clearvars -except shiftori base_keep bias_keep steps* stimrefs*...
                    trainsize testsize resp_raw_base resp_raw_bias keep* ori*...
                    base bias name b w osi_tmp n responsive

                osi_keep=find(oribias>=osi_tmp(b) & oribias<=osi_tmp(b+1));
                
                % keep units of chosen OSI and matched trials:
                resp_raw_base_sub=resp_raw_base(osi_keep,keep);
                resp_raw_bias_sub=resp_raw_bias(osi_keep,keep2);
                
                % exclude 'unresponsive' units for LDA training
                resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_bias_sub,2)>55,:);
                resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_bias_sub,2)>55,:);
                resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_base_sub,2)>55,:);
                resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_base_sub,2)>55,:);
                % store # of units in each bin
                n_units_f=length(osi_keep); 
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
                for o=1:length(ori_base)-1
                    % select 3 oris for comparison:
                    if o==1
                        n1=length(ori_base)-1;
                        n2=o+1;
                    elseif o==length(ori_base)-1
                        n1=o-1;
                        n2=1;
                    else
                        n1=o-1;
                        n2=o+1;
                    end
                    %         % select neighbor ori for 2 class:
                    %         if o==length(ori_base)-1
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
                for e=1:length(ori_base)-1
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
                    trainsize testsize resp_raw_base resp_raw_bias keep* ori*...
                    base bias name b w osi_tmp n responsive fit* pop* class* prob* n_units* par*
                save(sprintf('%s_f%d',name,b))
            end
                
        elseif w==2 % fixed bins, with overlap
            osi_tmp=[0.1 0.4; 0.2 0.5; 0.3 0.6; 0.4 0.7; 0.5 0.8; 0.6 0.9; 0.7 1];
            for b=1:size(osi_tmp,1)
                clearvars -except shiftori base_keep bias_keep steps* stimrefs*...
                    trainsize testsize resp_raw_base resp_raw_bias keep* ori*...
                    base bias name b w osi_tmp n responsive

                osi_keep=find(oribias>=osi_tmp(b,1) & oribias<=osi_tmp(b,2));
                
                % keep units of chosen OSI and matched trials:
                resp_raw_base_sub=resp_raw_base(osi_keep,keep);
                resp_raw_bias_sub=resp_raw_bias(osi_keep,keep2);
                
                % exclude 'unresponsive' units for LDA training
                resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_bias_sub,2)>55,:);
                resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_bias_sub,2)>55,:);
                resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_base_sub,2)>55,:);
                resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_base_sub,2)>55,:);
                % store # of units in each bin
                n_units_s=length(osi_keep); 
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
                for o=1:length(ori_base)-1
                    % select 3 oris for comparison:
                    if o==1
                        n1=length(ori_base)-1;
                        n2=o+1;
                    elseif o==length(ori_base)-1
                        n1=o-1;
                        n2=1;
                    else
                        n1=o-1;
                        n2=o+1;
                    end
                    %         % select neighbor ori for 2 class:
                    %         if o==length(ori_base)-1
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
                for e=1:length(ori_base)-1
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
                    trainsize testsize resp_raw_base resp_raw_bias keep* ori*...
                    base bias name b w osi_tmp n responsive fit* pop* class* prob* n_units* par*
                save(sprintf('%s_s%d',name,b))
            end
        
        elseif w==3 % variable bins with equal units per bin
            [R,I]=sort(oribias);
            L=round(linspace(1,length(oribias),4));
            for b=1:length(L)-1
                clearvars -except shiftori base_keep bias_keep steps* stimrefs*...
                    trainsize testsize resp_raw_base resp_raw_bias keep* ori*...
                    base bias name b w osi_tmp n responsive L I

                osi_keep=(I(L(b):L(b+1)));
                
                % keep units of chosen OSI and matched trials:
                resp_raw_base_sub=resp_raw_base(osi_keep,keep);
                resp_raw_bias_sub=resp_raw_bias(osi_keep,keep2);
                
                % exclude 'unresponsive' units for LDA training
                resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_bias_sub,2)>55,:);
                resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_bias_sub,2)>55,:);
                resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_base_sub,2)>55,:);
                resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_base_sub,2)>55,:);
                % store # of units in each bin
                n_units_p=length(osi_keep); 
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
                for o=1:length(ori_base)-1
                    % select 3 oris for comparison:
                    if o==1
                        n1=length(ori_base)-1;
                        n2=o+1;
                    elseif o==length(ori_base)-1
                        n1=o-1;
                        n2=1;
                    else
                        n1=o-1;
                        n2=o+1;
                    end
                    %         % select neighbor ori for 2 class:
                    %         if o==length(ori_base)-1
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
                for e=1:length(ori_base)-1
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
                    trainsize testsize resp_raw_base resp_raw_bias keep* ori* L I...
                    base bias name b w osi_tmp n responsive fit* pop* class* prob* n_units* par*
                save(sprintf('%s_p%d',name,b))
            end
                
        end
    end  
end
stop
clearvars -except responsive keep* ori* resp_raw_base resp_raw_bias base bias name t osi_tmp osi_keep n
    

%     %%  performance w/ different # of neurons
%     
%     tic
%     for o=1:length(ori_base)-1
%         % select 3 oris for comparison:
%         if o==1
%             n1=length(ori_base)-1;
%             n2=o+1;
%         elseif o==length(ori_base)-1
%             n1=o-1;
%             n2=1;
%         else
%             n1=o-1;
%             n2=o+1;
%         end
% %         % select neighbor ori for 2 class:
% %         if o==length(ori_base)-1
% %             n1=1;   % last ori w/ first ori (160-0)
% %         else
% %             n1=o+1; % next ori (e.g. 0-20, 20-40...)
% %         end
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
%                     tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp) stimrefsu(n2,tmp)];
%                     tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp) stimrefsb(n2,tmp)];
%                     x=setdiff([stimrefsu(o,:) stimrefsu(n1,:) stimrefsu(n2,:)],tmp2);
%                     y=setdiff([stimrefsb(o,:) stimrefsb(n1,:) stimrefsb(n2,:)],tmp3);
%                     xshuf=length(x)/3;
%                     % for 2 class:
% %                     tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
% %                     tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
% %                     x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
% %                     y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
% %                     xshuf=length(x)/2;
%                     
%                     trainu=units_up(:,tmp2);
%                     trainb=units_bp(:,tmp3);
%                     testu=units_up(:,x);
%                     testb=units_bp(:,y);
%                     
%                     % try to catch units that will creater error in Training data
%                     temp=sum(trainu,2);
%                     temp2=sum(trainb,2);
%                     zerou=ismember(0,temp);
%                     zerob=ismember(0,temp2);
%                     if zerou==1 || zerob==1     
%                         tmp=randperm(size(stimrefsu,2),trainsize);
%                         tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
%                         tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
%                         x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
%                         y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
%                         trainu=units_up(:,tmp2);
%                         trainb=units_bp(:,tmp3);
%                         testu=units_up(:,x);
%                         testb=units_bp(:,y);
%                         zerou=ismember(0,temp);
%                         zerob=ismember(0,temp2);
%                         if zerou==1 || zerob==1
%                             tmp=randperm(size(stimrefsu,2),trainsize);
%                             tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
%                             tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
%                             x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
%                             y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
%                             trainu=units_up(:,tmp2);
%                             trainb=units_bp(:,tmp3);
%                             testu=units_up(:,x);
%                             testb=units_bp(:,y);
%                             zerou=ismember(0,temp);
%                             zerob=ismember(0,temp2);
%                             if zerou==1 || zerob==1
%                                 tmp=randperm(size(stimrefsu,2),trainsize);
%                                 tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
%                                 tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
%                                 x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
%                                 y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
%                                 trainu=units_up(:,tmp2);
%                                 trainb=units_bp(:,tmp3);
%                                 testu=units_up(:,x);
%                                 testb=units_bp(:,y);
%                                 
%                             end
%                         end
%                     end
%                     
% %                     % shuffle train trials independently for each unit (row):
% %                     for k=1:size(trainu,1)
% % %                         tmpu1shuf=randperm(trainsize);
% % %                         tmpb1shuf=randperm(trainsize);
% % %                         tmpu2shuf=randperm(trainsize);
% % %                         tmpb2shuf=randperm(trainsize);
% % % %                         tmpu3shuf=randperm(trainsize);    % only need these for 3 class
% % % %                         tmpb3shuf=randperm(trainsize);
% % % %                         trainu_shuf(k,:)=units_up(k,[stimrefsu(o,tmp(tmpu1shuf)) stimrefsu(n1,tmp(tmpu2shuf)) stimrefsu(n2,tmp(tmpu3shuf))]);
% % % %                         trainb_shuf(k,:)=units_bp(k,[stimrefsb(o,tmp(tmpb1shuf)) stimrefsb(n1,tmp(tmpb2shuf)) stimrefsb(n2,tmp(tmpb3shuf))]);
% % %                         trainu_shuf(k,:)=units_up(k,[stimrefsu(o,tmp(tmpu1shuf)) stimrefsu(n1,tmp(tmpu2shuf))]);
% % %                         trainb_shuf(k,:)=units_bp(k,[stimrefsb(o,tmp(tmpb1shuf)) stimrefsb(n1,tmp(tmpb2shuf))]);
% %                         
% %                         tshufu1=randperm(xshuf);
% %                         tshufb1=randperm(xshuf);
% %                         tshufu2=randperm(xshuf)+xshuf;
% %                         tshufb2=randperm(xshuf)+xshuf;
% % %                         tshufu3=randperm(xshuf)+xshuf*2;  % only need these for 3 class
% % %                         tshufb3=randperm(xshuf)+xshuf*2;
% % %                         testu_shuf(k,:)=testu(k,[tshufu1 tshufu2 tshufu3]);
% % %                         testb_shuf(k,:)=testb(k,[tshufb1 tshufb2 tshufb3]);
% %                         testu_shuf(k,:)=testu(k,[tshufu1 tshufu2]);
% %                         testb_shuf(k,:)=testb(k,[tshufb1 tshufb2]);
% %                     end
% 
%                     % run classifier:
%     %                 [tmp_lda_u,tmp_err_u]=classify(units_u(:,testu)',units_u(:,trainu)',base2(trainu)');
%     %                 [tmp_lda_b,tmp_err_b]=classify(units_b(:,testb)',units_b(:,trainb)',bias2(trainb)');
%                     [tmp_lda_up,tmp_err_up,tmp_prob_u]=classify(testu',trainu',base_keep(tmp2)');
%                     [tmp_lda_bp,tmp_err_bp,tmp_prob_b]=classify(testb',trainb',bias_keep(tmp3)');
%                     tmp5=find(tmp_lda_up'==base_keep(x)); % percent correct on test trials
%                     tmp6=find(tmp_lda_bp'==bias_keep(y));
%                     % shuffled training trials:
% %                     [tmp_lda_up_shuf,tmp_err_up_shuf,tmp_prob_u_shuf]=classify(testu_shuf',trainu_shuf',base_keep(tmp2)');
% %                     [tmp_lda_bp_shuf,tmp_err_bp_shuf,tmp_prob_b_shuf]=classify(testb_shuf',trainb_shuf',bias_keep(tmp3)');
% %                     tmp7=find(tmp_lda_up_shuf'==base_keep(x)); % percent correct on test trials
% %                     tmp8=find(tmp_lda_bp_shuf'==bias_keep(y));
% 
%                     % store output
%     %                 pop_lda_u(idx,id,:)=[1-tmp_err_u length(tmp5)/length(tmp_lda_u)];
%     %                 pop_lda_b(idx,id,:)=[1-tmp_err_b length(tmp6)/length(tmp_lda_b)];
%                     pop_lda_up_train(o,idx,id1)=1-tmp_err_up;
%                     pop_lda_bp_train(o,idx,id1)=1-tmp_err_bp;
%                     pop_lda_up_test(o,idx,id1)=length(tmp5)/length(tmp_lda_up);
%                     pop_lda_bp_test(o,idx,id1)=length(tmp6)/length(tmp_lda_bp);
% %                     pop_lda_up_train_shuf(o,idx,id1)=1-tmp_err_up_shuf;
% %                     pop_lda_bp_train_shuf(o,idx,id1)=1-tmp_err_bp_shuf;
% %                     pop_lda_up_test_shuf(o,idx,id1)=length(tmp7)/length(tmp_lda_up_shuf);
% %                     pop_lda_bp_test_shuf(o,idx,id1)=length(tmp8)/length(tmp_lda_bp_shuf);
%                     % store classifications:
%                     class_up_test(o,idx,id1,:)=tmp_lda_up';
%                     class_bp_test(o,idx,id1,:)=tmp_lda_bp';
% %                     class_up_test_shuf(o,idx,id1,:)=tmp_lda_up_shuf';
% %                     class_bp_test_shuf(o,idx,id1,:)=tmp_lda_bp_shuf';
%                     % store posterior probabilities
%                     prob_up_test(o,idx,id1,:)=tmp_prob_u(:,1)'; % prob in class 1
%                     prob_bp_test(o,idx,id1,:)=tmp_prob_b(:,1)';
% %                     prob_up_test_shuf(o,idx,id1,:)=tmp_prob_u_shuf(:,1)';
% %                     prob_bp_test_shuf(o,idx,id1,:)=tmp_prob_b_shuf(:,1)';
%                     % structure: ori center, population size, [sampled neurons x trials: train/test]
%                     id1=id1+1;
%                     clear tmp* trainu* trainb* testu* testb*
%                 end
%             end
%             idx=idx+1;
%         end
%         disp('ori done')
%         save(name);
%     end
%     clear o e b j k tmp* id idx
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
%     % plot LDA performance, separated by center ori (3-class)
% %     figure
% %     supertitle('Acute, LDA performance vs population size')
% %     for e=1:length(ori_base)-1
% %         subplot(3,3,e); hold on
% %         title(sprintf('ori %g',e))
% %         hold on
% %         errorline(1+0.4:1:size(pop_lda_up_train,2)+0.4,pop_lda_up_train_mean(e,:),nanstd(squeeze(pop_lda_up_train(e,:,:))'),'k--')
% %         errorline(1+0.4:1:size(pop_lda_bp_train,2)+0.4,pop_lda_bp_train_mean(e,:),nanstd(squeeze(pop_lda_bp_train(e,:,:))'),'r--')
% %         errorline(1:size(pop_lda_up_test,2),pop_lda_up_test_mean(e,:),nanstd(squeeze(pop_lda_up_train(e,:,:))'),'k')
% %         errorline(1:size(pop_lda_bp_test,2),pop_lda_bp_test_mean(e,:),nanstd(squeeze(pop_lda_bp_train(e,:,:))'),'r')
% %     %     errorline(2:size(resp_raw_base,1),1-mean(pop_err_u,1),std(pop_err_u,1),'k')
% %     %     errorline(2:size(resp_raw_base,1),1-mean(pop_err_b,1),std(pop_err_b,1),'r')
% %         ylim([0 1])
% %         axis square; box off
% %         if e==1
% %             legend({'unif training','bias training','unif test','bias test'},'Location','southeast')
% %         end
% %         xlabel('# of units (steps of 3)')
% %         ylabel('% correct')
% % %     set(gca,'XTick',...
% %     end
%     
%     % fit performance data and find 75% performance mark
%     % (center ori, pop size, repeats (10x10=100), performance: train & test)
%     for e=1:length(ori_base)-1
%         % Xdata = # of steps (pop size); Ydata = % correct at that size
%         % only doing this for matched distribution
%         [par_up_train(e,:), gof_up_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean(e,:),[0.1 0.1 0.1],[]);
%         [par_bp_train(e,:), gof_bp_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean(e,:),[0.1 0.1 0.1],[]);
%         [par_up_test(e,:), gof_up_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean(e,:),[0.1 0.1 0.1],[]);
%         [par_bp_test(e,:), gof_bp_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean(e,:),[0.1 0.1 0.1],[]);
%         
% %         [par_up_train_shuf(e,:), gof_up_train_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean_shuf(e,:),[0.1 0.1 0.1],[]);
% %         [par_bp_train_shuf(e,:), gof_bp_train_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean_shuf(e,:),[0.1 0.1 0.1],[]);
% %         [par_up_test_shuf(e,:), gof_up_test_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean_shuf(e,:),[0.1 0.1 0.1],[]);
% %         [par_bp_test_shuf(e,:), gof_bp_test_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean_shuf(e,:),[0.1 0.1 0.1],[]);
%         
%         fit_up_train(e,:)=1-par_up_train(e,3)*exp(-(steps_fine/par_up_train(e,1)).^par_up_train(e,2));
%         fit_bp_train(e,:)=1-par_bp_train(e,3)*exp(-(steps_fine/par_bp_train(e,1)).^par_bp_train(e,2));
%         fit_up_test(e,:)=1-par_up_test(e,3)*exp(-(steps_fine/par_up_test(e,1)).^par_up_test(e,2));
%         fit_bp_test(e,:)=1-par_bp_test(e,3)*exp(-(steps_fine/par_bp_test(e,1)).^par_bp_test(e,2));
%         
% %         fit_up_train_shuf(e,:)=1-par_up_train_shuf(e,3)*exp(-(steps_fine/par_up_train_shuf(e,1)).^par_up_train_shuf(e,2));
% %         fit_bp_train_shuf(e,:)=1-par_bp_train_shuf(e,3)*exp(-(steps_fine/par_bp_train_shuf(e,1)).^par_bp_train_shuf(e,2));
% %         fit_up_test_shuf(e,:)=1-par_up_test_shuf(e,3)*exp(-(steps_fine/par_up_test_shuf(e,1)).^par_up_test_shuf(e,2));
% %         fit_bp_test_shuf(e,:)=1-par_bp_test_shuf(e,3)*exp(-(steps_fine/par_bp_test_shuf(e,1)).^par_bp_test_shuf(e,2));
%         
%         
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
%     end
%     
%     figure
%     supertitle('acute pnf fits')
%     for e=1:length(ori_base)-1
%         subplot(3,3,e); hold on
%         title(sprintf('ori %g',e))
% %         %         errorline(1+0.4:1:size(pop_lda_up_train,2)+0.4,pop_lda_up_train_mean(e,:),nanstd(squeeze(pop_lda_up_train(e,:,:))'),'k--')
% %         errorline(1+0.4:1:size(pop_lda_bp_train,2)+0.4,pop_lda_bp_train_mean(e,:),nanstd(squeeze(pop_lda_bp_train(e,:,:))'),'r--')
% %         errorline(1:size(pop_lda_up_test,2),pop_lda_up_test_mean(e,:),nanstd(squeeze(pop_lda_up_train(e,:,:))'),'k.')
% %         errorline(1:size(pop_lda_bp_test,2),pop_lda_bp_test_mean(e,:),nanstd(squeeze(pop_lda_bp_train(e,:,:))'),'r.')
% %         
% % %         errorline(1+0.4:1:size(pop_lda_up_train_shuf,2)+0.4,pop_lda_up_train_mean_shuf(e,:),nanstd(squeeze(pop_lda_up_train_shuf(e,:,:))'),'b--')
% % %         errorline(1+0.4:1:size(pop_lda_bp_train_shuf,2)+0.4,pop_lda_bp_train_mean_shuf(e,:),nanstd(squeeze(pop_lda_bp_train_shuf(e,:,:))'),'m--')
% %         errorline(1:size(pop_lda_up_test_shuf,2),pop_lda_up_test_mean_shuf(e,:),nanstd(squeeze(pop_lda_up_train_shuf(e,:,:))'),'b.')
% %         errorline(1:size(pop_lda_bp_test_shuf,2),pop_lda_bp_test_mean_shuf(e,:),nanstd(squeeze(pop_lda_bp_train_shuf(e,:,:))'),'m.')
% %         
% %     %     errorline(2:size(resp_raw_base,1),1-mean(pop_err_u,1),std(pop_err_u,1),'k')
% %     %     errorline(2:size(resp_raw_base,1),1-mean(pop_err_b,1),std(pop_err_b,1),'r')
%     
%         plot(steps_fine,fit_up_test(e,:),'k')
%         plot(steps_fine,fit_bp_test(e,:),'r')
% %         plot(steps_fine,fit_up_test_shuf(e,:),'b')
% %         plot(steps_fine,fit_bp_test_shuf(e,:),'m')
%         ylim([0 1])
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
%     disp('pnf done')
%     clearvars -except par* thresh* pop* num_units ori_base oribias oripref resp*...
%          stim* steps* shiftori fit* gof* name n class* trainsize testsize prob* t osi_tmp bias keep* base
%      
%      clear ans id tmp* x y uo bo units_* temp* ten* a b e i j k n1 n2
% 	
%     save(name);
%     end
% %     stop
% end