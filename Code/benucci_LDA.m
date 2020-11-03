%% LDA for acute benucci data, individual files
clear

% % % % 2:1
% load('129r001p173_corr','resp_*','ori*','base','bias','keep','keep2'); load('129r001p173_entropy','spont*');
% load('130l001p169_corr','resp_*','ori*','base','bias','keep','keep2'); load('130l001p169_entropy','spont*');
% load('140l001p107_corr','resp_*','ori*','base','bias','keep','keep2'); load('140l001p107_entropy','spont*');
% load('140l001p122_corr','resp_*','ori*','base','bias','keep','keep2'); load('140l001p122_entropy','spont*');
% load('140r001p105_corr','resp_*','ori*','base','bias','keep','keep2'); load('140r001p105_entropy','spont*');
% load('140r001p122_corr','resp_*','ori*','base','bias','keep','keep2'); load('140r001p122_entropy','spont*');

% % % % 4:1
% load('130l001p170_corr','resp_*','ori*','base','bias','keep','keep2'); load('130l001p170_entropy','spont*');
% load('140l001p108_corr','resp_*','ori*','base','bias','keep','keep2'); load('140l001p108_entropy','spont*');
% load('140l001p110_corr','resp_*','ori*','base','bias','keep','keep2'); load('140l001p110_entropy','spont*');
% load('140r001p107_corr','resp_*','ori*','base','bias','keep','keep2'); load('140r001p107_entropy','spont*');
% load('140r001p109_corr','resp_*','ori*','base','bias','keep','keep2'); load('140r001p109_entropy','spont*');
% load('141r001p114_corr','resp_*','ori*','base','bias','keep','keep2'); load('141r001p114_entropy','spont*');

% % % % awake time (4:1)
% load('140l113_awaketime_corr','resp_*','ori*','base','bias','keep','keep2'); load('140l113_awaketime_entropy','spont*');
% load('140r113_awaketime_corr','resp_*','ori*','base','bias','keep','keep2'); load('140r113_awaketime_entropy','spont*');
% 141r001p006
% 141r001p024
% 141r001p041
% % % % awake time (6:1)
% 141r001p007
% 141r001p025
% 141r001p039
% % % % awake time (6:1 fine ori)
% 141r001p009
% 141r001p027
% 141r001p038

% % % % low contrast
% load('lowcon114_corr','resp_*','ori*','base','bias','keep','keep2'); load('lowcon114_entropy','spont*');
% load('lowcon115_corr','resp_*','ori*','base','bias','keep','keep2'); load('lowcon115_entropy','spont*');
% load('lowcon116_corr','resp_*','ori*','base','bias','keep','keep2'); load('lowcon116_entropy','spont*');
% load('lowcon117_corr','resp_*','ori*','base','bias','keep','keep2'); load('lowcon117_entropy','spont*');

% % % % blank
% load('140l111_blank_corr','resp_*','ori*','base','bias','keep','keep2'); load('blankl111_entropy','spont*');
% load('140r111_blank_corr','resp_*','ori*','base','bias','keep','keep2'); load('blankr111_entropy','spont*');
% load('140r161_blank_corr','resp_*','ori*','base','bias','keep','keep2'); load('blankr161_entropy','spont*');

% % % % benucci
% load('140l118_benucci_corr','resp_*','ori*','base','bias','keep','keep2'); load('benuccil118_entropy','spont*');
% load('140r118_benucci_corr','resp_*','ori*','base','bias','keep','keep2'); load('benuccir118_entropy','spont*');

for n=[30 33]%[19 22 25 30 33]%1:33
    clearvars -except n
    if n==1
        load('129r001p173_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='129r001p173_lda_pnf';
    elseif n==2
        load('130l001p169_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='130l001p169_lda_pnf';
    elseif n==3
        load('140l001p107_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140l001p107_lda_pnf';
    elseif n==4
        load('140l001p122_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140l001p122_lda_pnf';
    elseif n==5
        load('140r001p105_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140r001p105_lda_pnf';
    elseif n==6
        load('140r001p122_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140r001p122_lda_pnf';
    elseif n==7
        load('130l001p170_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='130l001p170_lda_pnf';
    elseif n==8
        load('140l001p108_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140l001p108_lda_pnf';
    elseif n==9
        load('140l001p110_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140l001p110_lda_pnf';
    elseif n==10
        load('140r001p107_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140r001p107_lda_pnf';
    elseif n==11
        load('140r001p109_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140r001p109_lda_pnf';
    elseif n==12
        load('lowcon114_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='lowcon114_lda_pnf';
    elseif n==13
        load('lowcon115_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='lowcon115_lda_pnf';
    elseif n==14
        load('lowcon116_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='lowcon116_lda_pnf';
    elseif n==15
        load('lowcon117_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='lowcon117_lda_pnf';
    elseif n==16
        load('140l113_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140l113_awaketime_lda_pnf';
    elseif n==17
        load('140r113_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140r113_awaketime_lda_pnf';
    
    elseif n==18 % start of experiment 141 files
        load('141r001p006_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p006_awaketime_lda_pnf';
    elseif n==19
        load('141r001p007_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p007_awaketime6_lda_pnf_2class';
    elseif n==20
        load('141r001p009_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p009_awaketime_fine_lda_pnf';
    elseif n==21 % rotated AT 4:1 (80°)
        load('141r001p024_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p024_awaketime_lda_pnf';
    elseif n==22 % rotated AT 6:1 (80°)
        load('141r001p025_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p025_awaketime6_lda_pnf_2class';
    elseif n==23 % rotated AT fineori (90°??)
        load('141r001p027_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p027_awaketime_fine_lda_pnf';
    elseif n==24 % rotated fineori (40°)
        load('141r001p038_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p038_awaketime_fine_lda_pnf';
    elseif n==25 % rotated 6:1 (120°)
        load('141r001p039_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p039_awaketime6_lda_pnf_2class';
    elseif n==26 % rotated awaketime 4:1 (120°)
        load('141r001p041_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p041_awaketime_lda_pnf';
    elseif n==27
        load('141r001p114_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p114_lda_pnf';
        
    elseif n==28
        load('142l001p002_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p002_awaketime_lda_pnf';
    elseif n==29
        load('142l001p004_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p004_awaketime_fine_lda_pnf';
    elseif n==30
        load('142l001p006_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p006_awaketime6_lda_pnf_2class';
    elseif n==31
        load('142l001p007_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p007_awaketime_lda_pnf';
    elseif n==32
        load('142l001p009_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p009_awaketime_fine_lda_pnf';
    elseif n==33
        load('142l001p010_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p010_awaketime6_lda_pnf_2class';
    end
    
    %%  preprocessing first
    % select tuning criteria:
    osi_tmp=[0 0.35 1];
    
    for t=1:2
        osi_keep=find(oribias>=osi_tmp(t) & oribias<=osi_tmp(t+1));
        if t==1
            name=sprintf('%slowOSI',name(1:end-6));
        elseif t==2
            name=sprintf('%shiOSI',name(1:end-6));
        end
%     end
clearvars -except responsive keep* ori* resp_raw_base resp_raw_bias base bias name t osi_tmp osi_keep

    num_units=length(osi_keep);
    
%     num_units=sum(responsive); 
    val_mode=mode(bias);    % identify bias ori
    if val_mode==0         % align biased ori to first column
        shiftori=0;
    else
        shiftori=1;
    end
    
    % remove blanks trials
    [tmp]=find(base(keep)~=200);
    [tmp2]=find(bias(keep2)~=200);
    keep=keep(tmp);
%     resp_raw_base_sub=resp_raw_base(:,keep);
    resp_raw_base_sub=resp_raw_base(osi_keep,keep);
    keep2=keep2(tmp2);
%     resp_raw_bias_sub=resp_raw_bias(:,keep2);
    resp_raw_bias_sub=resp_raw_bias(osi_keep,keep2);
    base_keep=base(keep);
    bias_keep=bias(keep2);
%     [blank1]=find(base~=200);
%     [blank2]=find(bias~=200);
%     base2=base(blank1);
%     bias2=bias(blank2);
%     resp_raw_base=resp_raw_base(:,blank1);
%     resp_raw_bias=resp_raw_bias(:,blank2);
    
    % preallocate # of trials for test and train;
%     trainsize=sort(randperm(length(base2),round(0.9*length(base2))));  % random 90% of trials
%     testu=setdiff(1:length(base2),trainsize);                          % random 10% of trials
%     testb=setdiff(1:length(bias2),trainsize);
%     
%     temp1=base(keep); 
%     tmp4=randperm(length(temp1));
%     temp1=temp1(tmp4); 
%     resp_raw_base_sub=resp_raw_base_sub(:,tmp4);
%     temp2=bias(keep2);
%     tenpu=round(0.1*length(keep));
%     tenpb=round(0.1*length(keep2));

    clear tmp* responsive i
    %%  Over whole experiment: train on 90% of trials
    % match # of trials of each class in both environments for training
%     for e = 1:length(ori_base)-1
%         tmp(e)=length(find(base2==ori_base(e)));
%         tmp2(e)=length(find(bias2==ori_base(e)));
%     end
%     trainmatch=round(0.9*min([tmp tmp2])); % # of trials of each class to use
    
%     trainu=[];  % uniform training trials
%     trainb=[];  % bias training trials
%     for e = 1:length(ori_base)-1
%         tmp=find(base2(trainsize)==ori_base(e));
%         trainu=[trainu tmp(1:trainmatch)];
%         tmp2=find(bias2(trainsize)==ori_base(e));
%         trainb=[trainb tmp2(1:trainmatch)];
%     end
%     
%     [lda_u, lda_err_u]=classify(resp_raw_base(:,testu)',resp_raw_base(:,trainu)',base2(trainu)');
%     [lda_b, lda_err_b]=classify(resp_raw_bias(:,testb)',resp_raw_bias(:,trainb)',bias2(trainb)');
%     [lda_up, lda_err_up]=classify(resp_raw_base_sub(:,end-tenpu+1:end)',resp_raw_base_sub(:,1:end-tenpu)',temp1(1:end-tenpu)');
%     [lda_bp, lda_err_bp]=classify(resp_raw_bias_sub(:,end-tenpb+1:end)',resp_raw_bias_sub(:,1:end-tenpb)',temp2(1:end-tenpb)');
%     
%     tmp=find(lda_u'==base2(testu));
%     tmp2=find(lda_b'==bias2(testb));
%     tmp3=find(lda_up'==temp1(end-tenpu+1:end));
%     tmp4=find(lda_bp'==temp2(end-tenpb+1:end));
%     test_rate_u=length(tmp)/length(base(testu));
%     test_rate_b=length(tmp2)/length(base(testb));
%     test_rate_up=length(tmp3)/length(temp1(end-tenpu+1:end));
%     test_rate_bp=length(tmp4)/length(temp2(end-tenpb+1:end));
%     
%     figure
%     title('Acute, classify all trials/all units (LDA)')
%     hold on
%     plot(1-lda_err_u,test_rate_u,'k.','MarkerSize',10)
%     plot(1-lda_err_up,test_rate_up,'b.','MarkerSize',10)
%     plot(1-lda_err_b,test_rate_b,'r.','MarkerSize',10)
%     plot(1-lda_err_bp,test_rate_bp,'g.','MarkerSize',10)
%     ylim([0 1]); xlim([0 1])
%     xlabel('training: % correct')
%     ylabel('test: % correct')
%     % more appropriate legend: Uniform Responses trained on Uniform dist, Uniform
%     % trained on subsampled bias dist, Bias trained on bias, bias trained on
%     % subsampled bias dist
%     legend({'Uniform','U pred','Bias','B pred'},'Location','southeast')
%     axis square; box off
%     set(gca,'TickDir','out')
%     refline(1,0)
%     disp('full done')
    
%     for j = 1:10
%         trainu=[];
%         trainb=[];
%         testu=[];
%         testb=[];
%         trainclassu=[];
%         trainclassb=[];
%         testclassu=[];
%         testclassb=[];
%         for e = 1:length(ori_base)-1
%             tmp=find(base2==ori_base(e));   % all uniform trials of ori(e)
%             ushuf=randperm(length(tmp),trainmatch); % random subset of trials stored for training
%             trainu=[trainu tmp(1:trainmatch)];  % stored training trials
%             trainclassu=[trainclassu ones(1,trainmatch)*ori_base(e)]; % corresponding class for training
%             testu=[testu tmp(setdiff(1:length(tmp),ushuf))]; % trials not used in training
%             testclassu=[testclassu ones(1,length(tmp)-trainmatch)*ori_base(e)]; % class of test trials
%             tsu(e,:)=tmp(ushuf); % uniform trials stored for shuffling later
% 
%             tmp=find(bias2==ori_base(e));   % all bias trials of ori(e)
%             bshuf=randperm(length(tmp),trainmatch); % random subset of trials stored for training
%             trainb=[trainb tmp(1:trainmatch)];  % stored training trials
%             trainclassb=[trainclassb ones(1,trainmatch)*ori_base(e)]; % corresponding class for training
%             testb=[testb tmp(setdiff(1:length(tmp),bshuf))]; % trials not used in training
%             testclassb=[testclassb ones(1,length(tmp)-trainmatch)*ori_base(e)]; % class of test trials
%             tsb(e,:)=tmp(bshuf); % uniform trials stored for shuffling later
%         end
%         % trials aligned:
%         [lda_u, lda_err_u(j)]=classify(resp_raw_base(:,testu)',resp_raw_base(:,trainu)',trainclassu');
%         [lda_b, lda_err_b(j)]=classify(resp_raw_bias(:,testb)',resp_raw_bias(:,trainb)',trainclassb');
%         tmp=find(lda_u'==testclassu);
%         tmp2=find(lda_b'==testclassb);
%         test_rate_u(j)=length(tmp)/length(testclassu);
%         test_rate_b(j)=length(tmp2)/length(testclassb);
%         % trials shuffled (removes correlations):
%         for k=1:size(resp_raw_base,1)
%             temp=[];
%             temp2=[];
%             for m=1:size(tsb,1)
%                 temp=[temp tsu(m,randperm(length(tsu)))];
%                 temp2=[temp2 tsb(m,randperm(length(tsb)))];
%             end
%             trainu_shuf(k,:)=resp_raw_base(k,temp);
%             trainb_shuf(k,:)=resp_raw_bias(k,temp2);
%         end
%         [lda_u_shuf, lda_err_u_shuf(j)]=classify(resp_raw_base(:,testu)',trainu_shuf',trainclassu');
%         [lda_b_shuf, lda_err_b_shuf(j)]=classify(resp_raw_bias(:,testb)',trainb_shuf',trainclassb');
%         tmp_shuf=find(lda_u_shuf'==testclassu);
%         tmp2_shuf=find(lda_b_shuf'==testclassb);
%         test_rate_u_shuf(j)=length(tmp_shuf)/length(testclassu);
%         test_rate_b_shuf(j)=length(tmp2_shuf)/length(testclassb);
%     end
    
    %%  performance of each class vs its neighbors (3 classes)
%     clear tmp* temp* trainu* trainb* testu testb
%     lda_neighbors_u=nan*zeros(100,length(ori_base)-1,2); %10 repeats (trials selected) x 10 new units X stim oris X train/test
%     lda_neighbors_b=nan*zeros(100,length(ori_base)-1,2);
%     lda_neighbors_u_shuf=nan*zeros(100,length(ori_base)-1,2);
%     lda_neighbors_b_shuf=nan*zeros(100,length(ori_base)-1,2);
%     % throw out units that passed sorting but are too sparse
%     unit_sum_base=sum(resp_raw_base,2);
%     unit_sum_bias=sum(resp_raw_bias,2);
%     resp_raw_base2=resp_raw_base(unit_sum_base>500,:);
%     resp_raw_bias2=resp_raw_bias(unit_sum_base>500,:);
%     
%     for e = 1:length(ori_base)-1    % ignores "200" blank stim
%         id=1;
%         if e==1
%             n1=length(ori_base)-1;
%             n2=e+1;
%         elseif e==length(ori_base)-1
%             n1=e-1;
%             n2=1;
%         else
%             n1=e-1;
%             n2=e+1;
%         end
%         tmp=find(base2==ori_base(e));
%         tmp1=find(base2==ori_base(n1));
%         tmp2=find(base2==ori_base(n2));
%         temp=find(bias2==ori_base(e));
%         temp1=find(bias2==ori_base(n1));
%         temp2=find(bias2==ori_base(n2));
%         for j = 1:10
%             % train trials:
%             tmpu1=randperm(length(tmp),trainmatch);
%             tmpu2=randperm(length(tmp1),trainmatch);
%             tmpu3=randperm(length(tmp2),trainmatch);
%             tmpb1=randperm(length(temp),trainmatch);
%             tmpb2=randperm(length(temp1),trainmatch);
%             tmpb3=randperm(length(temp2),trainmatch);
%             trainu=[tmp(tmpu1) tmp1(tmpu2) tmp2(tmpu3)];
%             trainb=[temp(tmpb1) temp1(tmpb2) temp2(tmpb3)];
%             % shuffle train trials independently for each unit (row):
%             for k=1:size(resp_raw_base2,1)
%                 tmpu1shuf=randperm(trainmatch);
%                 tmpb1shuf=randperm(trainmatch);
%                 tmpu2shuf=randperm(trainmatch);
%                 tmpb2shuf=randperm(trainmatch);
%                 tmpu3shuf=randperm(trainmatch);
%                 tmpb3shuf=randperm(trainmatch);
%                 
%                 trainu_shuf(k,:)=resp_raw_base2(k,[tmp(tmpu1shuf) tmp1(tmpu2) tmp2(tmpu3)]);
%                 trainb_shuf(k,:)=resp_raw_bias2(k,[temp(tmpb1shuf) temp1(tmpb2) temp2(tmpb3)]);
%             end
%             % test trials:
%             testu1=tmp(setdiff(1:length(tmp),tmpu1));
%             testu2=tmp1(setdiff(1:length(tmp1),tmpu2));
%             testu3=tmp2(setdiff(1:length(tmp2),tmpu3));
%             testb1=temp(setdiff(1:length(temp),tmpb1));
%             testb2=temp1(setdiff(1:length(temp1),tmpb2));
%             testb3=temp2(setdiff(1:length(temp2),tmpb3));
%             testu=[testu1 testu2 testu3];
%             testb=[testb1 testb2 testb3];
%             for b=1:10
%                 if size(resp_raw_base2,1)>=20
%                     c=20; % may need to change this number. 40 units = ~75% max performance
%                 else
%                     c=size(resp_raw_base2,1);
%                 end
%                 
%                 num=randperm(size(resp_raw_base2,1),c);
%                 % trials aligned:
%                 units_u=resp_raw_base2(num,:);
%                 units_b=resp_raw_bias2(num,:);
%                 [lda_near_u, lda_near_err_u]=classify(units_u(:,testu)',units_u(:,trainu)',base2(trainu)');
%                 [lda_near_b, lda_near_err_b]=classify(units_b(:,testb)',units_b(:,trainb)',bias2(trainb)');
%                 tmp5=find(lda_near_u'==base2(testu));
%                 tmp6=find(lda_near_b'==bias2(testb));
%                 lda_neighbors_u(id,e,:)=[1-lda_near_err_u length(tmp5)/length(lda_near_u)];
%                 lda_neighbors_b(id,e,:)=[1-lda_near_err_b length(tmp6)/length(lda_near_b)];
%                 % trials shuffled:
%                 units_u2=trainu_shuf(num,:);
%                 units_b2=trainb_shuf(num,:);
%                 [lda_near_u_shuf, lda_near_err_u_shuf]=classify(units_u(:,testu)',units_u2',base2(trainu)');
%                 [lda_near_b_shuf, lda_near_err_b_shuf]=classify(units_b(:,testb)',units_b2',bias2(trainb)');
%                 tmp7=find(lda_near_u_shuf'==base2(testu));
%                 tmp8=find(lda_near_b_shuf'==bias2(testb));
%                 lda_neighbors_u_shuf(id,e,:)=[1-lda_near_err_u_shuf length(tmp7)/length(lda_near_u_shuf)];
%                 lda_neighbors_b_shuf(id,e,:)=[1-lda_near_err_b_shuf length(tmp8)/length(lda_near_b_shuf)];
%                 id=id+1;
%             end
%             clearvars testu* testb* tmpu* tmpb* trainu* trainb*
%         end
%     end
%     disp('3class done')

% new way ^
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% old way:

%     lda_neighbors_u=nan*zeros(20,9,2); % 20 repeats of sampled units x 9 oris/neighbors x [training test]
%     lda_neighbors_b=nan*zeros(20,9,2);
%     for e = 1:length(ori_base)-1    % ignores "200" blank stim
%         if e==1
%             n1=length(ori_base)-1;
%             n2=e+1;
%         elseif e==length(ori_base)-1
%             n1=e-1;
%             n2=1;
%         else
%             n1=e-1;
%             n2=e+1;
%         end
%         tmp=find(base2(trainsize)==ori_base(e));
%         tmp1=find(base2(trainsize)==ori_base(n1));
%         tmp2=find(base2(trainsize)==ori_base(n2));
%         temp=find(bias2(trainsize)==ori_base(e));
%         temp1=find(bias2(trainsize)==ori_base(n1));
%         temp2=find(bias2(trainsize)==ori_base(n2));
%         % match trial for all three oris in uniform and bias:
%         x=[tmp(1:trainmatch) tmp2(1:trainmatch) tmp1(1:trainmatch)];
%         x=sort(x);  % uniform training data
%         y=[temp(1:trainmatch) temp1(1:trainmatch) temp2(1:trainmatch)];
%         y=sort(y);  % bias training data
%         
%         for b=1:20
%             n=randperm(size(resp_raw_base,1),30);   % may need to change this number. 40 units = ~75% max performance
%             units_u=resp_raw_base(n,:);
%             units_b=resp_raw_bias(n,:);
%             [lda_near_u, lda_near_err_u]=classify(units_u(:,testu)',units_u(:,x)',base2(x)');
%             [lda_near_b, lda_near_err_b]=classify(units_b(:,testb)',units_b(:,y)',bias2(y)');
%             tmp=find(lda_near_u'==base2(testu));
%             tmp2=find(lda_near_b'==bias2(testb));
%             lda_neighbors_u(b,e,:)=[1-lda_near_err_u length(tmp)/length(lda_near_u)];
%             lda_neighbors_b(b,e,:)=[1-lda_near_err_b length(tmp2)/length(lda_near_b)];
%         end
%     end
%     %     nmatch=min([length(tmp) length(tmp1) length(tmp2) length(temp) length(temp1) length(temp2)]);
%     %     x=[tmp(1:nmatch) tmp1(1:nmatch) tmp2(1:nmatch)];
%     %     y=[temp(1:nmatch) temp1(1:nmatch) temp2(1:nmatch)];
%     %     tmpsize=round(0.9*length(x));
%     %     tmpsize2=round(0.9*length(y));
%     %     units_up=resp_raw_base_sub(n,x);
%     %     units_bp=resp_raw_bias_sub(n,y);
%     %     [lda_near_up, lda_near_err_up]=classify(units_up(:,tmpsize+1:end)',units_up(:,1:tmpsize)',uniform_dist(x(1:tmpsize))');
%     %     [lda_near_bp, lda_near_err_bp]=classify(units_bp(:,tmpsize2+1:end)',units_bp(:,1:tmpsize2)',biased_dist(y(1:tmpsize2))');
%     %     tmp3=find(lda_near_up==uniform_dist(x(tmpsize+1:end)));
%     %     tmp4=find(lda_near_bp==biased_dist(y(tmpsize2+1:end)));
%     %     lda_neighbors_up(e,:)=[1-lda_near_err_up length(tmp3)/length(lda_near_u)];
%     %     lda_neighbors_bp(e,:)=[1-lda_near_err_bp length(tmp4)/length(lda_near_b)];
%     
%     figure
%     supertitle('LDA, performance vs neighboring oris (30 units)')
%     subplot(121); hold on
%     title('training set performance')
%     if shiftori==1
%         errorline(1:9,squeeze(nanmean(lda_neighbors_u(:,:,1),1)),squeeze(nanstd(lda_neighbors_u(:,:,1),1)),'k')
%         errorline(1.1:1:9.1,squeeze(nanmean(lda_neighbors_b(:,:,1),1)),squeeze(nanstd(lda_neighbors_b(:,:,1),1)),'r')
%         % plot(1:9,lda_neighbors_up(:,1),'k--')
%         % plot(1:9,lda_neighbors_bp(:,1),'r--')
%     else
%         errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u(:,:,1),1)),4),circshift(squeeze(nanstd(lda_neighbors_u(:,:,1),1)),4),'k')
%         errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b(:,:,1),1)),4),circshift(squeeze(nanstd(lda_neighbors_b(:,:,1),1)),4),'r')
%         % plot(1:9,lda_neighbors_up(:,1),'k--')
%         % plot(1:9,lda_neighbors_bp(:,1),'r--')
%     end
%     ylim([0.0 1])
%     xlim([0 10])
%     set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
%     axis square; box off
%     legend({'unif','bias'},'Location','southeast')
%     xlabel('center ori +/-20')
%     ylabel('% correct')
%     subplot(122); hold on
%     title('Test set performance')
%     if shiftori==1
%         errorline(1:9,squeeze(nanmean(lda_neighbors_u(:,:,2),1)),squeeze(nanstd(lda_neighbors_u(:,:,2),1)),'k')
%         errorline(1.1:1:9.1,squeeze(nanmean(lda_neighbors_b(:,:,2),1)),squeeze(nanstd(lda_neighbors_b(:,:,2),1)),'r')
%         % plot(1:9,lda_neighbors_up(:,1),'k--')
%         % plot(1:9,lda_neighbors_bp(:,1),'r--')
%     else
%         errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u(:,:,2),1)),4),circshift(squeeze(nanstd(lda_neighbors_u(:,:,2),1)),4),'k')
%         errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b(:,:,2),1)),4),circshift(squeeze(nanstd(lda_neighbors_b(:,:,2),1)),4),'r')
%         % plot(1:9,lda_neighbors_up(:,1),'k--')
%         % plot(1:9,lda_neighbors_bp(:,1),'r--')
%     end
%     % plot(1:9,lda_neighbors_up(:,2),'k--')
%     % plot(1:9,lda_neighbors_bp(:,2),'r--')
%     ylim([0.0 1])
%     xlim([0 10])
%     set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
%     axis square; box off
%     xlabel('center ori +/-20')
%     ylabel('% correct')
%     disp('neighbors done')
    
    %%  performance w/ different # of neurons
    % skipping bc takes a long time to run - already chosen neighbors use 40
    
    % exclude 'unresponsive' units for LDA training
    resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_bias_sub,2)>500,:);
    resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_bias_sub,2)>500,:);
    num_units=size(resp_raw_base_sub,1);
    % steps of units for analysis
    steps=2:1:num_units; % MAY CHANGE THIS DEPENDING ON SHAPE
    steps_fine=2:.25:100;
    % find and store cases of each stim to call for training and test
    % (should be equal number of each if distribution created correctly)
    for b=1:length(ori_base)-1
        stimrefsu(b,:)=find(base_keep==ori_base(b));
        stimrefsb(b,:)=find(bias_keep==ori_base(b));
    end
    trainsize=round(0.9*size(stimrefsu,2));
    testsize=size(stimrefsu,2)-trainsize;
    
%     pop_lda_u=nan*zeros(length(steps),100,2);
%     pop_lda_b=nan*zeros(length(steps),100,2);
    pop_lda_up_train=nan*zeros(9,length(steps),200);
    pop_lda_bp_train=nan*zeros(9,length(steps),200);
    pop_lda_up_test=nan*zeros(9,length(steps),200);
    pop_lda_bp_test=nan*zeros(9,length(steps),200);
    pop_lda_up_train_shuf=nan*zeros(9,length(steps),200);
    pop_lda_bp_train_shuf=nan*zeros(9,length(steps),200);
    pop_lda_up_test_shuf=nan*zeros(9,length(steps),200);
    pop_lda_bp_test_shuf=nan*zeros(9,length(steps),200);
    %     class_up_train=nan*zeros(9,length(steps),200,trainsize*2);
%     class_bp_train=nan*zeros(9,length(steps),200,trainsize*2);
    class_up_test=nan*zeros(9,length(steps),200,testsize*2);
    class_bp_test=nan*zeros(9,length(steps),200,testsize*2);
%     class_up_train_shuf=nan*zeros(9,length(steps),200,trainsize*2);
%     class_bp_train_shuf=nan*zeros(9,length(steps),200,trainsize*2);
    class_up_test_shuf=nan*zeros(9,length(steps),200,testsize*2);
    class_bp_test_shuf=nan*zeros(9,length(steps),200,testsize*2);
%     prob_up_train=nan*zeros(9,length(steps),200,trainsize*2);
%     prob_bp_train=nan*zeros(9,length(steps),200,trainsize*2);
    prob_up_test=nan*zeros(9,length(steps),200,testsize*2);
    prob_bp_test=nan*zeros(9,length(steps),200,testsize*2);
%     prob_up_train_shuf=nan*zeros(9,length(steps),200,trainsize*2);
%     prob_bp_train_shuf=nan*zeros(9,length(steps),200,trainsize*2);
    prob_up_test_shuf=nan*zeros(9,length(steps),200,testsize*2);
    prob_bp_test_shuf=nan*zeros(9,length(steps),200,testsize*2);
    
    tic
    for o=1:length(ori_base)-1
        % select 3 oris for comparison:
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
        % select neighbor ori for 2 class:
        if o==length(ori_base)-1
            n1=1;   % last ori w/ first ori (160-0)
        else
            n1=o+1; % next ori (e.g. 0-20, 20-40...)
        end
        
        idx=1;  % index for population size 
        % loop over population size:
        for e=steps
            id1=1; % index for loop over neuron samples and trials (20x neurons & 10x trials =200);
            % loop over random neurons selected:
            for b=1:20 % index for train/test trials
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
%                     tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp) stimrefsu(n2,tmp)];
%                     tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp) stimrefsb(n2,tmp)];
%                     x=setdiff([stimrefsu(o,:) stimrefsu(n1,:) stimrefsu(n2,:)],tmp2);
%                     y=setdiff([stimrefsb(o,:) stimrefsb(n1,:) stimrefsb(n2,:)],tmp3);
%                     xshuf=length(x)/3;
                    % for 2 class:
                    tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                    tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                    x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                    y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                    xshuf=length(x)/2;
                    
                    trainu=units_up(:,tmp2);
                    trainb=units_bp(:,tmp3);
                    testu=units_up(:,x);
                    testb=units_bp(:,y);
                    
                    % shuffle train trials independently for each unit (row):
                    for k=1:size(trainu,1)
                        tmpu1shuf=randperm(trainsize);
                        tmpb1shuf=randperm(trainsize);
                        tmpu2shuf=randperm(trainsize);
                        tmpb2shuf=randperm(trainsize);
                        tmpu3shuf=randperm(trainsize);    % only need these for 3 class
                        tmpb3shuf=randperm(trainsize);
                        trainu_shuf(k,:)=units_up(k,[stimrefsu(o,tmp(tmpu1shuf)) stimrefsu(n1,tmp(tmpu2shuf)) stimrefsu(n2,tmp(tmpu3shuf))]);
                        trainb_shuf(k,:)=units_bp(k,[stimrefsb(o,tmp(tmpb1shuf)) stimrefsb(n1,tmp(tmpb2shuf)) stimrefsb(n2,tmp(tmpb3shuf))]);
%                         trainu_shuf(k,:)=units_up(k,[stimrefsu(o,tmp(tmpu1shuf)) stimrefsu(n1,tmp(tmpu2shuf))]);
%                         trainb_shuf(k,:)=units_bp(k,[stimrefsb(o,tmp(tmpb1shuf)) stimrefsb(n1,tmp(tmpb2shuf))]);
                        
                        tshufu1=randperm(xshuf);
                        tshufb1=randperm(xshuf);
                        tshufu2=randperm(xshuf)+xshuf;
                        tshufb2=randperm(xshuf)+xshuf;
                        tshufu3=randperm(xshuf)+xshuf*2;  % only need these for 3 class
                        tshufb3=randperm(xshuf)+xshuf*2;
                        testu_shuf(k,:)=testu(k,[tshufu1 tshufu2 tshufu3]);
                        testb_shuf(k,:)=testb(k,[tshufb1 tshufb2 tshufb3]);
%                         testu_shuf(k,:)=testu(k,[tshufu1 tshufu2]);
%                         testb_shuf(k,:)=testb(k,[tshufb1 tshufb2]);
                    end

                    % run classifier:
    %                 [tmp_lda_u,tmp_err_u]=classify(units_u(:,testu)',units_u(:,trainu)',base2(trainu)');
    %                 [tmp_lda_b,tmp_err_b]=classify(units_b(:,testb)',units_b(:,trainb)',bias2(trainb)');
                    [tmp_lda_up,tmp_err_up,tmp_prob_u]=classify(testu',trainu',base_keep(tmp2)');
                    [tmp_lda_bp,tmp_err_bp,tmp_prob_b]=classify(testb',trainb',bias_keep(tmp3)');
                    tmp5=find(tmp_lda_up'==base_keep(x)); % percent correct on test trials
                    tmp6=find(tmp_lda_bp'==bias_keep(y));
                    % shuffled training trials:
                    [tmp_lda_up_shuf,tmp_err_up_shuf,tmp_prob_u_shuf]=classify(testu_shuf',trainu_shuf',base_keep(tmp2)');
                    [tmp_lda_bp_shuf,tmp_err_bp_shuf,tmp_prob_b_shuf]=classify(testb_shuf',trainb_shuf',bias_keep(tmp3)');
                    tmp7=find(tmp_lda_up_shuf'==base_keep(x)); % percent correct on test trials
                    tmp8=find(tmp_lda_bp_shuf'==bias_keep(y));

                    % store output
    %                 pop_lda_u(idx,id,:)=[1-tmp_err_u length(tmp5)/length(tmp_lda_u)];
    %                 pop_lda_b(idx,id,:)=[1-tmp_err_b length(tmp6)/length(tmp_lda_b)];
                    pop_lda_up_train(o,idx,id1)=1-tmp_err_up;
                    pop_lda_bp_train(o,idx,id1)=1-tmp_err_bp;
                    pop_lda_up_test(o,idx,id1)=length(tmp5)/length(tmp_lda_up);
                    pop_lda_bp_test(o,idx,id1)=length(tmp6)/length(tmp_lda_bp);
                    pop_lda_up_train_shuf(o,idx,id1)=1-tmp_err_up_shuf;
                    pop_lda_bp_train_shuf(o,idx,id1)=1-tmp_err_bp_shuf;
                    pop_lda_up_test_shuf(o,idx,id1)=length(tmp7)/length(tmp_lda_up_shuf);
                    pop_lda_bp_test_shuf(o,idx,id1)=length(tmp8)/length(tmp_lda_bp_shuf);
                    % store classifications:
                    class_up_test(o,idx,id1,:)=tmp_lda_up';
                    class_bp_test(o,idx,id1,:)=tmp_lda_bp';
                    class_up_test_shuf(o,idx,id1,:)=tmp_lda_up_shuf';
                    class_bp_test_shuf(o,idx,id1,:)=tmp_lda_bp_shuf';
                    % store posterior probabilities
                    prob_up_test(o,idx,id1,:)=tmp_prob_u(:,1)'; % prob in class 1
                    prob_bp_test(o,idx,id1,:)=tmp_prob_b(:,1)';
                    prob_up_test_shuf(o,idx,id1,:)=tmp_prob_u_shuf(:,1)';
                    prob_bp_test_shuf(o,idx,id1,:)=tmp_prob_b_shuf(:,1)';
                    % structure: ori center, population size, [sampled neurons x trials: train/test]
                    id1=id1+1;
                    clear tmp* trainu* trainb* testu* testb*
                end
            end
            idx=idx+1;
        end
        disp('ori done')
        save(name);
    end
    clear o e b j k tmp* id idx
    toc
    
    pop_lda_up_train_mean=squeeze(nanmean(pop_lda_up_train(:,:,:),3));
    pop_lda_bp_train_mean=squeeze(nanmean(pop_lda_bp_train(:,:,:),3));
    pop_lda_up_test_mean=squeeze(nanmean(pop_lda_up_test(:,:,:),3));
    pop_lda_bp_test_mean=squeeze(nanmean(pop_lda_bp_test(:,:,:),3));
    pop_lda_up_train_mean_shuf=squeeze(nanmean(pop_lda_up_train_shuf(:,:,:),3));
    pop_lda_bp_train_mean_shuf=squeeze(nanmean(pop_lda_bp_train_shuf(:,:,:),3));
    pop_lda_up_test_mean_shuf=squeeze(nanmean(pop_lda_up_test_shuf(:,:,:),3));
    pop_lda_bp_test_mean_shuf=squeeze(nanmean(pop_lda_bp_test_shuf(:,:,:),3));
    
    
    % plot LDA performance, separated by center ori (3-class)
%     figure
%     supertitle('Acute, LDA performance vs population size')
%     for e=1:length(ori_base)-1
%         subplot(3,3,e); hold on
%         title(sprintf('ori %g',e))
%         hold on
%         errorline(1+0.4:1:size(pop_lda_up_train,2)+0.4,pop_lda_up_train_mean(e,:),nanstd(squeeze(pop_lda_up_train(e,:,:))'),'k--')
%         errorline(1+0.4:1:size(pop_lda_bp_train,2)+0.4,pop_lda_bp_train_mean(e,:),nanstd(squeeze(pop_lda_bp_train(e,:,:))'),'r--')
%         errorline(1:size(pop_lda_up_test,2),pop_lda_up_test_mean(e,:),nanstd(squeeze(pop_lda_up_train(e,:,:))'),'k')
%         errorline(1:size(pop_lda_bp_test,2),pop_lda_bp_test_mean(e,:),nanstd(squeeze(pop_lda_bp_train(e,:,:))'),'r')
%     %     errorline(2:size(resp_raw_base,1),1-mean(pop_err_u,1),std(pop_err_u,1),'k')
%     %     errorline(2:size(resp_raw_base,1),1-mean(pop_err_b,1),std(pop_err_b,1),'r')
%         ylim([0 1])
%         axis square; box off
%         if e==1
%             legend({'unif training','bias training','unif test','bias test'},'Location','southeast')
%         end
%         xlabel('# of units (steps of 3)')
%         ylabel('% correct')
% %     set(gca,'XTick',...
%     end
    
    % fit performance data and find 75% performance mark
    % (center ori, pop size, repeats (10x10=100), performance: train & test)
    for e=1:length(ori_base)-1
        % Xdata = # of steps (pop size); Ydata = % correct at that size
        % only doing this for matched distribution
        [par_up_train(e,:), gof_up_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean(e,:),[0.1 0.1 0.1],[]);
        [par_bp_train(e,:), gof_bp_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean(e,:),[0.1 0.1 0.1],[]);
        [par_up_test(e,:), gof_up_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean(e,:),[0.1 0.1 0.1],[]);
        [par_bp_test(e,:), gof_bp_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean(e,:),[0.1 0.1 0.1],[]);
        
        [par_up_train_shuf(e,:), gof_up_train_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean_shuf(e,:),[0.1 0.1 0.1],[]);
        [par_bp_train_shuf(e,:), gof_bp_train_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean_shuf(e,:),[0.1 0.1 0.1],[]);
        [par_up_test_shuf(e,:), gof_up_test_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean_shuf(e,:),[0.1 0.1 0.1],[]);
        [par_bp_test_shuf(e,:), gof_bp_test_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean_shuf(e,:),[0.1 0.1 0.1],[]);
        
        fit_up_train(e,:)=1-par_up_train(e,3)*exp(-(steps_fine/par_up_train(e,1)).^par_up_train(e,2));
        fit_bp_train(e,:)=1-par_bp_train(e,3)*exp(-(steps_fine/par_bp_train(e,1)).^par_bp_train(e,2));
        fit_up_test(e,:)=1-par_up_test(e,3)*exp(-(steps_fine/par_up_test(e,1)).^par_up_test(e,2));
        fit_bp_test(e,:)=1-par_bp_test(e,3)*exp(-(steps_fine/par_bp_test(e,1)).^par_bp_test(e,2));
        
        fit_up_train_shuf(e,:)=1-par_up_train_shuf(e,3)*exp(-(steps_fine/par_up_train_shuf(e,1)).^par_up_train_shuf(e,2));
        fit_bp_train_shuf(e,:)=1-par_bp_train_shuf(e,3)*exp(-(steps_fine/par_bp_train_shuf(e,1)).^par_bp_train_shuf(e,2));
        fit_up_test_shuf(e,:)=1-par_up_test_shuf(e,3)*exp(-(steps_fine/par_up_test_shuf(e,1)).^par_up_test_shuf(e,2));
        fit_bp_test_shuf(e,:)=1-par_bp_test_shuf(e,3)*exp(-(steps_fine/par_bp_test_shuf(e,1)).^par_bp_test_shuf(e,2));
        
        
%         if max(fit_up_train(e,:))<0.75 || max(fit_up_test(e,:))<0.75
%             thresh_u_train(e)=find(fit_up_train(e,:)>=.75*max(fit_up_train(e,:)),1,'first');
%             thresh_u_test(e)=find(fit_up_test(e,:)>=.75*max(fit_up_test(e,:)),1,'first');
%         else
%             thresh_u_train(e)=find(fit_up_train(e,:)>=.75,1,'first');
%             thresh_u_test(e)=find(fit_up_test(e,:)>=.75,1,'first');
%         end
%         if max(fit_bp_train(e,:))<0.75 || max(fit_bp_test(e,:))<0.75
%             thresh_b_train(e)=find(fit_bp_train(e,:)>=.75*max(fit_bp_train(e,:)),1,'first');
%             thresh_b_test(e)=find(fit_bp_test(e,:)>=.75*max(fit_bp_test(e,:)),1,'first');
%         else
%             thresh_b_train(e)=find(fit_bp_train(e,:)>=.75,1,'first');
%             thresh_b_test(e)=find(fit_bp_test(e,:)>=.75,1,'first');
%         end
%         if max(fit_up_train_shuf(e,:))<0.75 || max(fit_up_test_shuf(e,:))<0.75
%             thresh_u_train_shuf(e)=find(fit_up_train_shuf(e,:)>=.75*max(fit_up_train_shuf(e,:)),1,'first');
%             thresh_u_test_shuf(e)=find(fit_up_test_shuf(e,:)>=.75*max(fit_up_test_shuf(e,:)),1,'first');
%         else
%             thresh_u_train_shuf(e)=find(fit_up_train_shuf(e,:)>=.75,1,'first');
%             thresh_u_test_shuf(e)=find(fit_up_test_shuf(e,:)>=.75,1,'first');
%         end
%         if max(fit_bp_train_shuf(e,:))<0.75 || max(fit_bp_test_shuf(e,:))<0.75
%             thresh_b_train_shuf(e)=find(fit_bp_train_shuf(e,:)>=.75*max(fit_bp_train_shuf(e,:)),1,'first');
%             thresh_b_test_shuf(e)=find(fit_bp_test_shuf(e,:)>=.75*max(fit_bp_test_shuf(e,:)),1,'first');
%         else
%             thresh_b_train_shuf(e)=find(fit_bp_train_shuf(e,:)>=.75,1,'first');
%             thresh_b_test_shuf(e)=find(fit_bp_test_shuf(e,:)>=.75,1,'first');
%         end
    end
    
    figure
    supertitle('acute pnf fits')
    for e=1:length(ori_base)-1
        subplot(3,3,e); hold on
        title(sprintf('ori %g',e))
%         %         errorline(1+0.4:1:size(pop_lda_up_train,2)+0.4,pop_lda_up_train_mean(e,:),nanstd(squeeze(pop_lda_up_train(e,:,:))'),'k--')
%         errorline(1+0.4:1:size(pop_lda_bp_train,2)+0.4,pop_lda_bp_train_mean(e,:),nanstd(squeeze(pop_lda_bp_train(e,:,:))'),'r--')
%         errorline(1:size(pop_lda_up_test,2),pop_lda_up_test_mean(e,:),nanstd(squeeze(pop_lda_up_train(e,:,:))'),'k.')
%         errorline(1:size(pop_lda_bp_test,2),pop_lda_bp_test_mean(e,:),nanstd(squeeze(pop_lda_bp_train(e,:,:))'),'r.')
%         
% %         errorline(1+0.4:1:size(pop_lda_up_train_shuf,2)+0.4,pop_lda_up_train_mean_shuf(e,:),nanstd(squeeze(pop_lda_up_train_shuf(e,:,:))'),'b--')
% %         errorline(1+0.4:1:size(pop_lda_bp_train_shuf,2)+0.4,pop_lda_bp_train_mean_shuf(e,:),nanstd(squeeze(pop_lda_bp_train_shuf(e,:,:))'),'m--')
%         errorline(1:size(pop_lda_up_test_shuf,2),pop_lda_up_test_mean_shuf(e,:),nanstd(squeeze(pop_lda_up_train_shuf(e,:,:))'),'b.')
%         errorline(1:size(pop_lda_bp_test_shuf,2),pop_lda_bp_test_mean_shuf(e,:),nanstd(squeeze(pop_lda_bp_train_shuf(e,:,:))'),'m.')
%         
%     %     errorline(2:size(resp_raw_base,1),1-mean(pop_err_u,1),std(pop_err_u,1),'k')
%     %     errorline(2:size(resp_raw_base,1),1-mean(pop_err_b,1),std(pop_err_b,1),'r')
    
        plot(steps_fine,fit_up_test(e,:),'k')
        plot(steps_fine,fit_bp_test(e,:),'r')
        plot(steps_fine,fit_up_test_shuf(e,:),'b')
        plot(steps_fine,fit_bp_test_shuf(e,:),'m')
        ylim([0 1])
        axis square; box off
%         if e==1
%             legend({'unif training','bias training','unif test','bias test'},'Location','southeast')
%         end
        xlabel('# of units')
        ylabel('% correct')
%     set(gca,'XTick',...
    end
    
%     figure
%     if shiftori==1
%         plot(circshift(thresh_b_train./thresh_u_train,0),'k')
%         plot(circshift(thresh_b_test./thresh_u_test,0),'r')
%     else
%         plot(circshift(thresh_b_train./thresh_u_train,4),'k')
%         plot(circshift(thresh_b_test./thresh_u_test,4),'r')
%     end
%     axis square; box off
%     xlabel('center ori of 3-class')
%     ylabel('relative change in 75% threshold')
%     title('LDA Neurometric Threshold')
    
    disp('pnf done')
    clearvars -except par* thresh* pop* num_units ori_base oribias oripref resp*...
         stim* steps* shiftori fit* gof* name n class* trainsize testsize prob* t osi_tmp bias keep* base
    %%	test different # of trials
    % trial_err_u=nan*zeros(20,6);
    % trial_err_b=nan*zeros(20,6);
    % trial_class_u=cell(20,6);
    % trial_class_b=cell(20,6);
    % % tmp=base(test);
    % % tmp2=bias(test);
    % for b=1:20
    %     id=1;
    %     for e=[300 600 1200 2400 4800 9600]
    %         j=randperm(size(resp_raw_base,1),40);
    %
    %         k=randperm(length(trainu),e);
    %         k2=randperm(length(trainb),e);
    % %         k2=randperm(length(keep-tenp),e);
    %         units_u=resp_raw_base(j,trainu(k));
    %         units_b=resp_raw_bias(j,trainb(k2));
    % %         units_up=resp_raw_base_sub(j,k);
    % %         units_bp=resp_raw_bias_sub(j,k);
    %         tenp=0.1*e;
    %         uo=base2(trainu(k));
    %         bo=bias2(trainb(k2));
    %
    %
    %         [tmp_lda_u,tmp_err_u]=classify(units_u(:,end-tenp+1:end)',units_u(:,1:end-tenp)',uo(1:end-tenp)');
    %         [tmp_lda_b,tmp_err_b]=classify(units_b(:,end-tenp+1:end)',units_b(:,1:end-tenp)',bo(1:end-tenp)');
    % %         [tmp_lda_up,tmp_err_up]=classify(units_up(:,end-tenp+1:end)',units_up(:,1:end-tenp)',temp1(k2)');
    % %         [tmp_lda_bp,tmp_err_bp]=classify(units_bp(:,end-tenp+1:end)',units_bp(:,1:end-tenp)',temp2(k2)');
    %         trial_err_u(b,id)=tmp_err_u;
    %         trial_err_b(b,id)=tmp_err_b;
    % %         trial_err_up(a,id)=tmp_err_up;
    % %         trial_err_bp(a,id)=tmp_err_bp;
    %         trial_class_u{b,id}=[tmp_lda_u uo(end-tenp+1:end)'];
    %         trial_class_b{b,id}=[tmp_lda_b bo(end-tenp+1:end)'];
    % %         trial_class_up{a,id}=[tmp_lda_up uo(end-tenpercent+1:end)];
    % %         trial_class_bp{a,id}=[tmp_lda_bp bo(end-tenpercent+1:end)];
    %         id=id+1;
    %     end
    % end
    % figure
    % supertitle('Acute, LDA performance vs trial size (40 neurons)')
    % subplot(121); hold on
    % title('training set performance')
    % errorline(1:6,1-mean(trial_err_u,1),std(trial_err_u,1),'k')
    % errorline(1.1:1:6.1,1-mean(trial_err_b,1),std(trial_err_b,1),'r')
    % ylim([0 1])
    % % errorline(1-mean(trial_err_up,1),std(trial_err_up,1),'k--')
    % % errorline(1-mean(trial_err_bp,1),std(trial_err_bp,1),'r--')
    % set(gca,'XTick',1:6,'XTickLabel',{'300','600','1200','2400','4800','9600'},'TickDir','out')
    % axis square; box off
    % legend({'unif','bias'})%,'u pred','b pred'})
    % xlabel('# trials in training')
    % ylabel('% correct')
    %
    % subplot(122); hold on
    % title('Test set performance')
    % for b = 1:size(trial_err_u,1)
    %     for e = 1:size(trial_err_u,2)
    %         temp=squeeze(trial_class_u{b,e});
    %         tmp=find(temp(:,1)==temp(:,2));
    %         temp=squeeze(trial_class_b{b,e});
    %         tmp2=find(temp(:,1)==temp(:,2));
    % %         temp=squeeze(trial_class_up{a,e});
    % %         tmp3=find(temp(:,1)==temp(:,2));
    % %         temp=squeeze(trial_class_bp{a,e});
    % %         tmp4=find(temp(:,1)==temp(:,2));
    %
    %         test_rate_u_trl(b,e)=length(tmp)/length(temp);
    %         test_rate_b_trl(b,e)=length(tmp2)/length(temp);
    % %         test_rate_up_trl(a,e)=length(tmp3)/length(temp);
    % %         test_rate_bp_trl(a,e)=length(tmp4)/length(temp);
    %     end
    % end
    % errorline(1:6,mean(test_rate_u_trl,1),std(test_rate_u_trl,1),'k')
    % errorline(1.1:1:6.1,mean(test_rate_b_trl,1),std(test_rate_b_trl,1),'r')
    % % errorline(mean(test_rate_up_trl,1),std(test_rate_up_trl,1),'k--')
    % % errorline(mean(test_rate_bp_trl,1),std(test_rate_bp_trl,1),'r--')
    % ylim([0 1])
    % set(gca,'XTick',1:6,'XTickLabel',{'300','600','1200','2400','4800','9600'},'TickDir','out')
    % axis square; box off
    % legend({'unif','bias'})%,'u pred','b pred'})
    % xlabel('# trials in training')
    % ylabel('% correct')
    % disp('trials done')
    clear ans id tmp* x y uo bo units_* temp* ten* a b e i j k n1 n2
	
    save(name);
    end
    stop
end
stop
% if size(resp_raw_base,2)==15000     % 2:1, 4:1, lowcon, blank - 13501:end; 1:13500 (1500/15000)
    %     trainu=sort(randperm(length(base2),round(0.9*length(base2))));  % random 90% of trials
    %     testu=setdiff(1:length(base2),trainu);                          % random 10% of trials
    %     trainb=sort(randperm(length(bias2),round(0.9*length(bias2))));
    %     testb=setdiff(1:length(bias2),trainb);
    % elseif size(resp_raw_base,2)==7500      % awaketime - 6751:end; 1:6750 (750/7500)
    %     exp_type='awaketime';
    % %     test=6751:7500;     % last 10% of trials
    % %     train=1:6750;       % first 90% of trials
    %     train=sort(randperm(7500,6750));        % random 90% of trials
    %     test=setdiff(1:7500,train);             % random 10% of trials
    % elseif size(resp_raw_base,2)==49000     % benucci - 44101:end; 1:44100 (4900/49000)
    %     exp_type='benucci';
    % %     test=44101:49000;   % last 10% of trials
    % %     train=1:44100;      % first 90% of trials
    %     train=sort(randperm(49000,44100));      % random 90% of trials
    %     test=setdiff(1:49000,train);            % random 10% of trials
    % else
    %     error('trials/experiment type')
    % end

%% combine acute files....
% see acute_combined.m

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
    elseif a==15
%         load('cadetv1p366_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p366_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==16
%         load('cadetv1p371_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p371_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==17
%         load('cadetv1p384_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p384_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');;
    elseif a==18
%         load('cadetv1p385_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p385_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==19
%         load('cadetv1p392_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p392_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==20
%         load('cadetv1p403_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p403_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==21
%         load('cadetv1p419_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p419_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==22
%         load('cadetv1p432_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p432_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==23
%         load('cadetv1p437_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p437_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==24
%         load('cadetv1p438_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p438_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==25
%         load('cadetv1p460_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p460_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==26
%         load('cadetv1p467_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p467_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==27
%         load('cadetv1p468_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p468_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==28
        % before I made changes:
        load('cadetv1p422_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
        load('cadetv1p422_tuning','ori*','resp*','spont','tune*');
    end
    %%  preprocessing    
    % select tuning criteria:
    osi_tmp=[0 0.35 1];
    for t=1:3
        osi_keep=find(oribias>=osi_tmp(t) & oribias<=osi_tmp(t+1));
        if t==1
            savename=sprintf('%s_lda_pnf2_lowOSI',filename);
        elseif t==2
            savename=sprintf('%s_lda_pnf2_hiOSI',filename);
        end
%     end
clearvars -except osi_tmp osi_keep t savename filename keep* ori* resp_bias resp_uniform
    num_units=length(osi_keep);
    
    % entropy files already exclude non-responsive units
%     num_units=sum(responsive);
% %     ru=resp_uniform(:,:);
% %     rb=resp_bias(:,:);
% %     rusub=resp_uniform(:,keep);
% %     rbsub=resp_bias(:,keep2);
% %     spont=spont(responsive,:);
    
    % preallocate # of trials for test and train;
% %     l1=length(oris_u);
% %     l2=length(oris_b);
% %     l3=length(keep);
% %     r1=round(0.1*l1);
% %     r2=round(0.1*l2);
% %     r3=round(0.1*l3);
%     temp1=oris_u(keep);
%     temp2=oris_b(keep2);
%     tenpu=round(0.1*length(keep));
%     tenpb=round(0.1*length(keep2));
%     trainsizeu=sort(randperm(length(oris_u),round(0.9*length(oris_u))));
%     testu=setdiff(1:length(oris_u),trainsizeu);
%     trainsizeb=sort(randperm(length(oris_b),round(0.9*length(oris_b))));
%     testb=setdiff(1:length(oris_b),trainsizeb);
    
    %% Over whole experiment: train on 90% of trials
    % prediction = classify(test,train,stims,'linear');
    % match # of trials of each class in training sets
%     for e = 1:length(oris)
%         tmp(e)=length(find(oris_u(trainsizeu)==oris(e)));
%         tmp2(e)=length(find(oris_b(trainsizeb)==oris(e)));
%     end
%     trainmatch=min([tmp tmp2]);
%     trainu=[];
%     trainb=[];
%     for e = 1:length(oris)
%         tmp=find(oris_u(trainsizeu)==oris(e));
%         trainu=[trainu tmp(1:trainmatch)];
%         tmp2=find(oris_b(trainsizeb)==oris(e));
%         trainb=[trainb tmp2(1:trainmatch)];
%     end
%     
%     % 10% of trials used as test
%     [lda_u, lda_err_u]=classify(ru(:,testu)',ru(:,trainu)',oris_u(trainu)');
%     [lda_b, lda_err_b]=classify(rb(:,testb)',rb(:,trainb)',oris_b(trainb)');
%     [lda_up, lda_err_up]=classify(rusub(:,end-tenpu+1:end)',rusub(:,1:end-tenpu)',temp1(1:end-tenpu)');
%     [lda_bp, lda_err_bp]=classify(rbsub(:,end-tenpu+1:end)',rbsub(:,1:end-tenpu)',temp2(1:end-tenpu)');
%     tmp=find(lda_u'==oris_u(testu));
%     tmp2=find(lda_b'==oris_b(testb));
%     tmp3=find(lda_up'==temp1(end-tenpu+1:end));
%     tmp4=find(lda_bp'==temp2(end-tenpu+1:end));
%     test_rate_u=length(tmp)/length(oris_u(testu));
%     test_rate_b=length(tmp2)/length(oris_b(testb));
%     test_rate_up=length(tmp3)/length(temp1(end-tenpu+1:end));
%     test_rate_bp=length(tmp4)/length(temp2(end-tenpu+1:end));
% 
%    figure
%    title(sprintf('Awake %g, lda, whole exp',a))
%    hold on
%    plot(1-lda_err_u,test_rate_u,'k.','MarkerSize',10)
%    plot(1-lda_err_up,test_rate_up,'b.','MarkerSize',10)
%    plot(1-lda_err_b,test_rate_b,'r.','MarkerSize',10)
%    plot(1-lda_err_bp,test_rate_bp,'g.','MarkerSize',10)
%    xlabel('training: % correct')
%    ylabel('test: % correct')
%    legend({'Uniform','U pred','Bias','B pred'},'Location','southeast')
%    axis square; box off
%    set(gca,'TickDir','out')
%    refline(1,0)
%    disp('full done')

% old way ^
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% new way: proper 10x-CV and includes shuffled trials
    % find trial count of each ori:
%     clear tmp*
%     for e = 1:length(oris)
%         tmp(e)=length(find(oris_u==oris(e)));
%         tmp2(e)=length(find(oris_b==oris(e)));
%     end
%     % size of training set - 90% of trials of ori used least:
%     trainmatch=round(0.9*min([tmp tmp2])); 
%     
%     % 10x cross validate full experiment:
%     for j=1:10
%         trainu=[];
%         trainb=[];
%         testu=[];
%         testb=[];
%         trainclassu=[];
%         trainclassb=[];
%         testclassu=[];
%         testclassb=[];
%         for e = 1:length(oris)
%             tmp=find(oris_u==oris(e));  % all uniform trials of oris(e)
%             ushuf=randperm(length(tmp),trainmatch); % random subset of trials for training
%             trainu=[trainu tmp(ushuf)]; % store training
%             trainclassu=[trainclassu ones(1,trainmatch)*oris(e)]; % corresponding class for training
%             testu=[testu tmp(setdiff(1:length(tmp),ushuf))];    % trials of oris(e) not used in training
%             testclassu=[testclassu ones(1,length(tmp)-trainmatch)*oris(e)]; % class of test trials
%             tsu(e,:)=tmp(ushuf); % uniform trials stored for shuffling later
%             
%             tmp=find(oris_b==oris(e));
%             bshuf=randperm(length(tmp),trainmatch);
%             trainb=[trainb tmp(bshuf)];
%             trainclassb=[trainclassb ones(1,trainmatch)*oris(e)];
%             testb=[testb tmp(setdiff(1:length(tmp),bshuf))];
%             testclassb=[testclassb ones(1,length(tmp)-trainmatch)*oris(e)];
%             tsb(e,:)=tmp(bshuf); % bias trials stored for shuffling later
%         end
%         % trials aligned:
%         [lda_u, lda_err_u(j)]=classify(ru(:,testu)',ru(:,trainu)',trainclassu');
%         [lda_b, lda_err_b(j)]=classify(rb(:,testb)',rb(:,trainb)',trainclassb');
%         tmp=find(lda_u'==testclassu);
%         tmp2=find(lda_b'==testclassb);
%         test_rate_u(j)=length(tmp)/length(testclassu);
%         test_rate_b(j)=length(tmp2)/length(testclassb);
%         % trials shuffled (removes influence of correlations);
%         for k=1:size(ru,1)
%             temp=[];
%             temp2=[];
%             for m=1:size(tsb,1)
%                 temp=[temp tsu(m,randperm(length(tsu)))];
%                 temp2=[temp2 tsb(m,randperm(length(tsb)))];
%             end
%             trainu_shuf(k,:)=ru(k,temp);
%             trainb_shuf(k,:)=rb(k,temp2);
%         end
%         [lda_u_shuf, lda_err_u_shuf(j)]=classify(ru(:,testu)',trainu_shuf',trainclassu');
%         [lda_b_shuf, lda_err_b_shuf(j)]=classify(rb(:,testb)',trainb_shuf',trainclassb');
%         tmp_shuf=find(lda_u_shuf'==testclassu);
%         tmp2_shuf=find(lda_b_shuf'==testclassb);
%         test_rate_u_shuf(j)=length(tmp_shuf)/length(testclassu);
%         test_rate_b_shuf(j)=length(tmp2_shuf)/length(testclassb);
%     end
%     
% %     figure % this plot should have errorbars
% %     title(sprintf('Awake %g, lda, whole exp',a))
% %     hold on
% %     plot(1-mean(lda_err_u),mean(test_rate_u),'k.','MarkerSize',10)
% %     plot(1-mean(lda_err_b),mean(test_rate_b),'r.','MarkerSize',10)
% %     plot(1-mean(lda_err_u_shuf),mean(test_rate_u_shuf),'kx','MarkerSize',10)
% %     plot(1-mean(lda_err_b_shuf),mean(test_rate_b_shuf),'rx','MarkerSize',10)
% %     xlabel('training: Avg % correct')
% %     ylabel('test: Avg % correct')
% %     legend({'Uniform','Bias','U shuf','B shuf'},'Location','southeast')
% %     axis square; box off; xlim([0.5 1]); ylim([0.5 1])
% %     set(gca,'TickDir','out')
% %     refline(1,0)
%     disp('full done')
    
    %% % % % % % % % % % % performance vs neighbor oris
%     clear tmp* temp* trainu* trainb* testu testb
%     lda_neighbors_u=nan*zeros(100,9,2); %10 repeats x 10 new units X 9 oris X train/test
%     lda_neighbors_b=nan*zeros(100,9,2);
%     lda_neighbors_u_shuf=nan*zeros(100,9,2);
%     lda_neighbors_b_shuf=nan*zeros(100,9,2);
%     
%     for e = 1:length(oris)
%         id=1;
%         if e==1
%             n1=length(oris);
%             n2=e+1;
%         elseif e==length(oris)
%             n1=e-1;
%             n2=1;
%         else
%             n1=e-1;
%             n2=e+1;
%         end
%         tmp=find(oris_u==oris(e));
%         tmp1=find(oris_u==oris(n1));
%         tmp2=find(oris_u==oris(n2));
%         temp=find(oris_b==oris(e));
%         temp1=find(oris_b==oris(n1));
%         temp2=find(oris_b==oris(n2));
%   
%         for j = 1:10
%             % train trials:
%             tmpu1=randperm(length(tmp),trainmatch);
%             tmpu2=randperm(length(tmp1),trainmatch);
%             tmpu3=randperm(length(tmp2),trainmatch);
%             tmpb1=randperm(length(temp),trainmatch);
%             tmpb2=randperm(length(temp1),trainmatch);
%             tmpb3=randperm(length(temp2),trainmatch);
%             trainu=[tmp(tmpu1) tmp1(tmpu2) tmp2(tmpu3)];
%             trainb=[temp(tmpb1) temp1(tmpb2) temp2(tmpb3)];
%             % shuffle train trials independently for each unit (row):
%             for k=1:size(ru,1)
%                 tmpu1shuf=randperm(trainmatch);
%                 tmpb1shuf=randperm(trainmatch);
%                 tmpu2shuf=randperm(trainmatch);
%                 tmpb2shuf=randperm(trainmatch);
%                 tmpu3shuf=randperm(trainmatch);
%                 tmpb3shuf=randperm(trainmatch);
%                 
%                 trainu_shuf(k,:)=ru(k,[tmp(tmpu1shuf) tmp1(tmpu2) tmp2(tmpu3)]);
%                 trainb_shuf(k,:)=rb(k,[temp(tmpb1shuf) temp1(tmpb2) temp2(tmpb3)]);
%             end
%             % test trials:
%             testu1=tmp(setdiff(1:length(tmp),tmpu1));
%             testu2=tmp1(setdiff(1:length(tmp1),tmpu2));
%             testu3=tmp2(setdiff(1:length(tmp2),tmpu3));
%             testb1=temp(setdiff(1:length(temp),tmpb1));
%             testb2=temp1(setdiff(1:length(temp1),tmpb2));
%             testb3=temp2(setdiff(1:length(temp2),tmpb3));
%             testu=[testu1 testu2 testu3];
%             testb=[testb1 testb2 testb3];
%             
%             for b=1:10
%                 n=randperm(size(ru,1),30); % 40 like in acute is too high, 20 may be more appropriate for awake performance
%                 % trials aligned (i.e. w/ correlations)
%                 units_u=ru(n,:);
%                 units_b=rb(n,:);
%                 [lda_near_u, lda_near_err_u]=classify(units_u(:,testu)',units_u(:,trainu)',oris_u(trainu)');
%                 [lda_near_b, lda_near_err_b]=classify(units_b(:,testb)',units_b(:,trainb)',oris_b(trainb)');
%                 tmp5=find(lda_near_u'==oris_u(testu));
%                 tmp6=find(lda_near_b'==oris_b(testb));
%                 lda_neighbors_u(id,e,:)=[1-lda_near_err_u length(tmp5)/length(lda_near_u)];
%                 lda_neighbors_b(id,e,:)=[1-lda_near_err_b length(tmp6)/length(lda_near_b)];
%                 % trials shuffled (no correlations)
%                 units_u2=trainu_shuf(n,:);
%                 units_b2=trainb_shuf(n,:);
%                 [lda_near_u_shuf, lda_near_err_u_shuf]=classify(units_u(:,testu)',units_u2',oris_u(trainu)');
%                 [lda_near_b_shuf, lda_near_err_b_shuf]=classify(units_b(:,testb)',units_b2',oris_b(trainb)');
%                 tmp7=find(lda_near_u_shuf'==oris_u(testu));
%                 tmp8=find(lda_near_b_shuf'==oris_b(testb));
%                 lda_neighbors_u_shuf(id,e,:)=[1-lda_near_err_u_shuf length(tmp7)/length(lda_near_u_shuf)];
%                 lda_neighbors_b_shuf(id,e,:)=[1-lda_near_err_b_shuf length(tmp8)/length(lda_near_b_shuf)];
%                 id=id+1;
%             end
%             
%             clearvars testu* testb* tmpu* tmpb* trainu* trainb*
%         end        
%     end
%     disp('3class done')
    
% new way ^    
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     
% old way:
%     lda_neighbors_u=nan*zeros(30,9,2); %30 repeats of sampled units x 9 oris/neighbors x [training test]
%     lda_neighbors_b=nan*zeros(30,9,2);
%     for e = 1:length(oris)
%         if e==1
%             n1=length(oris);
%             n2=e+1;
%         elseif e==length(oris)
%             n1=e-1;
%             n2=1;
%         else
%             n1=e-1;
%             n2=e+1;
%         end
%         tmp=find(oris_u(trainsizeu)==oris(e));
%         tmp1=find(oris_u(trainsizeu)==oris(n1));
%         tmp2=find(oris_u(trainsizeu)==oris(n2));
%         temp=find(oris_b(trainsizeb)==oris(e));
%         temp1=find(oris_b(trainsizeb)==oris(n1));
%         temp2=find(oris_b(trainsizeb)==oris(n2));
%         % match trial for all three oris in uniform and bias:
%         x=[tmp(1:trainmatch) tmp2(1:trainmatch) tmp1(1:trainmatch)];
%         x=sort(x);  % uniform training data
%         y=[temp(1:trainmatch) temp1(1:trainmatch) temp2(1:trainmatch)];
%         y=sort(y);  % bias training data
%         
%         for b=1:30
%             n=randperm(size(ru,1),30); % 40 like in acute is too high, 20 may be more appropriate for awake performance
%             units_u=ru(n,:);
%             units_b=rb(n,:);
%             [lda_near_u, lda_near_err_u]=classify(units_u(:,testu)',units_u(:,x)',oris_u(x)');
%             [lda_near_b, lda_near_err_b]=classify(units_b(:,testb)',units_b(:,y)',oris_b(y)');
%             tmp=find(lda_near_u'==oris_u(testu));
%             tmp2=find(lda_near_b'==oris_b(testb));
%             lda_neighbors_u(b,e,:)=[1-lda_near_err_u length(tmp)/length(lda_near_u)];
%             lda_neighbors_b(b,e,:)=[1-lda_near_err_b length(tmp2)/length(lda_near_b)];
%         end
%     end
%     %     units_up=resp_raw_base_sub(n,x);
%     %     units_bp=resp_raw_bias_sub(n,y);
%     %     [lda_near_up, lda_near_err_up]=classify(units_up(:,tmpsize+1:end)',units_up(:,1:tmpsize)',uniform_dist(x(1:tmpsize))');
%     %     [lda_near_bp, lda_near_err_bp]=classify(units_bp(:,tmpsize2+1:end)',units_bp(:,1:tmpsize2)',biased_dist(y(1:tmpsize2))');
%     %     tmp3=find(lda_near_up==uniform_dist(x(tmpsize+1:end)));
%     %     tmp4=find(lda_near_bp==biased_dist(y(tmpsize2+1:end)));
%     %     lda_neighbors_up(e,:)=[1-lda_near_err_up length(tmp3)/length(lda_near_u)];
%     %     lda_neighbors_bp(e,:)=[1-lda_near_err_bp length(tmp4)/length(lda_near_b)];

%     figure
%     supertitle(sprintf('Awake %g, LDA, performance vs neighboring oris (30 units)',a))
%     subplot(121); hold on
%     title('Training')
%     errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u(:,:,1),1)),4),circshift(squeeze(nanstd(lda_neighbors_u(:,:,1),1)),4),'k-')
%     errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b(:,:,1),1)),4),circshift(squeeze(nanstd(lda_neighbors_b(:,:,1),1)),4),'r-')
%     errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u_shuf(:,:,1),1)),4),circshift(squeeze(nanstd(lda_neighbors_u_shuf(:,:,1),1)),4),'k--')
%     errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b_shuf(:,:,1),1)),4),circshift(squeeze(nanstd(lda_neighbors_b_shuf(:,:,1),1)),4),'r--')
%     % plot(1:10,lda_neighbors_up(:,1),'k--')
%     % plot(1:10,lda_neighbors_bp(:,1),'r--')
%     ylim([0 1])
%     xlim([0 10])
%     set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
%     axis square; box off
%     legend({'unif','bias','Ushuf','Bshuf'},'Location','southeast')
%     xlabel('center ori +/-20')
%     ylabel('% correct')
%     subplot(122); hold on
%     title('Test')
%     errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u(:,:,2),1)),4),circshift(squeeze(nanstd(lda_neighbors_u(:,:,2),1)),4),'k')
%     errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b(:,:,2),1)),4),circshift(squeeze(nanstd(lda_neighbors_b(:,:,2),1)),4),'r')
%     errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u_shuf(:,:,2),1)),4),circshift(squeeze(nanstd(lda_neighbors_u_shuf(:,:,2),1)),4),'k--')
%     errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b_shuf(:,:,2),1)),4),circshift(squeeze(nanstd(lda_neighbors_b_shuf(:,:,2),1)),4),'r--')
%     % plot(1:10,lda_neighbors_up(:,2),'k--')
%     % plot(1:10,lda_neighbors_bp(:,2),'r--')
%     ylim([0 1])
%     xlim([0 10])
%     set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
%     axis square; box off
%     xlabel('center ori +/-20')
%     ylabel('% correct')
%     clear tmp* units*
    
    %% % % % % % % % % % % performance w/ different # of neurons
    % population neurometric function
    
    % subsampled trials:
    base_keep=oris_u(keep);
    bias_keep=oris_b(keep2);
%     resp_raw_base_sub=resp_uniform(:,keep);
%     resp_raw_bias_sub=resp_bias(:,keep2);
    resp_raw_base_sub=resp_uniform(osi_keep,keep);
    resp_raw_bias_sub=resp_bias(osi_keep,keep2);
    
%     step to only include 'tuned' units
% tmp=(sum(resp_raw_bias_sub,2)>100);
% oribias_u=oribias_u(tmp);
    
    % exclude 'unresponsive' units for LDA training
    resp_raw_base_sub=resp_raw_base_sub(sum(resp_raw_bias_sub,2)>100,:);
    resp_raw_bias_sub=resp_raw_bias_sub(sum(resp_raw_bias_sub,2)>100,:);

% resp_raw_base_sub=resp_raw_base_sub(oribias_u>0.3,:);
% resp_raw_bias_sub=resp_raw_bias_sub(oribias_u>0.3,:);

    num_units=size(resp_raw_base_sub,1);
    % steps of units for analysis
    steps=2:1:num_units;
    steps_fine=2:.25:110;
    % find and store index of each stim to call for training and test
    % (should be equal number of each if distribution created correctly)
    for b=1:length(oris)
        stimrefsu(b,:)=find(base_keep==oris(b));
        stimrefsb(b,:)=find(bias_keep==oris(b));
    end
    trainsize=round(0.9*size(stimrefsu,2));
    testsize=size(stimrefsu,2)-trainsize;
    
%     pop_lda_u=nan*zeros(length(steps),100,2);
%     pop_lda_b=nan*zeros(length(steps),100,2);
    pop_lda_up_train=nan*zeros(9,length(steps),200);
    pop_lda_bp_train=nan*zeros(9,length(steps),200);
    pop_lda_up_test=nan*zeros(9,length(steps),200);
    pop_lda_bp_test=nan*zeros(9,length(steps),200);
    pop_lda_up_train_shuf=nan*zeros(9,length(steps),200);
    pop_lda_bp_train_shuf=nan*zeros(9,length(steps),200);
    pop_lda_up_test_shuf=nan*zeros(9,length(steps),200);
    pop_lda_bp_test_shuf=nan*zeros(9,length(steps),200);
%     class_up_train=nan*zeros(9,length(steps),200,trainsize*2);
%     class_bp_train=nan*zeros(9,length(steps),200,trainsize*2);
    class_up_test=nan*zeros(9,length(steps),200,testsize*2);
    class_bp_test=nan*zeros(9,length(steps),200,testsize*2);
%     class_up_train_shuf=nan*zeros(9,length(steps),200,trainsize*2);
%     class_bp_train_shuf=nan*zeros(9,length(steps),200,trainsize*2);
    class_up_test_shuf=nan*zeros(9,length(steps),200,testsize*2);
    class_bp_test_shuf=nan*zeros(9,length(steps),200,testsize*2);
%     prob_up_train=nan*zeros(9,length(steps),200,trainsize*2);
%     prob_bp_train=nan*zeros(9,length(steps),200,trainsize*2);
    prob_up_test=nan*zeros(9,length(steps),200,testsize*2);
    prob_bp_test=nan*zeros(9,length(steps),200,testsize*2);
%     prob_up_train_shuf=nan*zeros(9,length(steps),200,trainsize*2);
%     prob_bp_train_shuf=nan*zeros(9,length(steps),200,trainsize*2);
    prob_up_test_shuf=nan*zeros(9,length(steps),200,testsize*2);
    prob_bp_test_shuf=nan*zeros(9,length(steps),200,testsize*2);
    
    tic
    % loop over orientations (center)
    for o=1:length(oris)
        % select neighbor oris for 3 class:
%         if o==1
%             n1=length(oris);
%             n2=o+1;
%         elseif o==length(oris)
%             n1=o-1;
%             n2=1;
%         else
%             n1=o-1;
%             n2=o+1;
%         end
        % select neighbor ori for 2 class:
        if o==length(oris)
            n1=1;   % last ori w/ first ori (160-0)
        else
            n1=o+1; % next ori (e.g. 0-20, 20-40...)
        end
        
        idx=1;  % index for population size 
        % loop over population size:
        for e=steps
            id1=1; % index for loop over neuron samples and trials (20x neurons & 10x trials =200);
            % loop over random neurons selected:
            for b=1:20 % index for train/test trials
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
%                     tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp) stimrefsu(n2,tmp)];
%                     tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp) stimrefsb(n2,tmp)];
%                     x=setdiff([stimrefsu(o,:) stimrefsu(n1,:) stimrefsu(n2,:)],tmp2);
%                     y=setdiff([stimrefsb(o,:) stimrefsb(n1,:) stimrefsb(n2,:)],tmp3);
%                     xshuf=length(x)/3;
                    % for 2 class:
                    tmp2=[stimrefsu(o,tmp) stimrefsu(n1,tmp)];
                    tmp3=[stimrefsb(o,tmp) stimrefsb(n1,tmp)];
                    x=setdiff([stimrefsu(o,:) stimrefsu(n1,:)],tmp2);
                    y=setdiff([stimrefsb(o,:) stimrefsb(n1,:)],tmp3);
                    xshuf=length(x)/2;
                    
                    trainu=units_up(:,tmp2);
                    trainb=units_bp(:,tmp3);
                    testu=units_up(:,x);
                    testb=units_bp(:,y);
                    
                    % shuffle train and test trials independently for each unit (row):
                    for k=1:size(trainu,1)
                        tmpu1shuf=randperm(trainsize);
                        tmpb1shuf=randperm(trainsize);
                        tmpu2shuf=randperm(trainsize);
                        tmpb2shuf=randperm(trainsize);
%                         tmpu3shuf=randperm(trainsize);    % only need these for 3 class
%                         tmpb3shuf=randperm(trainsize);
%                         trainu_shuf(k,:)=units_up(k,[stimrefsu(o,tmp(tmpu1shuf)) stimrefsu(n1,tmp(tmpu2shuf)) stimrefsu(n2,tmp(tmpu3shuf))]);
%                         trainb_shuf(k,:)=units_bp(k,[stimrefsb(o,tmp(tmpb1shuf)) stimrefsb(n1,tmp(tmpb2shuf)) stimrefsb(n2,tmp(tmpb3shuf))]);
                        trainu_shuf(k,:)=units_up(k,[stimrefsu(o,tmp(tmpu1shuf)) stimrefsu(n1,tmp(tmpu2shuf))]);
                        trainb_shuf(k,:)=units_bp(k,[stimrefsb(o,tmp(tmpb1shuf)) stimrefsb(n1,tmp(tmpb2shuf))]);
                        
                        tshufu1=randperm(xshuf);
                        tshufb1=randperm(xshuf);
                        tshufu2=randperm(xshuf)+xshuf;
                        tshufb2=randperm(xshuf)+xshuf;
%                         tshufu3=randperm(xshuf)+xshuf*2;  % only need these for 3 class
%                         tshufb3=randperm(xshuf)+xshuf*2;
%                         testu_shuf(k,:)=testu(k,[tshufu1 tshufu2 tshufu3]);
%                         testb_shuf(k,:)=testb(k,[tshufb1 tshufb2 tshufb3]);
                        testu_shuf(k,:)=testu(k,[tshufu1 tshufu2]);
                        testb_shuf(k,:)=testb(k,[tshufb1 tshufb2]);
                    end

                    % run classifier:
    %                 [tmp_lda_u,tmp_err_u]=classify(units_u(:,testu)',units_u(:,trainu)',base2(trainu)');
    %                 [tmp_lda_b,tmp_err_b]=classify(units_b(:,testb)',units_b(:,trainb)',bias2(trainb)');
                    [tmp_lda_up,tmp_err_up,tmp_prob_u]=classify(testu',trainu',base_keep(tmp2)');
                    [tmp_lda_bp,tmp_err_bp,tmp_prob_b]=classify(testb',trainb',bias_keep(tmp3)');
                    % percent correct on test trials
                    tmp5=find(tmp_lda_up'==base_keep(x));
                    tmp6=find(tmp_lda_bp'==bias_keep(y));
                    % shuffled data:
                    [tmp_lda_up_shuf,tmp_err_up_shuf,tmp_prob_u_shuf]=classify(testu_shuf',trainu_shuf',base_keep(tmp2)');
                    [tmp_lda_bp_shuf,tmp_err_bp_shuf,tmp_prob_b_shuf]=classify(testb_shuf',trainb_shuf',bias_keep(tmp3)');
                    % percent correct on test trials
                    tmp7=find(tmp_lda_up_shuf'==base_keep(x));
                    tmp8=find(tmp_lda_bp_shuf'==bias_keep(y));

                    % store performance output
    %                 pop_lda_u(idx,id,:)=[1-tmp_err_u length(tmp5)/length(tmp_lda_u)];
    %                 pop_lda_b(idx,id,:)=[1-tmp_err_b length(tmp6)/length(tmp_lda_b)];
                    pop_lda_up_train(o,idx,id1)=1-tmp_err_up;
                    pop_lda_bp_train(o,idx,id1)=1-tmp_err_bp;
                    pop_lda_up_test(o,idx,id1)=length(tmp5)/length(tmp_lda_up);
                    pop_lda_bp_test(o,idx,id1)=length(tmp6)/length(tmp_lda_bp);
                    pop_lda_up_train_shuf(o,idx,id1)=1-tmp_err_up_shuf;
                    pop_lda_bp_train_shuf(o,idx,id1)=1-tmp_err_bp_shuf;
                    pop_lda_up_test_shuf(o,idx,id1)=length(tmp7)/length(tmp_lda_up_shuf);
                    pop_lda_bp_test_shuf(o,idx,id1)=length(tmp8)/length(tmp_lda_bp_shuf);
                    % store classifications:
                    class_up_test(o,idx,id1,:)=tmp_lda_up';
                    class_bp_test(o,idx,id1,:)=tmp_lda_bp';
                    class_up_test_shuf(o,idx,id1,:)=tmp_lda_up_shuf';
                    class_bp_test_shuf(o,idx,id1,:)=tmp_lda_bp_shuf';
                    % store posterior probabilities
                    prob_up_test(o,idx,id1,:)=tmp_prob_u(:,1)'; % prob in class 1
                    prob_bp_test(o,idx,id1,:)=tmp_prob_b(:,1)';
                    prob_up_test_shuf(o,idx,id1,:)=tmp_prob_u_shuf(:,1)';
                    prob_bp_test_shuf(o,idx,id1,:)=tmp_prob_b_shuf(:,1)';
                    % structure: ori center, population size, [sampled neurons x trials: train/test]
                    id1=id1+1;
                    clear tmp* trainu* trainb* testu* testb*
                end
            end
            idx=idx+1;
        end
        disp('ori comparison done')
    end
    clear o e b j k tmp* id1 idx
    toc
    
    pop_lda_up_train_mean=squeeze(nanmean(pop_lda_up_train(:,:,:),3));
    pop_lda_bp_train_mean=squeeze(nanmean(pop_lda_bp_train(:,:,:),3));
    pop_lda_up_test_mean=squeeze(nanmean(pop_lda_up_test(:,:,:),3));
    pop_lda_bp_test_mean=squeeze(nanmean(pop_lda_bp_test(:,:,:),3));
    pop_lda_up_train_mean_shuf=squeeze(nanmean(pop_lda_up_train_shuf(:,:,:),3));
    pop_lda_bp_train_mean_shuf=squeeze(nanmean(pop_lda_bp_train_shuf(:,:,:),3));
    pop_lda_up_test_mean_shuf=squeeze(nanmean(pop_lda_up_test_shuf(:,:,:),3));
    pop_lda_bp_test_mean_shuf=squeeze(nanmean(pop_lda_bp_test_shuf(:,:,:),3));
    
    
    % fit performance data and find 75% performance mark
    % (center ori, pop size, repeats (10x10=100), performance: train & test)
    for e=1:length(oris)
        % Xdata = # of steps (pop size); Ydata = % correct at that size
        % only doing this for matched distribution
        % previously used SigmoidFit - now using weibel function
        [par_up_train(e,:), gof_up_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean(e,:),[0.1 0.1 0.1],[]);
        [par_bp_train(e,:), gof_bp_train(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean(e,:),[0.1 0.1 0.1],[]);
        [par_up_test(e,:), gof_up_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean(e,:),[0.1 0.1 0.1],[]);
        [par_bp_test(e,:), gof_bp_test(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean(e,:),[0.1 0.1 0.1],[]);
        
        [par_up_train_shuf(e,:), gof_up_train_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_train_mean_shuf(e,:),[0.1 0.1 0.1],[]);
        [par_bp_train_shuf(e,:), gof_bp_train_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_train_mean_shuf(e,:),[0.1 0.1 0.1],[]);
        [par_up_test_shuf(e,:), gof_up_test_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_up_test_mean_shuf(e,:),[0.1 0.1 0.1],[]);
        [par_bp_test_shuf(e,:), gof_bp_test_shuf(e,:)]=lda_pnf_fit([10 0.6 0.5],pop_lda_bp_test_mean_shuf(e,:),[0.1 0.1 0.1],[]);
        
        fit_up_train(e,:)=1-par_up_train(e,3)*exp(-(steps_fine/par_up_train(e,1)).^par_up_train(e,2));
        fit_bp_train(e,:)=1-par_bp_train(e,3)*exp(-(steps_fine/par_bp_train(e,1)).^par_bp_train(e,2));
        fit_up_test(e,:)=1-par_up_test(e,3)*exp(-(steps_fine/par_up_test(e,1)).^par_up_test(e,2));
        fit_bp_test(e,:)=1-par_bp_test(e,3)*exp(-(steps_fine/par_bp_test(e,1)).^par_bp_test(e,2));
        
        fit_up_train_shuf(e,:)=1-par_up_train_shuf(e,3)*exp(-(steps_fine/par_up_train_shuf(e,1)).^par_up_train_shuf(e,2));
        fit_bp_train_shuf(e,:)=1-par_bp_train_shuf(e,3)*exp(-(steps_fine/par_bp_train_shuf(e,1)).^par_bp_train_shuf(e,2));
        fit_up_test_shuf(e,:)=1-par_up_test_shuf(e,3)*exp(-(steps_fine/par_up_test_shuf(e,1)).^par_up_test_shuf(e,2));
        fit_bp_test_shuf(e,:)=1-par_bp_test_shuf(e,3)*exp(-(steps_fine/par_bp_test_shuf(e,1)).^par_bp_test_shuf(e,2));
        
        % don't need this threshold info anymore:
%         if max(fit_up_train(e,:))<0.75 || max(fit_up_test(e,:))<0.75
%             thresh_u_train(e)=find(fit_up_train(e,:)>=.75*max(fit_up_train(e,:)),1,'first');
%             thresh_u_test(e)=find(fit_up_test(e,:)>=.75*max(fit_up_test(e,:)),1,'first');
%         else
%             thresh_u_train(e)=find(fit_up_train(e,:)>=.75,1,'first');
%             thresh_u_test(e)=find(fit_up_test(e,:)>=.75,1,'first');
%         end
%         if max(fit_bp_train(e,:))<0.75 || max(fit_bp_test(e,:))<0.75
%             thresh_b_train(e)=find(fit_bp_train(e,:)>=.75*max(fit_bp_train(e,:)),1,'first');
%             thresh_b_test(e)=find(fit_bp_test(e,:)>=.75*max(fit_bp_test(e,:)),1,'first');
%         else
%             thresh_b_train(e)=find(fit_bp_train(e,:)>=.75,1,'first');
%             thresh_b_test(e)=find(fit_bp_test(e,:)>=.75,1,'first');
%         end
%         if max(fit_up_train_shuf(e,:))<0.75 || max(fit_up_test_shuf(e,:))<0.75
%             thresh_u_train_shuf(e)=find(fit_up_train_shuf(e,:)>=.75*max(fit_up_train_shuf(e,:)),1,'first');
%             thresh_u_test_shuf(e)=find(fit_up_test_shuf(e,:)>=.75*max(fit_up_test_shuf(e,:)),1,'first');
%         else
%             thresh_u_train_shuf(e)=find(fit_up_train_shuf(e,:)>=.75,1,'first');
%             thresh_u_test_shuf(e)=find(fit_up_test_shuf(e,:)>=.75,1,'first');
%         end
%         if max(fit_bp_train_shuf(e,:))<0.75 || max(fit_bp_test_shuf(e,:))<0.75
%             thresh_b_train_shuf(e)=find(fit_bp_train_shuf(e,:)>=.75*max(fit_bp_train_shuf(e,:)),1,'first');
%             thresh_b_test_shuf(e)=find(fit_bp_test_shuf(e,:)>=.75*max(fit_bp_test_shuf(e,:)),1,'first');
%         else
%             thresh_b_train_shuf(e)=find(fit_bp_train_shuf(e,:)>=.75,1,'first');
%             thresh_b_test_shuf(e)=find(fit_bp_test_shuf(e,:)>=.75,1,'first');
%         end
        
        % previous code
%         if max(fit_up_train{e}(steps_fine)) <0.75 || max(fit_up_test{e}(steps_fine)) <0.75 
%             thresh_u_train(e)=steps_fine(find(fit_up_train{e}(steps_fine)>=.75*max(fit_up_train{e}(steps_fine)),1,'first'));
%             thresh_u_test(e)=steps_fine(find(fit_up_test{e}(steps_fine)>=.75*max(fit_up_test{e}(steps_fine)),1,'first'));
%         else
%             thresh_u_train(e)=steps_fine(find(fit_up_train{e}(steps_fine)>=.75,1,'first'));
%             thresh_u_test(e)=steps_fine(find(fit_up_test{e}(steps_fine)>=.75,1,'first'));
%         end
%         if max(fit_bp_train{e}(steps_fine)) <0.75 || max(fit_bp_test{e}(steps_fine)) <0.75
%             thresh_b_train(e)=steps_fine(find(fit_bp_train{e}(steps_fine)>=.75*max(fit_bp_train{e}(steps_fine)),1,'first'));
%             thresh_b_test(e)=steps_fine(find(fit_bp_test{e}(steps_fine)>=.75*max(fit_bp_test{e}(steps_fine)),1,'first'));
%         else
%             thresh_b_train(e)=steps_fine(find(fit_bp_train{e}(steps_fine)>=.75,1,'first'));
%             thresh_b_test(e)=steps_fine(find(fit_bp_test{e}(steps_fine)>=.75,1,'first'));
%         end
    end
    
    % plot LDA performance, separated by center ori (3-class)
    figure
    ori_labels=0:20:160;
    supertitle('Awake, LDA performance vs population size')
    for e=1:length(ori_labels)
        subplot(3,3,e); hold on
        title(sprintf('ori %g',ori_labels(e)))
        hold on
%         errorline(1+0.4:1:size(pop_lda_up_train,2)+0.4,pop_lda_up_train_mean(e,:),nanstd(squeeze(pop_lda_up_train(e,:,:))'),'k--')
%         errorline(1+0.4:1:size(pop_lda_bp_train,2)+0.4,pop_lda_bp_train_mean(e,:),nanstd(squeeze(pop_lda_bp_train(e,:,:))'),'r--')
%         errorline(1:size(pop_lda_up_test,2),pop_lda_up_test_mean(e,:),nanstd(squeeze(pop_lda_up_train(e,:,:))'),'k.')
%         errorline(1:size(pop_lda_bp_test,2),pop_lda_bp_test_mean(e,:),nanstd(squeeze(pop_lda_bp_train(e,:,:))'),'r.')
        
%         errorline(1+0.4:1:size(pop_lda_up_train_shuf,2)+0.4,pop_lda_up_train_mean_shuf(e,:),nanstd(squeeze(pop_lda_up_train_shuf(e,:,:))'),'b--')
%         errorline(1+0.4:1:size(pop_lda_bp_train_shuf,2)+0.4,pop_lda_bp_train_mean_shuf(e,:),nanstd(squeeze(pop_lda_bp_train_shuf(e,:,:))'),'m--')
%         errorline(1:size(pop_lda_up_test_shuf,2),pop_lda_up_test_mean_shuf(e,:),nanstd(squeeze(pop_lda_up_train_shuf(e,:,:))'),'b.')
%         errorline(1:size(pop_lda_bp_test_shuf,2),pop_lda_bp_test_mean_shuf(e,:),nanstd(squeeze(pop_lda_bp_train_shuf(e,:,:))'),'m.')
        
    %     errorline(2:size(resp_raw_base,1),1-mean(pop_err_u,1),std(pop_err_u,1),'k')
    %     errorline(2:size(resp_raw_base,1),1-mean(pop_err_b,1),std(pop_err_b,1),'r')
    
        plot(steps_fine,fit_up_test(e,:),'k')
        plot(steps_fine,fit_bp_test(e,:),'r')
        plot(steps_fine,fit_up_test_shuf(e,:),'b')
        plot(steps_fine,fit_bp_test_shuf(e,:),'m')
        
        axis([0 100 0.3 1])
        plot(0.33*ones(100,1),':k')
        axis square; box off
%         if e==1
%             legend({'unif training','bias training','unif test','bias test'},'Location','southeast')
%         end
        xlabel('# of units')
        ylabel('% correct')
%     set(gca,'XTick',...
    end
    
%     figure
%     oris=0:20:160;
%     supertitle('awake pnf fits')
%     for e=1:length(oris)
%         subplot(3,3,e); hold on
%         title(sprintf('ori %g',e))
%         plot(steps_fine,fit_up_train(e,:),'k')
%         plot(steps_fine,fit_up_test(e,:),'k--')
%         plot(steps_fine,fit_bp_train(e,:),'r')
%         plot(steps_fine,fit_bp_test(e,:),'r--')
%         if e==1
%             legend({'unif train','unif test','bias train','bias test'},'Location','southeast')
%             ylabel('% correct')
%             xlabel('# of units')
%         end
%         axis square
%         ylim([0.3 1])
%     end
    
%     figure
%     if shiftori==1
%         plot(circshift(thresh_b_train./thresh_u_train,0),'k')
%         plot(circshift(thresh_b_test./thresh_u_test,0),'r')
%     else
%         plot(circshift(thresh_b_train./thresh_u_train,4),'k')
%         plot(circshift(thresh_b_test./thresh_u_test,4),'r')
%     end
%     axis square; box off
%     xlabel('center ori of 3-class')
%     ylabel('relative change in 75% threshold')
%     title('LDA Neurometric Threshold')
    
    disp('file done')
    clearvars -except thresh* pop* num_units ori_base oripref resp* stim* steps* shiftori fit* gof* filename savename a par* class* trainsize testsize prob* t osi_tmp
    
% new way ^^
% % % % % % % % % % % % % % % % % % % % % % % % % % %     
% old way:
%     pop_err_u=nan*zeros(20,size(rb,1)-1);
%     pop_err_b=nan*zeros(20,size(rb,1)-1);
%     pop_err_up=nan*zeros(20,size(rb,1)-1);
%     pop_err_bp=nan*zeros(20,size(rb,1)-1);
%     pop_class_u=nan*zeros(20,size(rb,1)-1,round(0.1*length(oris_u)));%length(oris_u(end-r1+1:end)));
%     pop_class_b=nan*zeros(20,size(rb,1)-1,round(0.1*length(oris_b)));
%     pop_class_up=nan*zeros(20,size(rb,1)-1,round(0.1*length(keep)));%length(temp1(end-r3+1:end)));
%     pop_class_bp=nan*zeros(20,size(rb,1)-1,round(0.1*length(keep2)));
%     for b=1:20
%         for e=2:num_units
%             j=randperm(num_units,e);
%             units_u=ru(j,:);
%             units_b=rb(j,:);
%             units_up=rusub(j,:);
%             units_bp=rbsub(j,:);
%             
%             [tmp_lda_u,tmp_err_u]=classify(units_u(:,testu)',units_u(:,trainu)',oris_u(trainu)');
%             [tmp_lda_b,tmp_err_b]=classify(units_b(:,testb)',units_b(:,trainb)',oris_b(trainb)');
%             [tmp_lda_up,tmp_err_up]=classify(units_up(:,end-tenpu+1:end)',units_up(:,1:end-tenpu)',temp1(1:end-tenpu)');
%             [tmp_lda_bp,tmp_err_bp]=classify(units_bp(:,end-tenpu+1:end)',units_bp(:,1:end-tenpu)',temp2(1:end-tenpu)');
%             pop_err_u(b,e-1)=tmp_err_u;
%             pop_err_b(b,e-1)=tmp_err_b;
%             pop_err_up(b,e-1)=tmp_err_up;
%             pop_err_bp(b,e-1)=tmp_err_bp;
%             pop_class_u(b,e-1,:)=tmp_lda_u;
%             pop_class_b(b,e-1,:)=tmp_lda_b;
%             pop_class_up(b,e-1,:)=tmp_lda_up;
%             pop_class_bp(b,e-1,:)=tmp_lda_bp;
%         end
%     end
%     figure
%     supertitle(sprintf('Awake %g, LDA performance vs population size (all trials)',a))
%     subplot(221); hold on
%     title('Training')
%     errorline(2:num_units,1-mean(pop_err_u,1),std(pop_err_u,1),'k')
%     errorline(2:num_units,1-mean(pop_err_b,1),std(pop_err_b,1),'r')
%     ylim([0 1])
%     axis square; box off
%     legend({'unif','bias'})
%     xlabel('# units')
%     ylabel('% correct')
%     subplot(223); hold on
%     errorline(2:num_units,1-mean(pop_err_up,1),std(pop_err_up,1),'k--')
%     errorline(2:num_units,1-mean(pop_err_bp,1),std(pop_err_bp,1),'r--')
%     ylim([0 1])
%     title('Training predicted')
%     axis square; box off
%     xlabel('# units')
%     ylabel('% correct')
%     subplot(222); hold on
%     test_rate_u_pop=nan*zeros(size(pop_err_u,1),size(pop_err_u,2));
%     test_rate_b_pop=nan*zeros(size(pop_err_u,1),size(pop_err_u,2));
%     test_rate_bp_pop=nan*zeros(size(pop_err_u,1),size(pop_err_u,2));
%     test_rate_up_pop=nan*zeros(size(pop_err_u,1),size(pop_err_u,2));
%     title('Test')
%     for b = 1:size(pop_err_u,1)
%         for e = 1:size(pop_err_u,2)
%             tmp=find(squeeze(pop_class_u(b,e,:))'==oris_u(testu));
%             tmp2=find(squeeze(pop_class_b(b,e,:))'==oris_b(testb));
%             tmp3=find(squeeze(pop_class_up(b,e,:))'==temp1(end-tenpu+1:end));
%             tmp4=find(squeeze(pop_class_bp(b,e,:))'==temp2(end-tenpu+1:end));
%             
%             test_rate_u_pop(b,e)=length(tmp)/length(oris_u(testu));
%             test_rate_b_pop(b,e)=length(tmp2)/length(oris_b(testb));
%             test_rate_up_pop(b,e)=length(tmp3)/length(temp1(end-tenpu+1:end));
%             test_rate_bp_pop(b,e)=length(tmp4)/length(temp2(end-tenpu+1:end));
%         end
%     end
%     errorline(2:size(rb,1),mean(test_rate_u_pop,1),std(test_rate_u_pop,1),'k')
%     errorline(2:size(rb,1),mean(test_rate_b_pop,1),std(test_rate_b_pop,1),'r')
%     ylim([0 1])
%     axis square; box off
%     xlabel('# of units')
%     ylabel('% correct')
%     subplot(224); hold on
%     title('Test predicted')
%     errorline(2:size(rb,1),mean(test_rate_up_pop,1),std(test_rate_up_pop,1),'k--')
%     errorline(2:size(rb,1),mean(test_rate_bp_pop,1),std(test_rate_bp_pop,1),'r--')
%     ylim([0 1])
%     axis square; box off
%     disp('pop size done')
%     clear tmp* units*
    
    %% % % % % % % % % % % test different # of trials
%     trial_err_u=nan*zeros(20,6);
%     trial_err_b=nan*zeros(20,6);
%     trial_class_u=cell(20,6);
%     trial_class_b=cell(20,6);
%     for b=1:20
%         id=1;
%         for e=[500 750 1000 1250 1500 1750]
%             j=randperm(size(ru,1),20);   % 20 random neurons
%             k=randperm(size(ru,2),e);   % e random trials from uni
%             k2=randperm(size(rb,2),e);   % e random trials from bias
%             units_u=ru(j,k);
%             units_b=rb(j,k2);
%             %     units_up=rusub(j,k);
%             %     units_bp=rbsub(j,k);
%             tenp=round(0.1*e);
%             uo=oris_u(k);
%             bo=oris_b(k2);
%            
%             [tmp_lda_u,tmp_err_u]=classify(units_u(:,end-tenp+1:end)',units_u(:,1:end-tenp)',uo(1:end-tenp)');
%             [tmp_lda_b,tmp_err_b]=classify(units_b(:,end-tenp+1:end)',units_b(:,1:end-tenp)',bo(1:end-tenp)');
%     %     [tmp_lda_up,tmp_err_up]=classify(units_up(:,end-tenp+1:end)',units_up(:,1:end-tenp)',temp1(k2)');
%     %     [tmp_lda_bp,tmp_err_bp]=classify(units_bp(:,end-tenp+1:end)',units_bp(:,1:end-tenp)',temp2(k2)');
%             trial_err_u(b,id)=tmp_err_u;
%             trial_err_b(b,id)=tmp_err_b;
%             %     trial_err_up(a,id)=tmp_err_up;
%             %     trial_err_bp(a,id)=tmp_err_bp;
%             trial_class_u{b,id}=[tmp_lda_u uo(end-tenp+1:end)'];
%             trial_class_b{b,id}=[tmp_lda_b bo(end-tenp+1:end)'];
%             %     trial_class_up{a,id}=[tmp_lda_up uo(end-tenpercent+1:end)];
%             %     trial_class_bp{a,id}=[tmp_lda_bp bo(end-tenpercent+1:end)];
%             id=id+1;
%         end
%     end
%     figure
%     supertitle(sprintf('Awake %g, LDA performance vs trial size (20 neurons)',a))
%     subplot(121); hold on
%     title('Training')
%     errorline(1-nanmean(trial_err_u,1),nanstd(trial_err_u,1),'k')
%     errorline(1-nanmean(trial_err_b,1),nanstd(trial_err_b,1),'r')
%     % errorline(1-mean(trial_err_up,1),std(trial_err_up,1),'k--')
%     % errorline(1-mean(trial_err_bp,1),std(trial_err_bp,1),'r--')
%     ylim([0 1])
%     set(gca,'XTick',1:6,'XTickLabel',{'500','750','1000','1250','1500','1750'})
%     axis square; box off
%     legend({'unif','bias'})%,'u pred','b pred'})
%     xlabel('# trials included')
%     ylabel('% correct')
%     
%     subplot(122); hold on
%     title('Test')
%     for b = 1:size(trial_class_u,1)
%         for e = 1:size(trial_class_u,2)
%             temp=squeeze(trial_class_u{b,e});
%             tmp=find(temp(:,1)==temp(:,2));
%             temp=squeeze(trial_class_b{b,e});
%             tmp2=find(temp(:,1)==temp(:,2));
%             %         temp=squeeze(trial_class_up{a,e});
%             %         tmp3=find(temp(:,1)==temp(:,2));
%             %         temp=squeeze(trial_class_bp{a,e});
%             %         tmp4=find(temp(:,1)==temp(:,2));
%             
%             test_rate_u_trl(b,e)=length(tmp)/length(temp);
%             test_rate_b_trl(b,e)=length(tmp2)/length(temp);
%             %         test_rate_up_trl(a,e)=length(tmp3)/length(temp);
%             %         test_rate_bp_trl(a,e)=length(tmp4)/length(temp);
%         end
%     end
%     errorline(mean(test_rate_u_trl,1),std(test_rate_u_trl,1),'k')
%     errorline(mean(test_rate_b_trl,1),std(test_rate_b_trl,1),'r')
%     % errorline(mean(test_rate_up_trl,1),std(test_rate_up_trl,1),'k--')
%     % errorline(mean(test_rate_bp_trl,1),std(test_rate_bp_trl,1),'r--')
%     ylim([0 1])
%     set(gca,'XTick',1:6,'XTickLabel',{'500','750','1000','1250','1500','1750'})
%     axis square; box off
%     legend({'unif','bias'})%,'u pred','b pred'})
%     xlabel('# trials included')
%     ylabel('% correct')
%     clear tmp* units* ans id x y uo bo temp* a b e i j k n1 n2

%     savename=sprintf('%s_lda_pnf_2',filename);
    save(savename)
    end
end
stop
% see awake_combined for LDA data compiled


%% fixing things
% new fit function
% for n=1:33
%     clearvars -except n
%     if n==1
%         load('129r001p173_lda_pnf');
%         name='129r001p173_lda_pnf';
%     elseif n==2
%         load('130l001p169_lda_pnf');
%         name='130l001p169_lda_pnf';
%     elseif n==3
%         load('140l001p107_lda_pnf');
%         name='140l001p107_lda_pnf';
%     elseif n==4
%         load('140l001p122_lda_pnf');
%         name='140l001p122_lda_pnf';
%     elseif n==5
%         load('140r001p105_lda_pnf');
%         name='140r001p105_lda_pnf';
%     elseif n==6
%         load('140r001p122_lda_pnf');
%         name='140r001p122_lda_pnf';
%     elseif n==7
%         load('130l001p170_lda_pnf');
%         name='130l001p170_lda_pnf';
%     elseif n==8
%         load('140l001p108_lda_pnf');
%         name='140l001p108_lda_pnf';
%     elseif n==9
%         load('140l001p110_lda_pnf');
%         name='140l001p110_lda_pnf';
%     elseif n==10
%         load('140r001p107_lda_pnf');
%         name='140r001p107_lda_pnf';
%     elseif n==11
%         load('140r001p109_lda_pnf');
%         name='140r001p109_lda_pnf';
%     elseif n==12
%         load('lowcon114_lda_pnf');
%         name='lowcon114_lda_pnf';
%     elseif n==13
%         load('lowcon115_lda_pnf');
%         name='lowcon115_lda_pnf';
%     elseif n==14
%         load('lowcon116_lda_pnf');
%         name='lowcon116_lda_pnf';
%     elseif n==15
%         load('lowcon117_lda_pnf');
%         name='lowcon117_lda_pnf';
%     elseif n==16
%         load('140l113_awaketime_lda_pnf');
%         name='140l113_awaketime_lda_pnf';
%     elseif n==17
%         load('140r113_awaketime_lda_pnf');
%         name='140r113_awaketime_lda_pnf';
%     
%     elseif n==18 % start of experiment 141 files
%         load('141r001p006_awaketime_lda_pnf')
%         name='141r001p006_awaketime_lda_pnf';
%     elseif n==19
%         load('141r001p007_awaketime6_lda_pnf')
%         name='141r001p007_awaketime6_lda_pnf';
%     elseif n==20
%         load('141r001p009_awaketime_fine_lda_pnf')
%         name='141r001p009_awaketime_fine_lda_pnf';
%     elseif n==21 % rotated AT 4:1 (80°)
%         load('141r001p024_awaketime_lda_pnf')
%         name='141r001p024_awaketime_lda_pnf';
%     elseif n==22 % rotated AT 6:1 (80°)
%         load('141r001p025_awaketime6_lda_pnf')
%         name='141r001p025_awaketime6_lda_pnf';
%     elseif n==23 % rotated AT fineori (90°??)
%         load('141r001p027_awaketime_fine_lda_pnf')
%         name='141r001p027_awaketime_fine_lda_pnf';
%     elseif n==24 % rotated fineori (40°)
%         load('141r001p038_awaketime_fine_lda_pnf')
%         name='141r001p038_awaketime_fine_lda_pnf';
%     elseif n==25 % rotated 6:1 (120°)
%         load('141r001p039_awaketime6_lda_pnf')
%         name='141r001p039_awaketime6_lda_pnf';
%     elseif n==26 % rotated awaketime 4:1 (120°)
%         load('141r001p041_awaketime_lda_pnf')
%         name='141r001p041_awaketime_lda_pnf';
%     elseif n==27
%         load('141r001p114_lda_pnf')
%         name='141r001p114_lda_pnf';
%         
%     elseif n==28
%         load('142l001p002_awaketime_lda_pnf')
%         name='142l001p002_awaketime_lda_pnf';
%     elseif n==29
%         load('142l001p004_awaketime_fine_lda_pnf')
%         name='142l001p004_awaketime_fine_lda_pnf';
%     elseif n==30
%         load('142l001p006_awaketime6_lda_pnf')
%         name='142l001p006_awaketime6_lda_pnf';
%     elseif n==31
%         load('142l001p007_awaketime_lda_pnf')
%         name='142l001p007_awaketime_lda_pnf';
%     elseif n==32
%         load('142l001p009_awaketime_fine_lda_pnf')
%         name='142l001p009_awaketime_fine_lda_pnf';
%     elseif n==33
%         load('142l001p010_awaketime6_lda_pnf')
%         name='142l001p010_awaketime6_lda_pnf';
%     end

for a=15:27
    clearvars -except a oris
    if a==1
        load('cadetv1p194_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
    elseif a==15
        load('cadetv1p366_lda_pnf')
    elseif a==16
        load('cadetv1p371_lda_pnf')
    elseif a==17
        load('cadetv1p384_lda_pnf');
    elseif a==18
        load('cadetv1p385_lda_pnf')
    elseif a==19
        load('cadetv1p392_lda_pnf')
    elseif a==20
        load('cadetv1p403_lda_pnf')
    elseif a==21
        load('cadetv1p419_lda_pnf')
    elseif a==22
        load('cadetv1p432_lda_pnf')
    elseif a==23
        load('cadetv1p437_lda_pnf')
    elseif a==24
        load('cadetv1p438_lda_pnf')
    elseif a==25
        load('cadetv1p460_lda_pnf')
    elseif a==26
        load('cadetv1p467_lda_pnf')
    elseif a==27
        load('cadetv1p468_lda_pnf')
    elseif a==28
        % before I made changes:
        load('cadetv1p422_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
        load('cadetv1p422_tuning','ori*','resp*','spont','tune*');
    end
    clear fit* gof*
%     if n==32 || n==29 || n==20 || n==23 || n==24
%         ori_base=0:10:170;
%     else
        ori_base=0:20:160;
%     end
    for e=1:length(ori_base)
        % Xdata = # of steps (pop size); Ydata = % correct at that size
        % only doing this for matched distribution
%         id=1;
%         for m=[10 15 20 25]   % DON'T NEED TO DO THIS - ALL CONVERGE TO SAME PLACE
%             for n=[0.5 0.8 1.1 1.4]
                [par_up_train{e}, gof_up_train{e}]=lda_pnf_fit([20 0.8 0.5],pop_lda_up_train_mean(e,:),[],[]);
                [par_bp_train{e}, gof_bp_train{e}]=lda_pnf_fit([20 0.8 0.5],pop_lda_bp_train_mean(e,:),[],[]);
                [par_up_test{e}, gof_up_test{e}]=lda_pnf_fit([20 0.8 0.5],pop_lda_up_test_mean(e,:),[],[]);
                [par_bp_test{e}, gof_bp_test{e}]=lda_pnf_fit([20 0.8 0.5],pop_lda_bp_test_mean(e,:),[],[]);
%                 id=id+1;
%             end
%         end

        fit_up_train(e,:)=1-par_up_train{e}(3)*exp(-(steps_fine/par_up_train{e}(1)).^par_up_train{e}(2));
        fit_up_test(e,:)=1-par_up_test{e}(3)*exp(-(steps_fine/par_up_test{e}(1)).^par_up_test{e}(2));
        fit_bp_train(e,:)=1-par_bp_train{e}(3)*exp(-(steps_fine/par_bp_train{e}(1)).^par_bp_train{e}(2));
        fit_bp_test(e,:)=1-par_bp_test{e}(3)*exp(-(steps_fine/par_bp_test{e}(1)).^par_bp_test{e}(2));
%         fit_up_train(e,:)=1-0.5*exp(-(steps_fine/par_up_train{e}(1)).^par_up_train{e}(2));
%         fit_up_test(e,:)=1-0.5*exp(-(steps_fine/par_up_test{e}(1)).^par_up_test{e}(2));
%         fit_bp_train(e,:)=1-0.5*exp(-(steps_fine/par_bp_train{e}(1)).^par_bp_train{e}(2));
%         fit_bp_test(e,:)=1-0.5*exp(-(steps_fine/par_bp_test{e}(1)).^par_bp_test{e}(2));
        if max(fit_up_train(e,:))<0.75 || max(fit_up_test(e,:))<0.75
            thresh_u_train(e)=find(fit_up_train(e,:)>=.75*max(fit_up_train(e,:)),1,'first');
            thresh_u_test(e)=find(fit_up_test(e,:)>=.75*max(fit_up_test(e,:)),1,'first');
        else
            thresh_u_train(e)=find(fit_up_train(e,:)>=.75,1,'first');
            thresh_u_test(e)=find(fit_up_test(e,:)>=.75,1,'first');
        end
        if max(fit_bp_train(e,:))<0.75 || max(fit_bp_test(e,:))<0.75
            thresh_b_train(e)=find(fit_bp_train(e,:)>=.75*max(fit_bp_train(e,:)),1,'first');
            thresh_b_test(e)=find(fit_bp_test(e,:)>=.75*max(fit_bp_test(e,:)),1,'first');
        else
            thresh_b_train(e)=find(fit_bp_train(e,:)>=.75,1,'first');
            thresh_b_test(e)=find(fit_bp_test(e,:)>=.75,1,'first');
        end

    end
    
    figure
    oris=0:20:160;
    supertitle('awake pnf fits')
    for e=1:length(oris)
        subplot(3,3,e); hold on
        title(sprintf('ori %g',e))
        plot(fit_up_train(e,:),'k')
        plot(fit_up_test(e,:),'k--')
        plot(fit_bp_train(e,:),'r')
        plot(fit_bp_test(e,:),'r--')
        plot(steps,pop_lda_up_train_mean(e,:),'.k')
        plot(steps,pop_lda_up_test_mean(e,:),'xk')
        plot(steps,pop_lda_bp_train_mean(e,:),'r.')
        plot(steps,pop_lda_bp_test_mean(e,:),'rx')
        if e==1
            legend({'unif train fit','unif test fit','bias train fit','bias test fit'...
                ,'unif train data','unif test data','bias train data','bias test data'},'Location','southeast')
            ylabel('% correct')
            xlabel('# of units')
        end
        axis square
        ylim([0.3 1])
    end
    disp('file done')
%     stop
    clearvars -except thresh* pop* num_units ori_base oripref resp* stim* steps* shiftori fit* gof* filename a name n
    
    savename=sprintf('%s_lda_pnf',filename);
    save(savename)
    
%     clear ans id tmp* x y uo bo units_* temp* ten* a b e i j k n1 n2
%     save(name);
end