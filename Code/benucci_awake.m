%% Benucci, biased orientation frequency adaptation
clear

%% experiment files
% Original: cadet 2:1 - p345-349; 6:1 - p350-p355 <- bias in original are missing 160
% run details:
% (2,-2) size=2.5; ori=0:20:160; sf=2; tf=3, 100ms blank before 4*250ms gratings shown

% cadetv1p194, cadetv1p195, cadetv1p245, cadetv1p246 <- these have a blank
% in the uniform but not in the bias and shouldn't be used

% cadet 2:1 - p366,371,392,419,422,438,467      7 total
% cadet 6:1 - 384,385,403,432,437,460,468       7 total
% 422 crashes when loading .nev data

% run details:
% (2,-2.5) size=2.5; ori=0:20:160; sf=2; tf=3, 100ms blank before 6*150ms gratings shown

%%
% /Users/amir/....
data_folder=fullfile('E:\Documents\MATLAB\NEV and EXPO files\awake\cadet');
nEl=96;
% DELAY=0.046; % latency after stimulus onset for analysis window (from PSTH)
% elLabel=1:96;

% for a=[194 195 245 246] % pfile numbers original/old/flawed version
% for a = [345:353 355]  % second version: bugged - 2:1 and 6:1 bias versions
for a = [366 371 384 385 392 403 419 432 437 438 460 467 468]  % current versions (14 ttl)
% for a = 422
    clearvars -except data_folder nEl DELAY a elLabel
    filename1='cadetv1p';
    filename=sprintf('%s%g',filename1,a);
    stimType='cadet_benucci';
    try
%         savefile=fullfile(data_folder,sprintf('%s[%s].mat',filename,stimType));
%         load(savefile);
        expo_file=fullfile(data_folder,sprintf('%s[%s2].xml',filename,stimType));
        expo_data=ReadExpoXML(expo_file);
        savefile=fullfile(data_folder,sprintf('%s[%s2].mat',filename,stimType));
        save(savefile,'expo_data');
    catch
        expo_file=fullfile(data_folder,sprintf('%s[%s6].xml',filename,stimType));
        expo_data=ReadExpoXML(expo_file);
        savefile=fullfile(data_folder,sprintf('%s[%s6].mat',filename,stimType));
        save(savefile,'expo_data');
    end
    
    % % % % pull out relevant expo data
%     stim_conditions=expo_data.matrix.Dimensions{1}.Size;
    % view duration is 6 x 0.15s stim for 2:1 and 6:1. original version was 4 x 0.25s
    viewDuration=expo_data.slots.Until(3).DurationValue+expo_data.slots.Until(4).DurationValue...
        +expo_data.slots.Until(5).DurationValue+expo_data.slots.Until(6).DurationValue...
        +expo_data.slots.Until(7).DurationValue+expo_data.slots.Until(8).DurationValue;	
    spont_dur=expo_data.slots.Until(2).DurationValue; % .1s blank before stim for measuring spont firing
    tempi=strcmp(expo_data.blocks.Names,'Reward');
    reward_blockID=expo_data.blocks.IDs(tempi);
    % % % % work out which blocks send pulses to NEV
    tempi=find(strcmp(expo_data.blocks.routinesMap.RoutineNames,'Digital Output'));
    digOutput_routineID=expo_data.blocks.routinesMap.RoutineIDs(tempi);
    
    % % % % find passes that 1) sent a pulse, 2) were a reward, 3) It was a viewing
    nPass=size(expo_data.passes.events,2);  %pre-allocate variable
    passesWithPulse=false(1,nPass);         %pre-allocate variable
    for nn=1:nPass
        if sum(expo_data.passes.events{nn}.RoutineIDs==digOutput_routineID)
            tempi=find(expo_data.passes.events{nn}.RoutineIDs==digOutput_routineID);
            passesWithPulse(nn)=true;
        end
    end
    viewPass=(expo_data.passes.BlockIDs>=5); %expo_data.matrix.MatrixBaseID);
    rewardPass=(expo_data.passes.BlockIDs==reward_blockID);
    % NEV will not get a pulse if fixation broke before 10 ms
    tempi=expo_data.passes.EndTimes-expo_data.passes.StartTimes;
%     passTimes=tempi;        % for later to see time between successful fixations
    viewPass2=zeros(1,length(viewPass));
    for nn=1:length(viewPass)
        if viewPass(nn)==1 && tempi(nn)>=100
            viewPass2(nn)=1;
        else
            viewPass2(nn)=0;
        end
    end
    passesWithPulse=passesWithPulse & tempi>=10;
    rewardPass=rewardPass & tempi>=100;  % keep trials where fixation was maintained for the entire trial
    validViewPass=false(1,length(viewPass));
    for nn=1:length(viewPass)-6 % was 4 for old version. viewDuration was 1
        if viewPass2(nn)==1 && rewardPass(nn+6)==1
            validViewPass(nn)=1;
        end
    end
    % Get # of trials for stimulus shown the most
    tempval=expo_data.passes.BlockIDs(validViewPass);
%     [~,maxTrials]=mode(tempval);
    
    % % % % Summary stats
    nTrials=sum(viewPass);
    nValidTrials=sum(validViewPass);
    nRewards=sum(rewardPass);
    nPulse=sum(passesWithPulse);
    
    %% calculate # of uniform and bias trials
%     if a==194 % for old versions (would need to switch back a bunch of
%                                   this code to make this work again)
%         utrials=416; % # of passes before succ. trials switches to bias
%     elseif a==195
%         utrials=415;
%     elseif a==245
%         utrials=329;
%     elseif a==246
%         utrials=394;
%     end

    % index of first bias stim:
    first_bias=find(expo_data.passes.BlockIDs==6,1,'first');
    % find # of rewards before first_bias
    utrials=sum(rewardPass(1:first_bias));
    btrials=sum(rewardPass(first_bias:end));
    % check
    if utrials+btrials~=nValidTrials
        error('trial count')
    end
    
    %% load orientation data
    oris=0:20:160;
    trial_inds=find(validViewPass);
    oris_temp=nan*zeros(nValidTrials,6);
    for tt=1:nValidTrials
        tempi=trial_inds(tt);
        if tt<=utrials
%             oris_temp(tt,1)=expo_data.passes.events{tempi}.Data{6}(2); % Original version
            oris_temp(tt,1)=expo_data.passes.events{tempi}.Data{5}(2);
            oris_temp(tt,2)=expo_data.passes.events{tempi+1}.Data{5}(2);
            oris_temp(tt,3)=expo_data.passes.events{tempi+2}.Data{5}(2);
            oris_temp(tt,4)=expo_data.passes.events{tempi+3}.Data{5}(2);
            oris_temp(tt,5)=expo_data.passes.events{tempi+4}.Data{5}(2);
            oris_temp(tt,6)=expo_data.passes.events{tempi+5}.Data{5}(2);
        else
            oris_temp(tt,1)=expo_data.passes.events{tempi}.Data{7}(2);
            oris_temp(tt,2)=expo_data.passes.events{tempi+1}.Data{7}(2);
            oris_temp(tt,3)=expo_data.passes.events{tempi+2}.Data{7}(2);
            oris_temp(tt,4)=expo_data.passes.events{tempi+3}.Data{7}(2);
            oris_temp(tt,5)=expo_data.passes.events{tempi+4}.Data{7}(2);
            oris_temp(tt,6)=expo_data.passes.events{tempi+5}.Data{7}(2);
        end
    end
    oris_validTrials=[]; tmp2=[];
    for tt=1:nValidTrials
        oris_validTrials=[oris_validTrials oris_temp(tt,:)];
    end
    oris_u=oris_validTrials(1:utrials*6);
    oris_b=oris_validTrials(utrials*6+1:end);

%     figure; % ori distributions (uniform vs bias)
%     subplot(121); histogram(oris_u,0:20:180)
%     subplot(122); histogram(oris_b,0:20:180)
    
    if length(oris_validTrials(utrials*6+1:end))~=length(oris_b)
        error('trial mismatch')
    end
    
    %% extract nev data
    spikes=nev_reader(fullfile(data_folder,sprintf('%s-s.nev',filename)),0);
    nev_pulse=spikes(spikes(:,1)==2000,3);

    % Trellis software records anytime a digital input changes, so you will
    %                   have double the pulses (for rising and falling edge)
    % switch array
    %   case 'V1'
%           nev_pulse2=nev_pulse(1:2:end);   % Trellis recorded V1 in monyet. Blackrock recorded V1 in Cadet
    % end

    % Check if what you got from expo and nev match up and debug
    if length(nev_pulse)~=nPulse
        tmp_pulse=spikes(spikes(:,1)==2000 & spikes(:,2)==-1,3);
        if length(tmp_pulse)==nPulse
            disp('tmp_pulse is right')
            nev_pulse=tmp_pulse;
        elseif length(tmp_pulse)-1==nPulse
            disp('tmp_pulse minus 1 is right')
            nev_pulse=tmp_pulse(1:end-1);
        else
            nev_pulse=nev_pulse(1:end-1);
            if length(nev_pulse)~=nPulse
                error('Something wrong, expo and nev pulses don''t match up');
            else
                disp('nev_pulse-1 is right')
            end
        end
    end

    % Have a vector the length of passes, and fill in the times for all those pulses
    nev_passTimes=NaN*zeros(1,nPass);
    nev_passTimes(passesWithPulse)=nev_pulse; % corresponding time of each pulse - aligned with expo
    ii=1;
    for nn=1:length(nev_passTimes)
        if validViewPass(nn)==1
            nev_validViewTimes(ii)=nev_passTimes(nn);
            ii=ii+1;
        end
    end
    
    %% pre-process spike data
    % larger first dimension than needed; not 96 units, but simpler this way
    % spike count for all valid fixations
    % spikecounts structure: units x latency shifts x valid trials x spikes(6 stim)
    spikecounts=NaN*zeros(nEl,56,nValidTrials,6); % was 4 for original version   
    tempb=nev_passTimes(expo_data.passes.BlockIDs==2);  % find when pre-stim is on screen; see spont_dur
    spont_tmp=NaN*zeros(nEl,56,2);                          % [spontFR spontSEM]
    % % spike counts for valid trials and spontaneous
    idx=1;  % units
    for el=1:96
        temp=spikes(spikes(:,1)==el,:);
        sort_codes=unique(temp(:,2));
        sort_codes=sort_codes(sort_codes~=0 & sort_codes~=255);
%         sort_codes=0; % for unsorted
        for j=1:length(sort_codes)
            temp_spikes=temp(temp(:,2)==sort_codes(j),3);
            id2=1;
            for DELAY=0.02:0.002:0.13   % latency shifts (Matt Smith 2005)
                for k=1:nValidTrials
                % for original version:
%                 spikecounts(idx,k,1)=sum(temp_spikes>=(nev_validViewTimes(k) + DELAY)...
%                     & temp_spikes<(nev_validViewTimes(k) + 0.25 + DELAY));
%                 spikecounts(idx,k,2)=sum(temp_spikes>=(nev_validViewTimes(k) +0.25 + DELAY)...
%                     & temp_spikes<(nev_validViewTimes(k) + 0.5 + DELAY));
%                 spikecounts(idx,k,3)=sum(temp_spikes>=(nev_validViewTimes(k) +0.5 + DELAY)...
%                     & temp_spikes<(nev_validViewTimes(k) + 0.75 + DELAY));
%                 spikecounts(idx,k,4)=sum(temp_spikes>=(nev_validViewTimes(k) +0.75 + DELAY)...
%                     & temp_spikes<(nev_validViewTimes(k) + 1 + DELAY));
                
                    % for revised version:
                    spikecounts(idx,id2,k,1)=sum(temp_spikes>=(nev_validViewTimes(k) + DELAY)...
                        & temp_spikes<(nev_validViewTimes(k) + 0.15 + DELAY));
                    spikecounts(idx,id2,k,2)=sum(temp_spikes>=(nev_validViewTimes(k) + 0.15 + DELAY)...
                        & temp_spikes<(nev_validViewTimes(k) + 0.3 + DELAY));
                    spikecounts(idx,id2,k,3)=sum(temp_spikes>=(nev_validViewTimes(k) + 0.3 + DELAY)...
                        & temp_spikes<(nev_validViewTimes(k) + 0.45 + DELAY));
                    spikecounts(idx,id2,k,4)=sum(temp_spikes>=(nev_validViewTimes(k) + 0.45 + DELAY)...
                        & temp_spikes<(nev_validViewTimes(k) + 0.6 + DELAY));
                    spikecounts(idx,id2,k,5)=sum(temp_spikes>=(nev_validViewTimes(k) + 0.6 + DELAY)...
                        & temp_spikes<(nev_validViewTimes(k) + 0.75 + DELAY));
                    spikecounts(idx,id2,k,6)=sum(temp_spikes>=(nev_validViewTimes(k) + 0.75 + DELAY)...
                        & temp_spikes<(nev_validViewTimes(k) + 0.9 + DELAY));
                
                  % for calculating average delay:
%                 temp_times=find(temp_spikes>=(nev_validViewTimes(k))...
%                     & temp_spikes<(nev_validViewTimes(k) + 1.1));
%                 resp_times{idx,k}=temp_spikes(temp_times)-nev_validViewTimes(k);
                end
                % calculate spontaneous firing rate:
                temp2=[];
                for l=1:length(tempb)
                    temp2(l)=sum(temp_spikes>=tempb(l)+DELAY & temp_spikes<tempb(l)+DELAY+spont_dur);
                end
                spont_tmp(idx,id2,:)=[nanmean(temp2) std(temp2)/sqrt(length(temp2))];
                id2=id2+1;
            end
            idx=idx+1;
        end
        disp(el)
    end
    
    %% calculate tuning and responses to stim
    % choose best latency using TC variance
    spikecount=NaN*zeros(idx-1,56,nValidTrials*6);
%     spikecount=NaN*zeros(idx-1,nValidTrials*6);
%     spikecount_h1=NaN*zeros(idx-1,nValidTrials*2);  % spike count for first half of trials
%     spikecount_h2=NaN*zeros(idx-1,nValidTrials*2);  % spike count for second half of trials
%     spikecount_ep1=NaN*zeros(idx-1,nValidTrials*2); % spike count for first half of each fixation
%     spikecount_ep2=NaN*zeros(idx-1,nValidTrials*2); % spike count for second half of each fixation
%     resp_uniform=spikecount(:,1:utrials*6);
%     resp_bias=spikecount(:,utrials*6+1:end);
%     tune=NaN*zeros(idx-1,length(oris));             % mean spike count for each ori
%     tune_sem=NaN*zeros(idx-1,length(oris));         % sem for each ori
    stim_resp=cell(idx-1,length(oris));             % spike counts, individual trials by ori
    tune_u=nan*zeros(idx-1,length(oris));           % tuning during uniform
    tune_b=nan*zeros(idx-1,length(oris));           % tuning during bias
    tune_sem_u=nan*zeros(idx-1,length(oris));       % sem uniform
    tune_sem_b=nan*zeros(idx-1,length(oris));       % sem bias
    stim_resp_u=cell(idx-1,length(oris));           % spike counts for each trial, uniform
    stim_resp_b=cell(idx-1,length(oris));           % spike counts for each trial, bias
    spont=nan*zeros(idx-1,2);                       % spontaneous rate (mean and sem)
    resp_uniform=nan*zeros(idx-1,utrials*6);
    resp_bias=nan*zeros(idx-1,length(oris_validTrials(utrials*6+1:end)));
    latency=nan*zeros(size(spikecount,1));
    for j = 1:idx-1
        for e=1:56
            tmp=[];
            for k=1:nValidTrials
                tmp=[tmp; squeeze(spikecounts(j,e,k,:))];
            end
            spikecount(j,e,:)=tmp';
        end
    end
    
    for j=1:size(spikecount,1)
        tmp=zeros(1,56);
        for k = 1:56
            for i=1:length(oris)
                temp=find(oris_u==oris(i));
                resp_base_test(j,k,i)=mean(squeeze(spikecount(j,k,temp)));
                temp=find(oris_b==oris(i));
                resp_bias_test(j,k,i)=mean(squeeze(spikecount(j,k,temp+utrials*6)));
            end
            tmp(k)=var(squeeze(resp_base_test(j,k,1:9)));
        end
        [~,y]=find(tmp==max(tmp));
        disp(y)

        resp_uniform(j,:)=squeeze(spikecount(j,y,1:utrials*6));
        resp_bias(j,:)=squeeze(spikecount(j,y,utrials*6+1:end));
        tune_u(j,:)=squeeze(resp_base_test(j,y,:));
        tune_b(j,:)=squeeze(resp_bias_test(j,y,:));
        spont(j,:)=squeeze(spont_tmp(j,y,:));
        latency(j)=y*0.002+0.02;
    end
    
    % save spike counts to each stim on individual trials for corr analysis and FF
    for u=1:size(spikecount,1)
        for e=1:length(oris)
            [~,x]=find(oris_u==oris(e));
            tmp=resp_uniform(u,x);
%             tune(u,e)=mean(tmp);
%             tune_sem(u,e)=std(tmp)./sqrt(length(tmp));
%             tmp2=spikecount(u,x(x<=length(oris_u)));
%             tune_u2(u,e)=mean(tmp); % check that this matches tune_u (IT DOES)
            tune_sem_u(u,e)=std(tmp)./sqrt(length(tmp));
            [~,x]=find(oris_b==oris(e));
            tmp3=resp_bias(u,x);
%             tune_b2(u,e)=mean(tmp3);
            tune_sem_b(u,e)=std(tmp3)./sqrt(length(tmp3));
%             if isempty(tmp)
%                 stim_resp{u,e}=nan;
%             else
%                 stim_resp{u,e}=tmp;
%             end
            if isempty(tmp)
                stim_resp_u{u,e}=nan;
            else
                stim_resp_u{u,e}=tmp;
            end
            if isempty(tmp3)
                stim_resp_b{u,e}=nan;
            else
                stim_resp_b{u,e}=tmp3;
            end
            
            % fano:
            ff_u(u,e)=var(tmp)/mean(tmp);
            ff_b(u,e)=var(tmp3)/mean(tmp3);
            clear tmp tmp2 tmp3 
        end
    end

    % calculate tuning curve w/ vector sum
    for u = 1:size(spikecount,1)
%         tmp=tune(u,:);
%         [~,~, oribias(u), oripref(u), ~,~]=orivecfit(oris,tmp,spont(u,1)*1.5);
        tmp1=tune_u(u,:);
        [~,~, oribias_u(u), oripref_u(u),~,~]=orivecfit(oris,tmp1,spont(u,1)*1.5);
        tmp2=tune_b(u,:);
        [~,~, oribias_b(u), oripref_b(u),~,~]=orivecfit(oris(1:end-1),tmp2,spont(u,1)*1.5);
        clear tmp tmp1 tmp2
    end
    
    %% plot individual tuning curves
%     figure;
%     for u=1:size(spikecount,1)
%          hold on
%         plot(tune(u,:),'k')
%         plot(tune_u(u,:),'g')
%         plot(tune_b(u,:),'r')
%         pause; clf
%     end
% figure
% for j=1:56
%     hold on
%     plot(squeeze(resp_base_test(6,j,:)),'k')
%     plot(squeeze(resp_bias_test(6,j,:)),'r')
%     title(j)
%     axis square
%     pause; clf
% end

    %% plot Fanos:
%     figure; subplot(121)
%     histogram(ff_u(:))
%     axis square; box off
%     title('uniform')
%     xlabel('Fano')
%     subplot(122)
%     histogram(ff_b(:))
%     axis square; box off
%     title('bias')
    %% plot adaptation KERNEL:
%     figure
%     plot(geomean(tune_b./tune_u,1))
%     axis square; box off
%     xlabel('preferred ori')
%     ylabel('Response ratio (geomean)')

    clear u x e ii idx ans el elLabel j k l nn tt tmp* temp* oris_temp reward_blockID sort_codes...
        digOut* filename1 y spont_tmp resp_bias_test resp_base_test i id2 DELAY
    savename=sprintf('%s_tuning',filename);
    save(savename)
end

    %% PSTH - skip this if not looking at spontaneous firing or onset latency
% NOT USING THIS ANYMORE. CHANGED TO MATT SMITH METHOD.
%     stop % calculate psth and reset average DELAY
% %     figure out average delay and re-run analysis
%     bins=(0:0.01:1);
%     for i=1:size(resp_times,1)                        % units
%         tmp=[];
%         for j=1:size(resp_times,2)                % trials
%             tmp=[tmp;resp_times{i,j}];
%         end           
%        psth(i,:)=(1/diff(bins(1:2))).*histcounts(tmp,bins);%/size(resp_times,2);
%     end
%     figure
%     for i=1:size(resp_times,1)
%         plot(psth(i,:),'k.-')
%         pause; clf
%     end
%     temp=psth(:,1:50);
%     for i=1:size(resp_times,1)
%         temp(i,:)=temp(i,:)./max(temp(i,:));
%     end
%     meandelay=mean(temp,1);
%     figure; plot(meandelay,'k.-')
%     stoppsth