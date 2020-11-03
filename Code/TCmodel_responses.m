function [resp, resp_ssa, resp_gain, resp_full, resp_gh, resp_sh] = TCmodel_responses(k,amp,prefs,oris,test_ori)
    % k=tuning curve bw
    % amp=tuning curve amplitude
    % pref=orientation preferences of neurons
    % oris=orientations of stimuli
    % test=test orientations
    
%     prefs=(0:1:179);
%     oris=(0:1:179);
%     test_ori=(0:1:179);
%     k=2.8*(rand(length(prefs),1)+0.05);
%     amp=10*(rand(length(prefs),1));
    
    for i=1:length(prefs)
        tune(i,:)=amp(i)*exp(k(i)*(cos(2*deg2rad(oris-prefs(i)))-1))+(1.1*rand);
    end
    
    % %  ssa
%     ak_ssa=1.03-0.4*exp(3*(cos(deg2rad(90-oris))-1)); % previous
%     ak_ssa=1-0.00022*exp(6.194*(cos(deg2rad(90-oris))));   % from awake data
    ak_ssa=1-0.0003*exp(5.862*(cos(deg2rad(90-oris))));   % from awake data
    for i=1:length(prefs)
%         tune1(i,:)=(ak_ssa.*tune(i,:))+0.0134;
        tune1(i,:)=(ak_ssa.*tune(i,:))+0.0117;
    end

    % %  gain
    ak_gain=1.03-0.4*exp(3*(cos(deg2rad(90-prefs))-1));   % previous
%     ak_gain=1-0.0047*exp(3.574*(cos(deg2rad(90-prefs))));     % from awake data (ByResp)
%     ak_gain=1-0.0039*exp(3.491*(cos(deg2rad(90-prefs))));     % from awake data (ByPref)
    for i=1:length(prefs)
%         tune2(i,:)=(ak_gain(i).*tune(i,:))+0.0325;
        tune2(i,:)=(ak_gain(i).*tune(i,:))+0.0229;
    end

    % % 2 gains: ssa * gain (5 params - no alpha)
%     gain_two=(1-0.0043*exp(3.575*cos(deg2rad(90-prefs)))).*(1-1.175e-6*exp(10.457*cos(deg2rad(90-oris))));
%     gain_half=(1-0.0043*exp(3.575*cos(deg2rad(90-prefs))));
%     ssa_half=(1-1.175e-6*exp(10.457*cos(deg2rad(90-oris))));
    gain_two=(1-0.0028*exp(3.577*cos(deg2rad(90-prefs)))).*(1-2.4e-5*exp(7.834*cos(deg2rad(90-oris))));
    gain_half=(1-0.0028*exp(3.577*cos(deg2rad(90-prefs))));
    ssa_half=(1-2.4e-5*exp(7.834*cos(deg2rad(90-oris))));
    for i=1:length(prefs)
        tune2gain(i,:)=(gain_two(i).*tune(i,:))+0.0254;
        tune_gh(i,:)=(gain_half(i).*tune(i,:))+0.0254;
        tune_sh(i,:)=(ssa_half(i).*tune(i,:))+0.0254;
    end
    for i=1:length(test_ori)
        for j=1:length(prefs)
            resp(i,j,:)=poissrnd(tune(j,find(oris==test_ori(i))),1000,1);
            resp_ssa(i,j,:)=poissrnd(tune1(j,find(oris==test_ori(i))),1000,1);
            resp_gain(i,j,:)=poissrnd(tune2(j,find(oris==test_ori(i))),1000,1);
            resp_full(i,j,:)=poissrnd(tune2gain(j,find(oris==test_ori(i))),1000,1);
            resp_gh(i,j,:)=poissrnd(tune_gh(j,find(oris==test_ori(i))),1000,1);
            resp_sh(i,j,:)=poissrnd(tune_sh(j,find(oris==test_ori(i))),1000,1);
        end
    end
%     figure; 
%     hold on;
%     plot(ak_gain+0.0134);
%     plot(ak_ssa+0.0325);
%     plot(gain_two+0.0341)
%     plot(gain_half+0.0341);
%     plot(ssa_half+0.0341)
%     legend('N-s Ind','S-s Ind','Full','N-s full','S-s full')
%     title('kernels')
%     figure
%     subplot(131); hold on;
%     plot(tune(1:4:end,:)','k')
%     plot(tune1(1:4:end,:)','r')
%     plot(tune_gh(1:4:end,:)','m')
%     title('gain models')
%     axis square
%     subplot(132); hold on
%     plot(tune(1:4:end,:)','k')
%     plot(tune2(1:4:end,:)','b')
%     plot(tune_sh(1:4:end,:)','c')
%     title('ssa models')
%     axis square
%     subplot(133); hold on
%     plot(tune(1:4:end,:)','k')
%     plot(tune2gain(1:4:end,:)','g')
%     title('full model')
%     axis square
end
