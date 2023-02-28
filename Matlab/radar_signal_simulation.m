%% Intro
% Author:     jiawei li
% Date:       04.03.2022
% Environment R2021b
% Dependency  Radar Toolbox
%             Automated Driving Toolbox
% Ref         https://www.mathworks.com/help/radar/ug/radar-signal-simulation-and-processing-for-automated-driving.html
% Description randomly generate public road scenarios containing 1-5 targets (bicycle, car, truck), export ADC data 
%             (Xcubes.mat), range-Doppler data (Xbf.mat) and ground truth range, velocity, detected range, velocity, number of target,  
%             target type, factor (label.mat)

% output architectural dataset:
% -dataset
%     -record-24-May-2022-16.48.39
%         -all jpg (option)
%         -ref.avi (option)
%         -label.mat      |
%         -Xbf.mat        |->   label.mat to .csv and Xbf.mat, Xcubes.mat to .npy by prepare_dataset.py
%         -Xcubes.mat (option)    |
%     -record...


%% Load Radar and scenario Parameters from the configuration file.

clc, clear;                     % Clean work space
load_radar_setting;
load_scenario_setting;

display_scenario = 'off';       % 'off': do not show video display; 'on': show dispaly

save_Xcubes = 'no';            % 'yes': save Xcubes in mat; 'no': don't save Xcubes in mat

detect_object = 'no';          % 'yes': use CFAR to detect, 'no': no display, because no detect.

deletle_record = 'no';          % 'yes' or 'no': delete or not delete record if there is error 

if strcmp(lower(detect_object),'no') && strcmp(lower(display_scenario),'on')
    disp(['if detect_object set as no, display_scenario has to be set as  off. Because for displaying scenario,' ...
        'detecting object is needed.'])
    return
end

dataset_dir = './dataset/'; % data save path
date_time = string(datetime(now,'ConvertFrom','datenum'));
date_time = strrep(date_time,':','.');
date_time = strrep(date_time,' ','-');
record_dir = append(dataset_dir,'record-',date_time,'/');


%% Model Automotive Radar Transceiver

% Configure the FMCW waveform using the waveform parameters derived from
% the long-range requirements
waveform = phased.FMCWWaveform('SweepTime',tm,'SweepBandwidth',bw,...
    'SampleRate',fs,'SweepDirection','Up');
if strcmp(waveform.SweepDirection,'Down')
    sweepSlope = -sweepSlope;
end
txsig = waveform();

% Model the antenna element
antElmnt = phased.IsotropicAntennaElement('BackBaffled',true);

% Construct the send array
txArray = phased.ULA('Element',antElmnt,'NumElements',Nt,...
    'ElementSpacing',lambda/2*Ne);

% Construct the receive array
rxArray = phased.ULA('Element',antElmnt,'NumElements',Ne,...
    'ElementSpacing',lambda/2);

% Construct the synthetic virtual array
virtualArray = phased.ULA(Nt*Ne,lambda/2);

% Half-power beamwidth of the receive array
hpbw = beamwidth(rxArray,fc,'PropagationSpeed',c);  % currently not used

antAperture = 6.06e-4;                        % Antenna aperture (m^2)
antGain = aperture2gain(antAperture,lambda);  % Antenna gain (dB)
txPkPower = db2pow(5)*1e-3;                   % Tx peak power (W)
txGain = antGain;                             % Tx antenna gain (dB)
rxGain = antGain;                             % Rx antenna gain (dB)
rxNF = 68;                                    % Receiver noise figure (dB)
%rxNF=17  %SNR=39.688771
%rxNF=34  %SNR=30.572294
%rxNF=51  %SNR=20.164143
%rxNF=  %SNR=
%rxNF=  %SNR=

% Waveform transmitter
transmitter = phased.Transmitter('PeakPower',txPkPower,'Gain',txGain);

% Radiator for single transmit element
radiator = phased.Radiator('Sensor',txArray,'OperatingFrequency',fc,'WeightsInputPort',true);

% Collector for receive array
collector = phased.Collector('Sensor',rxArray,'OperatingFrequency',fc);

% Receiver preamplifier
receiver = phased.ReceiverPreamp('Gain',rxGain,'NoiseFigure',rxNF, 'SampleRate',fs);

% Define radar
radar = radarTransceiver('Waveform',waveform,'Transmitter',transmitter,...
    'TransmitAntenna',radiator,'ReceiveAntenna',collector,'Receiver',receiver, ...
    'ElectronicScanMode','Custom');


%% Define Radar Signal Processing Chain

% Direction-of-arrival estimator for linear phased array signals
doaest = phased.RootMUSICEstimator(...Nft
    'SensorArray',virtualArray,...
    'PropagationSpeed',c,'OperatingFrequency',fc,...
    'NumSignalsSource','Property','NumSignals',1);

% Scan beams in front of ego vehicle for range-angle image display
angscan = -80:80;
beamscan = phased.PhaseShiftBeamformer('Direction',[angscan;0*angscan],...
    'SensorArray',virtualArray,'OperatingFrequency',fc);

% Form forward-facing beam to detect objects in front of the ego vehicle
beamformer = phased.PhaseShiftBeamformer('SensorArray',virtualArray,...
    'PropagationSpeed',c,'OperatingFrequency',fc,'Direction',[0;0]);

% range-Doppler
Xcube = zeros(Nft,Ne,Nsweep);

% 4 blocks should have same dimentions.
Nr = 2^nextpow2(size(Xcube(:,:,1:2:end),1));   % range samples within a chirp approximating to next power of 2 number 500 -> 512
Nd = 2^nextpow2(size(Xcube(:,:,2:2:end),3));   % Doppler samples across each chirp within one block 32 --> 32

rngdopresp = phased.RangeDopplerResponse('PropagationSpeed',c,...
    'DopplerOutput','Speed','OperatingFrequency',fc,'SampleRate',fs,...
    'RangeMethod','FFT','PRFSource','Property',...
    'RangeWindow','Hann','PRF',1/(Nt*waveform.SweepTime),...
    'SweepSlope',waveform.SweepBandwidth/waveform.SweepTime,...
    'RangeFFTLengthSource','Property','RangeFFTLength',Nr,...
    'DopplerFFTLengthSource','Property','DopplerFFTLength',Nd,...
    'DopplerWindow','Hann');  

% Guard cell and training regions for range dimension
nGuardRng = 4;
nTrainRng = 4;
nCUTRng = 1+nGuardRng+nTrainRng;

% Guard cell and training regions for Doppler dimension
dopOver = round(Nd/Nsweep);
nGuardDop = 2*dopOver;
nTrainDop = 2*dopOver;
nCUTDop = 1+nGuardDop+nTrainDop;

% make lower CustomThresholdFactor if objects are not enough detected. normally: 13 for 1 non-ego; 12 for 2 non-ego 
cfar = phased.CFARDetector2D('GuardBandSize',[nGuardRng nGuardDop],...
    'TrainingBandSize',[nTrainRng nTrainDop],...
    'ThresholdFactor','Custom','CustomThresholdFactor',db2pow(13),...
    'NoisePowerOutputPort',true,'OutputFormat','Detection index');

% Perform CFAR processing over all of the range and Doppler cells
freqs = ((0:Nr-1)'/Nr-0.5)*fs;
rnggrid = beat2range(freqs,sweepSlope);
iRngCUT = find(rnggrid>0);
iRngCUT = iRngCUT((iRngCUT>=nCUTRng)&(iRngCUT<=Nr-nCUTRng+1));
iDopCUT = nCUTDop:(Nd-nCUTDop+1);
[iRng,iDop] = meshgrid(iRngCUT,iDopCUT);
idxCFAR = [iRng(:) iDop(:)]';

% Perform clustering algorithm to group detections
clusterer = clusterDBSCAN('Epsilon',2);

rmsRng = sqrt(12)*rangeRes;
rngestimator = phased.RangeEstimator('ClusterInputPort',true,...
    'VarianceOutputPort',true,'NoisePowerSource','Input port',...
    'RMSResolution',rmsRng);

% 'NumEstimates': The maximum number of estimates to report, specified as a positive integer. 
% When the number of requested estimates is greater than the number of columns in the detidx 
% argument of the step method, the remainder is filled with NaN.
dopestimator = phased.DopplerEstimator('ClusterInputPort',true,...
    'VarianceOutputPort',true,'NoisePowerSource','Input port',...
    'NumPulses',Nsweep,'NumEstimatesSource', 'Property', 'NumEstimates', 2, 'VarianceOutputPort', true);

tracker = radarTracker('FilterInitializationFcn',@initcvekf,...
    'AssignmentThreshold',50);


%% Simulate the Driving Scenario
try
    % Create driving scenario
    % load load_scenario_setting again, only the data loaded here will be
    % saved.
    [scenario,egoCar,radarParams] = ...
    helperAutoDrivingRadarSigProc('Setup Scenario',c,fc);
catch e
    fprintf(1,'Error:\n%s',e.message);
    return
end

% Initialize display for driving scenario example
helperAutoDrivingRadarSigProc('Initialize Display',egoCar,radarParams,...
    rxArray,fc,vMax,rangeMax);

% get informations about non-ego cars
tgtProfiles = actorProfiles(scenario);
tgtProfiles = tgtProfiles(2:end);
tgtHeight = [tgtProfiles.Height]; 
tgtRCS = [tgtProfiles.RCSPattern]; 
   
% 2 Modulations for MIMO: BPM and TDM
block_num = 1;          % initial block number is 1

if strcmp(modulation,'BPM')
    %weights of BPM
    w1 = [1;1];         % Chirp 1
    w2 = [1;-1];        % Chirp 2 
elseif strcmp(modulation,'TDM')
    %weights of TDM
    w1 = [1;0];         % Chirp 1 
    w2 = [0;1];         % Chirp 2      
else
    fprintf("Incorrect modulation setting, enter BPM or TDM! \n");
    return;
end

% creates a radar cross-section (RCS) signature
Signature = cell(1,numel(tgtHeight));
for i = 1:numel(tgtHeight)
    Signature{i}=rcsSignature('Pattern',tgtRCS(i));
end

if not(isfolder(record_dir))
    mkdir(record_dir);
end

if strcmp(lower(detect_object),'yes')
    video_name = append(record_dir,'ref');
    myVideo = VideoWriter(video_name); % open video file
    myVideo.FrameRate = 10;  % can adjust this, 5 - 10 works well
    open(myVideo)
end

time_array = [];
num_targets_array = [];
true_range = [];
r_det_array = [];
true_v = [];
permuted_Xcubes = zeros(frame_rate*running_time,Ne,Nsweep,Nft);
Xbfs = zeros(frame_rate*running_time,Nr,Nd);
iter_n = 1;
v_det_array = [];
rsest_array = [];
factor_array = [];
class_id_array = [];
while advance(scenario)
    % Get the current scenario time
    time = scenario.SimulationTime;
    % Get current target poses in ego vehicle's reference frame
    tgtPoses = targetPoses(egoCar);
    tgtPos = reshape([tgtPoses.Position],3,[]);
    % Position point targets at half of each target's height
    tgtPos(3,:) = tgtPos(3,:)+0.5*tgtHeight; 
    tgtVel = reshape([tgtPoses.Velocity],3,[]);

    % Assemble data cube at current scenario time
    for m = 1:Nsweep
        switch(block_num) 
            case 1 
                w0 = w1;
                block_num = block_num + 1;
            case 2
                w0 = w2;
                block_num = 1;
        end

        ntgt = size(tgtPos,2);
        tgtStruct = struct('Position',mat2cell(tgtPos(:).',1,repmat(3,1,ntgt)),...
            'Velocity',mat2cell(tgtVel(:).',1,repmat(3,1,ntgt)),...
            'Signature',Signature);
        [rxsig, info] = radar(tgtStruct,time+(m-1)*tm,w0);
        
        % Dechirp the received signal
        dechirpsig = dechirp(rxsig,txsig);

        % Decimate the return to reduce computation requirements
        for n = size(Xcube,2):-1:1
           Xcube(:,n,m) = decimate(dechirpsig(:,n),1,'fir');
        end
        
        % Move targets forward in time for next sweep
        tgtPos = tgtPos+tgtVel*tm;
    end
    
    % save permuted Xcube in mat file
    permuted_Xcube = permute(Xcube, [2 3 1]);
    permuted_Xcubes(iter_n,:,:,:) = permuted_Xcube;

    % Calculate the range-Doppler response
    [rd_cube1,rnggrid,dopgrid] = rngdopresp(Xcube(:,:,1:Nt:end));
    rd_cube2 = rngdopresp(Xcube(:,:,2:Nt:end));

    % DOPPLER PHASE COMPENSATION 
    phase_dp_grid = 4*pi*dopgrid*4*waveform.SweepTime/lambda;   % phase grid corresponding to the velocity grid
    
    % assuming in each Doppler bin, compensate with corresponding Doppler shift.
    compensator_line2 = transpose(exp((1j/2)*phase_dp_grid)); % TO COMPENSATE BLOCK 2
    size_rd_cube = size(rd_cube1); % for phase compensation indexing
   
    % Phase Compensation
    for i_row=1:size_rd_cube(1) % Range indexes
        for i_col=1:size_rd_cube(2) % RX indexes
            for i_depth=1:size_rd_cube(3) % 32 Doppler indexes
                rd_cube2(i_row,i_col,i_depth) = (rd_cube2(i_row,i_col,i_depth))*compensator_line2(1,i_depth);  
            end
        end
    end
    
    % HADAMARD DECODING 
    if strcmp(modulation,'BPM')
        resp_vrt_1 = (rd_cube1 + rd_cube2)/2;
        resp_vrt_2 = (rd_cube1 - rd_cube2)/2;  
    else %TDM
        resp_vrt_1 = rd_cube1;
        resp_vrt_2 = rd_cube2;        
    end

    resp_vrt = cat(2,resp_vrt_1,resp_vrt_2); 
    
    %beamforming with all virtual channels
    Xbf = cat(2,resp_vrt_1,resp_vrt_2);
    Xbf = permute(Xbf,[1 3 2]);
    Xbf = reshape(Xbf,Nr*Nd,Nt*Ne);
    Xbf = beamformer(Xbf);
    Xbf = reshape(Xbf,Nr,Nd);    
    Xbfs(iter_n,:,:) = Xbf;
    iter_n = iter_n + 1;

    true_r= tgtPos(1,:);
    true_vel = tgtVel(1,:);
    time_array = [time_array; time];
    num_targets_array = [num_targets_array; numel(tgtHeight)];
    class_id_array = [class_id_array; tgtPoses.ClassID];
    if strcmp(lower(detect_object),'yes')
        % Detect targets
        Xpow = abs(Xbf).^2;
     
        [detidx,noisepwr] = cfar(Xpow,idxCFAR);
        
        % Cluster detections
        [~,clusterIDs] = clusterer(detidx.');  

        % ensure detection order matches true target order
        try
            assert(max(clusterIDs)==num_non_ego_cars,"detected clusters not equal to true number of non-ego cars.")
        catch e 
            if strcmp(lower(deletle_record),'yes')
                close(myVideo);
                pause(0.5)
                rmdir(record_dir, 's');
                disp('Record deleted');
                return
            else
                disp('Record not deleted');
                break
            end
        end
        if (num_non_ego_cars == 2)
            First_clusterID = clusterIDs(1);
            Second_clusterID = 3 - First_clusterID; 
            First_range = rnggrid(detidx(1,1));
            if(abs(First_range - true_r(Second_clusterID)) <= abs(First_range - true_r(First_clusterID)))
                clusterIDs = 3 - clusterIDs; % swap clusterID of tar1 and tar2
            end
        end

        % Estimate azimuth, range, and radial speed measurements
        [azest,azvar,snrdB] = ...
            helperAutoDrivingRadarSigProc('Estimate Angle',doaest,...
            conj(resp_vrt),Xbf,detidx,noisepwr,clusterIDs);
        azvar = azvar+radarParams.RMSBias(1)^2;

        [rngest,rngvar] = rngestimator(Xbf,rnggrid,detidx,noisepwr,clusterIDs);
        rngvar = rngvar+radarParams.RMSBias(2)^2;
        % remove Nan
        rngest = rngest(~isnan(rngest));
        
        [rsest,rsvar] = dopestimator(Xbf,dopgrid,detidx,noisepwr,clusterIDs);
        % remove Nan
        rsest = rsest(~isnan(rsest));
        % Convert radial speed to range rate for use by the tracker
        rrest = -rsest;
        rrvar = rsvar(~isnan(rsvar));
        rrvar = rrvar+radarParams.RMSBias(3)^2;

        % Assemble object detections for use by tracker
        % jiawei add compare rngest and rrest because of [azest(iDet) rngest(iDet) rrest(iDet)]'. 
        if numel(rngest) <= numel(rrest)
            numDets = numel(rngest);
        else
            numDets = numel(rrest);
        end
    
        dets = cell(numDets,1);
        
        for iDet = 1:numDets
            dets{iDet} = objectDetection(time,...
                [azest(iDet) rngest(iDet) rrest(iDet)]',...
                'MeasurementNoise',diag([azvar(iDet) rngvar(iDet) rrvar(iDet)]),...
                'MeasurementParameters',{radarParams},...
                'ObjectAttributes',{struct('SNR',snrdB(iDet))});
        end
        
        % Track detections
        tracks = tracker(dets,time);
        
        % Update displays
        helperAutoDrivingRadarSigProc('Update Display',egoCar,dets,tracks,...
            dopgrid,rnggrid,Xbf,beamscan,resp_vrt, display_scenario);
        
        % Collect free space channel metrics
        metricsFS = helperAutoDrivingRadarSigProc('Collect Metrics',...
            radarParams,tgtPos,tgtVel,dets);

        if numDets ~= numel(tgtHeight)
            disp('Number of detected objects is not same as non-ego cars')
            if strcmp(lower(deletle_record),'yes')
                close(myVideo);
                pause(0.5)
                rmdir(record_dir, 's');
                disp('Record deleted');
                return
            else
                disp('Record not deleted');
                break
            end
        end
        
        true_range = [true_range;true_r];
        true_v = [true_v;true_vel];
        vTrue_n = true_vel/vMax;

        v_det = -rsest'; % convert closing speed to away speed.
        r_det = rngest';

        factor = [];
        diff_r = true_r - rngest;
        min_r = [];
        
        % calculate and match factor for every non-ego cars
        for i = 1:numel(tgtHeight)
            if (mul_n_array(end) < vTrue_n(i)) || (vTrue_n(i) < mul_n_array(1))
                disp('Some velocities are in range.')
                return
            end
            for j = 1:(size(mul_n_array,2)-1)
                if (mul_n_array(j) <= vTrue_n(i)) && (vTrue_n(i) < mul_n_array(j+1))
                    factor = [factor; factors(j)];
                end
            end
            [M_r,I_r] = min(abs(diff_r(:,i)));
            min_r = [min_r I_r];
            
        end
        diff_v = true_vel - (v_det+2*vMax*factor')';
        m_v = [];
        min_v = [];
       
        for i = 1:numel(tgtHeight)
            
            [M_v,I_v] = min(abs(diff_v(:,i)));
            m_v = [m_v M_v];
            min_v = [min_v I_v];
        end

        tolerance = m_v;

        fprintf('tgtVel  (radial speed: Vnon-ego - Vego    ): %f \n',true_vel)
        fprintf('v_det   (radial speed: Vnon-ego - Vego    ): %f \n',v_det)
        fprintf('v_rsest (radial speed: Vego     - Vnon-ego): %f \n',rsest)
        fprintf('factor: %i \n',factor)
        fprintf('tolerance: %f \n',tolerance)
        fprintf('**************************************************************\n')

        r_det_array = [r_det_array; r_det(:,min_r)];
        v_det_array = [v_det_array;v_det(:,min_v)];
        factor_array = [factor_array;factor'];
    
    else
        true_range = [true_range;true_r];
        true_v = [true_v;true_vel];

    end

    % save .jpg
    if strcmp(lower(detect_object),'yes')
        pause(0.01)
        frame = getframe(gcf);
        writeVideo(myVideo, frame);
    
        saveas(gcf, append(record_dir, 'ref-',sprintf("%.2f",time), '.jpg'))
    end
end

% save label.mat
label = [time_array num_targets_array];
try
    for m=1:numel(tgtHeight)
        if strcmp(lower(detect_object),'yes')
            label = [label true_range(:,m) r_det_array(:,m) true_v(:,m) v_det_array(:,m) factor_array(:,m) class_id_array(:,m)];
        else
            label = [label true_range(:,m) true_v(:,m) class_id_array(:,m)];
        end
    end
    save(append(record_dir,'label.mat'),'label');
catch e 
    close(myVideo);
    pause(0.5)
    rmdir(record_dir, 's');
    disp('Record deleted');
    return
end

% save Xbf.mat
save(append(record_dir,'Xbf.mat'),'Xbfs');

% save Xcube.mat
if strcmp(lower(save_Xcubes),'yes' )
    save(append(record_dir,'Xcubes.mat'),'permuted_Xcubes');
elseif strcmp(lower(save_Xcubes),'no' )
else
    disp('Please set save_Xcubes as yes or no.')
end

if strcmp(lower(detect_object),'yes')
    close(myVideo);
end

disp("gen record finished!");
