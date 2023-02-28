%% Radar Parameters

modulation = 'TDM';                 % 'TDM' or 'BPM'

fc = 77e9;                          % Center frequency (Hz) 
c = physconst('LightSpeed');        % Speed of light in air (m/s)
lambda = freq2wavelen(fc,c);        % Wavelength (m)

Nt = 2;                             % Tx number (currently only support Tx = 2)
Ne = 4;                             % Rx number
Nsweep = 128;                       % chirp number

rangeMax = 100;                     % Maximum range (m)
rangeRes = 2;                       % range resolution (m)

% sweep time should be at least 5 to 6 times the round trip time.
% ref: https://ww2.mathworks.cn/help/radar/ug/automotive-adaptive-cruise-control-using-fmcw-technology.html
tm = 40*range2time(rangeMax,c);      % Chirp duration (s)
% Determine the waveform bandwidth from the required range resolution
bw = rangeres2bw(rangeRes,c);       % Corresponding bandwidth (Hz)
                                                                                %ambiguous setting
vMax = lambda/4/Nt/tm;              % Maximum velocity (m/s)    -> 72.95 m/s    [18.24]
vRes = lambda/2/Nsweep/tm;          % velocity resolution (m/s) ->  4.56 m/s    [0.57]

% Set the sampling rate to satisfy both the range and velocity requirements
sweepSlope = bw/tm;                           % FMCW sweep slope (Hz/s)
fbeatMax = range2beat(rangeMax,sweepSlope,c); % Maximum beat frequency (Hz)
fdopMax = speed2dop(2*vMax,lambda);           % Maximum Doppler shift (Hz)
fifMax = fbeatMax+fdopMax;                    % Maximum received IF (Hz)
fs = max(2*fifMax,bw);                        % Sampling rate (Hz)
Nft = fs*tm;                                  % samples per chirp 

Nft = 1024;                                   % hacked samples per chirp
fs = Nft/tm;                                  % hacked Sampling rate (Hz)

save('load_radar_setting.mat', 'vMax')