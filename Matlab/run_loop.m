%% Collecte datas

% run 'radar_signal_simulation.m' n times to collecte datas.

clc, clear; 

% number of loops
n = 3;

for i = 1:n
    radar_signal_simulation;
    i = i+1;
end
