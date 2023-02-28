%% Scenario Parameters

% Scenario running time and sample time
% scenario_step %[s], scenario_stop %[s]
% frame_rate * running_time should be an integer.
frame_rate      = 24;   % [fps]
running_time    = 1;    % [s]
% if scenario can not run correct steps when running_time > 1, please change 'scenario_step/5'
% bigger but smaller than scenario_step in function getSimulateTime()
[scenario_step, scenario_stop] = getSimulateTime(frame_rate, running_time);

% number of non-ego cars (i.e. target number, <= 5)
num_non_ego_cars = 1;

% number of lanes
num_lane = 6;

% set lane offset [m]
l_offset_list = [0; 1; -1; 2; -2; 3];
l_offset = l_offset_list(1:num_non_ego_cars+1);

% Create road 
roadCenters = [0 0; 1000 0];

% set velocity of ego car and maxiaml velocity of non-ego cars [m/s]
v_ego = 13.9;
v_lim_car = 50;
v_lim_truck = 30;
v_lim_bicycle = 15;

% parameter for generate data within specific velocity range, please check
% also [speeds, waypoints] for making it
% possible velocity range (based on v_lim and vMax):
% car: [multiple_min(1), -3], [-3, -2], [-2, -1], [-1, 0], [0, 1], [1, multiple_max(1)]V_max
% truck: [multiple_min(2), -2], [-2, -1], [-1, 0], [0, multiple_max(2)]V_max
% bicycle: [multiple_min(3), -1], [-1, 0], [0, multiple_max(3)]V_max
% possible n times V_max
mul_n_array = [-5 -3 -1 1 3 5];
% possible factors (based on HPC)
factors =  [-2 -1 0 1 2];
load load_radar_setting.mat vMax
threshold=vMax;
multiple_min = [(-v_lim_car-v_ego)/threshold (-v_lim_truck-v_ego)/threshold (-v_lim_bicycle-v_ego)/threshold];
multiple_max = [(v_lim_car-v_ego)/threshold (v_lim_truck-v_ego)/threshold (v_lim_bicycle-v_ego)/threshold];
% multiple = [car truck bicycle]
multiple1=[1 -1 multiple_min(3)];
multiple2=[multiple_max(1) multiple_max(2) -1];

% set maxiaml acceleration of non-ego cars [m/s^2]
a_lim_car = 10;
a_lim_truck = 5;
a_lim_bicycle = 3;

% parameter for creating initial postion [m]
dist_diff = 90;

% set RCS of non-ego cars
car_RCS = 30;
truck_RCS = 100;
bicycle_RCS = 10;
%Converting Radar Cross Section in Square Meters to Decibels
car_rcs = SquareMetersToDecibels(car_RCS);
truck_rcs = SquareMetersToDecibels(truck_RCS);
bicycle_rcs = SquareMetersToDecibels(bicycle_RCS);

% set parameter of cars
% class_id ref： https://ww2.mathworks.cn/help/driving/ref/drivingscenario.vehicle.html
class_id = struct('Car', 1, 'Truck', 2, 'Bicycle', 3);
% parameters of ego car
ego_car_para = struct('ClassID',class_id.Car, 'Length', 4.5, 'Width', 1.9, 'Height', 1.5, ...
    'MaxVelocity', v_ego, 'MaxAcceleration', 0);
% parameters for non-ego cars
car_para = struct('ClassID',class_id.Car, 'Length', 4.5, 'Width', 1.9, 'Height', 1.5, ...
    'MaxVelocity', v_lim_car, 'MaxAcceleration', a_lim_car, 'RCSPattern',car_rcs);
truck_para = struct('ClassID',class_id.Truck,'Length', 12, 'Width', 2.4, 'Height', 3, ...
    'MaxVelocity', v_lim_truck, 'MaxAcceleration', a_lim_truck, 'RCSPattern',truck_rcs);
bicycle_para = struct('ClassID',class_id.Bicycle,'Length', 2, 'Width', 1, 'Height', 1.5, ...
    'MaxVelocity', v_lim_bicycle, 'MaxAcceleration', a_lim_bicycle, 'RCSPattern',bicycle_rcs);

% % random choose non-ego car
% % probability: Bicycle 20%, car 60%, truck 20%
non_ego_actors = ["Car" "Truck" "Bicycle"];
probability = [1 0 0];
non_egos = [];
% get all non-ego cars 
for i = 1:num_non_ego_cars
    non_ego = randsample(non_ego_actors,1,true,probability);
    non_egos = [non_egos non_ego];
end

% rename all non-ego cars
[C, ia, ic] = unique(non_egos, 'stable');
for n = 1: numel(C)
    inx_num = find(non_egos==C(n));
    for m = 1:numel(inx_num)
        a = inx_num(m);
        non_egos(a) = append(C(n), '_', num2str(m));
    end
end

% create non-ego cars
non_ego_car_paras = struct();
non_ego_speeds = [];
non_ego_waypoints = [];
for j = 1:numel(non_egos)
    if  contains(non_egos(j), 'Car')
        non_ego_car_paras.(non_egos(j)) = car_para;
        [speeds, waypoints] = getWaypointsAndSpeed(car_para, scenario_stop, scenario_step, dist_diff, false, false, ...
            false,threshold,multiple1(1),multiple2(1),v_ego);
    elseif contains(non_egos(j), 'Truck')
        non_ego_car_paras.(non_egos(j)) = truck_para;
        [speeds, waypoints] = getWaypointsAndSpeed(truck_para, scenario_stop, scenario_step, dist_diff, false, false, ...
            false,threshold,multiple1(2),multiple2(2),v_ego);
    elseif contains(non_egos(j), 'Bicycle')
        non_ego_car_paras.(non_egos(j)) = bicycle_para;
        [speeds, waypoints] = getWaypointsAndSpeed(bicycle_para, scenario_stop, scenario_step, dist_diff, false, false, ...
            false,threshold,multiple1(3),multiple2(3),v_ego);
    else
        fprintf(['The name of actor can only be "Bicycle", "Car" or "Truck". \n' ...
            'At least one of the actors is not correct. \n']);
    end
    non_ego_speeds = [non_ego_speeds; speeds];
    non_ego_waypoints = [non_ego_waypoints; waypoints];
end

%creat ego-car
[ego_speeds, ego_waypoints] = getWaypointsAndSpeed(v_ego, scenario_stop, scenario_step, 0, true, true);

% combined ego-car and non-ego cars
global all_speeds  all_waypoints
all_speeds = [ego_speeds; non_ego_speeds];
all_waypoints = [ego_waypoints; non_ego_waypoints];

% set radar
radarPos = [1.7 0 0.2]';
radarYaw = 0;
radarPitch = 0;
radarRoll = 0;
radarAxes = rotz(radarYaw)*roty(radarPitch)*rotx(radarRoll);
radar_params = struct( ...
    'Frame', drivingCoordinateFrameType.Spherical, ...
    'OriginPosition',radarPos, ...
    'OriginVelocity',zeros(3,1), ...
    'Orientation', radarAxes, ...
    'HasElevation', false, ...
    'HasVelocity', true, ...
    'RMSBias',[0 0.25 0.05]);


function [scenario_step, scenario_stop] = getSimulateTime(frame_rate, stop_time)
    scenario_step = 1/frame_rate;
    if stop_time <=1
        scenario_stop = stop_time + scenario_step;
    else
        scenario_stop = stop_time + scenario_step + scenario_step/5;
    end
end


function [speeds, waypoints]= getWaypointsAndSpeed(ObjectPara, scenario_stop, scenario_step, dist_diff, ifEgo, ...
    ifUniformMotion,ifvelocityrange,threshold,multiple1,multiple2,v_ego)
    % set ifvelocityrange as true, if specific velocity wanted
    % get initial velocicty and acceleration
    if ifEgo
        ini_v = ObjectPara;
        if ini_v < 0
            fprintf(['The Velocity of the ego car muss be positive or 0.\n']);
        end
    else
        if ifvelocityrange
            [v_min,v_max] = velocityrange(threshold,multiple1,multiple2,v_ego);
            ini_v = rand2(v_min, v_max);
        else
            ini_v = rand2(-ObjectPara.MaxVelocity, ObjectPara.MaxVelocity);
        end
    end

    if ifUniformMotion
        ini_a = 0;
    else
        ini_a = rand2(-ObjectPara.MaxAcceleration, ObjectPara.MaxAcceleration);
    end

    %get waypoints and speeds on waypoints
    steps = scenario_stop/scenario_step;
    cars_v = [];
    cars_s = [];
    for i = 1:steps-1
        t= i*scenario_step;
        v = abs(ini_v)+ini_a*t;
        s= abs(ini_v)*t + 0.5*ini_a*t^2;
        cars_v = [cars_v,v];
        cars_s = [cars_s,s];
    end

    speeds = [abs(ini_v) cars_v];

    if ini_v >= 0
        if ifEgo
            ini_dist = dist_diff;
        else
            ini_dist = rand2(10, 30);
        end
        waypoints = [ini_dist ini_dist+cars_s];
    else
        ini_v = abs(ini_v);
        if ifEgo
            ini_dist = dist_diff;
        else
            ini_dist = dist_diff-rand2(5, 15);
        end
        waypoints = [ini_dist ini_dist-cars_s];
    end
    
end


function rcs = SquareMetersToDecibels(RCS)
    rcs = [];
    formel = 10 * log10(RCS/1);
    rcs = [formel formel;formel formel];
end


% from https://github.com/tamaskis/rand2-MATLAB
function X = rand2(a,b,matrix_size,type)
    
    % sets default matrix size to 1-by-1 if matrix_size is not input (i.e.
    % rand2 returns a single number by default)
    if nargin < 3 || isempty(matrix_size)
        matrix_size = [1,1];
    end
    
    % sets datatype to double (default) if there less than 4 input
    % arguments (because in this case, we logically know that typename is
    % not specified)
    if nargin < 4
        type = 'double';
    end
    
    % rounds lower and upper bounds if integer data type specified
    if strcmpi(type,'int')
        a = ceil(a);
        b = floor(b);
    end

    % returns matrix of random integers between a and b
    if strcmpi(type,'int')
        X = randi([a,b],matrix_size);
        
    % returns matrix of random floating-point numbers (either single or
    % double precision, as specified by "typename") between a and b
    else
        X = a+(b-a)*rand(matrix_size,type);
        
    end

end


function [v_min,v_max] = velocityrange(threshold,multiple1,multiple2,v_ego)
    if multiple1==multiple2
        disp('multiple1 and multiple2 should not be same.')
        return
    end

    if multiple1 > multiple2
        v_min=threshold*multiple2+v_ego;
        v_max=threshold*multiple1+v_ego;
    else
        v_min=threshold*multiple1+v_ego;
        v_max=threshold*multiple2+v_ego;
    end
end
        