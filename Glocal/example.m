clc;
addpath('clf');
addpath('lib');
addpath(genpath('lib/manopt'));
addpath('evl');

[in_result, out_result] = run_arts();

disp("============== 70% label missing: ================");
disp("Recovery Result:");
disp(in_result(1));
disp("Prediction Result:");
disp(out_result(1));

disp("============== 30% label missing: ================");
disp("Recovery Result:");
disp(in_result(2));
disp("Prediction Result:");
disp(out_result(2));