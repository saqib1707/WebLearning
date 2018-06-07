addpath('lib');
clear
close all
clc

% =============================================================== %
% Initialisation
BUFFER_SIZE = 360;
tsamp = 50e-3;       % (in sec)
fsamp = 1/tsamp;
l_size = 100;
Sample_size = zeros(4, 2*BUFFER_SIZE);

training_window_size = zeros(4, 2*BUFFER_SIZE);
next_training_window_size = zeros(3,1);
next_training_window_size_temp = zeros(3,1);

Adaptive_Threshold = zeros(4,2*BUFFER_SIZE);

Accel_data = zeros(4,BUFFER_SIZE);
Accel_motion_data = zeros(4, BUFFER_SIZE);
Accel_motion_data_cpy = zeros(4, BUFFER_SIZE);

differentiation_1 = zeros(4, BUFFER_SIZE);
integeral_1 = zeros(3,BUFFER_SIZE);
velocity = zeros(3,BUFFER_SIZE);
motion = zeros(3,BUFFER_SIZE);
% mean = zeros(3,1);
% diff = zeros(3,1);
% myvar=zeros(3,1);
% mystd=zeros(3,1);
sum_x = sum_y = sum_z = 0;

alph = 0.02;
k = 3;
wsize = 0;
j  = 0;

initial_training_window_size = 50;
x_itr = 1;
y_itr = 1;
training_window_size(1,x_itr) = initial_training_window_size;
x_itr += 1;
training_window_size(2,y_itr) = initial_training_window_size;
y_itr += 1;


% =============================================================== %
%	collecting acceleration data from data file
filepath = '..\dataset\U.csv';
data = dlmread(filepath);
Accel_data(1,:) = (data(1:BUFFER_SIZE,1))';
Accel_data(2,:) = (data(1:BUFFER_SIZE,2))';
Accel_data(3,:) = (data(1:BUFFER_SIZE,3))';
Accel_data(4,:) = (data(1:BUFFER_SIZE,7))';

%================================================================ %
% Initial Calculation for Acceleration Values
mean_a = zeros(3,1);
std_a = zeros(3,1);
thresh_a = zeros(3,1);
mean_da = zeros(3,1);
std_da = zeros(3,1);
thresh_da = zeros(3,1);


% for i = 1:initial_training_window_size
% 	SUM(1,1) += abs(Accel_data(1,i));
% 	SUM(2,1) += abs(Accel_data(2,i));
% end

% MEAN(1,1) = SUM(1,1)/initial_training_window_size;
% MEAN(2,1) = SUM(2,1)/initial_training_window_size;
mean_a(1:2,1) = mean(abs(Accel_data(1:2,1:initial_training_window_size)),dim=2);

% DIFF = zeros(3,1);
% for i = 1:initial_training_window_size
% 	DIFF(1,1) = Accel_data(1,i) - MEAN(1,1);
% 	DIFF(2,1) = Accel_data(2,i) - MEAN(2,1);

% 	VAR(1,1) += (DIFF(1,1))^2;
% 	VAR(2,1) += (DIFF(2,1))^2;
% end
% STD(1,1) = sqrt(VAR(1,1)/initial_training_window_size);
% STD(2,1) = sqrt(VAR(2,1)/initial_training_window_size);

std_a(1:2,1) = std(Accel_data(1:2,1:initial_training_window_size), 1, dim=2);    % second argument = 0 for (N-1)  and 1 for N

% thresh_accel_x = MEAN(1,1) + k*STD(1,1);
% thresh_accel_y = MEAN(2,1) + k*STD(2,1);
thresh_a(1:2,1) = mean_a(1:2,1) + k*std_a(1:2,1);

% =============================================================== %
% Initial Calculation for Delta Acceleration Values
% differentiation_1(1,1) = (Accel_data(1,1)-0);
% differentiation_1(2,1) = (Accel_data(2,1)-0);
differentiation_1(1:2,1) = Accel_data(1:2,1)-0;
for i = 2:initial_training_window_size
	training_window_size(1,x_itr) = initial_training_window_size;
	x_itr += 1;
	training_window_size(2,y_itr) = initial_training_window_size;
	y_itr += 1;

	% differentiation_1(1,i) = (Accel_data(1,i) - Accel_data(1,i-1));
	% differentiation_1(2,i) = (Accel_data(2,i) - Accel_data(2,i-1));
	differentiation_1(1:2,i) = Accel_data(1:2,i) - Accel_data(1:2,i-1);

	% sum_x = sum_x + abs(differentiation_1(1,i));
	% sum_y = sum_y + abs(differentiation_1(2,i));
end

% mean(1,1) = sum_x/(initial_training_window_size);
% mean(2,1) = sum_y/(initial_training_window_size);
mean_da(1:2,1) = mean(abs(differentiation_1(1:2,1:initial_training_window_size)), dim=2);

% for i = 1:initial_training_window_size
% 	diff(1,1)=differentiation_1(1,i)-mean(1,1);
% 	diff(2,1)=differentiation_1(2,i)-mean(2,1);

% 	myvar(1,1)+=(diff(1,1))^2;
% 	myvar(2,1)+=(diff(2,1))^2;
% end

% mystd(1,1)=sqrt((myvar(1,1)/initial_training_window_size));
% mystd(2,1)=sqrt((myvar(2,1)/initial_training_window_size));
std_da(1:2,1) = std(differentiation_1(1:2,1:initial_training_window_size), 1, dim=2);     % second argument = 0 for (N-1)  and 1 for N

% threshold_x = (k * mystd(1,1) + mean(1,1));
% threshold_y = (k * mystd(2,1) + mean(2,1));
thresh_da(1:2,1) = mean_da(1:2,1) + k*std_da(1:2,1);

% next_training_window_size(1,1) = log(2/alpha)* (1/(2 * (mean(1,1) + k*mystd(1,1))^2));
% next_training_window_size(1,1) = ceil(next_training_window_size(1,1));
next_training_window_size(1:2,1) = floor(log(2/alph)/(2*(thresh_da(1:2,1)).^2));
% disp("next_training_window_size_x = "), disp(next_training_window_size(1,1));
training_window_size(1,x_itr) = next_training_window_size(1,1);
x_itr += 1;

% next_training_window_size(2,1) = log(2/alpha)* (1/(2 * (mean(2,1) + k*mystd(2,1))^2));
% next_training_window_size(2,1) = ceil(next_training_window_size(2,1));
% disp("next_training_window_size_y = "), disp(next_training_window_size(2,1));
training_window_size(2,y_itr) = next_training_window_size(2,1);
y_itr += 1;

% next_training_window_size_temp(1,1) = next_training_window_size(1,1);
% next_training_window_size_temp(2,1) = next_training_window_size(2,1);
next_training_window_size_temp(1:2,1) = next_training_window_size(1:2,1);
% ==================================X-AXIS=================== %

j = training_window_size(1,1) + 1;
while(j <= BUFFER_SIZE)
	temp_mean = mean_da(1,1);
  temp_a_mean = mean_a(1,1);
	temp_th = thresh_da(1,1);
  temp_a_th = thresh_a(1,1);

  SUM_X = 0;
	sum_x = 0;
	mean_da(1,1) = 0;
  mean_a(1,1) = 0;
	myvar(1,1) = 0;
  VAR(1,1) = 0;
	mystd(1,1) = 0;
  STD(1,1) = 0;
	wsize = 0;
	start = j;

	while(j <= BUFFER_SIZE && wsize < next_training_window_size(1,1))
		training_window_size(1,x_itr) = next_training_window_size_temp(1,1);
		Adaptive_Threshold(1,x_itr) = temp_th;
		x_itr += 1;
    	differentiation_1(1,j) = (Accel_data(1,j) - Accel_data(1,j-1));
		
		if (abs(differentiation_1(1,j)) < thresh_da && abs(Accel_data(1,j)) < thresh_a)
	      wsize += 1;
	      sum_x = sum_x + (differentiation_1(1,j));
	      SUM_X = SUM_X + Accel_data(1,j);
    	end

		if(abs(differentiation_1(1,j)) < thresh_da && abs(Accel_data(1,j)) < thresh_a)
			Accel_motion_data(1,j) = 0;
		elseif(abs(differentiation_1(1,j)) < thresh_da && abs(Accel_data(1,j)) > thresh_a)
			Accel_motion_data(1,j) = Accel_motion_data(1,j-1);
		else
			Accel_motion_data(1,j) = Accel_data(1,j) - mean_a(1,1);
		end

		if(abs(differentiation_1(1,j)) > thresh_da)
			velocity(1,j) = velocity(1,j-1) + (Accel_data(4,j)-Accel_data(4,j-1))*Accel_motion_data(1,j)/1000;
		else
			velocity(1,j) = 0;
		end

		motion(1,j) = motion(1,j-1) + (Accel_data(4,j)-Accel_data(4,j-1))*velocity(1,j)/1000;
		j = j + 1;
	end

	mean_da(1,1) = sum_x/(wsize);
  	mean_a(1,1) = SUM_X/(wsize);
	wsize = 0;
	j = start;
	
	while(j <= BUFFER_SIZE && wsize < next_training_window_size(1,1))
    	if(abs(differentiation_1(1,j)) < threshold_x && abs(Accel_data(1,j)) < thresh_accel_x)
	      diff(1,1) = differentiation_1(1,j)-mean(1,1);
	      DIFF(1,1) = Accel_data(1,j)-MEAN(1,1);
	      myvar(1,1) += diff(1,1)^2;
	      VAR(1,1) += DIFF(1,1)^2;
	      wsize += 1;
	  	end
		j = j + 1;
  	end

	mystd(1,1)=sqrt((myvar(1,1)/(wsize)));
  	STD(1,1)=sqrt(VAR(1,1)/(wsize));

	thresh_da = (k * std_da(1,1) + mean_da(1,1));
    thresh_a = (k * std_a(1,1)) + mean_a(1,1);

	next_training_window_size_temp(1,1) = next_training_window_size(1,1);
	next_training_window_size(1,1) = floor(log(2/alph)/(2*thresh_da.^2));
	% next_training_window_size(1,1) = ceil(next_training_window_size(1,1));

	if(threshold_x == 0 || next_training_window_size(1,1) <= 2)
		disp("threshold is zero or next_training_window_size_x is zero");
		next_training_window_size(1,1) = 50;
		threshold_x = temp_th;
		mean_da(1,1) = temp_mean;
		continue;
	end;
	next_training_window_size_temp(1,1) = next_training_window_size(1,1);
end

% ==========================Y-AXIS====================== %
j = training_window_size(2,1) + 1;
while(j <= BUFFER_SIZE)

	temp_mean = mean(2,1);
  temp_a_mean = MEAN(2,1);
	temp_th = threshold_y;
  temp_a_th = thresh_accel_y;

  SUM_Y = 0;
	sum_x = 0;
	mean(2,1) = 0;
  MEAN(2,1) = 0;
	myvar(2,1) = 0;
  VAR(2,1) = 0;
	mystd(2,1) = 0;
  STD(2,1) = 0;
	wsize = 0;
	start = j;

  %disp("j = "), disp(j);
	while(j <= BUFFER_SIZE && wsize < next_training_window_size(2,1))
		training_window_size(2,y_itr) = next_training_window_size_temp(2,1);
		Adaptive_Threshold(2,y_itr) = temp_th;
		y_itr += 1;
		differentiation_1(2,j) = (Accel_data(2,j) - Accel_data(2,j-1));
		% disp("final diff values "),disp(differentiation_1(2,j));
		% disp("final_threshold_y "),disp(threshold_y);
		% disp("final accel values "),disp(Accel_data(2,j));
		% disp("final_thresh_accel_y "),disp(thresh_accel_y);
		% disp("\n");
		if (abs(differentiation_1(2,j)) < threshold_y && abs(Accel_data(2,j)) < thresh_accel_y)
      wsize += 1;
      sum_y = sum_y + (differentiation_1(2,j));
      SUM_Y = SUM_Y + Accel_data(2,j);
    end

		if(abs(differentiation_1(2,j)) < threshold_y && abs(Accel_data(2,j)) < thresh_accel_y)
			Accel_motion_data(2,j) = 0;
		elseif(abs(differentiation_1(2,j)) < threshold_y && abs(Accel_data(2,j)) > thresh_accel_y)
			Accel_motion_data(2,j) = Accel_motion_data(2,j-1);
			% disp(1);
		else
			Accel_motion_data(2,j) = Accel_data(2,j) - MEAN(2,1);
			% disp(2);
		end
		% disp(Accel_motion_data(2,j));

		if(abs(differentiation_1(2,j)) > threshold_y)
			velocity(2,j) = velocity(2,j-1) + (Accel_data(4,j)-Accel_data(4,j-1))*Accel_motion_data(2,j)/1000;
		else
			velocity(2,j) = 0;
		end

		motion(2,j) = motion(2,j-1) + (Accel_data(4,j)-Accel_data(4,j-1))*velocity(2,j)/1000;

%		if(abs(differentiation_1(2,j)) < threshold_y)
%			Accel_motion_data(2,j) = 0;
%		end
%		if(abs(differentiation_1(2,j)) > threshold_y)
%			Accel_motion_data(2,j) = Accel_data(2,j) - MEAN(2,1);
%		end

		j = j + 1;
	endwhile

	mean(2,1) = sum_y/(wsize);
  MEAN(2,1) = SUM_Y/(wsize);
	wsize = 0;
	j = start;

	while(j <= BUFFER_SIZE && wsize < next_training_window_size(2,1))
    if(abs(differentiation_1(2,j)) < threshold_y && abs(Accel_data(2,j)) < thresh_accel_y)
      diff(2,1) = differentiation_1(2,j)-mean(2,1);
      DIFF(2,1) = Accel_data(2,j)-MEAN(2,1);
      myvar(2,1) += diff(2,1)^2;
      VAR(2,1) += DIFF(2,1)^2;
      wsize += 1;
	  end
		j = j + 1;
  endwhile

	mystd(2,1)=sqrt((myvar(2,1)/(wsize)));
  STD(2,1)=sqrt(VAR(2,1)/(wsize));
	%disp("mystd = "), disp(mystd(1,1));

	threshold_y = (k * mystd(2,1) + mean(2,1));
  thresh_accel_y = (k * STD(2,1)) + MEAN(2,1);

	next_training_window_size_temp(2,1) = next_training_window_size(2,1);

	next_training_window_size(2,1) = log(2/alph) * (1/(2 * (mean(2,1) + k*mystd(2,1))^2));
	next_training_window_size(2,1) = ceil(next_training_window_size(2,1));

	if(threshold_y == 0 || next_training_window_size(2,1) <= 2)
		disp("threshold is zero or next_training_window_size_x is zero");
		next_training_window_size(2,1) = 50;
		threshold_y = temp_th;
		mean(2,1) = temp_mean;
		continue;
	end;

	next_training_window_size_temp(2,1) = next_training_window_size(2,1);
endwhile

% =========================== PLOTS =====================================
%Hoeffding_convergence

% hfig=(figure);
% scrsz = get(0,'ScreenSize');
% set(hfig,'position',scrsz);
%
% subplot(4,1,1)
% p1 = plot(1:BUFFER_SIZE, Accel_data(1,:),'r');
% hold on;
% grid on;
% p2 = plot(1:BUFFER_SIZE, Accel_data(2,:),'b');
% hold on;
% grid on;
% ylabel ("Accel Input");
% title(['Accel Input with motion and non-motion areas']);
%
% subplot(4,1,2)
% p1 = plot(1:BUFFER_SIZE, training_window_size(1,1:BUFFER_SIZE),'r');
% hold on;
% grid on;
% p2 = plot(1:BUFFER_SIZE, training_window_size(2,1:BUFFER_SIZE),'b');
% hold on;
% grid on;
% ylabel ("Training Window Size");
% title(['Training Window Size']);
%
% subplot(4,1,3)
% p1 = plot(1:BUFFER_SIZE, Adaptive_Threshold(1,1:BUFFER_SIZE),'r');
% hold on;
% grid on;
% p2 = plot(1:BUFFER_SIZE, Adaptive_Threshold(2,1:BUFFER_SIZE),'b');
% hold on;
% grid on;
% ylabel ("Adaptive Threshold");
% title(['Adaptive Threshold']);
%
% subplot(4,1,4)
% p1 = plot(1:BUFFER_SIZE, Accel_motion_data(1,:),'r');
% hold on;
% grid on;
% p2 = plot(1:BUFFER_SIZE, Accel_motion_data(2,:),'b');
% hold on;
% grid on;
% ylabel ("Motion Identified Accel Output");
% title(['Motion Identified Accel Output']);
%
% hfig=(figure);
% scrsz = get(0,'ScreenSize');
% set(hfig,'position',scrsz);
%
% subplot(2,1,1)
% p1 = plot(1:BUFFER_SIZE, Accel_data(1,1:BUFFER_SIZE),'r');
% hold on;
% grid on;
% p2 = plot(1:BUFFER_SIZE, Accel_data(2,1:BUFFER_SIZE),'b');
% hold on;
% grid on;
% ylabel ("Zoomed in Accel-data");
% title(['Zoomed in Accel-data']);
%
% subplot(2,1,2)
% p1 = plot(1:BUFFER_SIZE, Accel_motion_data(1,1:BUFFER_SIZE),'r');
% hold on;
% grid on;
% p2 = plot(1:BUFFER_SIZE, Accel_motion_data(2,1:BUFFER_SIZE),'b');
% hold on;
% grid on;
% ylabel ("Zoomed in Accel-motion-data");
% title(['Zoomed in Accel-motion-data']);
%
% hfig=(figure);
% scrsz = get(0,'ScreenSize');
% set(hfig,'position',scrsz);

subplot(1,1,1)
p1 = plot(motion(1,:), motion(2,:),'r');
hold on;
grid on;
ylabel ("Y");
xlabel ("X");
title(['Actual Drawn Character']);
