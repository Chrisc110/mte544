beta1 = pi/2;
beta2 = (7*pi)/6;
beta3 = 11*pi/6;

l = 0.25;
r = 0.1;

x1 = l;
y1 = 0;
x2 = -l*sin(pi/6);
y2 =  l*cos(pi/6);
x3 = -l*sin(pi/6);
y3 = -l* cos(pi/6);

syms vx vy omega theta t

%4bI
u1 = -2;
u2 = 1;
u3 = 1;

% 4bII1
% u1 = 10*cos(theta + pi/2) + 10*3^(1/2)*sin(theta + pi/2);
% u2 = 10*cos(theta + (7*pi)/6) + 10*3^(1/2)*sin(theta + (7*pi)/6)
% u3 = 10*cos(theta + (11*pi)/6) + 10*3^(1/2)*sin(theta + (11*pi)/6);

%4bII2
% u1 = 10*cos(t)*sin(theta + pi/2) + 10*cos(theta + pi/2)*sin(t);
% u2 = 10*cos(t)*sin(theta + (7*pi)/6) + 10*cos(theta + (7*pi)/6)*sin(t);
% u3 = 10*cos(t)*sin(theta + (11*pi)/6) + 10*cos(theta + (11*pi)/6)*sin(t);

G = 1/r * [cos(theta + beta1), sin(theta + beta1), x1*sin(beta1)-y1*cos(beta1);
           cos(theta + beta2), sin(theta + beta2), x2*sin(beta2)-y2*cos(beta2);
           cos(theta + beta3), sin(theta + beta3), x3*sin(beta3)-y3*cos(beta3)];

%%%%%%%%%%%%%%% 4a %%%%%%%%%%%%%%%%%
u = [u1; u2; u3];

qdot = inv(G) * u;

vx = qdot(1);
vy = qdot(2);
omega = qdot(3);

% Integrate to get positions and orientation
x = int(vx, 0, t);
y = int(vy, 0, t);
theta1 = int(omega, 0, t);

% Time vector
time = linspace(0, 30, 1000);  % Time from 0 to 30 seconds

% Substitute time into the expressions
x_vals = double(subs(x, t, time));
y_vals = double(subs(y, t, time));
theta_vals = double(subs(theta1, t, time));


% Plot the results
figure;

% Plot x position vs time
subplot(4,1,1);
plot(time, x_vals, 'r', 'LineWidth', 2);
xlabel('Time [s]');
ylabel('x [m]');
title('x Position vs Time');

% Plot y position vs time
subplot(4,1,2);
plot(time, y_vals, 'g', 'LineWidth', 2);
xlabel('Time [s]');
ylabel('y [m]');
title('y vs Time');

% Plot theta position vs time
subplot(4,1,3);
plot(time, theta_vals, 'b', 'LineWidth', 2);
xlabel('Time [s]');
ylabel('theta [rad]');
title('theta vs Time');

% Plot theta position vs time
subplot(4,1,4);
plot(x_vals, y_vals, 'm', 'LineWidth', 2);
xlabel('x [m]');
ylabel('y [m]');
title('x vs y');

% Show grid for clarity
grid on;


%%%%%%%%%%%%% 4b %%%%%%%%%%%%%%%
% 1)
qdot = [1; tan(pi/3); 0];
u = G * qdot;

% 2)
qdot = [sin(t); cos(t); 0];
u = G * qdot;

