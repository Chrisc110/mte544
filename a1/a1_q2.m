psi = -pi/2;
theta = -pi/3;
gamma = pi/4;
alpha = pi/4;

l3 = 1;
l2 = 1;
l1 = 1;
x  = 1;
y  = 1;

G_e3 = [cos(psi),   -sin(psi),   l3; sin(psi),   cos(psi),   0; 0, 0, 1];
G_32 = [cos(theta), -sin(theta), l2; sin(theta), cos(theta), 0; 0, 0, 1];
G_21 = [cos(gamma), -sin(gamma), l1; sin(gamma), cos(gamma), 0; 0, 0, 1];
G_1s = [cos(alpha), -sin(alpha),  x; sin(alpha), cos(alpha), y; 0, 0, 1];

G_es = G_e3*G_32*G_21*G_1s;

%%%%%%%%%%% Q2b %%%%%%%%%%%%
p1_e = [0;0;1];
p2_e = [1;2;1];

p1_s = G_es * p1_e
p2_s = G_es * p2_e

