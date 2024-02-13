L = 1;
x = linspace(0,L,100);
t = linspace(0,1.5,421);
global n p
rng(1)
samples = 1000
n_val = -0.5+rand(samples,1);
p_val = -0.5+rand(samples,1);
sol =zeros(samples,421,100);
m=0;
for i=1:samples
    i
    n = n_val(i,1);
    p = p_val(i,1);
    sol1 = pdepe(m,@burgerpde,@burgeric,@burgerbc,x,t);
    sol(i,:,:) = sol1;
end

% mat_ics =zeros(1000,241,241);
% global n p

% for i=1:1000
%     n = n_val(i,1);
%     p = p_val(i,1);
%     mat_ics1 = repmat(sin(n*pi*x)+ cos(p*pi*x),241,1);
%     mat_ics(i,:,:) = mat_ics1;
% end
    
    
save u_sol2_burger.mat t x sol