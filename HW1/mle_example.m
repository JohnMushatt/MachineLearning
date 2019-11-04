
m = 3;
subplot(2,2,1)
x = randn(1,1)+m;
ms= -4:0.0001:10;
for i=1:length(ms)
    y(i) = -0.5*(x-ms(i))'*(x-ms(i));
end

plot(ms,y,'k','linewidth',3);hold on
plot(x,0.01*randn(size(x)),'rx','markersize',12,'linewidth',3);
xlabel('m')
ylabel('log likelihood')
title('N=1')
set(gca,'fontsize', 18)
grid minor
ylim([-1000 1])

subplot(2,2,2)
x = randn(10,1)+m;

for i=1:length(ms)
    y(i) = -0.5*(x-ms(i))'*(x-ms(i));
end

plot(ms,y,'k','linewidth',3);hold on
plot(x,0.01*randn(size(x)),'rx','markersize',12,'linewidth',3);
xlabel('m')
ylabel('log likelihood')
title('N=10')
set(gca,'fontsize', 18)
grid minor
ylim([-1000 1])

subplot(2,2,3)
x = randn(100,1)+m;

for i=1:length(ms)
    y(i) = -0.5*(x-ms(i))'*(x-ms(i));
end

plot(ms,y,'k','linewidth',3);hold on
plot(x,0.01*randn(size(x)),'rx','markersize',12,'linewidth',3);
xlabel('m')
ylabel('log likelihood')
title('N=100')
set(gca,'fontsize', 18)
grid minor
ylim([-1000 1])

subplot(2,2,4)
x = randn(800,1)+m;

for i=1:length(ms)
    y(i) = -0.5*(x-ms(i))'*(x-ms(i));
end

plot(ms,y,'k','linewidth',3);hold on
plot(x,0.01*randn(size(x)),'rx','markersize',12,'linewidth',3);
xlabel('m')
ylabel('log likelihood')
title('N=800')
set(gca,'fontsize', 18)
grid minor
ylim([-1000 1])