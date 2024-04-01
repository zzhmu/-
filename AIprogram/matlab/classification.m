clear
clc
% n_data=ones(2,100);
data_number=100
x0=randn(2,data_number)+2;
y0=zeros(1,data_number);
x1=randn(2,data_number)-2;
y1=ones(1,data_number);
x=[x0,x1];
% x=x./(sum(x));
y=[y0,y1]-0.5;
% y=[y;~y];

lr=0.05
lmd=0
W1=randn(10, 2);
W2=randn(1, 10);
% subplot(121)
% scatter(x(1,:),x(2,:))
b1=randn(10,1);
b2=randn(1,1);
a0=x;
% for j=1:100
for i=1:100
    z1=W1*a0+b1;
    a1=F(z1);
    z2=W2*a1+b2;
    a2=F(z2);
%     loss=L(a2,y);
    
    derivatives2=dL(a2,y).*dF(z2);
    derivatives1=dF(z1).*(W2'*derivatives2);
    dL_dW2=derivatives2*a1';
    dL_dW1=derivatives1*a0';
    W2=W2-lr*dL_dW2-lr*lmd*W2;
    W1=W1-lr*dL_dW1-lr*lmd*W1;
    b1=b1-lr*sum(derivatives1,2)/2/data_number;
    b2=b2-lr*sum(derivatives2,2)/2/data_number;

end
% end
% m=[-2;-2];
m=[-2;-2];
z1=W1*m+b1;
a1=F(z1);
z2=W2*a1+b2;
a2=F(z2)
function activation=F(x)
    activation=1./(1+exp(-x));
%     activation=max(0,x);
end

function derivatives=dF(x)
    derivatives=exp(-x)./(1+exp(-x)).^2;
%     derivatives=(x>0).*x;

end

function loss=L(pre,tar)
    loss=-tar(1,:).*log(pre(1,:))-tar(2,:).*log(pre(2,:));
end

function dloss=dL(pre,tar)
    dloss=-tar./pre;
end