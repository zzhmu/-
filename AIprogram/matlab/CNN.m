clear
clc

fid = fopen('train-labels.idx1-ubyte', 'rb');
trainLabels = fread(fid, inf, 'uint8', 'l');
trainLabels = trainLabels(9:end);
fclose(fid);
% read test labels
fid = fopen('t10k-labels.idx1-ubyte', 'rb');
testLabels = fread(fid, inf, 'uint8', 'l');
testLabels = testLabels(9:end);
fclose(fid);
% read train images
fid = fopen('train-images.idx3-ubyte', 'rb');
trainImages = fread(fid, inf, 'uint8', 'l');
trainImages = trainImages(17:end);
fclose(fid);
trainData = reshape(trainImages, 28,28, size(trainImages,1) / 784);
% read train images
fid = fopen('t10k-images.idx3-ubyte', 'rb');
testImages = fread(fid, inf, 'uint8', 'l');
testImages = testImages(17:end);
fclose(fid);
testData = reshape(testImages, 28,28, size(testImages,1) / 784);
% clear testImages
% clear trainImages
%%
% clear
% clc
EPOCH = 1 
BATCH_SIZE = 50
LR = 0.001  
filter1=rand(5,5,16);
filter2=rand(5,5,16,32);
b1=rand(5,5,16);
b2=rand(5,5,16,32);
stride=1;

conv2d1=zeros(28,28,16);
conv2d2=zeros(14,14,32);
huiju1=zeros(14,14,16);
huiju2=zeros(7,7,32);
for i=1:500
    fg=padarray(trainData(:,:,i), [2 2]);
    lb=trainLabels(i);
    %第一次卷积
    for m=1:28
        for n=1:28
            conv2d1(m,n,:)=sum(sum(fg(m:m+4,n:n+4).*filter1+b1));
        end
    end
    z1=ReLU(conv2d1);
    %第一次汇聚
    for p=1:14
        for q=1:14
        huiju1(p,q,:)=max(max(z1(2*p-1:2*p,2*q-1:2*q,:)));
        end
    end
    %第二次卷积
    huiju1=padarray(huiju1, [2 2]);
    for m=1:14
        for n=1:14
            conv2d2(m,n,:,:)=sum(sum(sum(huiju1(m:m+4,n:n+4,:).*filter2+b2)));
        end
    end
    z2=ReLU(conv2d2);
    %第二次汇聚
    for p=1:7
        for q=1:7
          huiju2(p,q,:)=max(max(z2(2*p-1:2*p,2*q-1:2*q,:)));
        end
    end
    huiju2=reshape(huiju2,7*7*32,1);
%     filter(1,16,5,1,2,fg);
end









function activation=ReLU(x)
    activation=max(0,x);
end