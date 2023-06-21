%% Q1 part 1
clc
clear all
my_pic0=imread('D:\Term4\Signal khalaj\HWcu\2\q1.jpg');
my_pic=my_pic0(:,:,1);
n1=1;
fft_my_pic=fftshift(fft2(double(my_pic)));
X=-282:1:282;
Y=X;
kc=1;
[X,Y]=meshgrid(X,Y);
sigma=300;
z=1./(2.*pi.*sigma.^2).*exp((-X.^2-Y.^2)./(2.*sigma));
z=ones(size(z)).*max(z,[],"all")-z;
z=z.*(1/max(z,[],"all"));
filtered=z.*fft_my_pic;
final=n1*(abs(ifft2(fftshift(kc.* filtered+fft_my_pic))));
%===================================================
my_pic2=my_pic0(:,:,2);
fft_my_pic2=fftshift(fft2(double(my_pic2)));
filtered2=z.*fft_my_pic2;
final2=n1*(abs(ifft2(fftshift(kc.* filtered2+fft_my_pic2))));
%===================================================
my_pic3=my_pic0(:,:,3);
fft_my_pic3=fftshift(fft2(double(my_pic3)));
filtered3=z.*fft_my_pic3;
final3=n1*(abs(ifft2(fftshift(kc.* filtered3+fft_my_pic3))));
%===================================================
result=zeros(565,565,3);
result(:,:,1)=final;
result(:,:,2)=final2;
result(:,:,3)=final3;
montage({ uint8(my_pic0),uint8(result)}, 'Size',[1,2]);
maxa=max(result,[],"all");
mina=min(result,[],"all");
result=(result-mina.*ones(565,565,3)).*(255./maxa);
result=result+1.*(ones(565,565,3));
result=uint8(result);
%montage({ uint8(my_pic0),uint8(result)}, 'Size',[1,2]);
%% plotting filter
subplot(2,1,1);
mesh(z);
subplot(2,1,2);
mesh(z);
%% plot fft2
subplot(2,1,1)
mesh(abs(fftshift(fft2(double(result(:,:,2))))));
title("fft2 after filtering")
subplot(2,1,2)
mesh(abs(fftshift(fft2(double(my_pic0(:,:,2))))));
title("fft2 before filtering")
%% part2
clc
clear all
my_pic0=imread('D:\Term4\Signal khalaj\HWcu\2\q1.jpg');
my_pic=my_pic0(:,:,1);
n1=1;
n2=0.5;
fft_my_pic=fftshift(fft2(double(my_pic)));
X=-282:1:282;
Y=X;
[X,Y]=meshgrid(X,Y);
z=X.^2+Y.^2;
filtered=z.*fft_my_pic;
final=n1*(abs(ifft2(fftshift(4.*pi^2.* filtered))));
%===================================================
my_pic2=my_pic0(:,:,2);
fft_my_pic2=fftshift(fft2(double(my_pic2)));
filtered2=z.*fft_my_pic2;
final2=n1*(abs(ifft2(fftshift(4.*pi^2.* filtered2))));
%===================================================
my_pic3=my_pic0(:,:,3);
fft_my_pic3=fftshift(fft2(double(my_pic3)));

filtered3=z.*fft_my_pic3;
final3=n1*(abs(ifft2(fftshift(4.*pi^2.* filtered3))));
%===================================================
result=zeros(565,565,3);
result(:,:,1)=final;
result(:,:,2)=final2;
result(:,:,3)=final3;
maxa=max(result,[],"all");
mina=min(result,[],"all");
result=(result-mina.*ones(565,565,3)).*(800./maxa);
result=result.*(result>20);
result=result+10.*(ones(565,565,3));
result=uint8(result);
highpassed=result;
result=double(result);
k=0.3;
result=k.*result+(1).*double(my_pic0);
maxa=max(result,[],"all");
mina=0;
result=(result-mina.*ones(565,565,3)).*(255./maxa);
max(result,[],"all");
montage({ uint8(my_pic0),uint8(result),uint8(highpassed)}, 'Size',[1,3]);

%mesh(highpassed(:,:,2))
%mesh(z);
%% Q2
clear
clc
my_pic2=( imread('D:\Term4\Signal khalaj\HWcu\2\pic.jpg'));
pout_imadjust = adapthisteq(my_pic2,'NBins',32384,'Distribution','uniform','ClipLimit',0.01);
imshowpair(my_pic2,pout_imadjust,'montage');
%% histogram of images
subplot(2,1,1);
imhist(uint8(pout_imadjust));
subplot(2,1,2);
imhist(uint8(my_pic2));
%% Q3 adding photos in frequency domain :
clc; clear; close all;
einstein = imread("D:\Term4\Signal khalaj\HWcu\2\einstein.jpg");
marilyn = imread("D:\Term4\Signal khalaj\HWcu\2\marilyn.jpg");
einstein=rgb2gray(einstein);
marilyn=marilyn(1:234,1:225);
einstein = imrotate(einstein,10,'bilinear','crop');
fft_my_pic2=fftshift(fft2(double(einstein)));
fft_my_pic3=fftshift(fft2(double(marilyn)));
X=-112:1:112;
Y=-116:1:117;
[X,Y]=meshgrid(X,Y);
D0=80;
n=5;
n1=1;
z=X.^2+Y.^2;
param1=200;
param2=400;
z1=z>param1;
z2=z<param2;
z3=(param1<z)&(z<param2);
z1=z1-z3./2;
z2=z2-z3./2;
% sum in frequency domain
filtered2=z1.*fft_my_pic2+z2.*fft_my_pic3;
Output=(abs(ifft2(fftshift( filtered2))));
hp=(abs(ifft2(fftshift( z1.*fft_my_pic2))));
lp=(abs(ifft2(fftshift( z2.*fft_my_pic3))));
montage({uint8(hp), uint8(lp),uint8(Output)}, 'Size',[1,3]);
%% ploting filters
subplot (2,1,1);
mesh(z1);
title('Low pass filter');
subplot(2,1,2);
mesh(z2);
title('High pass filter');
%% Adding photos with 1/2 coefficeint
clc
clear
albert= imread("D:\Term4\Signal khalaj\HWcu\2\einstein.jpg");
marilyn = imread("D:\Term4\Signal khalaj\HWcu\2\marilyn.jpg");
albert=rgb2gray(albert);
marilyn=marilyn(1:234,1:225);
albert = imrotate(albert,10,'bilinear','crop');
albert_HI_AF=double(highpass(albert,80));
marilyn_LowKey=double(lowpass(marilyn,80));
hybrid=uint8((marilyn_LowKey+albert_HI_AF)./2);
% modifying contrast for better output
hybrid= adapthisteq(hybrid,'NBins',32384,'Distribution','uniform','ClipLimit',0.005);
montage({ uint8(albert_HI_AF),uint8(marilyn_LowKey),hybrid}, 'Size',[1,3]);
%% Adding photos with 1/2 coefficeint with matlab filters
low=imgaussfilt(marilyn,2);
high=1.25*albert- imgaussfilt(albert,4);
hybrid2=(high./1.25+low)./2;
montage({ uint8(high),uint8(low),uint8(hybrid2)}, 'Size',[1,3]);
%%
function preh=highpass(input,sigma)
X=-112:1:112;
Y=-116:1:117;
[X,Y]=meshgrid(X,Y);
z=1./(2.*pi.*sigma.^2).*exp((-X.^2-Y.^2)./(2.*sigma));

z=ones(size(z)).*max(z,[],"all")-z;
%mesh(z);
coef=230;
albert = double(input);
albert=albert(1:234,1:225);
%mesh((abs(ifft2(fftshift(z.*fftshift(fft2(albert)))))));
preh=(abs(ifft2(fftshift(z.*fftshift(fft2(albert))))));
%change image pixels rang to 0-230
preh=preh.*( coef./  (max(preh,[],"all")   ));
prehdark=preh<30;
prehmid=(preh>30)&(preh<60);
preh=preh+30.*prehdark+prehmid.*15;
preh=uint8(preh);
end

function prel=lowpass(input,sigma)
X=-112:1:112;
Y=-116:1:117;
[X,Y]=meshgrid(X,Y);
z=1./(2.*pi.*sigma.^2).*exp((-X.^2-Y.^2)./(2.*sigma));
%mesh(z);
coef=230;
marilyn = double(input);
marilyn=marilyn(1:234,1:225);
%mesh((abs(ifft2(fftshift(z.*fftshift(fft2(marilyn)))))));
prel=(abs(ifft2(fftshift(z.*fftshift(fft2(marilyn))))));
%change image pixels rang to 0-230
prel=prel.*( coef./  (max(prel,[],"all")   ));
end