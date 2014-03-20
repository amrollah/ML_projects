clc;clear; close all
%better
path(path, 'SVM_multiclass\SM_KM');
path(path, 'libsvm-3.172\matlab');

constructDictionary = false;
constructTrainingFeatures = false;
constructTestFeatures=true;
Bayesian = true;
SVM = false;

d =importdata('handout\training.csv');
cityCode=d.data(:,1);
countryCode=d.data(:,2);
cityNames=lower(d.textdata);

mergeCode=str2num(strcat(num2str(cityCode),num2str(countryCode)));

%% constructing Dictionary
if(constructDictionary)
    %construct the dictionary
    Dic_extra=strsplit(cityNames{1,1},' ');
    for i=2:size(cityNames,1)
        temp=strsplit(cityNames{i,1},' ');
        for j=1:size(temp,2)
            a=ismember(temp{1,j},Dic_extra);
            if(~a)
                Dic_extra = horzcat(Dic_extra,temp{1,j});
            end
        end
    end


  num=1;
  Dic{num} = Dic_extra{1,1};
  num = num + 1;
  
  for l=1:size(Dic_extra,2)
      b=1;
      LevenDist=zeros((size(Dic,2)-1),1); %without consider the word itself
      for i=1:size(Dic,2)
            LevenDist(i,1)= levenshtein(Dic_extra{1,l},Dic{1,i});
      end
      
      [f inx]=min(LevenDist);
      if((f <= 2 && length(Dic_extra{1,l}) > 3) || f == 0)
          b=0;
      end
      if(b)
          Dic{num} = Dic_extra{1,l};
          num=num+1;
      end
  end
  
save Dic Dic
end

load Dic

%% constructing the basic feature matrix

if(constructTrainingFeatures)
xapp = zeros( size(cityNames,1) ,size(Dic,2));
LevenDist=zeros(1,size(Dic,2));
for i=1:size(xapp,1)
    
    C = strsplit(cityNames{i},' ');
    for j=1:length(C)
    b=0;    
      
      [flag inx]=ismember(C{1,j},Dic);
      if(flag) 
          b=1;
      else
          
      LevenDist = zeros(size(Dic,2),1);  
      for k=1:size(Dic,2)
            LevenDist(k,1)= levenshtein(C{1,j},Dic{1,k});
      end
      
      [f inx]=min(LevenDist);
      if((f <= 2 && length(C{1,j}) > 3))
          b=1;
      end
      
      end
      
      if(b)
          xapp(i,inx)=xapp(i,inx)+1;
      end
    end
    i
end
save xapp xapp
end
load xapp


%% reforming the city codes


%% classification
yapp=mergeCode;
ycity=cityCode;

%country code
O1 = NaiveBayes.fit(xapp, yapp, 'Distribution', 'mn');
%city code
conUniq=unique(yapp);
for i=1:length(conUniq)
      ind=find(conUniq(i)==yapp);
      x=xapp(ind,:);
      y=ycity(ind);
      Classi{i}=NaiveBayes.fit(x, y, 'Distribution', 'mn');
end
    
%% testing

[a, cityNames, alldata] = xlsread('handout\testing.csv');
cityNames=lower(cityNames);

if(constructTestFeatures)

xtest = zeros( size(cityNames,1) ,size(Dic,2));
for i=1:size(xtest,1)
    C = strsplit(cityNames{i},' ');
    for j=1:length(C)

      b=0;    
      [flag inx]=ismember(C{1,j},Dic);
      if(flag) 
          b=1;
      else
        LevenDist = zeros(size(Dic,2),1);  
        for k=1:size(Dic,2)
            LevenDist(k,1)= levenshtein(C{1,j},Dic{1,k});
        end
        [f inx]=min(LevenDist);
        if((f <= 2 && length(C{1,j}) > 3))
          b=1;
        end
      end
      
      if(b)
          xtest(i,inx)=xtest(i,inx)+1;
      end
    end
    i
end
save xtest2 xtest 
end
load xtest
%% prediction

if(Bayesian)
canoutPut=zeros(size(xtest,1),2);
C1 = O1.predict(xtest);
C1=num2str(C1);
city=str2num(C1(:,1:6));
country=str2num(C1(:,7:9));
outPut(:,2)=country;
outPut(:,1)=city;
dlmwrite('val-pred.txt',outPut,'delimiter',',','precision','%d')
end

%% Changing the labels to 1:15 
realCode=yapp;
repCode=zeros(size(yapp));
code=unique(realCode);
num=1;
for i=1:length(code)
    inx=find(realCode==code(i));
    repCode(inx,1)=num;
    num=num+1;
end

%%
if(SVM)
    yapp=repCode;
% %-----------------------------------------------------
%   Learning and Learning Parameters
% c = 1000;
% lambda = 1e-7;
% kerneloption= 2;
% kernel='gaussian';
% verbose = 1;
% 
% %---------------------One Against All algorithms----------------
% nbclass=3;
% [xsup,w,b,nbsv]=svmmulticlassoneagainstall(xapp',yapp,nbclass,c,lambda,kernel,kerneloption,verbose);
% 
% 
% [xtesta1,xtesta2]=meshgrid([-4:0.1:4],[-4:0.1:4]);
% [na,nb]=size(xtesta1);
% xtest1=reshape(xtesta1,1,na*nb);
% xtest2=reshape(xtesta2,1,na*nb);
% xtest=[xtest1;xtest2]';
% [ypred,maxi] = svmmultival(xtest,xsup,w,b,nbsv,kernel,kerneloption);


result = multisvm2(xapp,yapp,xtest);

end

% c = 1000;
% lambda = 1e-7;
% kerneloption= 1;
% kernel='poly';
% verbose = 0;
% 
% %---------------------One Against One algorithms----------------
% nbclass=122;
% kerneloptionm.matrix=svmkernel(xapp,kernel,kerneloption);
% [xsup,w,b,nbsv,classifier,pos]=svmmulticlassoneagainstone([],yapp,nbclass,c,lambda,'numerical',kerneloptionm,verbose);
% 
% kerneloptionm.matrix=svmkernel(xtest,kernel,kerneloption,xapp(pos,:));
% [ypred,maxi] = svmmultivaloneagainstone([],[],w,b,nbsv,'numerical',kerneloptionm);
% 
% end
% 
% 
% load ypred
% code=unique(ypred);
% 
% for i=1:length(code)
%     inx=find(ypred==code(i));
%     a=find(ypred(inx(1))==repCode);
%     ypred(inx)=realCode(a(1));
% end
% 
% C1=ypred;
% % Write 
% C1=num2str(C1);
% city=str2num(C1(:,1:6));
% country=str2num(C1(:,7:9));
% outPut(:,2)=country;
% outPut(:,1)=city;
% dlmwrite('val-pred.txt',outPut,'delimiter',',','precision','%d')


