load xtest
load xapp
dlmwrite('xapp.txt',xapp,'delimiter',',','precision','%d')
dlmwrite('xtest.txt',xtest,'delimiter',',','precision','%d')
mergeCode=str2num(strcat(num2str(cityCode),num2str(countryCode)));
dlmwrite('yapp.txt',mergeCode,'delimiter',',','precision','%d')
