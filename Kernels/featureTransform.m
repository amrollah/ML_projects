%index - index of the feature that you want to transform it
%X - all the input data

function X=featureTransform(X,index,d,increment)

for i=1:length(d)
    inx1=find(X(:,index)==d(i));
    X(inx1,index)=(i-1)*increment+X(inx1,13)/max(X(inx1,13))*(increment);
end

end