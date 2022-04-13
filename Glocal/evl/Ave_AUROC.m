function ret=Ave_AUROC(pred,target)

label_num=size(target,2);

ret=zeros(label_num,1);
for aa=1:label_num
    [tp,fp]=roc(target(:,aa),pred(:,aa));
    
    ret(aa,1)=auroc(tp,fp);
end

ret=mean(ret);