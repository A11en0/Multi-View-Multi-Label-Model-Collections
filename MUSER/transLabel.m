function [Yn] = transLabel(Y, params)
%TRANSLABEL 此处显示有关此函数的摘要
%   此处显示详细说明;
    Yn = Y;
    if params==0
        for i=1: size(Y,1)
            for j=1: size(Y,2)
                if( Yn(i,j)==-1)
                    Yn(i,j)=0;
                else
                    Yn(i,j)=1;
                end
            end
        end
    elseif params==-1
        for i=1: size(Y,1)
            for j=1: size(Y,2)
                if( Yn(i,j)<=0)
                    Yn(i,j)=-1;
                else
                    Yn(i,j)=1;
                end
            end
        end   
    else
        for i=1: size(Y,1)
            for j=1: size(Y,2)
                if( Yn(i,j)<=params)
                    Yn(i,j)=0;
                else
                    Yn(i,j)=1;
                end
            end
        end  
    end
    
end

