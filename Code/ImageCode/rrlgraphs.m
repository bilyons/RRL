A = readtable('dat_comb_eps_05');
B = readtable('dat_comb_eps_05_obs');
C = readtable('dat_comb_eps_05_tri');
D = readtable('dat_comb_eps_05_tbc');
E = readtable('dat_comb_hi_squ');
F = readtable('dat_2_hi_squ');
G = readtable('dat_comb_hi_obs');
H = readtable('dat_2_hi_obs');
I = readtable('dat_comb_hi_tri');
J = readtable('dat_2_hi_tri');
K = readtable('dat_comb_hi_tbc');
L = readtable('dat_2_hi_tbc');

Data1=zeros(21,21);
Data2=zeros(21,21);
Data3=zeros(21,21);
Data4=zeros(21,21);
Data5=zeros(21,21);
Data6=zeros(21,21);
Data7=zeros(21,21);
Data8=zeros(21,21);
Data9=zeros(21,21);
Data10=zeros(21,21);
Data11=zeros(21,21);
Data12=zeros(21,21);

for i = 1:441;
    Data1((A.Var1(i)+1), (A.Var2(i)+1)) = A.Var3(i);
    Data2((B.Var1(i)+1), (B.Var2(i)+1)) = B.Var3(i);
    Data3((C.Var1(i)+1), (C.Var2(i)+1)) = C.Var3(i);
    Data4((D.Var1(i)+1), (D.Var2(i)+1)) = D.Var3(i);
    Data5((E.Var1(i)+1), (E.Var2(i)+1)) = E.Var3(i);
    Data6((F.Var1(i)+1), (F.Var2(i)+1)) = F.Var3(i);
    Data7((G.Var1(i)+1), (G.Var2(i)+1)) = G.Var3(i);
    Data8((H.Var1(i)+1), (H.Var2(i)+1)) = H.Var3(i);
    Data9((I.Var1(i)+1), (I.Var2(i)+1)) = I.Var3(i);
    Data10((J.Var1(i)+1), (J.Var2(i)+1)) = J.Var3(i);
    Data11((K.Var1(i)+1), (K.Var2(i)+1)) = K.Var3(i);
    Data12((L.Var1(i)+1), (L.Var2(i)+1)) = L.Var3(i);    
end

comp1 = Data5-Data6;
comp2 = Data7-Data8;
comp3 = Data9-Data10;
comp4 = Data11-Data12;