function [var_calculo] = calculo_LEA(ypk,duk,Par)
    
pressao_intake =   (ypk(1)*Par.Vol_1.rho_1*Par.g);                                                       
pressao_fundo = (Par.Vol_1.rho_1*Par.g*(Par.Vol_1.h_r-Par.Vol_1.h_p)+Par.Vol_1.rho_1*Par.g*ypk(1));      
vazao_choke = (((duk(2)).*((ypk(2)-duk(3)).^0.5)*Par.k_choke));
vazao_reservatorio = (Par.IP*(duk(4) - pressao_fundo));

var_calculo = [pressao_intake pressao_fundo vazao_choke vazao_reservatorio]; 
end

