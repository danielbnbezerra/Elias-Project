# DEFINIR DIRETÓRIO DOS ARQUIVOS .g2.tar
dir_arq = "DadosGFS"

# DEFINIR ORDER ID

orderID = "HAS012378593"

# DEFINIR DATAS

dia_inicio = 10
mes_inicio = 4
ano_inicio = 2014

dia_fim = 9
mes_fim = 4
ano_fim = ano_inicio+1

dias = ["0"+str(x) for x in list(range(1,10))]+[str(x) for x in list(range(10,32))]
meses = ["0"+str(x) for x in list(range(1,10))]+[str(x) for x in list(range(10,13))]
anos = [str(x) for x in list(range(ano_inicio,ano_fim+1))]

# BAIXAR ARQUIVOS COM POWERSHELL 7.3

f = open("baixar_com_powershell_"+orderID+".txt", "a")
f.write("$baseUri = "+"\"https://www.ncei.noaa.gov/pub/has/model/"+orderID+"\"\n")
f.write("$files = @("+"\n")
for ano in anos:
    if (ano==str(ano_inicio)):
        for mes in meses:
            for dia in dias:
                if ((int(mes)==mes_inicio and int(dia)>=dia_inicio) or (int(mes)>mes_inicio)):
                    f.write("    @{\n        Uri = \"$baseUri/gfsanl_4_"+ano+mes+dia+"00.g2.tar"+"\"\n")
                    f.write("        OutFile = \""+dir_arq+"\\gfsanl_4_"+ano+mes+dia+"00.g2.tar"+"\"\n"+"    },\n")
    elif (ano==str(ano_fim)):
        for mes in meses:
            for dia in dias:
                if ((int(mes)==mes_fim and int(dia)<=dia_fim) or (int(mes)<mes_fim)):
                    f.write("    @{\n        Uri = \"$baseUri/gfsanl_4_"+ano+mes+dia+"00.g2.tar"+"\"\n")
                    if (int(mes)==mes_fim and int(dia)==dia_fim):
                        f.write("        OutFile = \""+dir_arq+"\\gfsanl_4_"+ano+mes+dia+"00.g2.tar"+"\"\n"+"    }\n")
                    else:
                        f.write("        OutFile = \""+dir_arq+"\\gfsanl_4_"+ano+mes+dia+"00.g2.tar"+"\"\n"+"    },\n")
                    
f.write(""")

$jobs = @()

foreach ($file in $files) {
    $jobs += Start-ThreadJob -Name $file.OutFile -ScriptBlock {
        $params = $using:file
        Invoke-WebRequest @params
    }
}

Write-Host "Downloads started..."
Wait-Job -Job $jobs

foreach ($job in $jobs) {
    Receive-Job -Job $job
}""")
f.close()
