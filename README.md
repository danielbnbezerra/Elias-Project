# Elias: Ferramenta de Prognóstico Hidrológico com Redes Neurais

Elias é uma aplicação de desktop desenvolvida em Python para prognóstico de vazão em bacias hidrográficas. Através de uma interface gráfica intuitiva, permite a pesquisadores e engenheiros carregarem séries temporais próprias, configurar, treinar e avaliar modelos de redes neurais avançados sem necessidade de programar.

---

## Principais Funcionalidades

- **Interface gráfica amigável** construída em CustomTkinter.
- **Suporte a múltiplos modelos de Deep Learning:**
  - N-BEATS
  - N-HiTS
  - LHC (LSTM Customizado em PyTorch Lightning)
- **Upload simplificado** de séries temporais em `.csv`.
- **Geração automática** de covariáveis e divisão configurável entre treino/validação.
- **Configuração de hiperparâmetros:** modos pré-configurados ou ajuste manual.
- **Visualização detalhada** com gráficos comparativos, curva de aprendizado e resíduos.
- **Exportação abrangente:** relatórios completos em PDF, gráficos em alta resolução, modelos treinados e scripts reprodutíveis.

---

## Tecnologias Utilizadas

- Python 3.11+
- CustomTkinter, Tkinter
- Darts, PyTorch, PyTorch Lightning
- Pandas, NumPy, Scikit-learn
- Matplotlib, ReportLab
- Statsmodels, Pymannkendall

---

## Instalação

`pip install -r requirements.txt`
`pyinstaller -F --noconsole C:\Users\Usuário\PycharmProjects\Elias-Project\main.py --hidden-import pytorch_lightning --add-data="D:\Dan\Programas\miniconda3\envs\tcc\Lib\site-packages\pytorch_lightning\version.info;pytorch_lightning" --add-data="D:\Dan\Programas\miniconda3\envs\tcc\Lib\site-packages\lightning_fabric\version.info;lightning_fabric" --exclude-module xgboost`
