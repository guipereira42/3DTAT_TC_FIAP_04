import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

#carga e tratamento dos dados
@st.cache_data
def carregar_dataframe():
    url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'
    df_list = pd.read_html(url)
    df_price_hist = df_list[2][1:]  # Supondo que a tabela desejada é a terceira na lista
    df_price_hist.columns = ['Data', 'Preço']
    df_price_hist['Data'] = pd.to_datetime(df_price_hist['Data'], format='%d/%m/%Y')
    df_price_hist['Preço'] = df_price_hist['Preço'].astype(float) / 100
    df_price_hist.set_index('Data', inplace=True)
    df_price_hist.sort_index(ascending=False, inplace=True)
    return df_price_hist


#criação da página introdução
def introducao():
    st.markdown("""            
        <h2 style="color: #fb8500; text-align: center; text-decoration: underline; font-weight: bold"> Estudo e Predição do Valor do Barril de Petróleo</h2>
        <h2 style="text-align: center;">Objetivo</h2>
            <p style="text-indent: 2em; text-align: justify; font-size: 20px">
                O petróleo é uma matéria-prima de extrema importância para a humanidade. Estudar, interpretar e até predizer o comportamento
                da variação do valor do barril dessa substância é de suma importância.<br>
            <p style="text-indent: 2em; text-align: justify; font-size: 20px">
                O objetivo deste estudo é apresentar uma análise exploratória dos dados disponibilizados pelo Instituto de Pesquisa Econômica 
                Aplicada (IPEA) e, a partir do aprendizado de máquina, predizer o valor do barril do petróleo a partir da sua série histórica.
        <h2 style="text-align: center;">Introdução</h2>
            <p style="text-indent: 2em; text-align: justify; font-size: 20px">
                Entre as commodities mais relevantes do mundo, o petróleo se destaca entre as mais utilizadas.
                Historicamente, a mais de 50 anos ele é uma das principais fontes de energia. Além disso, tratando-se de derivados de petróleo temos 
                uma gama de produtos que são essenciais mundialmente como: gasolina, diesel, asfalto, tintas, comésticos e plástico.
            <p style="text-indent: 2em; text-align: justify; font-size: 20px">
                Dada a sua importância, vamos entender como o preço dessa matéria-prima é definido.
            <p style="text-indent: 2em; text-align: justify; font-size: 20px">
                Por ser uma comodity, o seu valor mundial é padronizado de acordo com a sua classificação em <b>BRENT</b> e <b>WTI</b>. Segue a explicação e comparação desses tipos de petróleo:
            <ul>
                <li><p style="font-size: 20px"><span style="font-weight: bold; color: orange;">Brent: </span> considerado mais comum, serve como referência para cerca de dois terços do petróleo mundial.</li>
                <li><p style="font-size: 20px"><span style="font-weight: bold; color: orange">West Texas Intermediate: </span> petróleo com qualidade superior ao Brent. É extraído no Golfo do México nos Estados Unidos da América.</li>
            </ul>

        <div style="text-align: center;">
        <img src="https://www.focus-economics.com/app/uploads/2022/10/focuseconomics_wti_vs_brent.jpg" alt="Petrol" style="width:50%;padding:10px;">
        <figcaption style="font-style: italic; color: #888;">Disponível em: https://www.focus-economics.com/blog/difference-between-wti-and-brent/</figcaption>
        <br>
            <p style="text-indent: 2em; text-align: justify; font-size: 20px">                    
                De forma bem simplista, o que faz o valor desses tipos de petróleo variar é antiga lei da oferta e da procura que é regida por alguns fatores. Vamos conhecer alguns deles.
        <h4 style="text-align: justify;">Fatores que influenciam o valor do petróleo</h4>
        <ul>
            <li><p style="font-size: 20px; text-align: justify"><span style="font-weight: bold; color: orange;">Aumento do preço: </span> redução proposital da produção pelos países dominantes da comodity para 
                aumentar a demanda e consequentemente o valor. Tensões políticas em  países importantes na produção mundial de petróleo como, por exemplo, a região do Oriente Médio, característica por conflitos 
                étnico-religiosos. A alta do dólar visto que o barril do petróleo é comercializado com base na moeda americana.</li>
            <li><p style="font-size: 20px; text-align: justify"><span style="font-weight: bold; color: orange">Redução do preço: </span> analogamente a escassez do petróleo, quando há o excesso de produção, ou seja, 
                aumento da oferta, o valor do barril de petróleo reduz. Em 2020, época da pandemia de COVID-19, o barril do WTI chegou a ficar com o valor negativo de U$$ -37,00. A redução do valor do dólar e o avanço 
                tecnológico são fatores da redução do preço do petróleo. O primeiro se justifica pelo mesmo motivo do aumento da moeda, o segundo explica com o barateamento do custo de produção e transporte através do avanço 
                das tecnologias.</li>
        </ul>
        <h4 style="text-align: justify;">Qual o impacto do preço do petróleo na economia?</h4>
            <p style="text-indent: 2em; text-align: justify; font-size: 20px">
                Apesar de cadeia produtiva do petróleo ser extensa, o impacto com efeito mais visível em primeiro momento seria sobre os transportes em geral. O aumento nos valores dos combustíveis fósseis refletiria 
                diretamente no custo de produção da maioria dos produtos existentes atualmente.
            <p style="text-indent: 2em; text-align: justify; font-size: 20px">
                Para investidores, o preço do petróleo serve como um termômetro geral a econômia global, pois, afeta diretamente o desempenho das empresas.
                Se há uma alta do valor, há uma especulação que os insumos ficarão mais caros diminuindo a receita das empresas e possivelmente a redução de lucros.
                Já a baixa do valor do petróleo gera especulações de que os custos de produção de empresas dependentes da comodity reduzirá, melhorando o desempenho.
        <p style="text-indent: 2em; text-align: justify; font-size: 20px">
                Com todos esses fatores explicados, resaltando a importância que o valor do barril de petróleo tem para a população mundial, entraremos num profundo estudo para entender o comportamento dos dados 
                da evolução do preço do petróleo no decorrer do tempo e a partir de técnicas de tratamentos de dados e machine learning, identificar os padrões e tentar realizar predições para o valor futuro dessa 
                matéria-prima.

        <hr style="border: none; height: 1px; background-color: #fb8500; margin-top: 10px; margin-bottom: 10px;">
    """, unsafe_allow_html=True)
    
#criação 
def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df["rolling_std_7"] = df["Preço"].shift(5).rolling(window=7).std()
    df["rolling_mean_7"] = df["Preço"].shift(5).rolling(window=7).mean()
    for lag in range(5, 13):
      df[f"lag_{lag}"] = df["Preço"].shift(lag)

    df.dropna(inplace=True)
    return df


def analise():
    css = '''
        <style>
            .stTabs [data-baseweb="tab-list"] button {
                flex: 1;
                text-align: center;
                border-bottom: 3px solid transparent;
                padding-bottom: 15px;
            }
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
                border-bottom: 1px solid #fb8500;
            }
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size: 30px;
                color: #fb8500;
                font-weight: bold;
                margin: 0;
            }
        </style>
        '''
    st.markdown(css, unsafe_allow_html=True)

    df_price_hist = carregar_dataframe()
    
    analises = ['Fonte de Dados','Análise Exploratória']
    tabs = st.tabs(analises)

    with tabs[0]:
        st.markdown("""            
        <p style="text-indent: 2em; font-size: 20px">
            A base de dados foi extraída diretamente do site do Instituto de Pesquisa Econômica Aplicada (IPEA). Segue a lista de informações resumidas disponibilizadas no site: 
        <br>
        <ul>
            <li><p style="font-size: 20px";><span style="font-weight: bold; color: orange;">Nome: </span> Preço por barril do petróleo bruto Brent (FOB)(EIA366_PBRENT366);</li>
            <li><p style="font-size: 20px"><span style="font-weight: bold; color: orange">Frequência: </span> diária de 04/01/1986 até 01/07/2024;</li>
            <li><p style="font-size: 20px"><span style="font-weight: bold; color: orange">Unidade: </span> US$;</li>
            <li><p style="font-size: 20px"; text-align: justify><span style="font-weight: bold; color: orange">Comentário: </span> Preço por barril do petróleo bruto tipo Brent. Produzido no Mar do Norte (Europa), 
                Brent é uma classe de petróleo bruto que serve como benchmark para o preço internacional de diferentes tipos de petróleo. Neste caso, é valorado no chamado preço FOB (free on board), 
                que não inclui despesa de frete e seguro no preço. Mais informações: https://www.eia.gov/dnav/pet/TblDefs/pet_pri_spt_tbldef2.asp;</li>                
        </ul>
        <br>
        <hr style="border: none; height: 1px; background-color: #fb8500; margin-top: 10px; margin-bottom: 10px;">
    """, unsafe_allow_html=True)
        
    with tabs[1]:
        st.markdown(""" 
            <p style="text-indent: 2em; font-size: 20px; text-align: justify">
                A seguir, visualizaremos um gráfico de linha com marcações dos acontecimentos históricos, possibilitando o melhor entendimento da variação dos valores do barril de petróleo. 
        """, unsafe_allow_html=True)

        coluna1, coluna2 = st.columns([4,1])
        
        with coluna1:
            fig, ax = plt.subplots(figsize=(18, 6))
            sns.lineplot(data=df_price_hist, x='Data', y='Preço', ax=ax)
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y')) 
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            plt.axvspan('1990-06-01', '1990-11-01', color='#6a994e', alpha=0.5)
            plt.axvspan('2001-01-01', '2001-01-30', color='#0496ff', alpha=0.5)
            plt.axvspan('2007', '2009', color='#f8961e', alpha=0.5)
            plt.axvspan('2010-01-01', '2012-01-01', color='#f3722c', alpha=0.5)
            plt.axvspan('2020-01-01', '2022-03-01', color='#f94144', alpha=0.5)
            plt.title('Série histórica do preço do barril de petróleo Brent', fontsize = 20)
            plt.xlabel('Ano', fontsize = 20)
            plt.ylabel('Preço (US$)', fontsize = 20)
            st.pyplot(fig) 

        with coluna2:
            st.write(df_price_hist)
        
        st.markdown("""
            <ul>
                <li><p style="font-size: 20px; text-align: justify"><span style="font-weight: bold; color: #6a994e">(1990-1995) Guerra do Golfo: </span> O conflito da Guerra do Golfo, causada pela invasão do Kuwait pelo Iraque, interrompeu a produção e exportações da região, fazendo com que os preços do petróleo subissem. </li>
                <li><p style="font-size: 20px; text-align: justify"><span style="font-weight: bold; color: #0496ff">(2001) Atentados 11 de Setembro: </span> Com o ataque terrorista ao Estados Unidos da América, ocorreu o aumento temporário do petróleo devido ao receio de interrupções e instabilidade no fornecimento. Após algum tempo os valores voltaram a estabilizar.</li>
                <li><p style="font-size: 20px; text-align: justify"><span style="font-weight: bold; color: #f8961e">(2007-2009) Grande Recessão: </span> Agora o efeito deste evendo foi na desvalorização da commodity. Com a crise financeira global, a demanda pelo barril do petróleo caiu drasticamente, consequentemente seu valor caiu junto.</li>
                <li><p style="font-size: 20px; text-align: justify"><span style="font-weight: bold; color: #f3722c">(2010-2012) Primavera Árabe: </span> Problemas políticos no Oriente Médio e no Norte da África desencadearam uma instabilidade em países produtores de petróleo, incluindo a Libia. Dessa forma, novamente pelo temor mundial na interrupção de fornecimento de petróleo, os valores dispararam e ficaram altos.</li>
                <li><p style="font-size: 20px; text-align: justify"><span style="font-weight: bold; color: #f94144">(2020-2022) Pandemia COVID 19: </span> Com a crise sanitária ocorrida em 2020 pela pandemia, os fatores de isolamento e a redução de diversas atividades dependentes de petróleo fez com que essa matéria-prima ficasse muito barata pela baixa procura e até negativa no terceiro trimestre de 2020 com a crise de falta de lugar de armazenamento para o excedente de produção de petróleo.</li>
            <ul>
        """, unsafe_allow_html=True)             

def predicao():
    css = '''
        <style>
            .stTabs [data-baseweb="tab-list"] button {
                flex: 1;
                text-align: center;
                border-bottom: 3px solid transparent;
                padding-bottom: 15px;
            }
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
                border-bottom: 1px solid #fb8500;
            }
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size: 30px;
                color: #fb8500;
                font-weight: bold;
                margin: 0;
            }
        </style>
        '''
    st.markdown(css, unsafe_allow_html=True)

    df_price_hist = carregar_dataframe()
    analises = ['Modelagem','Forecast']
    tabs = st.tabs(analises)


    with tabs[0]:
        st.markdown("""            
        <p style="text-indent: 2em; font-size: 20px">
            O modelo de machine learning escolhido foi o XGBoost que é baseado em árvore de decisão e utiliza uma estrutura de gradient boosting. 
            Com a função TimeSeriesSplit, a base de dados original foi separada em quatro janelas de treino e teste, usando uma janela de teste de 90 dias. 
            A seguir segue a representação gráfica dos períodos utilizados para treino e teste:    
    """, unsafe_allow_html=True)
    
        color_pal = sns.color_palette()
        plt.style.use('fivethirtyeight')
        
        df_ml = df_price_hist[df_price_hist.index >= '2020-01-01'].copy()
        tss = TimeSeriesSplit(n_splits=4, test_size=90, gap=0)
        df_ml = df_ml.sort_index()

        fig, axs = plt.subplots(4, 1, figsize=(20, 10), sharex=True)

        fold = 0
        for train_idx, val_idx in tss.split(df_ml):
            train = df_ml.iloc[train_idx]
            test = df_ml.iloc[val_idx]
            train['Preço'].plot(ax=axs[fold],
                                label='Training Set',
                                title=f'Data Train/Test Split Fold {fold}')
            test['Preço'].plot(ax=axs[fold],
                                label='Test Set')
            axs[fold].axvline(test.index.min(), color='black', ls='--')
            fold += 1
        st.pyplot(fig)

        st.markdown("""            
        <p style="text-indent: 2em; font-size: 20px">
            Na etapa de feature engineering foram criadas variáveis qualificadoras de sazonalidade e valores períodos anteriores para auxiliar no modelo de machine learning. 
            Segue a tabela com os valores do dataset original e suas features:    
        """, unsafe_allow_html=True)

        df_ml = create_features(df_ml)
        st.write(df_ml)


        st.markdown("""            
        <p style="text-indent: 2em; font-size: 20px">
            Para o modelo de predição as features criadas possuem um grau de importância, a seguir segue um gráfico de barras com o ranking de importância de cada feature:    
        """, unsafe_allow_html=True)
        
        fold = 0
        preds = []
        scores = []
        for train_idx, val_idx in tss.split(df_ml):
            train = df_ml.iloc[train_idx]
            test = df_ml.iloc[val_idx]

            train = create_features(train)
            test = create_features(test)

            FEATURES = df_ml.columns.tolist()
            FEATURES.remove('Preço')
            TARGET = 'Preço'

            X_train = train[FEATURES]
            y_train = train[TARGET]

            X_test = test[FEATURES]
            y_test = test[TARGET]

            reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                                n_estimators=400,
                                early_stopping_rounds=50,
                                objective='reg:linear',
                                max_depth=3,
                                learning_rate=0.01)
            reg.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=100)

            y_pred = reg.predict(X_test)
            preds.append(y_pred)
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(score)
    
        fi = pd.DataFrame(data=reg.feature_importances_,
                        index=reg.feature_names_in_,
                        columns=['importance'])
        
        fi_sorted = fi.sort_values('importance')

        # Criar o gráfico utilizando matplotlib
        fig, ax = plt.subplots(figsize=(18, 6))
        fi_sorted.plot(kind='barh', ax=ax, title='Feature Importance')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Exibir o gráfico no Streamlit
        st.pyplot(fig)

        st.markdown("""            
        <p style="text-indent: 2em; font-size: 20px">
            Métrica de acurácia mean square error(MSE)</p>
        <ul>    
            <li><p style="font-size: 20px"><span style="font-weight: bold; color: #fb8500">MSE Score Geral: </span> 4.1064</li>
            <li><p style="font-size: 20px"><span style="font-weight: bold; color: #fb8500">MSE Score por janela de teste: </span>[4.53748402563118, 4.917437875586603, 3.1639861402296052, 3.806598383608038]</li>    
        <ul>
        """, unsafe_allow_html=True)

    with tabs[1]:
        df_price_hist = carregar_dataframe()
        
        st.markdown(""" 
            <p style="text-indent: 2em; font-size: 20px; text-align: justify">
               Com base na série histórica, o forecast para os próximos 5 dias flutuará entre os valores US$83,34 e US$84,58. 
               Considerando os valores dos 60 dias plotados, a previsão apresenta uma diferença significativa que pode culminar em prejuízo na utilização de estoques adquiridos em abril.   
        """, unsafe_allow_html=True)

        latest_date = df_price_hist.index.max()
        future_dates = pd.date_range(start=latest_date, periods=6, freq='B')
        future_df = pd.DataFrame({'Data': future_dates, 'Preço': 0})
        future_df = future_df.set_index('Data')
        df_price_hist = pd.concat([df_price_hist, future_df[1:]], ignore_index=False)
        df_price_hist.sort_index(inplace=True)

        df_pred = create_features(df_price_hist)
        df_pred['future'] = df_pred.apply(lambda x: 'futuro' if x['Preço'] == 0 else 'histórico', axis=1)
        df_pred['Preço'] = reg.predict(df_pred[FEATURES])
        
        df_last_60_days = df_pred.tail(60)

        fig = px.line(df_last_60_days, x= df_last_60_days.index, y='Preço', color='future', markers=True)
        
        fig.update_layout(
            title='Price History - Last 60 Days',
            xaxis_title='Date',
            yaxis_title='Price',
            )
        st.plotly_chart(fig)
          

#criação da página conclusão
def conclusao():
    st.markdown("""
     <h2 style="font-weight: bold; color: '#fb8500';">Conclusão</h2>
        <p style="text-indent: 2em; text-align: justify; font-size: 20px">
            A variação do preço do barril do petróleo leva em consideração diversas variáveis geopolíticas e econômicas. 
            Com a criação deste aplicativo para prever o comportamento do valor do petróleo em momentos futuros, apenas com a data e valor de uma base de dados e criando métricas para compor o treinamento,
            nota-se o quão complexo deve ser esse cálculo para retornar valores fidedignos. 
            Apesar do acesso às informações mundiais estar cada vez mais disponível e rápido, a construção de um modelo de predição para a análise futura do preço do petróleo aceitável e confiável é extremamente 
            mais complicado do que o apresentado. Mesmo para um modelo de predição simples, a previsão de valores para períodos mais próximos mostra-se mais acertiva.
            Em caso de eventos extremos, recomenda-se que um novo treinamento seja realizado com o modelo.
        """, unsafe_allow_html=True)
    
def referencia():
     st.markdown("""   
    <h2 style="font-weight: bold; color: '#fb8500';">Referências</h2>
    
    <ul>
        <li><p style="text-align: justify; font-size: 20px">https://warren.com.br/magazine/preco-do-petroleo/ acesso em: 29 de junho de 2024</li>
        <li><p style="text-align: justify; font-size: 20px">https://docs.streamlit.io/ acesso em: 2 de julho de 2024 </li>
        <li><p style="text-align: justify; font-size: 20px">RIBEIRO, Cássio. A oscilação do preço do petróleo: uma análise sobre o período entre 2010-2015. 2019. Disponível em: https://www.researchgate.net/profile/Cassio-Ribeiro/publication/330948314_A_oscilacao_do_preco_do_petroleo_uma_analise_sobre_o_periodo_entre_2010-2015/links/5d483236a6fdcc370a7ccbd4/A-oscilacao-do-preco-do-petroleo-uma-analise-sobre-o-periodo-entre-2010-2015.pdf. Acesso em: 10 jul. 2024. 3</li>
        <li><p style="text-align: justify; font-size: 20px">RAMOS, Júlia Fernandes. A oscilação do preço do petróleo: uma análise sobre o período entre 2010-2015. Rio de Janeiro, 2019. Disponível em: http://ftp.econ.puc-rio.br/uploads/adm/trabalhos/files/Julia_Fernandes_Ramos.pdf. Acesso em: 14 jul. 2024.</li>
        <li><p style="text-align: justify; font-size: 20px">FORTI, Maira Cristina Rebelato. Caracterização econômica da indústria do petróleo brasileira e estudo da influência das variáveis econômicas sob sua produção. São Carlos, 2013. Disponível em: https://www.cienciaseconomicas.ufscar.br/arquivos/acervo-monografias/monografias-2013/2013-1-maira-cristina-rebelato-forti-caracterizacao-economica-da-industria-do-petroleo-brasileira-e-estudo-da-influencia-das-variaveis-economicas-sob-sua-producao.pdf. Acesso em: 14 jul. 2024.</li>
        <li><p style="text-align: justify; font-size: 20px">YERGIN, Daniel. The prize: The epic quest for oil, money and power. New York: Free Press, 1991.</li>
        <li><p style="text-align: justify; font-size: 20px">ELLIS, Christopher; SINGH, Sanjaya. The financial crisis and the impact on the oil industry. Journal of Business & Economic Policy, v. 1, n. 2, p. 1-12, 2014.</li>
        <li><p style="text-align: justify; font-size: 20px">APOSTOLAKIS, Giorgos; BENTZEN, Thomas; LARSEN, Bert. The impact of COVID-19 on oil prices. Energy Policy, v. 147, 2020.</li>
    </ul>
 """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Análise do Preço do Petróleo Brent", layout="wide")
    #caminho_arquivo = 'petroleo.xlsx'
    #dados = carregar_dados(caminho_arquivo)

    with st.sidebar:
        selecionado = option_menu(
            menu_title="Menu Principal",  
            options=["Introdução", "Análise Exploratória", "Predição","Conclusão", "Referências"],  
            icons=["book", "database", "clock", "pencil", 'paperclip'],  
            menu_icon="list",  
            default_index=0,
            styles={"nav-link-selected": {"background-color": "#fb8500", "color": "white"} }
        )

    if selecionado == "Introdução":
        introducao()
    elif selecionado == "Análise Exploratória":
        analise()
    elif selecionado == "Predição":
        predicao()
    elif selecionado == "Conclusão":
        conclusao()
    elif selecionado == "Referências":
        referencia()

if __name__ == "__main__":
    main()
