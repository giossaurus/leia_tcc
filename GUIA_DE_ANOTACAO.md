# Guia de Anotação – Projeto LeIA (v3)

Este documento define as regras para a anotação da **Intenção Pedagógica** das perguntas do dataset.  
O objetivo é garantir **consistência** na rotulagem, permitindo treinar o Módulo Classificador (NLU) de forma confiável.

---

## Categorias de Intenção

### 1. Conceitual
- **Definição:** Pergunta que exige a compreensão de um conceito, definição ou ideia.  
- **Exemplos Positivos:**
  - “A incompatibilidade sanguínea está associada a que componente presente no plasma do receptor?”  
  - “Qual é a função das mitocôndrias?”
- **Exemplos Negativos:**
  - “Quantos diferentes cruzamentos sanguíneos podem ocorrer considerando ABO e Rh?” → Procedimental  
  - “Compare o papel das mitocôndrias e dos cloroplastos.” → Comparativo  

---

### 2. Procedimental
- **Definição:** Pergunta que exige cálculo, aplicação de regra ou execução de um passo a passo.  
- **Exemplos Positivos:**
  - “Quantos modelos de unhas diferentes podem ser combinados a partir das opções de cor, formato e acabamento?”  
  - “Calcule a área de um triângulo de base 8 cm e altura 5 cm.”
- **Exemplos Negativos:**
  - “Explique o que é área de um triângulo.” → Conceitual  
  - “Compare área e perímetro.” → Comparativo  

---

### 3. Análise de Exemplo
- **Definição:** Pergunta que apresenta um caso, trecho, gráfico ou situação concreta e pede interpretação com base teórica.  
- **Exemplos Positivos:**
  - Texto literário de Lima Barreto e pergunta: “Com base nesse trecho, o autor está…”  
  - Questão de História que mostra um cartaz de época e pede para identificar a ideologia representada.
- **Exemplos Negativos:**
  - “Defina ideologia.” → Conceitual  
  - “Compare diferentes ideologias políticas.” → Comparativo  

---

### 4. Comparativo
- **Definição:** Pergunta que pede identificar semelhanças/diferenças ou articular conceitos distintos.  
- **Exemplos Positivos:**
  - “A partir da leitura dos textos, qual conclusão pode ser extraída sobre a atuação das ‘mãos invisíveis’ do mercado durante a crise de 2007-2009?”  
  - “Qual a diferença entre clima tropical e equatorial?”
- **Exemplos Negativos:**
  - “Explique o que é clima tropical.” → Conceitual  
  - “Resolva o gráfico de temperaturas.” → Procedimental  

---

## Boas Práticas na Anotação
- Sempre leia o **texto de referência + enunciado completo** antes de rotular.  
- Pergunte-se: *o que o aluno precisa fazer para responder?*  
- Escolha **apenas uma categoria por questão**.  
- Em caso de dúvida, registre o exemplo para revisão posterior.  

---

**Versão:** v3 — setembro de 2025  
**Responsável:** Giovanni Della Déa  