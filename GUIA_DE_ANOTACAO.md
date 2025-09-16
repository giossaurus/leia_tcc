# Guia de Anotação para o Projeto LeIA

Este documento define as regras para a anotação da **intenção pedagógica** das perguntas do dataset `simulados_enem`. O objetivo é manter a consistência e a alta qualidade dos rótulos que treinarão o Módulo Classificador (NLU) do agente LeIA.

---

## Definição das Categorias de Intenção

### 1. Dúvida Conceitual
- **Definição:** O aluno quer entender o "o quê", o "porquê" ou a definição de um conceito, teoria, evento ou termo. A pergunta foca no entendimento teórico.
- **Palavras-chave comuns:** "O que é?", "Defina", "Explique", "Por que...", "Qual a função de...".
- **Exemplo 1 (História):** "O que foi a política do Café com Leite na República Velha?"
- **Exemplo 2 (Biologia):** "Qual a função das mitocôndrias na célula?"

### 2. Dúvida Procedimental
- **Definição:** O aluno quer saber "como fazer" algo. A pergunta foca em um método, um passo a passo, um cálculo ou a aplicação de uma fórmula.
- **Palavras-chave comuns:** "Como calcular?", "Como resolver?", "Qual a fórmula para?", "Como se faz?".
- **Exemplo 1 (Matemática):** "Como se calcula a área de um triângulo?"
- **Exemplo 2 (Química):** "Como fazer o balanceamento de uma equação química?"

### 3. Busca por Exemplo
- **Definição:** O aluno já tem uma ideia do conceito, mas pede uma aplicação prática ou um caso concreto para solidificar o entendimento.
- **Palavras-chave comuns:** "Me dê um exemplo de...", "Em que situação se aplica?", "Cite um caso...".
- **Exemplo 1 (Gramática):** "Pode me dar um exemplo de uma oração subordinada substantiva?"
- **Exemplo 2 (Física):** "Qual um exemplo de aplicação da Terceira Lei de Newton no dia a dia?"

### 4. Comparação
- **Definição:** O aluno pede para que sejam destacadas as semelhanças e/ou diferenças entre dois ou more conceitos, teorias, eventos, etc.
- **Palavras-chave comuns:** "Qual a diferença entre...", "Compare A e B", "...em relação a...".
- **Exemplo 1 (Geografia):** "Qual a diferença entre o clima tropical e o equatorial?"
- **Exemplo 2 (Filosofia):** "Compare o pensamento de Platão e Aristóteles sobre o conhecimento."

### 5. Tentativa de Cola
- **Definição:** O aluno não demonstra interesse em aprender o processo ou o conceito, mas sim em obter a resposta direta e final de uma questão específica, muitas vezes referenciando um número ou uma lista.
- **Palavras-chave comuns:** "Qual a resposta da questão 5?", "Me dá o gabarito da lista", "A resposta é a letra A?".
- **Exemplo 1:** "Qual a alternativa correta da questão sobre a Revolução Industrial?"
- **Exemplo 2:** "Me passa a resposta da 3 da prova de 2018."

---

### Regra de Ouro
Na dúvida entre duas categorias, escolheremos aquela que melhor representa a **necessidade primária** do aluno. Se uma pergunta conceitual pede um exemplo no final, ela ainda é primariamente uma **Dúvida Conceitual**. O objetivo é capturar a intenção principal para que o LeIA possa agir da forma mais pedagogicamente adequada.