"""
FreireAgent - Agente Pedagógico Dinâmico Multi-Modelo

Implementa a persona freiriana de forma agnóstica ao modelo NLG.
Suporta qualquer modelo LLM que o ModelLoader consiga carregar.

Baseado no agent_langchain_final.py, mas refatorado para:
1. Aceitar qualquer modelo NLG no construtor
2. Memória explícita manual (não depende de LangChain Memory)
3. Branching lógico (scaffolding detection)
4. Framework ReAct (Reason + Act) com ferramentas
5. Prompts padronizados via FreirePromptBuilder
"""

import sys
from pathlib import Path
import time
import datetime
from typing import Dict, List, Optional, Any
import logging

# Configurar imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.classifier.classifier import NLUClassifier
from src.orchestrator.core.prompts.prompt_builder import FreirePromptBuilder

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_community.tools import DuckDuckGoSearchRun

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Gerenciador de memória explícita da conversa.
    Armazena o histórico de mensagens e fornece métodos de acesso.
    """

    def __init__(self, max_history: int = 10):
        """
        Args:
            max_history: Número máximo de turnos a manter no histórico
        """
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_user_message(self, content: str) -> None:
        """Adiciona mensagem do usuário ao histórico."""
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """Adiciona mensagem da assistente ao histórico."""
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()

    def get_recent_history(self, n: int = 3) -> List[Dict[str, str]]:
        """
        Retorna os últimos N turnos da conversa.

        Args:
            n: Número de turnos a retornar

        Returns:
            Lista de dicionários com role e content
        """
        return self.messages[-(n * 2):] if self.messages else []

    def get_history_as_text(self, n: int = 3) -> str:
        """
        Retorna o histórico formatado como texto.

        Args:
            n: Número de turnos a incluir

        Returns:
            String formatada com o histórico
        """
        recent = self.get_recent_history(n)
        if not recent:
            return ""

        lines = []
        for msg in recent:
            role_name = "Aluno" if msg["role"] == "user" else "LeIA"
            lines.append(f"{role_name}: {msg['content']}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Limpa todo o histórico."""
        self.messages.clear()

    def _trim_history(self) -> None:
        """Remove mensagens antigas se exceder max_history."""
        if len(self.messages) > self.max_history * 2:
            # Mantém apenas as últimas max_history trocas (user + assistant = 2 msgs)
            self.messages = self.messages[-(self.max_history * 2):]


class FreireAgent:
    """
    Agente pedagógico baseado na filosofia de Paulo Freire.
    Agnóstico ao modelo NLG - funciona com qualquer LLM.
    """

    def __init__(
        self,
        nlu_classifier: NLUClassifier,
        nlg_llm: HuggingFacePipeline,
        model_name: str,
        enable_tools: bool = False,
        session_id: Optional[str] = None
    ):
        """
        Inicializa o FreireAgent.

        Args:
            nlu_classifier: Classificador de intenções de perguntas (NLU)
            nlg_llm: Pipeline LangChain do modelo NLG
            model_name: Nome/ID do modelo NLG (para logging)
            enable_tools: Se True, habilita ferramentas (ReAct)
            session_id: ID da sessão (opcional)
        """
        logger.info(f"Inicializando FreireAgent com modelo: {model_name}")

        self.nlu = nlu_classifier
        self.llm = nlg_llm
        self.model_name = model_name
        self.session_id = session_id or f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Memória explícita
        self.memory = ConversationMemory(max_history=10)

        # Prompts padronizados
        self.prompt_builder = FreirePromptBuilder

        # Ferramentas (ReAct framework)
        self.enable_tools = enable_tools
        self.tools = {}
        if enable_tools:
            self._initialize_tools()

        # Chains do LangChain
        self._build_chains()

        logger.info("FreireAgent inicializado com sucesso")

    def _initialize_tools(self) -> None:
        """Inicializa as ferramentas disponíveis (ReAct framework)."""
        try:
            # Ferramenta de busca na web
            self.tools["search"] = DuckDuckGoSearchRun()
            logger.info("Ferramenta de busca (DuckDuckGo) inicializada")
        except Exception as e:
            logger.warning(f"Erro ao inicializar ferramentas: {str(e)}")
            self.enable_tools = False

    def _build_chains(self) -> None:
        """Constrói as chains do LangChain."""
        # Chain de condensação de pergunta (com contexto)
        condense_prompt = self.prompt_builder.get_condense_question_prompt()
        self.condense_chain = condense_prompt | self.llm | StrOutputParser()

        # Chain de scaffolding (suporte adicional)
        scaffolding_prompt = self.prompt_builder.get_scaffolding_prompt()
        self.scaffolding_chain = scaffolding_prompt | self.llm | StrOutputParser()

        logger.info("Chains construídas com sucesso")

    def _condense_question(self, question: str) -> str:
        """
        Condensa a pergunta com base no histórico da conversa.

        Args:
            question: Pergunta atual do usuário

        Returns:
            Pergunta condensada (independente de contexto)
        """
        chat_history = self.memory.get_history_as_text(n=3)

        if not chat_history:
            # Sem histórico, retorna a pergunta original
            return question

        try:
            condensed = self.condense_chain.invoke({
                "chat_history": chat_history,
                "question": question
            })
            return condensed.strip()
        except Exception as e:
            logger.warning(f"Erro ao condensar pergunta: {str(e)}. Usando pergunta original.")
            return question

    def _execute_scaffolding(self) -> str:
        """
        Executa a rota de scaffolding (suporte adicional).

        Returns:
            Resposta de scaffolding
        """
        logger.info("Executando rota de SCAFFOLDING")

        chat_history = self.memory.get_history_as_text(n=5)

        try:
            response = self.scaffolding_chain.invoke({
                "chat_history": chat_history
            })
            return response.strip()
        except Exception as e:
            logger.error(f"Erro ao executar scaffolding: {str(e)}")
            # Fallback manual
            return "Vejo que você está com dificuldade. Vamos voltar um passo: o que você já tentou até agora? Me conta um pouco do seu raciocínio."

    def _execute_standard_response(self, intent: str, question: str) -> str:
        """
        Executa a rota padrão (baseada na intenção NLU).

        Args:
            intent: Intenção classificada pelo NLU
            question: Pergunta (já condensada)

        Returns:
            Resposta pedagógica
        """
        logger.info(f"Executando rota PADRÃO (Intenção: {intent})")

        # Obter prompt da intenção
        intent_prompt = self.prompt_builder.get_intent_prompt(intent)

        # Criar chain
        chain = intent_prompt | self.llm | StrOutputParser()

        try:
            response = chain.invoke({"question": question})
            return response.strip()
        except Exception as e:
            logger.error(f"Erro ao executar resposta padrão: {str(e)}")
            # Fallback manual
            return f"Interessante pergunta! Me conta: o que você já sabe sobre isso? Qual sua primeira impressão?"

    def _classify_intent(self, user_input: str) -> tuple[str, float]:
        """
        Classifica a intenção do usuário usando o NLU.

        Args:
            user_input: Mensagem do usuário

        Returns:
            Tupla (intent, confidence)
        """
        nlu_result = self.nlu.predict(user_input)
        intent = nlu_result['label']
        nlu_confidence = nlu_result['confidence']
        logger.info(f"Intenção NLU: {intent} (confiança: {nlu_confidence:.2%})")
        return intent, nlu_confidence

    def _route_and_execute(
        self,
        user_input: str,
        condensed_question: str,
        intent: str
    ) -> tuple[str, str]:
        """
        Roteia para a estratégia apropriada e executa a resposta.

        Args:
            user_input: Mensagem original do usuário
            condensed_question: Pergunta condensada
            intent: Intenção classificada

        Returns:
            Tupla (response, agent_trace)
        """
        # Detectar necessidade de scaffolding
        if self.prompt_builder.detect_scaffolding_trigger(user_input):
            agent_trace = "EXECUTED_SCAFFOLDING"
            response = self._execute_scaffolding()
            return response, agent_trace

        # Detectar necessidade de ferramenta (ReAct)
        if self.enable_tools and self._should_use_tool(condensed_question):
            agent_trace = "EXECUTED_REACT_TOOL"
            react_result = self._execute_react_tool(condensed_question)

            if react_result["success"]:
                response = react_result["result"]
                agent_trace = react_result["trace"]
            else:
                # Fallback para rota padrão se ReAct falhar
                logger.warning("ReAct falhou, usando rota padrão")
                agent_trace = "EXECUTED_STANDARD (ReAct fallback)"
                response = self._execute_standard_response(intent, condensed_question)

            return response, agent_trace

        # Rota padrão (baseada na intenção)
        agent_trace = f"EXECUTED_STANDARD ({intent})"
        response = self._execute_standard_response(intent, condensed_question)
        return response, agent_trace

    def _build_result(
        self,
        user_input: str,
        condensed_question: str,
        intent: str,
        nlu_confidence: float,
        agent_trace: str,
        response: str,
        latency_ms: float
    ) -> Dict[str, Any]:
        """
        Constrói o dicionário de resultado completo para logging.

        Args:
            user_input: Mensagem original do usuário
            condensed_question: Pergunta condensada
            intent: Intenção classificada
            nlu_confidence: Confiança da classificação NLU
            agent_trace: Trace da execução
            response: Resposta gerada
            latency_ms: Latência em milissegundos

        Returns:
            Dicionário com todos os metadados da interação
        """
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": self.session_id,
            "model_name": self.model_name,
            "user_input": user_input,
            "condensed_question": condensed_question,
            "nlu_label": intent,
            "nlu_confidence": float(nlu_confidence),
            "agent_trace": agent_trace,
            "response": response,
            "latency_ms": latency_ms,
            "chat_history": self.memory.get_recent_history(n=3),
        }

    def _execute_react_tool(self, question: str) -> Dict[str, Any]:
        """
        Executa o framework ReAct (Reason + Act) com ferramentas.

        NOTA: Implementação simplificada. O ReAct completo requer um loop de
        reasoning mais sofisticado (verificar se o Llama 8B consegue usar ferramentas
        sem entrar em loop, ao contrário do Gemma).

        Args:
            question: Pergunta que requer busca externa

        Returns:
            Dicionário com resultado da ferramenta e trace
        """
        logger.info("Executando framework REACT (ferramenta de busca)")

        if not self.enable_tools or "search" not in self.tools:
            return {
                "success": False,
                "trace": "REACT_DISABLED",
                "result": None
            }

        try:
            # Buscar informação externa
            search_result = self.tools["search"].run(question)

            # Criar prompt que incorpora o resultado da busca
            react_prompt = f"""**CONTEXTO DA BUSCA:**
{search_result}

**INSTRUÇÕES:**
Use o contexto acima para guiar o aluno com uma pergunta. NUNCA copie informações diretamente.
Transforme os dados encontrados em uma pergunta-guia que ajude o aluno a descobrir a resposta.

**Pergunta do Aluno:** {question}

**Sua Resposta (pergunta-guia baseada no contexto):"""

            # Gerar resposta com contexto
            response = self.llm.invoke(react_prompt)

            return {
                "success": True,
                "trace": "REACT_SUCCESS",
                "result": response.strip(),
                "search_context": search_result[:200]  # Primeiros 200 chars para log
            }

        except Exception as e:
            logger.error(f"Erro no ReAct: {str(e)}")
            return {
                "success": False,
                "trace": "REACT_ERROR",
                "result": None,
                "error": str(e)
            }

    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Processa uma mensagem do usuário e retorna a resposta.

        Workflow:
        1. Classificar intenção (NLU)
        2. Condensar pergunta (se houver histórico)
        3. Rotear e executar resposta (scaffolding, ReAct, ou padrão)
        4. Atualizar memória
        5. Retornar resultado com metadados

        Args:
            user_input: Mensagem do usuário

        Returns:
            Dicionário com resposta e metadados para logging
        """
        start_time = time.time()
        logger.info(f"\n[ALUNO]: {user_input}")

        # 1. Classificar intenção (NLU)
        intent, nlu_confidence = self._classify_intent(user_input)

        # 2. Condensar pergunta (se houver histórico)
        condensed_question = self._condense_question(user_input)
        if condensed_question != user_input:
            logger.info(f"Pergunta condensada: {condensed_question}")

        # 3. Rotear e executar resposta
        response, agent_trace = self._route_and_execute(
            user_input, condensed_question, intent
        )

        # 4. Atualizar memória explícita
        self.memory.add_user_message(user_input)
        self.memory.add_assistant_message(response)

        # 5. Calcular latência e preparar resultado
        latency_ms = round((time.time() - start_time) * 1000, 2)
        result = self._build_result(
            user_input, condensed_question, intent, nlu_confidence,
            agent_trace, response, latency_ms
        )

        logger.info(f"[LEIA]: {response}")
        logger.info(f"Latência: {latency_ms}ms | Trace: {agent_trace}")

        return result

    def _should_use_tool(self, question: str) -> bool:
        """
        Determina se a pergunta requer uso de ferramenta externa (busca).

        HEURÍSTICA SIMPLES: Detecta perguntas sobre eventos recentes, dados específicos, etc.

        Args:
            question: Pergunta a analisar

        Returns:
            True se deve usar ferramenta, False caso contrário
        """
        # Keywords que indicam necessidade de busca externa
        search_keywords = [
            "quando aconteceu",
            "em que ano",
            "data de",
            "onde fica",
            "localização",
            "notícia",
            "recente",
            "atual",
            "hoje",
            "agora",
        ]

        question_lower = question.lower()
        return any(keyword in question_lower for keyword in search_keywords)

    def get_memory_state(self) -> Dict[str, Any]:
        """
        Retorna o estado atual da memória.

        Returns:
            Dicionário com informações da memória
        """
        return {
            "session_id": self.session_id,
            "num_messages": len(self.memory.messages),
            "recent_history": self.memory.get_recent_history(n=3),
        }

    def reset_memory(self) -> None:
        """Limpa a memória da conversa."""
        self.memory.clear()
        logger.info("Memória resetada")


# --- Bloco de Teste ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("\n=== Teste do FreireAgent ===\n")
    print("NOTA: Este teste requer os módulos NLU e NLG carregados.")
    print("Execute experiment_runner.py para testes completos.\n")

    # Importar loaders
    from src.orchestrator.core.loaders.nlu_loader import load_nlu_model
    from src.orchestrator.core.loaders.model_loader import load_nlg_model

    try:
        # Carregar NLU
        print("Carregando NLU...")
        nlu = load_nlu_model()

        # Carregar NLG (Gemma para teste rápido)
        print("Carregando NLG (Gemma 3-1B)...")
        llm, tokenizer, device_info = load_nlg_model(
            "google/gemma-3-1b-it",
            use_quantization=True,
            quantization_bits=4
        )
        print(f"Modelo carregado em {device_info}")

        # Criar agente
        print("\nCriando FreireAgent...")
        agent = FreireAgent(
            nlu_classifier=nlu,
            nlg_llm=llm,
            model_name="gemma-3-1b-it",
            enable_tools=False  # Desabilitar ferramentas para teste rápido
        )

        # Teste de conversa
        print("\n=== Iniciando Teste de Conversa ===\n")

        test_interactions = [
            "O que é fotossíntese?",
            "Como eu calculo a área de um círculo?",
            "não sei",
        ]

        for user_msg in test_interactions:
            result = agent.chat(user_msg)
            print(f"\n[ALUNO]: {user_msg}")
            print(f"[LEIA]: {result['response']}")
            print(f"  (Trace: {result['agent_trace']}, Latência: {result['latency_ms']}ms)")

        print("\n✓ Teste concluído!")

    except Exception as e:
        print(f"\n✗ Erro durante o teste: {str(e)}")
        import traceback
        traceback.print_exc()
