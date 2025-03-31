import random
import logging
from typing import Dict, List, Tuple, Optional
import spacy
from dataclasses import dataclass
import asyncio
from datetime import datetime
import json


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("support_ai")


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Модель spaCy не найдена. Загрузка модели...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


@dataclass
class QueryResult:
    query: str
    intent: str
    confidence: float
    status: str
    message: str
    timestamp: datetime = datetime.now()
   
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "intent": self.intent,
            "confidence": self.confidence,
            "status": self.status,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }
   
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)




class KnowledgeBase:
    def __init__(self):
        self._data = {
            "billing": [
                "Счет выставляется в начале каждого месяца.",
                "Оплату можно произвести через наш сайт или мобильное приложение.",
                "Для получения счета на email, пожалуйста, обновите ваши контактные данные."
            ],
            "technical_issue": [
                "Попробуйте перезагрузить устройство.",
                "Проверьте соединение с интернетом.",
                "Обновите приложение до последней версии."
            ],
            "account": [
                "Вы можете изменить пароль в настройках аккаунта.",
                "Для удаления аккаунта обратитесь в службу поддержки.",
                "Информация о вашей подписке доступна в разделе 'Мой аккаунт'."
            ],
            "product_info": [
                "Наш премиум план включает дополнительные функции.",
                "Гарантия на продукт составляет 2 года.",
                "Доставка занимает 3-5 рабочих дней."
            ]
        }
       


        self._stats = {intent: 0 for intent in self._data}
   
    async def get_response(self, intent: str) -> Optional[str]:
        """Асинхронное получение ответа по намерению"""
        await asyncio.sleep(0.05)
       
        if intent in self._data:
            self._stats[intent] += 1
            return random.choice(self._data[intent])
        return None
   
    def get_stats(self) -> Dict[str, int]:
        """Получение статистики запросов"""
        return self._stats
   
    def add_entry(self, intent: str, response: str) -> bool:
        """Добавление нового ответа в базу знаний"""
        if intent in self._data:
            self._data[intent].append(response)
        else:
            self._data[intent] = [response]
            self._stats[intent] = 0
        return True


class IntentTrainingData:
    def __init__(self):
        self._data = {
            "billing": [
                "Когда придет мой счет?",
                "Как я могу оплатить?",
                "Не получил счет на email",
                "Вопрос по оплате",
                "Когда списывают деньги?",
                "Почему сумма больше обычной?",
            ],
            "technical_issue": [
                "Приложение не работает",
                "Не могу войти в систему",
                "Ошибка при загрузке",
                "Сайт не открывается",
                "Выдает ошибку при запуске",
                "Не загружается страница",
            ],
            "account": [
                "Как изменить пароль?",
                "Хочу удалить аккаунт",
                "Где посмотреть информацию о подписке?",
                "Настройки аккаунта",
                "Как поменять email?",
                "Где мои личные данные?",
            ],
            "product_info": [
                "Что входит в премиум план?",
                "Сколько длится гарантия?",
                "Когда будет доставка?",
                "Информация о продукте",
                "Какие функции доступны?",
                "Сравнение тарифов",
            ]
        }
   
    def get_training_data(self) -> Dict[str, List[str]]:
        return self._data
   
    def add_example(self, intent: str, example: str) -> None:
        """Добавление нового примера для обучения"""
        if intent in self._data:
            self._data[intent].append(example)
        else:
            self._data[intent] = [example]


class EscalationService:
    def __init__(self):
        self.available_operators = ["Оператор 1", "Оператор 2", "Оператор 3"]
        self.escalation_queue = []
   
    async def escalate_query(self, query: str, intent: str, confidence: float) -> str:
        """Эскалация запроса человеку-оператору"""
        await asyncio.sleep(0.1) # work imitation
       
        escalation_id = f"ESC-{len(self.escalation_queue) + 1}"
        operator = random.choice(self.available_operators)
       
        self.escalation_queue.append({
            "id": escalation_id,
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "assigned_to": operator,
            "timestamp": datetime.now().isoformat()
        })
       
        return f"Ваш запрос передан оператору {operator}. Идентификатор обращения: {escalation_id}"


class CustomerSupportAI:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.training_data = IntentTrainingData()
        self.escalation_service = EscalationService()
       
        self.intent_vectors = self._prepare_intent_vectors()
       
        self.query_cache = {}
       
        self.processed_queries = 0
        self.successful_responses = 0
        self.escalated_queries = 0
       
    def _prepare_intent_vectors(self) -> Dict[str, List]:
        """Подготовка векторных представлений обучающих фраз для каждого намерения"""
        intent_vectors = {}
        for intent, examples in self.training_data.get_training_data().items():
            intent_vectors[intent] = [nlp(example) for example in examples]
        return intent_vectors
   
    def update_intent_models(self) -> None:
        """Обновление моделей намерений после добавления новых примеров"""
        self.intent_vectors = self._prepare_intent_vectors()
        logger.info("Модели намерений обновлены")
   
    def identify_intent(self, query: str) -> Tuple[str, float]:
        """Определение намерения пользователя с использованием NLP"""
        query_doc = nlp(query)
       
        best_score = 0
        identified_intent = "unknown"
       
        for intent, examples in self.intent_vectors.items():
            for example in examples:
                similarity = query_doc.similarity(example)
                if similarity > best_score:
                    best_score = similarity
                    identified_intent = intent
       
        return identified_intent, best_score
   
    async def query_knowledge_base(self, intent: str) -> Optional[str]:
        """Получение ответа из базы знаний по определенному намерению"""
        return await self.knowledge_base.get_response(intent)
   
    async def process_query(self, customer_query: str, confidence_threshold: float = 0.6) -> QueryResult:
        """Обработка запроса клиента"""
        self.processed_queries += 1
       
        if customer_query in self.query_cache:
            logger.info(f"Найден ответ в кэше для запроса: {customer_query}")
            return self.query_cache[customer_query]
       
        intent, confidence = self.identify_intent(customer_query)
        logger.info(f"Определено намерение: {intent} с уверенностью {confidence:.2f}")
       
        if confidence < confidence_threshold:
            self.escalated_queries += 1
            escalation_message = await self.escalation_service.escalate_query(
                customer_query, intent, confidence
            )
           
            result = QueryResult(
                query=customer_query,
                intent=intent,
                confidence=confidence,
                status="escalated",
                message=f"Извините, я не могу полностью понять ваш вопрос. {escalation_message}"
            )
        else:
            answer = await self.query_knowledge_base(intent)
           
            if answer:
                self.successful_responses += 1
                result = QueryResult(
                    query=customer_query,
                    intent=intent,
                    confidence=confidence,
                    status="answered",
                    message=answer
                )
            else:
                self.escalated_queries += 1
                escalation_message = await self.escalation_service.escalate_query(
                    customer_query, intent, confidence
                )
               
                result = QueryResult(
                    query=customer_query,
                    intent=intent,
                    confidence=confidence,
                    status="escalated",
                    message=f"У меня нет подходящего ответа на ваш вопрос. {escalation_message}"
                )


        self.query_cache[customer_query] = result
       
        return result
   
    def get_stats(self) -> Dict:
        """Получение статистики работы системы"""
        return {
            "processed_queries": self.processed_queries,
            "successful_responses": self.successful_responses,
            "escalated_queries": self.escalated_queries,
            "success_rate": self.successful_responses / max(1, self.processed_queries),
            "knowledge_base_stats": self.knowledge_base.get_stats()
        }


async def main():
    support_ai = CustomerSupportAI()
   
    test_queries = [
        "Когда мне придет счет за этот месяц?",
        "Мое приложение выдает ошибку при запуске",
        "Как мне поменять пароль от аккаунта?",
        "Что входит в премиум подписку?",
        "Абракадабра что это такое вообще",
    ]
   
    print("Система поддержки клиентов с ИИ")
    print("-" * 50)
   
    tasks = [support_ai.process_query(query) for query in test_queries]
    results = await asyncio.gather(*tasks)
   
    for query, result in zip(test_queries, results):
        print(f"\nЗапрос клиента: {query}")
        print(f"Определенное намерение: {result.intent} (уверенность: {result.confidence:.2f})")
        print(f"Статус: {result.status}")
        print(f"Ответ: {result.message}")
   
    print("\nДобавление нового примера и ответа в систему...")
    support_ai.training_data.add_example("product_info", "Какие есть планы подписки?")
    support_ai.knowledge_base.add_entry("product_info", "У нас есть базовый, стандартный и премиум планы подписки.")
    support_ai.update_intent_models()
   
    print("\nСтатистика работы системы:")
    stats = support_ai.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
